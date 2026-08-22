"""Stage B unconditional shared-SIREN baseline training."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn

from mcar.data import Normalization
from mcar.evaluation.residual_metrics import solid_angle_weighted_residual_metrics
from mcar.models.siren import Siren, SirenConfig
from mcar.paths import project_root
from mcar.q26_condition import normalize_split
from mcar.training.train_film_siren import (
    coordinate_block,
    file_sha256,
    git_state,
    read_common_grid,
    set_seed,
    split_subject_paths,
)
from mcar.training.train_siren import frequency_coordinates


@dataclass(frozen=True)
class SharedSubject:
    subject_id: int
    subject_label: str
    split: str
    path: Path


@dataclass(frozen=True)
class ValidationResult:
    aggregate_mae_db: float
    aggregate_rmse_db: float
    median_mae_db: float
    std_mae_db: float
    minimum_mae_db: float
    maximum_mae_db: float
    mca_zero_residual_mae_db: float
    per_subject: list[dict[str, float | int | str]]


def subjects_from_paths(
    rows: list[tuple[int, str, Path]], split: str
) -> list[SharedSubject]:
    return [SharedSubject(subject_id, label, split, path) for subject_id, label, path in rows]


def validate_identity(subject: SharedSubject, handle: h5py.File) -> None:
    actual_id = int(np.asarray(handle.attrs["subject_id"]).item())
    actual_split = normalize_split(handle.attrs["split"])
    if actual_id != subject.subject_id or actual_split != normalize_split(subject.split):
        raise ValueError(f"Subject identity mismatch for {subject.path}")


@torch.no_grad()
def predict_subject(
    model: Siren,
    directions: np.ndarray,
    frequency_coordinate: torch.Tensor,
    direction_indices: np.ndarray,
    device: torch.device,
    directions_per_block: int,
) -> np.ndarray:
    frequency_count = int(frequency_coordinate.shape[0])
    prediction = np.empty((2, direction_indices.size, frequency_count), dtype=np.float32)
    for start in range(0, direction_indices.size, directions_per_block):
        stop = min(direction_indices.size, start + directions_per_block)
        selected = direction_indices[start:stop]
        xyz = torch.from_numpy(directions[selected, 2:5]).to(device)
        query = coordinate_block(xyz, frequency_coordinate)
        normalized = model(query)
        prediction[:, start:stop, :] = (
            normalized.reshape(stop - start, frequency_count, 2)
            .permute(2, 0, 1)
            .float()
            .cpu()
            .numpy()
        )
    return prediction


@torch.no_grad()
def evaluate_validation(
    model: Siren,
    subjects: list[SharedSubject],
    directions: np.ndarray,
    frequency_coordinate: torch.Tensor,
    interpolation_indices: np.ndarray,
    direction_weights: np.ndarray,
    target_mean: float,
    target_std: float,
    device: torch.device,
    directions_per_block: int,
) -> ValidationResult:
    model.eval()
    rows: list[dict[str, float | int | str]] = []
    maes: list[float] = []
    rmses: list[float] = []
    baselines: list[float] = []
    selected_weights = direction_weights[interpolation_indices]
    selected_mask = np.ones(interpolation_indices.size, dtype=bool)
    for subject in subjects:
        normalized_prediction = predict_subject(
            model, directions, frequency_coordinate, interpolation_indices, device, directions_per_block
        )
        prediction_db = normalized_prediction * target_std + target_mean
        with h5py.File(subject.path, "r") as handle:
            validate_identity(subject, handle)
            target_db = np.asarray(
                handle["target_residual_db"][:, interpolation_indices, :], dtype=np.float32
            )
        metrics = solid_angle_weighted_residual_metrics(
            prediction_db, target_db, selected_weights, selected_mask
        )
        baseline = solid_angle_weighted_residual_metrics(
            np.zeros_like(target_db), target_db, selected_weights, selected_mask
        )
        maes.append(metrics.residual_mae_db)
        rmses.append(metrics.residual_rmse_db)
        baselines.append(baseline.residual_mae_db)
        rows.append(
            {
                "subject_id": subject.subject_id,
                "subject_label": subject.subject_label,
                "weighted_mae_db": metrics.residual_mae_db,
                "weighted_rmse_db": metrics.residual_rmse_db,
                "mca_zero_residual_weighted_mae_db": baseline.residual_mae_db,
            }
        )
    values = np.asarray(maes, dtype=np.float64)
    return ValidationResult(
        aggregate_mae_db=float(np.mean(values)),
        aggregate_rmse_db=float(np.mean(rmses)),
        median_mae_db=float(np.median(values)),
        std_mae_db=float(np.std(values, ddof=0)),
        minimum_mae_db=float(np.min(values)),
        maximum_mae_db=float(np.max(values)),
        mca_zero_residual_mae_db=float(np.mean(baselines)),
        per_subject=rows,
    )


def checkpoint_payload(
    model: Siren,
    optimizer: torch.optim.Optimizer,
    configuration: dict[str, Any],
    cycle: int,
    metrics: ValidationResult,
    normalization: Normalization,
    normalization_path: Path,
) -> dict[str, Any]:
    return {
        "cycle": cycle,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "siren_configuration": asdict(model.configuration),
        "experiment_configuration": configuration,
        "validation_metrics": asdict(metrics),
        "output_unit": "normalized_residual",
        "residual_db_conversion": {
            "target_mean": normalization.target_mean,
            "target_std": normalization.target_std,
            "normalization_path": str(normalization_path),
            "normalization_sha256": file_sha256(normalization_path),
        },
        "condition_inputs_read": 0,
    }


def run(configuration: dict[str, Any], root: Path, config_path: Path) -> None:
    seed = int(configuration["seed"])
    set_seed(seed)
    if bool(configuration.get("require_cuda", True)) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a formal Stage B baseline run")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = git_state(root)
    if bool(configuration.get("require_clean_git", True)) and state["dirty"]:
        raise RuntimeError("Formal Stage B runs require a clean Git worktree")
    forbidden = {"q26_csv", "q26_normalization", "condition_encoder"}
    if forbidden.intersection(configuration):
        raise ValueError("Unconditional config contains condition inputs")

    dataset_root = (root / str(configuration["dataset_root"])).resolve()
    split_csv = (root / str(configuration["subject_split_csv"])).resolve()
    target_normalization_path = dataset_root / "training_statistics.json"
    target_normalization = Normalization.from_json(target_normalization_path)
    train_subjects = subjects_from_paths(
        split_subject_paths(dataset_root, split_csv, "train"), "train"
    )
    validation_subjects = subjects_from_paths(
        split_subject_paths(dataset_root, split_csv, "val"), "val"
    )
    directions, frequency, interpolation_mask, direction_weights = read_common_grid(
        train_subjects[0].path
    )
    interpolation_indices = np.flatnonzero(interpolation_mask)
    mapping = configuration["frequency_mapping"]
    frequency_coordinate = torch.from_numpy(
        frequency_coordinates(
            frequency,
            str(mapping["mode"]),
            float(mapping["frequency_minimum_hz"]),
            float(mapping["frequency_maximum_hz"]),
        )
    ).to(device)
    model = Siren(SirenConfig(**configuration["model"])).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(configuration["optimizer"]["learning_rate"]),
        weight_decay=float(configuration["optimizer"]["weight_decay"]),
    )
    cycles = int(configuration["cycles"])
    validation_interval = int(configuration["validation_interval_cycles"])
    directions_per_step = int(configuration["directions_per_step"])
    validation_block = int(configuration["validation_directions_per_block"])
    gradient_clip = float(configuration["gradient_clip"])
    if cycles < 1 or validation_interval < 1 or cycles % validation_interval != 0:
        raise ValueError("cycles must be positive and divisible by validation interval")
    if directions_per_step < 1 or directions_per_step > interpolation_indices.size:
        raise ValueError("Invalid directions_per_step")

    output_dir = root / "artifacts" / "training" / str(configuration["run_name"])
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    provenance = {
        "configuration": configuration,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "git": state,
        "split_csv_sha256": file_sha256(split_csv),
        "target_normalization_sha256": file_sha256(target_normalization_path),
        "train_subject_ids": [item.subject_id for item in train_subjects],
        "validation_subject_ids": [item.subject_id for item in validation_subjects],
        "train_subject_count": len(train_subjects),
        "validation_subject_count": len(validation_subjects),
        "condition_inputs_read": 0,
        "test_subjects_read": 0,
        "device": str(device),
    }
    (output_dir / "configuration.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    rng = np.random.default_rng(seed)
    history: list[dict[str, float | int]] = []
    ledger: list[dict[str, Any]] = []
    best_mae = float("inf")
    best_cycle = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for cycle in range(1, cycles + 1):
        model.train()
        losses: list[float] = []
        gradient_norms: list[float] = []
        cycle_started = time.perf_counter()
        for subject_index in rng.permutation(len(train_subjects)):
            subject = train_subjects[int(subject_index)]
            selected = np.sort(
                rng.choice(interpolation_indices, size=directions_per_step, replace=False)
            )
            with h5py.File(subject.path, "r") as handle:
                validate_identity(subject, handle)
                target_db = np.asarray(
                    handle["target_residual_db"][:, selected, :], dtype=np.float32
                )
            xyz = torch.from_numpy(directions[selected, 2:5]).to(device)
            query = coordinate_block(xyz, frequency_coordinate)
            target = torch.from_numpy(target_db).to(device)
            target = (
                (target - target_normalization.target_mean) / target_normalization.target_std
            ).permute(1, 2, 0).reshape(-1, 2)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(query)
            loss = nn.functional.mse_loss(prediction.float(), target.float())
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"Non-finite loss at cycle {cycle}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            if not bool(torch.isfinite(gradient_norm).item()):
                raise FloatingPointError(f"Non-finite gradient at cycle {cycle}")
            optimizer.step()
            losses.append(float(loss.detach().item()))
            gradient_norms.append(float(gradient_norm.detach().item()))

        record: dict[str, float | int] = {
            "cycle": cycle,
            "optimizer_steps_completed": cycle * len(train_subjects),
            "train_mse_normalized": float(np.mean(losses)),
            "gradient_norm_mean": float(np.mean(gradient_norms)),
            "gradient_norm_max": float(np.max(gradient_norms)),
            "cycle_elapsed_seconds": time.perf_counter() - cycle_started,
        }
        if cycle % validation_interval == 0:
            validation_started = time.perf_counter()
            result = evaluate_validation(
                model,
                validation_subjects,
                directions,
                frequency_coordinate,
                interpolation_indices,
                direction_weights,
                target_normalization.target_mean,
                target_normalization.target_std,
                device,
                validation_block,
            )
            record["validation_weighted_mae_db"] = result.aggregate_mae_db
            record["validation_weighted_rmse_db"] = result.aggregate_rmse_db
            ledger.append(
                {
                    "cycle": cycle,
                    "metric": "interpolation_solid_angle_weighted_residual_mae_db",
                    "aggregate": asdict(result),
                    "elapsed_seconds": time.perf_counter() - validation_started,
                    "condition_inputs_read": 0,
                    "test_subjects_read": 0,
                }
            )
            if result.aggregate_mae_db < best_mae:
                best_mae = result.aggregate_mae_db
                best_cycle = cycle
                torch.save(
                    checkpoint_payload(
                        model,
                        optimizer,
                        configuration,
                        cycle,
                        result,
                        target_normalization,
                        target_normalization_path,
                    ),
                    output_dir / "best.pt",
                )
                with (output_dir / "best_validation_per_subject.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=result.per_subject[0].keys())
                    writer.writeheader()
                    writer.writerows(result.per_subject)
            model.train()
        history.append(record)
        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = sorted({key for item in history for key in item})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        (output_dir / "validation_ledger.json").write_text(
            json.dumps(ledger, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"cycle={cycle:03d} mse={record['train_mse_normalized']:.6f} "
            + (
                f"val_mae={record['validation_weighted_mae_db']:.4f}dB "
                if "validation_weighted_mae_db" in record
                else ""
            )
            + f"time={record['cycle_elapsed_seconds']:.1f}s"
        )

    if best_cycle == 0:
        raise AssertionError("No validation checkpoint was produced")
    last_result = ValidationResult(**ledger[-1]["aggregate"])
    torch.save(
        checkpoint_payload(
            model,
            optimizer,
            configuration,
            cycles,
            last_result,
            target_normalization,
            target_normalization_path,
        ),
        output_dir / "last.pt",
    )
    decision = "RETEST" if best_cycle == cycles else "KEEP"
    report = {
        "status": "completed",
        "run_name": configuration["run_name"],
        "seed": seed,
        "cycles": cycles,
        "optimizer_steps": cycles * len(train_subjects),
        "best_cycle": best_cycle,
        "best_validation_weighted_mae_db": best_mae,
        "decision": decision,
        "decision_reason": (
            "Best validation checkpoint occurred at the final evaluation."
            if decision == "RETEST"
            else "Best validation checkpoint occurred before the final evaluation."
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / (1024.0**2) if device.type == "cuda" else 0.0
        ),
        "best_checkpoint_sha256": file_sha256(output_dir / "best.pt"),
        "last_checkpoint_sha256": file_sha256(output_dir / "last.pt"),
        "condition_inputs_read": 0,
        "test_subjects_read": 0,
        "git": state,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    arguments = parser.parse_args()
    root = project_root()
    config_path = arguments.config.resolve()
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    if configuration.get("experiment_type") != "joint_unconditional_siren_stage_b":
        raise ValueError("Configuration is not a Stage B unconditional shared-SIREN run")
    run(configuration, root, config_path)


if __name__ == "__main__":
    main()
