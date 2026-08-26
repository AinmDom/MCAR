"""Joint Stage B training for Q26-conditioned FiLM-SIREN."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn

from mcar.data import Normalization
from mcar.evaluation.residual_metrics import (
    WeightedResidualMetrics,
    solid_angle_weighted_residual_metrics,
)
from mcar.models.film_siren import (
    ConditionEncoderConfig,
    FilmSiren,
    FilmSirenConfig,
)
from mcar.paths import project_root
from mcar.q26_condition import (
    Q26MagnitudeNormalization,
    build_q26_condition,
    normalize_split,
)
from mcar.training.train_siren import frequency_coordinates


CONDITIONING_SCOPES = {
    "global",
    "global_zero_local",
    "local_mca",
    "global_plus_local_mca",
}


@dataclass(frozen=True)
class SubjectCondition:
    subject_id: int
    subject_label: str
    split: str
    path: Path
    normalized_magnitude: np.ndarray
    xyz: np.ndarray
    mask: np.ndarray


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
    modulation_statistics: dict[str, object]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def git_state(root: Path) -> dict[str, str | bool]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_subject_paths(
    dataset_root: Path,
    split_csv: Path,
    split: str,
) -> list[tuple[int, str, Path]]:
    split = normalize_split(split)
    if split == "test":
        raise PermissionError("Stage B training entry point never permits test access")
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if normalize_split(row.get("split", "")) == split
        ]
    expected = {"train": 262, "val": 44}[split]
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} {split} subjects, found {len(rows)}")
    output: list[tuple[int, str, Path]] = []
    for row in rows:
        label = row["subject_id"]
        if not label.startswith("P") or not label[1:].isdigit():
            raise ValueError(f"Invalid subject label {label!r}")
        subject_id = int(label[1:])
        path = dataset_root / "subjects" / label / "q26.h5"
        if not path.is_file():
            raise FileNotFoundError(path)
        output.append((subject_id, label, path))
    if len({item[0] for item in output}) != expected:
        raise ValueError(f"Duplicate {split} subject in split CSV")
    return output


def load_condition_cache(
    subject_paths: list[tuple[int, str, Path]],
    dataset_root: Path,
    split_csv: Path,
    q26_csv: Path,
    normalization: Q26MagnitudeNormalization,
    split: str,
) -> list[SubjectCondition]:
    cache: list[SubjectCondition] = []
    for subject_id, label, path in subject_paths:
        condition = build_q26_condition(
            dataset_root,
            split_csv,
            subject_id,
            q26_csv,
        )
        cache.append(
            SubjectCondition(
                subject_id=subject_id,
                subject_label=label,
                split=split,
                path=path,
                normalized_magnitude=normalization.normalize(
                    condition.binaural_magnitude_db
                ),
                xyz=condition.xyz,
                mask=condition.mask,
            )
        )
    return cache


def load_subject_identity_cache(
    subject_paths: list[tuple[int, str, Path]],
    split: str,
) -> list[SubjectCondition]:
    """Build local-only subject records without constructing Q26 conditions."""
    return [
        SubjectCondition(
            subject_id=subject_id,
            subject_label=label,
            split=split,
            path=path,
            normalized_magnitude=np.empty((0,), dtype=np.float32),
            xyz=np.empty((0, 3), dtype=np.float32),
            mask=np.empty((0,), dtype=bool),
        )
        for subject_id, label, path in subject_paths
    ]


def read_common_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        directions = np.asarray(handle["direction_features"][:], dtype=np.float32)
        frequency = np.asarray(handle["frequency_hz"][:], dtype=np.float32).reshape(-1)
        interpolation_mask = np.asarray(
            handle["interpolation_evaluation_mask"][:]
        ).reshape(-1).astype(bool)
    if directions.shape != (793, 6):
        raise ValueError(f"Unexpected direction grid {directions.shape}")
    if frequency.shape != (463,):
        raise ValueError(f"Unexpected frequency grid {frequency.shape}")
    if interpolation_mask.shape != (793,) or np.count_nonzero(interpolation_mask) != 767:
        raise ValueError("Expected exactly 767 interpolation directions")
    return directions, frequency, interpolation_mask, directions[:, 5].astype(np.float64)


def validate_subject_identity(subject: SubjectCondition, handle: h5py.File) -> None:
    actual_id = int(np.asarray(handle.attrs["subject_id"]).item())
    actual_split = normalize_split(handle.attrs["split"])
    if actual_id != subject.subject_id or actual_split != subject.split:
        raise ValueError(
            f"Subject identity mismatch for {subject.path}: "
            f"id={actual_id}, split={actual_split}"
        )


def coordinate_block(
    xyz: torch.Tensor,
    frequency_coordinate: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            xyz[:, None, :].expand(-1, frequency_coordinate.shape[0], -1),
            frequency_coordinate[None, :, :].expand(xyz.shape[0], -1, -1),
        ),
        dim=-1,
    ).reshape(-1, xyz.shape[1] + frequency_coordinate.shape[1])


def conditioned_coordinate_block(
    xyz: torch.Tensor,
    frequency_coordinate: torch.Tensor,
    conditioning_scope: str,
    local_mca_db: np.ndarray | None,
    normalization: Normalization,
) -> torch.Tensor:
    """Build the frozen five-coordinate query plus paired local-MCA channels.

    The bounded gate uses the same seven-dimensional first-layer interface for
    both candidates.  ``global_zero_local`` appends zeros without reading MCA,
    while ``global_plus_local_mca`` appends train-statistics-normalized left and
    right MCA values for each direction/frequency query.
    """
    if conditioning_scope not in CONDITIONING_SCOPES:
        raise ValueError(f"Unknown conditioning_scope {conditioning_scope!r}")
    base = coordinate_block(xyz, frequency_coordinate)
    if conditioning_scope == "global":
        if local_mca_db is not None:
            raise ValueError("Legacy global scope does not accept local MCA")
        return base
    if conditioning_scope == "global_zero_local":
        if local_mca_db is not None:
            raise ValueError("global_zero_local must not read local MCA")
        local = torch.zeros(
            (base.shape[0], 2),
            dtype=base.dtype,
            device=base.device,
        )
    else:
        if local_mca_db is None:
            raise ValueError("global_plus_local_mca requires local MCA")
        values = np.asarray(local_mca_db, dtype=np.float32)
        expected = (2, xyz.shape[0], frequency_coordinate.shape[0])
        if values.shape != expected:
            raise ValueError(f"Expected local MCA shape {expected}, found {values.shape}")
        if not bool(np.all(np.isfinite(values))):
            raise ValueError("local MCA must be finite")
        normalized = (values - normalization.mca_mean) / normalization.mca_std
        local = (
            torch.from_numpy(normalized)
            .to(device=base.device, dtype=base.dtype)
            .permute(1, 2, 0)
            .reshape(-1, 2)
        )
    return torch.cat((base, local), dim=-1)


def condition_tensors(
    subject: SubjectCondition,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(subject.normalized_magnitude).unsqueeze(0).to(device),
        torch.from_numpy(subject.xyz).unsqueeze(0).to(device),
        torch.from_numpy(subject.mask).unsqueeze(0).to(device),
    )


@torch.no_grad()
def predict_subject(
    model: FilmSiren,
    subject: SubjectCondition,
    directions: np.ndarray,
    frequency_coordinate: torch.Tensor,
    direction_indices: np.ndarray,
    device: torch.device,
    directions_per_block: int,
    latent: torch.Tensor | None = None,
    conditioning_scope: str = "global",
    local_mca_db: np.ndarray | None = None,
    normalization: Normalization | None = None,
) -> np.ndarray:
    if latent is None:
        magnitude, condition_xyz, mask = condition_tensors(subject, device)
        latent = model.encode_condition(magnitude, condition_xyz, mask)
    frequency_count = int(frequency_coordinate.shape[0])
    prediction = np.empty(
        (2, direction_indices.size, frequency_count),
        dtype=np.float32,
    )
    for start in range(0, direction_indices.size, directions_per_block):
        stop = min(direction_indices.size, start + directions_per_block)
        selected = direction_indices[start:stop]
        xyz = torch.from_numpy(directions[selected, 2:5]).to(device)
        local_block = (
            None
            if local_mca_db is None
            else local_mca_db[:, start:stop, :]
        )
        if conditioning_scope == "global":
            query = coordinate_block(xyz, frequency_coordinate)
        else:
            if normalization is None:
                raise ValueError("Gate prediction requires normalization")
            query = conditioned_coordinate_block(
                xyz,
                frequency_coordinate,
                conditioning_scope,
                local_block,
                normalization,
            )
        normalized = model(query, latent)
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
    model: FilmSiren,
    subjects: list[SubjectCondition],
    directions: np.ndarray,
    frequency_coordinate: torch.Tensor,
    interpolation_indices: np.ndarray,
    direction_weights: np.ndarray,
    target_mean: float,
    target_std: float,
    device: torch.device,
    directions_per_block: int,
    conditioning_scope: str = "global",
    normalization: Normalization | None = None,
) -> ValidationResult:
    model.eval()
    rows: list[dict[str, float | int | str]] = []
    model_mae: list[float] = []
    model_rmse: list[float] = []
    baseline_mae: list[float] = []
    validation_latents: list[torch.Tensor] = []
    selected_weights = direction_weights[interpolation_indices]
    selected_mask = np.ones(interpolation_indices.size, dtype=bool)
    local_enabled = conditioning_scope in {"local_mca", "global_plus_local_mca"}
    global_enabled = conditioning_scope != "local_mca"
    if conditioning_scope != "global" and normalization is None:
        raise ValueError("Gate validation requires normalization")
    for subject in subjects:
        if global_enabled:
            magnitude, condition_xyz, condition_mask = condition_tensors(subject, device)
            latent = model.encode_condition(
                magnitude,
                condition_xyz,
                condition_mask,
            )
        else:
            latent = torch.zeros(
                (1, model.configuration.latent_dimension),
                dtype=frequency_coordinate.dtype,
                device=device,
            )
        validation_latents.append(latent)
        with h5py.File(subject.path, "r") as handle:
            validate_subject_identity(subject, handle)
            target_db = np.asarray(
                handle["target_residual_db"][:, interpolation_indices, :],
                dtype=np.float32,
            )
            local_mca_db = (
                np.asarray(
                    handle["mca_logmag_db"][:, interpolation_indices, :],
                    dtype=np.float32,
                )
                if local_enabled
                else None
            )
        normalized_prediction = predict_subject(
            model,
            subject,
            directions,
            frequency_coordinate,
            interpolation_indices,
            device,
            directions_per_block,
            latent,
            conditioning_scope,
            local_mca_db,
            normalization,
        )
        prediction_db = normalized_prediction * target_std + target_mean
        metrics = solid_angle_weighted_residual_metrics(
            prediction_db,
            target_db,
            selected_weights,
            selected_mask,
        )
        baseline = solid_angle_weighted_residual_metrics(
            np.zeros_like(target_db),
            target_db,
            selected_weights,
            selected_mask,
        )
        model_mae.append(metrics.residual_mae_db)
        model_rmse.append(metrics.residual_rmse_db)
        baseline_mae.append(baseline.residual_mae_db)
        rows.append(
            {
                "subject_id": subject.subject_id,
                "subject_label": subject.subject_label,
                "weighted_mae_db": metrics.residual_mae_db,
                "weighted_rmse_db": metrics.residual_rmse_db,
                "mca_zero_residual_weighted_mae_db": baseline.residual_mae_db,
            }
        )
    values = np.asarray(model_mae, dtype=np.float64)
    return ValidationResult(
        aggregate_mae_db=float(np.mean(values)),
        aggregate_rmse_db=float(np.mean(model_rmse)),
        median_mae_db=float(np.median(values)),
        std_mae_db=float(np.std(values, ddof=0)),
        minimum_mae_db=float(np.min(values)),
        maximum_mae_db=float(np.max(values)),
        mca_zero_residual_mae_db=float(np.mean(baseline_mae)),
        per_subject=rows,
        modulation_statistics=model.modulation_statistics(
            torch.cat(validation_latents, dim=0)
        ),
    )


def checkpoint_payload(
    model: FilmSiren,
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
        "film_siren_configuration": asdict(model.configuration),
        "condition_encoder_configuration": asdict(model.encoder_configuration),
        "experiment_configuration": configuration,
        "validation_metrics": asdict(metrics),
        "output_unit": "normalized_residual",
        "residual_db_conversion": {
            "target_mean": normalization.target_mean,
            "target_std": normalization.target_std,
            "normalization_path": str(normalization_path),
            "normalization_sha256": file_sha256(normalization_path),
        },
    }


def run(configuration: dict[str, Any], root: Path, config_path: Path) -> None:
    seed = int(configuration["seed"])
    set_seed(seed)
    require_cuda = bool(configuration.get("require_cuda", True))
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a formal Stage B run")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = git_state(root)
    if bool(configuration.get("require_clean_git", True)) and state["dirty"]:
        raise RuntimeError("Formal Stage B runs require a clean Git worktree")

    dataset_root = (root / configuration["dataset_root"]).resolve()
    split_csv = (root / configuration["subject_split_csv"]).resolve()
    q26_csv = (root / configuration["q26_csv"]).resolve()
    q26_normalization_path = (
        root / configuration["q26_normalization"]
    ).resolve()
    target_normalization_path = dataset_root / "training_statistics.json"
    target_normalization = Normalization.from_json(target_normalization_path)
    q26_normalization = Q26MagnitudeNormalization.from_json(
        q26_normalization_path
    )

    train_paths = split_subject_paths(dataset_root, split_csv, "train")
    validation_paths = split_subject_paths(dataset_root, split_csv, "val")
    conditioning_scope = str(configuration.get("conditioning_scope", "global"))
    if conditioning_scope not in CONDITIONING_SCOPES:
        raise ValueError(f"Unknown conditioning_scope {conditioning_scope!r}")
    global_condition_enabled = conditioning_scope != "local_mca"
    if global_condition_enabled:
        train_subjects = load_condition_cache(
            train_paths,
            dataset_root,
            split_csv,
            q26_csv,
            q26_normalization,
            "train",
        )
        validation_subjects = load_condition_cache(
            validation_paths,
            dataset_root,
            split_csv,
            q26_csv,
            q26_normalization,
            "val",
        )
    else:
        train_subjects = load_subject_identity_cache(train_paths, "train")
        validation_subjects = load_subject_identity_cache(validation_paths, "val")
    directions, frequency, interpolation_mask, direction_weights = read_common_grid(
        train_subjects[0].path
    )
    interpolation_indices = np.flatnonzero(interpolation_mask)
    mapping = configuration["frequency_mapping"]
    frequency_coordinate_array = frequency_coordinates(
        frequency,
        str(mapping["mode"]),
        float(mapping["frequency_minimum_hz"]),
        float(mapping["frequency_maximum_hz"]),
    )
    frequency_coordinate = torch.from_numpy(frequency_coordinate_array).to(device)

    encoder_configuration = ConditionEncoderConfig(
        **configuration["condition_encoder"]
    )
    model_configuration = FilmSirenConfig(**configuration["model"])
    expected_coordinate_dimension = 5 if conditioning_scope == "global" else 7
    if model_configuration.coordinate_dimension != expected_coordinate_dimension:
        raise ValueError(
            f"conditioning_scope={conditioning_scope!r} requires "
            f"coordinate_dimension={expected_coordinate_dimension}"
        )
    local_mca_enabled = conditioning_scope in {"local_mca", "global_plus_local_mca"}
    model = FilmSiren(model_configuration, encoder_configuration).to(device)
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

    output_dir = root / "artifacts" / "training" / configuration["run_name"]
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    provenance = {
        "configuration": configuration,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "git": state,
        "split_csv_sha256": file_sha256(split_csv),
        "q26_csv_sha256": file_sha256(q26_csv),
        "q26_normalization_sha256": file_sha256(q26_normalization_path),
        "target_normalization_sha256": file_sha256(target_normalization_path),
        "train_subject_ids": [item.subject_id for item in train_subjects],
        "validation_subject_ids": [item.subject_id for item in validation_subjects],
        "train_subject_count": len(train_subjects),
        "validation_subject_count": len(validation_subjects),
        "conditioning_scope": conditioning_scope,
        "global_condition_input_enabled": global_condition_enabled,
        "local_mca_input_enabled": local_mca_enabled,
        "test_subjects_read": 0,
        "device": str(device),
    }
    (output_dir / "configuration.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    rng = np.random.default_rng(seed)
    history: list[dict[str, float | int]] = []
    validation_ledger: list[dict[str, Any]] = []
    best_mae = float("inf")
    best_cycle = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for cycle in range(1, cycles + 1):
        model.train()
        cycle_losses: list[float] = []
        gradient_norms: list[float] = []
        cycle_started = time.perf_counter()
        for subject_index in rng.permutation(len(train_subjects)):
            subject = train_subjects[int(subject_index)]
            selected = np.sort(
                rng.choice(
                    interpolation_indices,
                    size=directions_per_step,
                    replace=False,
                )
            )
            with h5py.File(subject.path, "r") as handle:
                validate_subject_identity(subject, handle)
                target_db = np.asarray(
                    handle["target_residual_db"][:, selected, :],
                    dtype=np.float32,
                )
                local_mca_db = (
                    np.asarray(
                        handle["mca_logmag_db"][:, selected, :],
                        dtype=np.float32,
                    )
                    if local_mca_enabled
                    else None
                )
            xyz = torch.from_numpy(directions[selected, 2:5]).to(device)
            query = conditioned_coordinate_block(
                xyz,
                frequency_coordinate,
                conditioning_scope,
                local_mca_db,
                target_normalization,
            )
            target = torch.from_numpy(target_db).to(device)
            target = (
                (target - target_normalization.target_mean)
                / target_normalization.target_std
            ).permute(1, 2, 0).reshape(-1, 2)
            optimizer.zero_grad(set_to_none=True)
            if global_condition_enabled:
                magnitude, condition_xyz, mask = condition_tensors(subject, device)
                prediction = model.forward_from_condition(
                    query,
                    magnitude,
                    condition_xyz,
                    mask,
                )
            else:
                latent = torch.zeros(
                    (1, model_configuration.latent_dimension),
                    dtype=query.dtype,
                    device=device,
                )
                prediction = model(query, latent)
            loss = nn.functional.mse_loss(prediction.float(), target.float())
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"Non-finite loss at cycle {cycle}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                gradient_clip,
            )
            if not bool(torch.isfinite(gradient_norm).item()):
                raise FloatingPointError(f"Non-finite gradient at cycle {cycle}")
            optimizer.step()
            cycle_losses.append(float(loss.detach().item()))
            gradient_norms.append(float(gradient_norm.detach().item()))

        record: dict[str, float | int] = {
            "cycle": cycle,
            "optimizer_steps_completed": cycle * len(train_subjects),
            "train_mse_normalized": float(np.mean(cycle_losses)),
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
                conditioning_scope,
                target_normalization,
            )
            record["validation_weighted_mae_db"] = result.aggregate_mae_db
            record["validation_weighted_rmse_db"] = result.aggregate_rmse_db
            ledger_entry = {
                "cycle": cycle,
                "metric": "interpolation_solid_angle_weighted_residual_mae_db",
                "aggregate": asdict(result),
                "elapsed_seconds": time.perf_counter() - validation_started,
                "test_subjects_read": 0,
            }
            validation_ledger.append(ledger_entry)
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
                    "w",
                    newline="",
                    encoding="utf-8",
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
            json.dumps(validation_ledger, indent=2) + "\n",
            encoding="utf-8",
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
    last_result = ValidationResult(**validation_ledger[-1]["aggregate"])
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
            torch.cuda.max_memory_allocated() / (1024.0**2)
            if device.type == "cuda"
            else 0.0
        ),
        "best_checkpoint_sha256": file_sha256(output_dir / "best.pt"),
        "last_checkpoint_sha256": file_sha256(output_dir / "last.pt"),
        "conditioning_scope": conditioning_scope,
        "condition_inputs_read": (
            len(train_subjects) + len(validation_subjects)
            if global_condition_enabled
            else 0
        ),
        "local_mca_inputs_read": (
            cycles * len(train_subjects)
            + (cycles // validation_interval) * len(validation_subjects)
            if local_mca_enabled
            else 0
        ),
        "test_subjects_read": 0,
        "git": state,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


def main() -> None:
    arguments = parse_arguments()
    root = project_root()
    config_path = arguments.config.resolve()
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    if configuration.get("experiment_type") != "joint_film_siren_stage_b":
        raise ValueError("Configuration is not a Stage B joint FiLM-SIREN run")
    run(configuration, root, config_path)


if __name__ == "__main__":
    main()
