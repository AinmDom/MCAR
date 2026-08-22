"""Train the frozen Stage-B FiLM-SIREN with the Stage-C objective."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch

from mcar.data import Normalization
from mcar.evaluation.residual_metrics import solid_angle_weighted_residual_metrics
from mcar.models.film_siren import ConditionEncoderConfig, FilmSiren, FilmSirenConfig
from mcar.paths import project_root
from mcar.q26_condition import Q26MagnitudeNormalization
from mcar.training.film_siren_stage_c import (
    StageCLossConfiguration,
    calculate_stage_c_losses,
    strict_metadata_to_device,
)
from mcar.training.train_film_siren import (
    SubjectCondition,
    condition_tensors,
    coordinate_block,
    file_sha256,
    git_state,
    load_condition_cache,
    predict_subject,
    read_common_grid,
    set_seed,
    split_subject_paths,
    validate_subject_identity,
)
from mcar.training.train_mlp_v2 import (
    LossMetrics,
    add_metrics,
    average_metrics,
    make_erb_center_frequencies_hz,
    make_erb_weights,
)
from mcar.training.train_siren import frequency_coordinates


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    return parser.parse_args()


def loss_configuration(raw: Mapping[str, Any]) -> StageCLossConfiguration:
    values = dict(raw)
    if "notch_radii_bins" in values:
        values["notch_radii_bins"] = tuple(values["notch_radii_bins"])
    result = StageCLossConfiguration(**values)
    result.validate()
    return result


def horizontal_interpolation_indices(
    directions: np.ndarray, interpolation_mask: np.ndarray
) -> np.ndarray:
    mask = interpolation_mask & (np.abs(directions[:, 1]) <= 1e-6)
    indices = np.flatnonzero(mask)
    if indices.size != 72:
        raise ValueError(f"Expected 72 horizontal interpolation directions, found {indices.size}")
    return indices


def read_block(
    subject: SubjectCondition,
    indices: np.ndarray,
    *,
    strict_ild: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray | int] | None]:
    with h5py.File(subject.path, "r") as handle:
        validate_subject_identity(subject, handle)
        target = np.asarray(
            handle["target_residual_db"][:, indices, :], dtype=np.float32
        )
        mca = np.asarray(handle["mca_logmag_db"][:, indices, :], dtype=np.float32)
        direction_features = np.asarray(
            handle["direction_features"][indices, :], dtype=np.float32
        )
        if not strict_ild:
            return target, mca, direction_features, None
        strict = handle["strict_ild"]
        metadata: dict[str, np.ndarray | int] = {
            "mca_selected_phase_rad": np.asarray(
                strict["mca_selected_phase_rad"][:, indices, :], dtype=np.float32
            ),
            "mca_outside_real": np.asarray(
                strict["mca_outside_real"][:, indices, :], dtype=np.float32
            ),
            "mca_outside_imag": np.asarray(
                strict["mca_outside_imag"][:, indices, :], dtype=np.float32
            ),
            "selected_bin_indices_zero_based": np.squeeze(
                strict["selected_bin_indices_zero_based"][:]
            ).astype(np.int64),
            "outside_bin_indices_zero_based": np.squeeze(
                strict["outside_bin_indices_zero_based"][:]
            ).astype(np.int64),
            "reference_ild_db": np.asarray(
                strict["reference_ild_db"][indices], dtype=np.float32
            ).reshape(-1),
            "single_sided_frequency_count": int(
                np.asarray(strict.attrs["single_sided_frequency_count"]).item()
            ),
            "hrir_length": int(np.asarray(strict.attrs["hrir_length"]).item()),
        }
    return target, mca, direction_features, metadata


def prediction_block(
    model: FilmSiren,
    latent: torch.Tensor,
    directions: np.ndarray,
    frequency_coordinate: torch.Tensor,
    indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    xyz = torch.from_numpy(directions[indices, 2:5]).to(device)
    query = coordinate_block(xyz, frequency_coordinate)
    prediction = model(query, latent)
    return prediction.reshape(indices.size, frequency_coordinate.shape[0], 2).permute(
        2, 0, 1
    )


def tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(value).to(device, non_blocking=True)


def one_loss(
    model: FilmSiren,
    latent: torch.Tensor,
    subject: SubjectCondition,
    directions: np.ndarray,
    frequency_coordinate: torch.Tensor,
    frequency_hz: torch.Tensor,
    global_indices: np.ndarray,
    horizontal_indices: np.ndarray,
    normalization: Normalization,
    log_erb_weights: torch.Tensor,
    erb_centers: torch.Tensor,
    objective: StageCLossConfiguration,
    device: torch.device,
) -> tuple[torch.Tensor, LossMetrics]:
    global_target, global_mca, global_features, _ = read_block(
        subject, global_indices, strict_ild=False
    )
    horizontal_target, horizontal_mca, horizontal_features, strict = read_block(
        subject, horizontal_indices, strict_ild=True
    )
    if strict is None:
        raise AssertionError("Strict ILD metadata was not loaded")
    global_prediction = prediction_block(
        model, latent, directions, frequency_coordinate, global_indices, device
    )
    horizontal_prediction = prediction_block(
        model, latent, directions, frequency_coordinate, horizontal_indices, device
    )
    global_target_tensor = tensor(global_target, device)
    return calculate_stage_c_losses(
        global_prediction,
        (global_target_tensor - normalization.target_mean) / normalization.target_std,
        global_target_tensor,
        tensor(global_mca, device),
        tensor(global_features, device),
        horizontal_prediction,
        tensor(horizontal_target, device),
        tensor(horizontal_mca, device),
        tensor(horizontal_features, device),
        strict_metadata_to_device(strict, device),
        frequency_hz,
        log_erb_weights,
        erb_centers,
        normalization.target_mean,
        normalization.target_std,
        objective,
    )


@torch.no_grad()
def evaluate_validation(
    model: FilmSiren,
    subjects: list[SubjectCondition],
    directions: np.ndarray,
    frequency_coordinate: torch.Tensor,
    frequency_hz: torch.Tensor,
    interpolation_indices: np.ndarray,
    horizontal_indices: np.ndarray,
    direction_weights: np.ndarray,
    normalization: Normalization,
    log_erb_weights: torch.Tensor,
    erb_centers: torch.Tensor,
    objective: StageCLossConfiguration,
    device: torch.device,
    directions_per_block: int,
) -> dict[str, Any]:
    model.eval()
    accumulated = LossMetrics()
    residual_mae: list[float] = []
    residual_rmse: list[float] = []
    rows: list[dict[str, float | int | str]] = []
    horizontal_positions = np.flatnonzero(
        np.isin(interpolation_indices, horizontal_indices)
    )
    selected_weights = direction_weights[interpolation_indices]
    latents: list[torch.Tensor] = []
    for subject in subjects:
        magnitude, condition_xyz, condition_mask = condition_tensors(subject, device)
        latent = model.encode_condition(magnitude, condition_xyz, condition_mask)
        latents.append(latent)
        prediction_normalized = predict_subject(
            model,
            subject,
            directions,
            frequency_coordinate,
            interpolation_indices,
            device,
            directions_per_block,
            latent,
        )
        global_target, global_mca, global_features, _ = read_block(
            subject, interpolation_indices, strict_ild=False
        )
        horizontal_target, horizontal_mca, horizontal_features, strict = read_block(
            subject, horizontal_indices, strict_ild=True
        )
        if strict is None:
            raise AssertionError("Strict ILD metadata was not loaded")
        prediction = tensor(prediction_normalized, device)
        target = tensor(global_target, device)
        _, metrics = calculate_stage_c_losses(
            prediction,
            (target - normalization.target_mean) / normalization.target_std,
            target,
            tensor(global_mca, device),
            tensor(global_features, device),
            prediction[:, horizontal_positions, :],
            tensor(horizontal_target, device),
            tensor(horizontal_mca, device),
            tensor(horizontal_features, device),
            strict_metadata_to_device(strict, device),
            frequency_hz,
            log_erb_weights,
            erb_centers,
            normalization.target_mean,
            normalization.target_std,
            objective,
        )
        add_metrics(accumulated, metrics)
        raw_metrics = solid_angle_weighted_residual_metrics(
            prediction_normalized * normalization.target_std
            + normalization.target_mean,
            global_target,
            selected_weights,
            np.ones(interpolation_indices.size, dtype=bool),
        )
        residual_mae.append(raw_metrics.residual_mae_db)
        residual_rmse.append(raw_metrics.residual_rmse_db)
        rows.append(
            {
                "subject_id": subject.subject_id,
                "subject_label": subject.subject_label,
                "objective_total": metrics.total,
                "weighted_residual_mae_db": raw_metrics.residual_mae_db,
                "weighted_residual_rmse_db": raw_metrics.residual_rmse_db,
                "erb_mae_db": metrics.erb_mae_db,
                "contralateral_high_frequency_mae_db": (
                    metrics.contralateral_high_frequency_mae_db
                ),
                "strict_ild_mae_db": metrics.ild_mae_db,
                "spectral_band_ild_mae_db": metrics.spectral_band_ild_mae_db,
            }
        )
    averaged = average_metrics(accumulated, len(subjects))
    return {
        "objective_metrics": asdict(averaged),
        "aggregate_weighted_residual_mae_db": float(np.mean(residual_mae)),
        "aggregate_weighted_residual_rmse_db": float(np.mean(residual_rmse)),
        "per_subject": rows,
        "modulation_statistics": model.modulation_statistics(torch.cat(latents, dim=0)),
    }


def learning_rate_for_cycle(
    base_learning_rate: float,
    scheduler_name: str,
    cycle: int,
    horizon: int,
    warmup_cycles: int,
) -> float:
    if scheduler_name == "constant":
        return base_learning_rate
    if scheduler_name == "cosine":
        progress = (cycle - 1) / max(horizon, 1)
        return base_learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
    if scheduler_name == "warmup_cosine":
        if warmup_cycles < 1 or warmup_cycles >= horizon:
            raise ValueError("warmup_cosine requires 0 < warmup_cycles < horizon")
        if cycle <= warmup_cycles:
            return base_learning_rate * cycle / warmup_cycles
        progress = (cycle - warmup_cycles) / (horizon - warmup_cycles)
        return base_learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unsupported scheduler {scheduler_name!r}")


def make_optimizer(
    model: FilmSiren, configuration: Mapping[str, Any]
) -> torch.optim.Optimizer:
    name = str(configuration["name"])
    kwargs = {
        "lr": float(configuration["learning_rate"]),
        "weight_decay": float(configuration["weight_decay"]),
    }
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "AdamW":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    raise ValueError(f"Unsupported optimizer {name!r}")


def stage_c_checkpoint_payload(
    model: FilmSiren,
    optimizer: torch.optim.Optimizer,
    configuration: dict[str, Any],
    cycle: int,
    metrics: dict[str, Any],
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
        "validation_metrics": metrics,
        "training_stage": "film_siren_stage_c",
        "selection_metric": "mean_stage_c_objective_total",
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
        raise RuntimeError("CUDA is required for a formal Stage C run")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = git_state(root)
    if bool(configuration.get("require_clean_git", True)) and state["dirty"]:
        raise RuntimeError("Formal Stage C runs require a clean Git worktree")

    dataset_root = (root / configuration["dataset_root"]).resolve()
    split_csv = (root / configuration["subject_split_csv"]).resolve()
    q26_csv = (root / configuration["q26_csv"]).resolve()
    q26_normalization_path = (root / configuration["q26_normalization"]).resolve()
    target_normalization_path = dataset_root / "training_statistics.json"
    normalization = Normalization.from_json(target_normalization_path)
    q26_normalization = Q26MagnitudeNormalization.from_json(q26_normalization_path)
    train_subjects = load_condition_cache(
        split_subject_paths(dataset_root, split_csv, "train"),
        dataset_root,
        split_csv,
        q26_csv,
        q26_normalization,
        "train",
    )
    validation_subjects = load_condition_cache(
        split_subject_paths(dataset_root, split_csv, "val"),
        dataset_root,
        split_csv,
        q26_csv,
        q26_normalization,
        "val",
    )
    directions, frequency, interpolation_mask, direction_weights = read_common_grid(
        train_subjects[0].path
    )
    interpolation_indices = np.flatnonzero(interpolation_mask)
    horizontal_indices = horizontal_interpolation_indices(directions, interpolation_mask)
    mapping = configuration["frequency_mapping"]
    frequency_coordinate = torch.from_numpy(
        frequency_coordinates(
            frequency,
            str(mapping["mode"]),
            float(mapping["frequency_minimum_hz"]),
            float(mapping["frequency_maximum_hz"]),
        )
    ).to(device)
    frequency_hz = torch.from_numpy(frequency).to(device)
    erb_weights = torch.from_numpy(make_erb_weights(frequency)).to(device)
    log_erb_weights = torch.log(torch.clamp(erb_weights, min=1e-12)).view(
        1, 1, erb_weights.shape[0], erb_weights.shape[1]
    )
    erb_centers = torch.from_numpy(make_erb_center_frequencies_hz()).to(device)

    model = FilmSiren(
        FilmSirenConfig(**configuration["model"]),
        ConditionEncoderConfig(**configuration["condition_encoder"]),
    ).to(device)
    optimizer = make_optimizer(model, configuration["optimizer"])
    objective = loss_configuration(configuration["objective"])
    cycles = int(configuration["cycles"])
    validation_interval = int(configuration["validation_interval_cycles"])
    global_count = int(configuration["global_directions_per_step"])
    horizontal_count = int(configuration["horizontal_directions_per_step"])
    gradient_clip = float(configuration["gradient_clip"])
    scheduler = configuration["scheduler"]
    scheduler_name = str(scheduler["name"])
    scheduler_horizon = int(scheduler["horizon_cycles"])
    warmup_cycles = int(scheduler.get("warmup_cycles", 0))
    if cycles < 1 or validation_interval < 1 or cycles % validation_interval != 0:
        raise ValueError("cycles must be positive and divisible by validation interval")
    if not 1 <= global_count <= interpolation_indices.size:
        raise ValueError("Invalid global_directions_per_step")
    if not 1 <= horizontal_count <= horizontal_indices.size:
        raise ValueError("Invalid horizontal_directions_per_step")
    learning_rate_for_cycle(
        float(configuration["optimizer"]["learning_rate"]),
        scheduler_name,
        1,
        scheduler_horizon,
        warmup_cycles,
    )

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
        "train_subject_ids": [subject.subject_id for subject in train_subjects],
        "validation_subject_ids": [subject.subject_id for subject in validation_subjects],
        "train_subject_count": len(train_subjects),
        "validation_subject_count": len(validation_subjects),
        "global_interpolation_direction_count": int(interpolation_indices.size),
        "horizontal_interpolation_direction_count": int(horizontal_indices.size),
        "test_subjects_read": 0,
        "device": str(device),
    }
    (output_dir / "configuration.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    rng = np.random.default_rng(seed)
    history: list[dict[str, float | int]] = []
    ledger: list[dict[str, Any]] = []
    best_total = float("inf")
    best_cycle = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for cycle in range(1, cycles + 1):
        learning_rate = learning_rate_for_cycle(
            float(configuration["optimizer"]["learning_rate"]),
            scheduler_name,
            cycle,
            scheduler_horizon,
            warmup_cycles,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        accumulated = LossMetrics()
        gradient_norms: list[float] = []
        cycle_started = time.perf_counter()
        for subject_index in rng.permutation(len(train_subjects)):
            subject = train_subjects[int(subject_index)]
            global_indices = np.sort(
                rng.choice(interpolation_indices, size=global_count, replace=False)
            )
            horizontal_batch = np.sort(
                rng.choice(horizontal_indices, size=horizontal_count, replace=False)
            )
            magnitude, condition_xyz, condition_mask = condition_tensors(subject, device)
            optimizer.zero_grad(set_to_none=True)
            latent = model.encode_condition(magnitude, condition_xyz, condition_mask)
            loss, metrics = one_loss(
                model,
                latent,
                subject,
                directions,
                frequency_coordinate,
                frequency_hz,
                global_indices,
                horizontal_batch,
                normalization,
                log_erb_weights,
                erb_centers,
                objective,
                device,
            )
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(f"Non-finite loss at cycle {cycle}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            if not bool(torch.isfinite(gradient_norm).item()):
                raise FloatingPointError(f"Non-finite gradient at cycle {cycle}")
            optimizer.step()
            add_metrics(accumulated, metrics)
            gradient_norms.append(float(gradient_norm.detach().item()))
        train_metrics = average_metrics(accumulated, len(train_subjects))
        record: dict[str, float | int] = {
            "cycle": cycle,
            "optimizer_steps_completed": cycle * len(train_subjects),
            "learning_rate": learning_rate,
            **{f"train_{key}": value for key, value in asdict(train_metrics).items()},
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
                frequency_hz,
                interpolation_indices,
                horizontal_indices,
                direction_weights,
                normalization,
                log_erb_weights,
                erb_centers,
                objective,
                device,
                int(configuration["validation_directions_per_block"]),
            )
            validation_metrics = result["objective_metrics"]
            record.update(
                {f"validation_{key}": value for key, value in validation_metrics.items()}
            )
            record["validation_weighted_residual_mae_db"] = result[
                "aggregate_weighted_residual_mae_db"
            ]
            entry = {
                "cycle": cycle,
                "selection_metric": "mean_stage_c_objective_total",
                "aggregate": result,
                "elapsed_seconds": time.perf_counter() - validation_started,
                "test_subjects_read": 0,
            }
            ledger.append(entry)
            current_total = float(validation_metrics["total"])
            if current_total < best_total:
                best_total = current_total
                best_cycle = cycle
                payload = stage_c_checkpoint_payload(
                    model,
                    optimizer,
                    configuration,
                    cycle,
                    result,
                    normalization,
                    target_normalization_path,
                )
                torch.save(payload, output_dir / "best.pt")
                rows = result["per_subject"]
                with (output_dir / "best_validation_per_subject.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
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
            f"cycle={cycle:03d} total={train_metrics.total:.6f} "
            + (
                f"val_total={record['validation_total']:.6f} "
                if "validation_total" in record
                else ""
            )
            + f"lr={learning_rate:.3e} time={record['cycle_elapsed_seconds']:.1f}s",
            flush=True,
        )

    if best_cycle == 0:
        raise AssertionError("No validation checkpoint was produced")
    last_result = ledger[-1]["aggregate"]
    payload = stage_c_checkpoint_payload(
        model,
        optimizer,
        configuration,
        cycles,
        last_result,
        normalization,
        target_normalization_path,
    )
    torch.save(payload, output_dir / "last.pt")
    decision = "RETEST" if best_cycle == cycles else "KEEP"
    report = {
        "status": "completed",
        "run_name": configuration["run_name"],
        "seed": seed,
        "cycles": cycles,
        "optimizer_steps": cycles * len(train_subjects),
        "best_cycle": best_cycle,
        "best_validation_objective_total": best_total,
        "decision": decision,
        "scheduler_horizon_cycles": scheduler_horizon,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / (1024.0**2)
            if device.type == "cuda"
            else 0.0
        ),
        "best_checkpoint_sha256": file_sha256(output_dir / "best.pt"),
        "last_checkpoint_sha256": file_sha256(output_dir / "last.pt"),
        "test_subjects_read": 0,
        "git": state,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    arguments = parse_arguments()
    root = project_root()
    config_path = arguments.config.resolve()
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    if configuration.get("experiment_type") != "joint_film_siren_stage_c":
        raise ValueError("Configuration is not a Stage C FiLM-SIREN run")
    run(configuration, root, config_path)


if __name__ == "__main__":
    main()
