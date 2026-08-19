"""Train a plain SIREN on one SONICOM subject's residual field."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
from mcar.losses import (
    high_frequency_spectral_difference_mae,
    multi_scale_notch_depth_mae,
)
from mcar.models.siren import Siren, SirenConfig, parameter_count
from mcar.paths import project_root


@dataclass(frozen=True)
class SubjectField:
    subject_id: int
    subject_label: str
    path: Path
    target_residual_db: np.ndarray
    mca_logmag_db: np.ndarray
    direction_features: np.ndarray
    frequency_hz: np.ndarray
    eligible_direction_indices: np.ndarray


@dataclass(frozen=True)
class FieldMetrics:
    residual_mae_db: float
    residual_rmse_db: float
    residual_mae_above_8khz_db: float
    residual_mae_above_10khz_db: float
    first_spectral_difference_mae_db_per_bin: float
    second_spectral_difference_mae_db_per_bin2: float
    notch_depth_mae_db: float


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    sampled_train_mse_normalized: float
    full_field_residual_mae_db: float
    full_field_residual_rmse_db: float
    full_field_mae_above_8khz_db: float
    full_field_mae_above_10khz_db: float
    first_spectral_difference_mae_db_per_bin: float
    second_spectral_difference_mae_db_per_bin2: float
    notch_depth_mae_db: float
    gradient_norm_mean: float
    gradient_norm_max: float
    learning_rate: float
    elapsed_seconds: float


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
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

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


def subject_label(subject_id: int) -> str:
    if subject_id < 1:
        raise ValueError("subject_id must be positive")
    return f"P{subject_id:04d}"


def verify_train_subject(split_csv: Path, subject_id: int) -> str:
    """Verify the split before constructing or opening an HDF5 path."""
    label = subject_label(subject_id)
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle) if row["subject_id"] == label
        ]
    if len(rows) != 1:
        raise ValueError(f"Expected one split row for {label}, found {len(rows)}")
    split = rows[0]["split"].lower()
    if split != "train":
        raise PermissionError(
            f"Single-subject SIREN training only permits train subjects; "
            f"{label} belongs to {split!r}"
        )
    return label


def load_subject_field(
    dataset_root: Path,
    split_csv: Path,
    subject_id: int,
    *,
    interpolation_only: bool,
) -> SubjectField:
    label = verify_train_subject(split_csv, subject_id)
    path = dataset_root / "subjects" / label / "q26.h5"
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        actual_split = str(handle.attrs["split"])
        actual_subject_id = int(np.asarray(handle.attrs["subject_id"]).item())
        if actual_split != "train" or actual_subject_id != subject_id:
            raise ValueError(
                f"HDF5 identity mismatch for {path}: "
                f"split={actual_split!r}, subject={actual_subject_id}"
            )
        target = np.asarray(handle["target_residual_db"], dtype=np.float32)
        mca = np.asarray(handle["mca_logmag_db"], dtype=np.float32)
        directions = np.asarray(handle["direction_features"], dtype=np.float32)
        frequency_hz = np.squeeze(
            np.asarray(handle["frequency_hz"], dtype=np.float32)
        )
        if interpolation_only:
            eligible = np.flatnonzero(
                np.squeeze(handle["interpolation_evaluation_mask"][:]).astype(
                    bool
                )
            )
        else:
            eligible = np.arange(target.shape[1], dtype=np.int64)
    if target.shape != mca.shape or target.ndim != 3 or target.shape[0] != 2:
        raise ValueError(f"Unexpected residual field shape {target.shape}")
    if directions.shape != (target.shape[1], 6):
        raise ValueError("Direction metadata does not match the residual field")
    if frequency_hz.shape != (target.shape[2],):
        raise ValueError("Frequency metadata does not match the residual field")
    for name, values in (
        ("target_residual_db", target),
        ("mca_logmag_db", mca),
        ("direction_features", directions),
        ("frequency_hz", frequency_hz),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite values in {name} for {label}")
    return SubjectField(
        subject_id=subject_id,
        subject_label=label,
        path=path,
        target_residual_db=target,
        mca_logmag_db=mca,
        direction_features=directions,
        frequency_hz=frequency_hz,
        eligible_direction_indices=eligible,
    )


def erb_rate(frequency_hz: np.ndarray) -> np.ndarray:
    return 21.4 * np.log10(1.0 + 0.00437 * frequency_hz)


def frequency_coordinates(
    frequency_hz: np.ndarray,
    mode: str,
    minimum_hz: float,
    maximum_hz: float,
) -> np.ndarray:
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if maximum_hz <= minimum_hz:
        raise ValueError("maximum_hz must exceed minimum_hz")
    tolerance = max(1e-5, 1e-7 * maximum_hz)
    if (
        float(np.min(frequency)) < minimum_hz - tolerance
        or float(np.max(frequency)) > maximum_hz + tolerance
    ):
        raise ValueError("Frequency samples fall outside the frozen mapping")
    linear = 2.0 * (frequency - minimum_hz) / (maximum_hz - minimum_hz) - 1.0
    if mode == "linear":
        result = linear[:, None]
    elif mode in {"erb", "dual"}:
        rates = erb_rate(frequency)
        minimum_rate = float(erb_rate(np.asarray([minimum_hz]))[0])
        maximum_rate = float(erb_rate(np.asarray([maximum_hz]))[0])
        erb = 2.0 * (rates - minimum_rate) / (maximum_rate - minimum_rate) - 1.0
        result = erb[:, None] if mode == "erb" else np.stack((linear, erb), axis=1)
    else:
        raise ValueError(f"Unsupported frequency_mode {mode!r}")
    return result.astype(np.float32)


def coordinate_block(
    direction_xyz: torch.Tensor,
    frequency_coordinate: torch.Tensor,
) -> torch.Tensor:
    direction_count = direction_xyz.shape[0]
    frequency_count = frequency_coordinate.shape[0]
    spatial = direction_xyz[:, None, :].expand(-1, frequency_count, -1)
    spectral = frequency_coordinate[None, :, :].expand(direction_count, -1, -1)
    return torch.cat((spatial, spectral), dim=-1).reshape(
        direction_count * frequency_count, -1
    )


@torch.no_grad()
def predict_field(
    model: Siren,
    field: SubjectField,
    frequency_coordinate: torch.Tensor,
    normalization: Normalization,
    device: torch.device,
    directions_per_block: int,
    use_amp: bool,
) -> torch.Tensor:
    model.eval()
    predictions: list[torch.Tensor] = []
    eligible = field.eligible_direction_indices
    for start in range(0, eligible.size, directions_per_block):
        indices = eligible[start : start + directions_per_block]
        xyz = torch.from_numpy(field.direction_features[indices, 2:5]).to(device)
        query = coordinate_block(xyz, frequency_coordinate)
        with torch.amp.autocast("cuda", enabled=use_amp):
            normalized = model(query)
        direction_count = len(indices)
        residual_db = (
            normalized.float() * normalization.target_std
            + normalization.target_mean
        )
        predictions.append(
            residual_db.reshape(direction_count, field.frequency_hz.size, 2)
            .permute(2, 0, 1)
            .cpu()
        )
    return torch.cat(predictions, dim=1)


def evaluate_field(
    predicted_residual_db: torch.Tensor,
    field: SubjectField,
    *,
    direction_indices: np.ndarray | None = None,
) -> FieldMetrics:
    if direction_indices is None:
        indices = field.eligible_direction_indices
        predicted = predicted_residual_db
    else:
        indices = direction_indices
        if predicted_residual_db.shape[1] != field.eligible_direction_indices.size:
            raise ValueError(
                "predicted_residual_db must cover all eligible directions "
                "when direction_indices is provided"
            )
        # predict_field emits directions in eligible order. Subsetting by
        # source indices is only valid when eligible is the full source order,
        # which is exactly the matrix-mode contract.
        if not np.array_equal(
            field.eligible_direction_indices,
            np.arange(predicted_residual_db.shape[1], dtype=np.int64),
        ):
            raise ValueError(
                "direction_indices requires full-grid prediction order"
            )
        predicted = predicted_residual_db[:, indices, :]
    target = torch.from_numpy(field.target_residual_db[:, indices, :]).float()
    mca = torch.from_numpy(field.mca_logmag_db[:, indices, :]).float()
    frequencies = torch.from_numpy(field.frequency_hz).float()
    weights = torch.from_numpy(field.direction_features[indices, 5]).float()
    errors = predicted.float() - target
    absolute = torch.abs(errors)
    above_8khz = frequencies > 8000.0
    above_10khz = frequencies > 10000.0
    corrected = mca + predicted.float()
    reference = mca + target
    first_difference, second_difference = high_frequency_spectral_difference_mae(
        corrected,
        reference,
        weights,
        frequencies,
        4000.0,
    )
    notch = multi_scale_notch_depth_mae(
        corrected,
        reference,
        weights,
        frequencies,
        4000.0,
        18000.0,
        (4, 8, 16),
        1.0,
        0.5,
    )
    return FieldMetrics(
        residual_mae_db=float(torch.mean(absolute).item()),
        residual_rmse_db=float(torch.sqrt(torch.mean(torch.square(errors))).item()),
        residual_mae_above_8khz_db=float(
            torch.mean(absolute[..., above_8khz]).item()
        ),
        residual_mae_above_10khz_db=float(
            torch.mean(absolute[..., above_10khz]).item()
        ),
        first_spectral_difference_mae_db_per_bin=float(first_difference.item()),
        second_spectral_difference_mae_db_per_bin2=float(second_difference.item()),
        notch_depth_mae_db=float(notch.item()),
    )


def save_checkpoint(
    path: Path,
    model: Siren,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    configuration: dict[str, Any],
    metrics: FieldMetrics,
    normalization: Normalization,
    normalization_path: Path,
    normalization_sha256: str,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "siren_configuration": model.configuration.to_dict(),
            "experiment_configuration": configuration,
            "field_metrics": asdict(metrics),
            "output_unit": "normalized_residual",
            "residual_db_conversion": {
                "target_mean": normalization.target_mean,
                "target_std": normalization.target_std,
                "normalization_path": str(normalization_path),
                "normalization_sha256": normalization_sha256,
            },
        },
        path,
    )


def measure_inference_milliseconds(
    model: Siren,
    field: SubjectField,
    frequency_coordinate: torch.Tensor,
    device: torch.device,
    directions_per_block: int,
    use_amp: bool,
    repetitions: int,
) -> float:
    def run_once() -> None:
        for start in range(
            0, field.eligible_direction_indices.size, directions_per_block
        ):
            indices = field.eligible_direction_indices[
                start : start + directions_per_block
            ]
            xyz = torch.from_numpy(field.direction_features[indices, 2:5]).to(
                device
            )
            query = coordinate_block(xyz, frequency_coordinate)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                model(query)

    run_once()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repetitions):
        run_once()
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - started) / repetitions


@dataclass(frozen=True)
class MatrixEpochRecord:
    epoch: int
    sampled_train_mse_normalized: float
    holdout_residual_mae_db: float
    holdout_residual_rmse_db: float
    holdout_mae_above_8khz_db: float
    holdout_mae_above_10khz_db: float
    holdout_first_spectral_difference_mae_db_per_bin: float
    holdout_second_spectral_difference_mae_db_per_bin2: float
    holdout_notch_depth_mae_db: float
    gradient_norm_mean: float
    gradient_norm_max: float
    learning_rate: float
    elapsed_seconds: float


def load_subject_ids(subjects_csv: Path) -> list[int]:
    """Read the frozen Stage A2 subject lock file into numeric subject ids."""
    with subjects_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Subject lock file is empty")
    subject_ids: list[int] = []
    for row in rows:
        label = row["subject_id"].strip()
        if not label.startswith("P") or not label[1:].isdigit():
            raise ValueError(f"Invalid subject label {label!r}")
        subject_ids.append(int(label[1:]))
    return subject_ids


def load_holdout(holdout_csv: Path) -> tuple[np.ndarray, str]:
    """Read and hash-verify the frozen Stage A2 holdout direction lock file."""
    with holdout_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError("Holdout lock file must contain exactly one row")
    row = rows[0]
    indices = np.asarray(
        [int(item) for item in row["source_indices_zero_based"].split(",")],
        dtype=np.int64,
    )
    if indices.size == 0 or len(set(indices.tolist())) != indices.size:
        raise ValueError("Holdout indices must be non-empty and unique")
    payload = json.dumps(indices.tolist()).encode("utf-8")
    expected = hashlib.sha256(payload).hexdigest().upper()
    actual = row["sha256"].strip().upper()
    if actual != expected:
        raise ValueError(
            f"Holdout hash mismatch: expected {expected}, got {actual}"
        )
    return indices, actual


def run_matrix(
    configuration: dict[str, Any],
    root: Path,
    config_path: Path,
) -> None:
    """Run the frozen Stage A2 matrix: one independent plain SIREN per subject.

    Each subject is trained on the 793-minus-holdout directions and evaluated
    on the fixed 64-direction holdout every epoch. Per-subject artifacts are
    written below ``artifacts/training/<run_name>/<PXXXX>/`` and a matrix
    summary is written to the run root. Validation/test HDF5 files are never
    opened.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SIREN matrix")
    subjects_csv = (root / configuration["subjects_csv"]).resolve()
    holdout_csv = (root / configuration["holdout_csv"]).resolve()
    subject_ids = load_subject_ids(subjects_csv)
    if not subject_ids:
        raise ValueError("Subject lock file contains no subjects")
    holdout_indices, holdout_sha256 = load_holdout(holdout_csv)

    dataset_root = (root / configuration["dataset_root"]).resolve()
    split_csv = (root / configuration["subject_split_csv"]).resolve()
    normalization_path = dataset_root / "training_statistics.json"
    normalization = Normalization.from_json(normalization_path)
    normalization_sha256 = file_sha256(normalization_path)

    frequency_mapping = configuration["frequency_mapping"]
    frequency_mode = str(frequency_mapping["mode"])
    if frequency_mode not in {"linear", "erb", "dual"}:
        raise ValueError(f"Unsupported frequency_mode {frequency_mode!r}")
    minimum_hz = float(frequency_mapping["frequency_minimum_hz"])
    maximum_hz = float(frequency_mapping["frequency_maximum_hz"])
    seed = int(configuration["seed"])
    set_seed(seed)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    hidden_width = int(configuration["model"]["hidden_width"])
    sine_layer_count = int(configuration["model"]["sine_layer_count"])
    first_omega = float(configuration["model"]["first_omega"])
    hidden_omega = float(configuration["model"]["hidden_omega"])
    learning_rate = float(configuration["optimizer"]["learning_rate"])
    weight_decay = float(configuration["optimizer"]["weight_decay"])
    epochs = int(configuration["epochs"])
    steps_per_epoch = int(configuration["steps_per_epoch"])
    directions_per_batch = int(configuration["directions_per_batch"])
    evaluation_block = int(configuration["evaluation_directions_per_block"])
    gradient_clip = float(configuration["gradient_clip"])
    use_amp = bool(configuration["automatic_mixed_precision"])
    inference_repetitions = int(configuration["inference_repetitions"])
    interpolation_only = bool(configuration.get("interpolation_only", False))

    output_dir = root / "artifacts" / "training" / configuration["run_name"]
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing matrix directory {output_dir}"
        )
    output_dir.mkdir(parents=True)
    current_git_state = git_state(root)
    matrix_configuration = {
        "experiment_type": "multi_subject_siren_matrix",
        "experiment_configuration": configuration,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "subject_ids": subject_ids,
        "subjects_csv": str(subjects_csv),
        "subjects_csv_sha256": file_sha256(subjects_csv),
        "holdout_csv": str(holdout_csv),
        "holdout_csv_sha256": file_sha256(holdout_csv),
        "holdout_indices": holdout_indices.tolist(),
        "holdout_sha256": holdout_sha256,
        "normalization_path": str(normalization_path),
        "normalization_sha256": normalization_sha256,
        "git": current_git_state,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "test_subjects_read": 0,
    }
    (output_dir / "matrix_configuration.json").write_text(
        json.dumps(matrix_configuration, indent=2) + "\n", encoding="utf-8"
    )

    per_subject_reports: list[dict[str, Any]] = []
    for subject_id in subject_ids:
        set_seed(seed)
        # Per-subject scaler: AMP state (growth/scale) must not leak across
        # subjects even though current runs are FP32 (GradScaler no-op).
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        label = subject_label(subject_id)
        field = load_subject_field(
            dataset_root,
            split_csv,
            subject_id,
            interpolation_only=interpolation_only,
        )
        total_directions = int(field.target_residual_db.shape[1])
        if not math.isclose(
            float(np.min(field.frequency_hz)), minimum_hz, abs_tol=1e-5
        ):
            raise ValueError("Configured frequency minimum does not match dataset")
        if not math.isclose(
            float(np.max(field.frequency_hz)), maximum_hz, abs_tol=1e-5
        ):
            raise ValueError("Configured frequency maximum does not match dataset")
        frequency_coordinate_array = frequency_coordinates(
            field.frequency_hz, frequency_mode, minimum_hz, maximum_hz
        )
        frequency_coordinate = torch.from_numpy(
            frequency_coordinate_array
        ).to(device)
        if np.max(holdout_indices) >= total_directions:
            raise ValueError("Holdout index exceeds the field direction count")
        train_indices = np.setdiff1d(
            np.arange(total_directions, dtype=np.int64),
            holdout_indices,
            assume_unique=True,
        )
        model = Siren(
            SirenConfig(
                input_dimension=3 + frequency_coordinate_array.shape[1],
                hidden_width=hidden_width,
                sine_layer_count=sine_layer_count,
                first_omega=first_omega,
                hidden_omega=hidden_omega,
                output_dimension=2,
            )
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        subject_output_dir = output_dir / label
        subject_output_dir.mkdir(parents=True)
        (subject_output_dir / "configuration.json").write_text(
            json.dumps(
                {
                    "matrix_run": configuration["run_name"],
                    "matrix_config_sha256": matrix_configuration["config_sha256"],
                    "subject_id": subject_id,
                    "subject_label": label,
                    "frequency_mode": frequency_mode,
                    "hidden_width": hidden_width,
                    "sine_layer_count": sine_layer_count,
                    "first_omega": first_omega,
                    "hidden_omega": hidden_omega,
                    "epochs": epochs,
                    "steps_per_epoch": steps_per_epoch,
                    "seed": seed,
                    "holdout_sha256": holdout_sha256,
                    "train_direction_count": int(
                        np.arange(total_directions, dtype=np.int64).size
                    ),
                    "test_subjects_read": 0,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if directions_per_batch > train_indices.size:
            raise ValueError("directions_per_batch exceeds train directions")

        initial_prediction = predict_field(
            model,
            field,
            frequency_coordinate,
            normalization,
            device,
            evaluation_block,
            use_amp,
        )
        initial_holdout = evaluate_field(
            initial_prediction, field, direction_indices=holdout_indices
        )
        print(
            f"[{configuration['run_name']}] {label} parameters="
            f"{parameter_count(model)} train={train_indices.size} "
            f"holdout={holdout_indices.size} "
            f"init_holdout_rmse={initial_holdout.residual_rmse_db:.4f}dB"
        )

        rng = np.random.default_rng(seed)
        history: list[MatrixEpochRecord] = []
        best_holdout_rmse = float("inf")
        best_epoch = 0
        nonfinite_detected = False
        run_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        target_mean = normalization.target_mean
        target_std = normalization.target_std
        for epoch in range(1, epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            sampled_losses: list[float] = []
            gradient_norms: list[float] = []
            for _ in range(steps_per_epoch):
                indices = np.sort(
                    rng.choice(
                        train_indices,
                        size=directions_per_batch,
                        replace=False,
                    )
                )
                xyz = torch.from_numpy(field.direction_features[indices, 2:5]).to(
                    device
                )
                query = coordinate_block(xyz, frequency_coordinate)
                target_db = torch.from_numpy(
                    field.target_residual_db[:, indices, :]
                ).to(device)
                target_normalized = (target_db - target_mean) / target_std
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    predicted = model(query)
                    predicted = predicted.reshape(
                        len(indices), field.frequency_hz.size, 2
                    ).permute(2, 0, 1)
                    loss = nn.functional.mse_loss(
                        predicted.float(), target_normalized.float()
                    )
                if not bool(torch.isfinite(loss).item()):
                    nonfinite_detected = True
                    raise FloatingPointError(
                        f"Non-finite training loss at epoch {epoch} ({label})"
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=gradient_clip
                )
                if not bool(torch.isfinite(gradient_norm).item()):
                    nonfinite_detected = True
                    raise FloatingPointError(
                        f"Non-finite gradient norm at epoch {epoch} ({label})"
                    )
                scaler.step(optimizer)
                scaler.update()
                sampled_losses.append(float(loss.detach().item()))
                gradient_norms.append(float(gradient_norm.detach().item()))

            prediction = predict_field(
                model,
                field,
                frequency_coordinate,
                normalization,
                device,
                evaluation_block,
                use_amp,
            )
            holdout_metrics = evaluate_field(
                prediction, field, direction_indices=holdout_indices
            )
            record = MatrixEpochRecord(
                epoch=epoch,
                sampled_train_mse_normalized=float(np.mean(sampled_losses)),
                holdout_residual_mae_db=holdout_metrics.residual_mae_db,
                holdout_residual_rmse_db=holdout_metrics.residual_rmse_db,
                holdout_mae_above_8khz_db=(
                    holdout_metrics.residual_mae_above_8khz_db
                ),
                holdout_mae_above_10khz_db=(
                    holdout_metrics.residual_mae_above_10khz_db
                ),
                holdout_first_spectral_difference_mae_db_per_bin=(
                    holdout_metrics.first_spectral_difference_mae_db_per_bin
                ),
                holdout_second_spectral_difference_mae_db_per_bin2=(
                    holdout_metrics.second_spectral_difference_mae_db_per_bin2
                ),
                holdout_notch_depth_mae_db=holdout_metrics.notch_depth_mae_db,
                gradient_norm_mean=float(np.mean(gradient_norms)),
                gradient_norm_max=float(np.max(gradient_norms)),
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                elapsed_seconds=time.perf_counter() - epoch_started,
            )
            history.append(record)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "siren_configuration": model.configuration.to_dict(),
                    "experiment_configuration": configuration,
                    "holdout_metrics": asdict(holdout_metrics),
                    "output_unit": "normalized_residual",
                    "residual_db_conversion": {
                        "target_mean": normalization.target_mean,
                        "target_std": normalization.target_std,
                        "normalization_path": str(normalization_path),
                        "normalization_sha256": normalization_sha256,
                    },
                },
                subject_output_dir / "last.pt",
            )
            if holdout_metrics.residual_rmse_db < best_holdout_rmse:
                best_holdout_rmse = holdout_metrics.residual_rmse_db
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "siren_configuration": model.configuration.to_dict(),
                        "experiment_configuration": configuration,
                        "holdout_metrics": asdict(holdout_metrics),
                        "output_unit": "normalized_residual",
                        "residual_db_conversion": {
                            "target_mean": normalization.target_mean,
                            "target_std": normalization.target_std,
                            "normalization_path": str(normalization_path),
                            "normalization_sha256": normalization_sha256,
                        },
                    },
                    subject_output_dir / "best.pt",
                )
            with (subject_output_dir / "history.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=asdict(record).keys()
                )
                writer.writeheader()
                writer.writerows(asdict(item) for item in history)
            print(
                f"epoch={epoch:03d} mse={record.sampled_train_mse_normalized:.6f} "
                f"holdout_rmse={record.holdout_residual_rmse_db:.4f}dB "
                f"time={record.elapsed_seconds:.1f}s"
            )

        best_payload = torch.load(
            subject_output_dir / "best.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(best_payload["model_state"])
        best_holdout_prediction = predict_field(
            model,
            field,
            frequency_coordinate,
            normalization,
            device,
            evaluation_block,
            use_amp,
        )
        best_holdout_metrics = evaluate_field(
            best_holdout_prediction, field, direction_indices=holdout_indices
        )
        best_full_metrics = evaluate_field(best_holdout_prediction, field)
        inference_ms = measure_inference_milliseconds(
            model,
            field,
            frequency_coordinate,
            device,
            evaluation_block,
            use_amp,
            inference_repetitions,
        )
        elapsed = time.perf_counter() - run_started
        decision = (
            "RETEST"
            if best_epoch == epochs or nonfinite_detected
            else "KEEP"
        )
        if nonfinite_detected:
            decision_reason = "A non-finite value was detected during training."
        elif best_epoch == epochs:
            decision_reason = (
                "The best holdout checkpoint occurred at the final epoch; "
                "extend the budget before trusting the plateau."
            )
        else:
            decision_reason = (
                "Holdout residual-field RMSE reached its best value before "
                "the budget ended without non-finite values."
            )
        report = {
            "status": "completed",
            "experiment_id": configuration["experiment_id"],
            "model_version": configuration["model_version"],
            "frequency_mode": frequency_mode,
            "first_omega": first_omega,
            "hidden_width": hidden_width,
            "sine_layer_count": sine_layer_count,
            "hidden_omega": hidden_omega,
            "subject_id": subject_id,
            "subject_label": label,
            "split": "train",
            "test_subjects_read": 0,
            "seed": seed,
            "parameter_count": parameter_count(model),
            "automatic_mixed_precision": use_amp,
            "output_unit": "normalized_residual",
            "train_direction_count": int(train_indices.size),
            "holdout_direction_count": int(holdout_indices.size),
            "initial_holdout_metrics": asdict(initial_holdout),
            "best_epoch": best_epoch,
            "best_holdout_metrics": asdict(best_holdout_metrics),
            "best_full_field_metrics": asdict(best_full_metrics),
            "completed_epochs": epochs,
            "elapsed_seconds": elapsed,
            "peak_cuda_allocated_mib": (
                torch.cuda.max_memory_allocated() / (1024.0**2)
            ),
            "full_subject_inference_milliseconds": inference_ms,
            "nan_or_inf": nonfinite_detected,
            "checkpoint_path": str(subject_output_dir / "best.pt"),
            "checkpoint_sha256": file_sha256(subject_output_dir / "best.pt"),
            "last_checkpoint_sha256": file_sha256(
                subject_output_dir / "last.pt"
            ),
            "result_path": str(subject_output_dir),
            "git": current_git_state,
            "decision": decision,
            "decision_reason": decision_reason,
        }
        (subject_output_dir / "training_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        per_subject_reports.append(report)
        print(
            f"[{configuration['run_name']}] {label} best_epoch={best_epoch} "
            f"holdout_rmse={best_holdout_metrics.residual_rmse_db:.4f}dB "
            f"full_rmse={best_full_metrics.residual_rmse_db:.4f}dB"
        )

    summary = {
        "status": "completed",
        "run_name": configuration["run_name"],
        "experiment_id": configuration["experiment_id"],
        "model_version": configuration["model_version"],
        "frequency_mode": frequency_mode,
        "first_omega": first_omega,
        "hidden_width": hidden_width,
        "sine_layer_count": sine_layer_count,
        "hidden_omega": hidden_omega,
        "seed": seed,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "test_subjects_read": 0,
        "holdout_sha256": holdout_sha256,
        "subjects": per_subject_reports,
        "aggregate_holdout_rmse_db": float(
            np.mean(
                [
                    item["best_holdout_metrics"]["residual_rmse_db"]
                    for item in per_subject_reports
                ]
            )
        ),
        "aggregate_holdout_mae_db": float(
            np.mean(
                [
                    item["best_holdout_metrics"]["residual_mae_db"]
                    for item in per_subject_reports
                ]
            )
        ),
        "aggregate_full_field_rmse_db": float(
            np.mean(
                [
                    item["best_full_field_metrics"]["residual_rmse_db"]
                    for item in per_subject_reports
                ]
            )
        ),
        "aggregate_full_field_mae_db": float(
            np.mean(
                [
                    item["best_full_field_metrics"]["residual_mae_db"]
                    for item in per_subject_reports
                ]
            )
        ),
    }
    (output_dir / "matrix_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    arguments = parse_arguments()
    root = project_root()
    config_path = arguments.config.resolve()
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    experiment_type = configuration.get("experiment_type")
    if experiment_type == "multi_subject_siren_matrix":
        run_matrix(configuration, root, config_path)
        return
    if experiment_type != "single_subject_siren_baseline":
        raise ValueError(
            "Unsupported experiment_type; expected "
            "'single_subject_siren_baseline' or 'multi_subject_siren_matrix'"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the recorded SIREN baseline")

    seed = int(configuration["seed"])
    set_seed(seed)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    dataset_root = (root / configuration["dataset_root"]).resolve()
    split_csv = (root / configuration["subject_split_csv"]).resolve()
    normalization_path = dataset_root / "training_statistics.json"
    normalization = Normalization.from_json(normalization_path)
    normalization_sha256 = file_sha256(normalization_path)
    field = load_subject_field(
        dataset_root,
        split_csv,
        int(configuration["subject_id"]),
        interpolation_only=bool(configuration["interpolation_only"]),
    )

    frequency_mapping = configuration["frequency_mapping"]
    minimum_hz = float(frequency_mapping["frequency_minimum_hz"])
    maximum_hz = float(frequency_mapping["frequency_maximum_hz"])
    if not math.isclose(float(np.min(field.frequency_hz)), minimum_hz, abs_tol=1e-5):
        raise ValueError("Configured frequency minimum does not match the dataset")
    if not math.isclose(float(np.max(field.frequency_hz)), maximum_hz, abs_tol=1e-5):
        raise ValueError("Configured frequency maximum does not match the dataset")
    frequency_coordinate_array = frequency_coordinates(
        field.frequency_hz,
        str(frequency_mapping["mode"]),
        minimum_hz,
        maximum_hz,
    )
    frequency_coordinate = torch.from_numpy(frequency_coordinate_array).to(device)

    model_config = SirenConfig(
        input_dimension=3 + frequency_coordinate_array.shape[1],
        hidden_width=int(configuration["model"]["hidden_width"]),
        sine_layer_count=int(configuration["model"]["sine_layer_count"]),
        first_omega=float(configuration["model"]["first_omega"]),
        hidden_omega=float(configuration["model"]["hidden_omega"]),
        output_dimension=2,
    )
    model = Siren(model_config).to(device)
    optimizer_name = str(configuration["optimizer"]["name"])
    if optimizer_name != "Adam":
        raise ValueError(f"Unsupported baseline optimizer {optimizer_name!r}")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(configuration["optimizer"]["learning_rate"]),
        weight_decay=float(configuration["optimizer"]["weight_decay"]),
    )
    use_amp = bool(configuration["automatic_mixed_precision"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    directions_per_batch = int(configuration["directions_per_batch"])
    if directions_per_batch > field.eligible_direction_indices.size:
        raise ValueError("directions_per_batch exceeds eligible directions")
    evaluation_block = int(configuration["evaluation_directions_per_block"])
    epochs = int(configuration["epochs"])
    steps_per_epoch = int(configuration["steps_per_epoch"])
    gradient_clip = float(configuration["gradient_clip"])

    output_dir = root / "artifacts" / "training" / configuration["run_name"]
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing experiment directory {output_dir}"
        )
    output_dir.mkdir(parents=True)
    current_git_state = git_state(root)
    runtime_configuration = {
        "experiment_configuration": configuration,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "normalization_path": str(normalization_path),
        "normalization_sha256": normalization_sha256,
        "git": current_git_state,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model_configuration": model_config.to_dict(),
        "parameter_count": parameter_count(model),
        "subject_file": str(field.path),
        "eligible_direction_count": int(field.eligible_direction_indices.size),
        "total_direction_count": int(field.target_residual_db.shape[1]),
        "frequency_count": int(field.frequency_hz.size),
        "output_unit": "normalized_residual",
        "test_subjects_read": 0,
    }
    (output_dir / "configuration.json").write_text(
        json.dumps(runtime_configuration, indent=2) + "\n", encoding="utf-8"
    )

    initial_prediction = predict_field(
        model,
        field,
        frequency_coordinate,
        normalization,
        device,
        evaluation_block,
        use_amp,
    )
    initial_metrics = evaluate_field(initial_prediction, field)
    (output_dir / "initial_metrics.json").write_text(
        json.dumps(asdict(initial_metrics), indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"device={runtime_configuration['device']} "
        f"subject={field.subject_label} parameters={parameter_count(model)} "
        f"directions={field.eligible_direction_indices.size} "
        f"frequencies={field.frequency_hz.size} amp={use_amp}"
    )
    print(
        f"initial mae={initial_metrics.residual_mae_db:.4f}dB "
        f"rmse={initial_metrics.residual_rmse_db:.4f}dB "
        f"hf10={initial_metrics.residual_mae_above_10khz_db:.4f}dB"
    )

    rng = np.random.default_rng(seed)
    history: list[EpochRecord] = []
    best_rmse = float("inf")
    best_epoch = 0
    nonfinite_detected = False
    run_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    target_mean = normalization.target_mean
    target_std = normalization.target_std
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        sampled_losses: list[float] = []
        gradient_norms: list[float] = []
        for _ in range(steps_per_epoch):
            indices = np.sort(
                rng.choice(
                    field.eligible_direction_indices,
                    size=directions_per_batch,
                    replace=False,
                )
            )
            xyz = torch.from_numpy(field.direction_features[indices, 2:5]).to(
                device
            )
            query = coordinate_block(xyz, frequency_coordinate)
            target_db = torch.from_numpy(
                field.target_residual_db[:, indices, :]
            ).to(device)
            target_normalized = (target_db - target_mean) / target_std
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                predicted = model(query)
                predicted = predicted.reshape(
                    len(indices), field.frequency_hz.size, 2
                ).permute(2, 0, 1)
                loss = nn.functional.mse_loss(
                    predicted.float(), target_normalized.float()
                )
            if not bool(torch.isfinite(loss).item()):
                nonfinite_detected = True
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=gradient_clip
            )
            if not bool(torch.isfinite(gradient_norm).item()):
                nonfinite_detected = True
                raise FloatingPointError(
                    f"Non-finite gradient norm at epoch {epoch}"
                )
            scaler.step(optimizer)
            scaler.update()
            sampled_losses.append(float(loss.detach().item()))
            gradient_norms.append(float(gradient_norm.detach().item()))

        prediction = predict_field(
            model,
            field,
            frequency_coordinate,
            normalization,
            device,
            evaluation_block,
            use_amp,
        )
        field_metrics = evaluate_field(prediction, field)
        record = EpochRecord(
            epoch=epoch,
            sampled_train_mse_normalized=float(np.mean(sampled_losses)),
            full_field_residual_mae_db=field_metrics.residual_mae_db,
            full_field_residual_rmse_db=field_metrics.residual_rmse_db,
            full_field_mae_above_8khz_db=(
                field_metrics.residual_mae_above_8khz_db
            ),
            full_field_mae_above_10khz_db=(
                field_metrics.residual_mae_above_10khz_db
            ),
            first_spectral_difference_mae_db_per_bin=(
                field_metrics.first_spectral_difference_mae_db_per_bin
            ),
            second_spectral_difference_mae_db_per_bin2=(
                field_metrics.second_spectral_difference_mae_db_per_bin2
            ),
            notch_depth_mae_db=field_metrics.notch_depth_mae_db,
            gradient_norm_mean=float(np.mean(gradient_norms)),
            gradient_norm_max=float(np.max(gradient_norms)),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            elapsed_seconds=time.perf_counter() - epoch_started,
        )
        history.append(record)
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            configuration,
            field_metrics,
            normalization,
            normalization_path,
            normalization_sha256,
        )
        if field_metrics.residual_rmse_db < best_rmse:
            best_rmse = field_metrics.residual_rmse_db
            best_epoch = epoch
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                configuration,
                field_metrics,
                normalization,
                normalization_path,
                normalization_sha256,
            )
        with (output_dir / "history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=asdict(record).keys())
            writer.writeheader()
            writer.writerows(asdict(item) for item in history)
        print(
            f"epoch={epoch:03d} mse={record.sampled_train_mse_normalized:.6f} "
            f"mae={record.full_field_residual_mae_db:.4f}dB "
            f"rmse={record.full_field_residual_rmse_db:.4f}dB "
            f"hf10={record.full_field_mae_above_10khz_db:.4f}dB "
            f"grad={record.gradient_norm_mean:.3f} "
            f"time={record.elapsed_seconds:.1f}s"
        )

    best_checkpoint = output_dir / "best.pt"
    best_payload = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state"])
    final_prediction = predict_field(
        model,
        field,
        frequency_coordinate,
        normalization,
        device,
        evaluation_block,
        use_amp,
    )
    final_metrics = evaluate_field(final_prediction, field)
    inference_ms = measure_inference_milliseconds(
        model,
        field,
        frequency_coordinate,
        device,
        evaluation_block,
        use_amp,
        int(configuration["inference_repetitions"]),
    )
    elapsed = time.perf_counter() - run_started
    improvement = (
        initial_metrics.residual_rmse_db - final_metrics.residual_rmse_db
    )
    budget_ended_at_best = best_epoch == epochs
    decision = (
        "RETEST"
        if budget_ended_at_best or nonfinite_detected
        else "KEEP"
    )
    if nonfinite_detected:
        decision_reason = "A non-finite value was detected during training."
    elif budget_ended_at_best:
        decision_reason = (
            "The implementation is trainable and improved the residual-field "
            "fit, but the best checkpoint occurred at the final epoch. Extend "
            "the budget before treating this as a representation ceiling."
        )
    else:
        decision_reason = (
            "The fixed training-subject residual-field RMSE improved without "
            "non-finite values and reached its best value before the budget "
            "ended."
        )
    report = {
        "status": "completed",
        "experiment_id": configuration["experiment_id"],
        "model_version": configuration["model_version"],
        "subject_id": field.subject_id,
        "subject_label": field.subject_label,
        "split": "train",
        "test_subjects_read": 0,
        "seed": seed,
        "parameter_count": parameter_count(model),
        "automatic_mixed_precision": use_amp,
        "output_unit": "normalized_residual",
        "initial_metrics": asdict(initial_metrics),
        "best_epoch": best_epoch,
        "best_metrics": asdict(final_metrics),
        "completed_epochs": epochs,
        "elapsed_seconds": elapsed,
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / (1024.0**2)
        ),
        "full_subject_inference_milliseconds": inference_ms,
        "nan_or_inf": nonfinite_detected,
        "checkpoint_path": str(best_checkpoint),
        "checkpoint_sha256": file_sha256(best_checkpoint),
        "last_checkpoint_sha256": file_sha256(output_dir / "last.pt"),
        "result_path": str(output_dir),
        "git": current_git_state,
        "decision": decision,
        "decision_reason": decision_reason,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
