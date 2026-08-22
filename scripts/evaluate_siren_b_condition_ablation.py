"""Evaluate frozen Stage B winner checkpoints under condition interventions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from mcar.data import Normalization
from mcar.evaluation.residual_metrics import solid_angle_weighted_residual_metrics
from mcar.models.film_siren import ConditionEncoderConfig, FilmSiren, FilmSirenConfig
from mcar.paths import project_root
from mcar.q26_condition import Q26MagnitudeNormalization
from mcar.training.train_film_siren import (
    condition_tensors,
    file_sha256,
    git_state,
    load_condition_cache,
    predict_subject,
    read_common_grid,
    split_subject_paths,
    validate_subject_identity,
)
from mcar.training.train_siren import frequency_coordinates


SEEDS = (20260821, 20260822, 20260823)
PROTOCOL = "experiments/film_siren/STAGE_B_CONDITION_ABLATION_PROTOCOL.md"


def cyclic_donor_indices(subject_ids: list[int]) -> list[int]:
    if len(subject_ids) < 2 or subject_ids != sorted(subject_ids):
        raise ValueError("Expected at least two sorted validation subject IDs")
    return list(range(1, len(subject_ids))) + [0]


@torch.no_grad()
def mean_train_latent(model: FilmSiren, subjects: list, device: torch.device) -> torch.Tensor:
    latents = []
    model.eval()
    for subject in subjects:
        magnitude, xyz, mask = condition_tensors(subject, device)
        latents.append(model.encode_condition(magnitude, xyz, mask))
    return torch.mean(torch.cat(latents, dim=0), dim=0, keepdim=True)


@torch.no_grad()
def evaluate_mode(
    model: FilmSiren,
    validation_subjects: list,
    donor_indices: list[int] | None,
    fixed_latent: torch.Tensor | None,
    directions: np.ndarray,
    frequency_coordinate: torch.Tensor,
    interpolation_indices: np.ndarray,
    direction_weights: np.ndarray,
    normalization: Normalization,
    device: torch.device,
    block_size: int,
    mode: str,
) -> tuple[float, float, list[dict[str, object]]]:
    if (donor_indices is None) == (fixed_latent is None):
        raise ValueError("Specify exactly one condition intervention")
    selected_weights = direction_weights[interpolation_indices]
    selected_mask = np.ones(interpolation_indices.size, dtype=bool)
    rows: list[dict[str, object]] = []
    maes: list[float] = []
    rmses: list[float] = []
    for target_index, target_subject in enumerate(validation_subjects):
        if fixed_latent is not None:
            latent = fixed_latent
            donor_label = "train_mean"
        else:
            donor = validation_subjects[donor_indices[target_index]]
            magnitude, xyz, mask = condition_tensors(donor, device)
            latent = model.encode_condition(magnitude, xyz, mask)
            donor_label = donor.subject_label
        normalized_prediction = predict_subject(
            model,
            target_subject,
            directions,
            frequency_coordinate,
            interpolation_indices,
            device,
            block_size,
            latent,
        )
        prediction_db = (
            normalized_prediction * normalization.target_std + normalization.target_mean
        )
        with h5py.File(target_subject.path, "r") as handle:
            validate_subject_identity(target_subject, handle)
            target_db = np.asarray(
                handle["target_residual_db"][:, interpolation_indices, :], dtype=np.float32
            )
        metrics = solid_angle_weighted_residual_metrics(
            prediction_db, target_db, selected_weights, selected_mask
        )
        maes.append(metrics.residual_mae_db)
        rmses.append(metrics.residual_rmse_db)
        rows.append(
            {
                "mode": mode,
                "target_subject": target_subject.subject_label,
                "condition_donor": donor_label,
                "weighted_mae_db": metrics.residual_mae_db,
                "weighted_rmse_db": metrics.residual_rmse_db,
            }
        )
    return float(np.mean(maes)), float(np.mean(rmses)), rows


def evaluate_seed(root: Path, seed: int, device: torch.device) -> tuple[dict[str, object], list[dict[str, object]]]:
    name = f"sonicom_film_siren_b_conditioning_check_all_seed{seed}_e150"
    artifact_dir = root / "artifacts" / "training" / name
    checkpoint_path = artifact_dir / "best.pt"
    report = json.loads((artifact_dir / "training_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "completed" or report.get("test_subjects_read") != 0:
        raise ValueError(f"Unsafe/incomplete checkpoint report for {name}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    configuration = checkpoint["experiment_configuration"]
    model = FilmSiren(
        FilmSirenConfig(**checkpoint["film_siren_configuration"]),
        ConditionEncoderConfig(**checkpoint["condition_encoder_configuration"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset_root = (root / configuration["dataset_root"]).resolve()
    split_csv = (root / configuration["subject_split_csv"]).resolve()
    q26_csv = (root / configuration["q26_csv"]).resolve()
    q26_normalization_path = (root / configuration["q26_normalization"]).resolve()
    q26_normalization = Q26MagnitudeNormalization.from_json(q26_normalization_path)
    normalization = Normalization.from_json(dataset_root / "training_statistics.json")
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
    validation_subjects.sort(key=lambda subject: subject.subject_id)
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
    donor_indices = cyclic_donor_indices(
        [subject.subject_id for subject in validation_subjects]
    )
    shuffled_mae, shuffled_rmse, shuffled_rows = evaluate_mode(
        model,
        validation_subjects,
        donor_indices,
        None,
        directions,
        frequency_coordinate,
        interpolation_indices,
        direction_weights,
        normalization,
        device,
        int(configuration["validation_directions_per_block"]),
        "condition_shuffle",
    )
    train_mean = mean_train_latent(model, train_subjects, device)
    mean_mae, mean_rmse, mean_rows = evaluate_mode(
        model,
        validation_subjects,
        None,
        train_mean,
        directions,
        frequency_coordinate,
        interpolation_indices,
        direction_weights,
        normalization,
        device,
        int(configuration["validation_directions_per_block"]),
        "train_mean_latent",
    )
    normal_mae = float(report["best_validation_weighted_mae_db"])
    summary = {
        "seed": seed,
        "run_name": name,
        "best_cycle": int(report["best_cycle"]),
        "normal_weighted_mae_db": normal_mae,
        "shuffle_weighted_mae_db": shuffled_mae,
        "shuffle_delta_vs_normal_db": shuffled_mae - normal_mae,
        "shuffle_relative_degradation_percent": 100.0 * (shuffled_mae / normal_mae - 1.0),
        "shuffle_weighted_rmse_db": shuffled_rmse,
        "train_mean_latent_weighted_mae_db": mean_mae,
        "train_mean_delta_vs_normal_db": mean_mae - normal_mae,
        "train_mean_relative_degradation_percent": 100.0 * (mean_mae / normal_mae - 1.0),
        "train_mean_latent_weighted_rmse_db": mean_rmse,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "test_subjects_read": 0,
    }
    for row in shuffled_rows + mean_rows:
        row["seed"] = seed
    return summary, shuffled_rows + mean_rows


def main() -> None:
    root = project_root()
    state = git_state(root)
    if state["dirty"]:
        raise RuntimeError("Formal Stage B ablation requires a clean Git worktree")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal Stage B ablation")
    device = torch.device("cuda")
    summaries: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        summary, rows = evaluate_seed(root, seed, device)
        summaries.append(summary)
        subject_rows.extend(rows)
        print(json.dumps(summary, indent=2))
    output_dir = root / "results" / "sonicom_film_siren_b_condition_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "per_subject.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=subject_rows[0].keys())
        writer.writeheader()
        writer.writerows(subject_rows)
    aggregate = {
        "protocol": PROTOCOL,
        "seeds": list(SEEDS),
        "normal_mean_weighted_mae_db": float(np.mean([row["normal_weighted_mae_db"] for row in summaries])),
        "shuffle_mean_weighted_mae_db": float(np.mean([row["shuffle_weighted_mae_db"] for row in summaries])),
        "train_mean_latent_mean_weighted_mae_db": float(np.mean([row["train_mean_latent_weighted_mae_db"] for row in summaries])),
        "shuffle_worse_seed_count": sum(row["shuffle_delta_vs_normal_db"] > 0.0 for row in summaries),
        "train_mean_worse_seed_count": sum(row["train_mean_delta_vs_normal_db"] > 0.0 for row in summaries),
        "git": state,
        "test_subjects_read": 0,
    }
    (output_dir / "decision.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
