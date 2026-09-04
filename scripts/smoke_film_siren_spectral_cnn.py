"""One-step train-split smoke test for the Stage-D spectral refiner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mcar.data import Normalization
from mcar.models.film_siren import ConditionEncoderConfig, FilmSiren, FilmSirenConfig
from mcar.models.film_siren_spectral_cnn import (
    FilmSirenSpectralCNN,
    SpectralRefinerConfig,
)
from mcar.paths import project_root
from mcar.q26_condition import Q26MagnitudeNormalization
from mcar.training.train_film_siren import (
    condition_tensors,
    file_sha256,
    load_condition_cache,
    read_common_grid,
    split_subject_paths,
)
from mcar.training.train_film_siren_stage_c import (
    horizontal_interpolation_indices,
    loss_configuration,
    one_loss,
)
from mcar.training.train_mlp_v2 import (
    make_erb_center_frequencies_hz,
    make_erb_weights,
)
from mcar.training.train_siren import frequency_coordinates


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--directions", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    root = project_root()
    configuration = json.loads(arguments.config.read_text(encoding="utf-8"))
    seed = int(configuration["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(arguments.device)
    dataset_root = (root / configuration["dataset_root"]).resolve()
    split_csv = (root / configuration["subject_split_csv"]).resolve()
    q26_csv = (root / configuration["q26_csv"]).resolve()
    q26_normalization_path = (root / configuration["q26_normalization"]).resolve()
    normalization_path = dataset_root / "training_statistics.json"
    normalization = Normalization.from_json(normalization_path)
    q26_normalization = Q26MagnitudeNormalization.from_json(
        q26_normalization_path
    )
    subjects = load_condition_cache(
        split_subject_paths(dataset_root, split_csv, "train")[:1],
        dataset_root,
        split_csv,
        q26_csv,
        q26_normalization,
        "train",
    )
    directions, frequency, interpolation_mask, _ = read_common_grid(subjects[0].path)
    if not 1 <= arguments.directions <= 72:
        raise ValueError("directions must be between 1 and 72")
    interpolation_indices = np.flatnonzero(interpolation_mask)[:arguments.directions]
    horizontal_indices = horizontal_interpolation_indices(
        directions, interpolation_mask
    )[:arguments.directions]
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
    normalized_log_frequency = torch.from_numpy(
        (
            (np.log10(frequency) - normalization.log_frequency_mean)
            / normalization.log_frequency_std
        ).astype(np.float32)
    ).to(device)
    erb_weights = torch.from_numpy(make_erb_weights(frequency)).to(device)
    log_erb_weights = torch.log(torch.clamp(erb_weights, min=1e-12)).view(
        1, 1, erb_weights.shape[0], erb_weights.shape[1]
    )
    erb_centers = torch.from_numpy(make_erb_center_frequencies_hz()).to(device)

    backbone = FilmSiren(
        FilmSirenConfig(**configuration["model"]),
        ConditionEncoderConfig(**configuration["condition_encoder"]),
    )
    checkpoint_path = (root / configuration["initial_film_checkpoint"]).resolve()
    actual_hash = file_sha256(checkpoint_path).upper()
    if actual_hash != configuration["initial_film_checkpoint_sha256"]:
        raise ValueError("Initial checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    backbone.load_state_dict(checkpoint["model_state"])
    refiner_values = dict(configuration["spectral_refiner"])
    refiner_values["dilation_schedule"] = tuple(refiner_values["dilation_schedule"])
    model = FilmSirenSpectralCNN(
        backbone,
        SpectralRefinerConfig(**refiner_values),
        freeze_film_siren=True,
    ).to(device)
    model.train()
    magnitude, condition_xyz, condition_mask = condition_tensors(subjects[0], device)
    latent = model.encode_condition(magnitude, condition_xyz, condition_mask)
    loss, metrics = one_loss(
        model,
        latent,
        subjects[0],
        directions,
        frequency_coordinate,
        frequency_hz,
        interpolation_indices,
        horizontal_indices,
        normalization,
        log_erb_weights,
        erb_centers,
        loss_configuration(configuration["objective"]),
        device,
        str(configuration["conditioning_scope"]),
        normalized_log_frequency,
    )
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("Smoke loss is not finite")
    loss.backward()
    if any(parameter.grad is not None for parameter in model.film_siren.parameters()):
        raise AssertionError("Frozen FiLM-SIREN received gradients")
    output_gradient = model.spectral_cnn.output.weight.grad
    if output_gradient is None or not bool(torch.all(torch.isfinite(output_gradient))):
        raise FloatingPointError("Spectral output gradient is missing or non-finite")
    print(
        json.dumps(
            {
                "status": "passed",
                "split": "train",
                "subject_id": subjects[0].subject_id,
                "global_direction_count": int(interpolation_indices.size),
                "horizontal_direction_count": int(horizontal_indices.size),
                "loss": float(loss.detach().item()),
                "objective_total": metrics.total,
                "full_sphere_lsd_db": metrics.full_sphere_lsd_db,
                "lsd_loss_db": metrics.lsd_loss_db,
                "film_siren_frozen": model.film_siren_frozen,
                "spectral_output_gradient_norm": float(
                    torch.linalg.vector_norm(output_gradient).item()
                ),
                "test_subjects_read": 0,
                "cuda_peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
