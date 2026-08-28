"""Checkpoint-backed smoke test for the Stage-E bounded correction model."""

from __future__ import annotations

import json

import torch

from mcar.models.bounded_mcar_film_correction import (
    BoundedCorrectionConfig,
    BoundedMcarFilmCorrection,
)
from mcar.models.film_siren import ConditionEncoderConfig, FilmSiren, FilmSirenConfig
from mcar.paths import project_root
from mcar.training.train_film_siren import file_sha256
from mcar.training.train_film_siren_stage_c import load_frozen_mcar


def main() -> None:
    root = project_root()
    config_path = (
        root
        / "configs/experiments/sonicom_bounded_mcar_film_correction_d1_seed20260821_e40.json"
    )
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    film = FilmSiren(
        FilmSirenConfig(**configuration["model"]),
        ConditionEncoderConfig(**configuration["condition_encoder"]),
    ).to(device)
    film_path = root / configuration["initial_film_checkpoint"]
    actual_film_hash = file_sha256(film_path).upper()
    if actual_film_hash != configuration["initial_film_checkpoint_sha256"]:
        raise RuntimeError("FiLM checkpoint hash mismatch")
    film_checkpoint = torch.load(film_path, map_location=device, weights_only=False)
    film.load_state_dict(film_checkpoint["model_state"])

    raw = dict(configuration["bounded_mcar_film_correction"])
    components = raw.pop("components")
    previous, _ = load_frozen_mcar(
        root / components[0]["checkpoint"],
        components[0]["checkpoint_sha256"],
        device,
    )
    candidate, _ = load_frozen_mcar(
        root / components[1]["checkpoint"],
        components[1]["checkpoint_sha256"],
        device,
    )
    model = BoundedMcarFilmCorrection(
        film,
        previous,
        candidate,
        BoundedCorrectionConfig(**raw),
    ).to(device)

    torch.manual_seed(20260829)
    direction_count = 2
    frequency_count = 463
    coordinates = torch.randn(direction_count * frequency_count, 7, device=device)
    point_features = torch.randn(
        direction_count, 2, frequency_count, 7, device=device
    )
    latent = torch.randn(1, configuration["model"]["latent_dimension"], device=device)
    final, base, correction, gate = model.forward_grid(
        coordinates, latent, point_features
    )
    identity_error = float((final - base).abs().max().item())
    if identity_error != 0.0 or float(correction.abs().max().item()) != 0.0:
        raise RuntimeError("Zero-initialized model is not exactly MCAR")
    final.square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in model.gate_output.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
        raise RuntimeError("Gate gradients are missing or nonfinite")

    print(
        json.dumps(
            {
                "status": "passed",
                "device": str(device),
                "zero_initialization_identity_error": identity_error,
                "maximum_absolute_gate": float(gate.abs().max().item()),
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                "test_subjects_read": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
