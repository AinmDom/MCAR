"""Two-step CUDA smoke for the frozen Stage B default context.

The smoke reads one train subject and never evaluates a validation target.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import torch
from torch import nn

from mcar.data import Normalization
from mcar.models.film_siren import (
    ConditionEncoderConfig,
    FilmSiren,
    FilmSirenConfig,
)
from mcar.paths import project_root
from mcar.q26_condition import Q26MagnitudeNormalization, build_q26_condition
from mcar.training.train_film_siren import coordinate_block, set_seed
from mcar.training.train_siren import frequency_coordinates


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the FiLM-SIREN smoke")
    root = project_root()
    dataset_root = root / "data" / "processed" / "sonicom_residual_q26_v1"
    split_csv = root / "configs" / "data" / "sonicom_subject_split_v1.csv"
    q26_csv = root / "configs" / "data" / "sonicom_sparse_grid_q26_v1.csv"
    q26_normalization_path = (
        root / "configs" / "data" / "siren_b_q26_normalization_v1.json"
    )
    subject_id = 2
    path = dataset_root / "subjects" / "P0002" / "q26.h5"
    condition = build_q26_condition(
        dataset_root,
        split_csv,
        subject_id,
        q26_csv,
    )
    condition_normalization = Q26MagnitudeNormalization.from_json(
        q26_normalization_path
    )
    target_normalization = Normalization.from_json(
        dataset_root / "training_statistics.json"
    )
    with h5py.File(path, "r") as handle:
        directions = np.asarray(handle["direction_features"][:], dtype=np.float32)
        frequency = np.asarray(handle["frequency_hz"][:], dtype=np.float32).reshape(-1)
        interpolation = np.asarray(
            handle["interpolation_evaluation_mask"][:]
        ).reshape(-1).astype(bool)
    frequency_coordinate = frequency_coordinates(
        frequency,
        "dual",
        float(frequency.min()),
        float(frequency.max()),
    )
    set_seed(20260821)
    device = torch.device("cuda")
    model = FilmSiren(
        FilmSirenConfig(
            coordinate_dimension=5,
            hidden_width=256,
            sine_layer_count=6,
            first_omega=20.0,
            hidden_omega=20.0,
            latent_dimension=128,
            modulation_variant="phase",
            placement="hidden",
        ),
        ConditionEncoderConfig(latent_dimension=128),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    magnitude = torch.from_numpy(
        condition_normalization.normalize(condition.binaural_magnitude_db)
    ).unsqueeze(0).to(device)
    condition_xyz = torch.from_numpy(condition.xyz).unsqueeze(0).to(device)
    mask = torch.from_numpy(condition.mask).unsqueeze(0).to(device)
    frequency_tensor = torch.from_numpy(frequency_coordinate).to(device)
    eligible = np.flatnonzero(interpolation)
    rng = np.random.default_rng(20260821)
    losses: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        selected = np.sort(rng.choice(eligible, size=16, replace=False))
        with h5py.File(path, "r") as handle:
            target_db = np.asarray(
                handle["target_residual_db"][:, selected, :],
                dtype=np.float32,
            )
        xyz = torch.from_numpy(directions[selected, 2:5]).to(device)
        query = coordinate_block(xyz, frequency_tensor)
        target = torch.from_numpy(target_db).to(device)
        target = (
            (target - target_normalization.target_mean)
            / target_normalization.target_std
        ).permute(1, 2, 0).reshape(-1, 2)
        optimizer.zero_grad(set_to_none=True)
        prediction = model.forward_from_condition(
            query,
            magnitude,
            condition_xyz,
            mask,
        )
        loss = nn.functional.mse_loss(prediction.float(), target.float())
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("FiLM-SIREN smoke produced a non-finite value")
        losses.append(float(loss.detach().item()))
    print(
        json.dumps(
            {
                "status": "passed",
                "subject": "P0002",
                "split": "train",
                "steps": 2,
                "losses": losses,
                "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()
                / (1024.0**2),
                "validation_targets_read": 0,
                "test_subjects_read": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
