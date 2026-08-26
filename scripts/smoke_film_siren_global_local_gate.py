"""CUDA smoke for the Stage B global/local-MCA bounded gate.

Reads one train subject only; validation targets and test subjects are untouched.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import torch
from torch import nn

from mcar.data import Normalization
from mcar.models.film_siren import ConditionEncoderConfig, FilmSiren, FilmSirenConfig
from mcar.paths import project_root
from mcar.q26_condition import Q26MagnitudeNormalization, build_q26_condition
from mcar.training.train_film_siren import conditioned_coordinate_block, set_seed
from mcar.training.train_siren import frequency_coordinates


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the global/local-MCA gate smoke")
    root = project_root()
    dataset_root = root / "data" / "processed" / "sonicom_residual_q26_v1"
    split_csv = root / "configs" / "data" / "sonicom_subject_split_v1.csv"
    q26_csv = root / "configs" / "data" / "sonicom_sparse_grid_q26_v1.csv"
    path = dataset_root / "subjects" / "P0002" / "q26.h5"
    condition = build_q26_condition(dataset_root, split_csv, 2, q26_csv)
    q26_normalization = Q26MagnitudeNormalization.from_json(
        root / "configs" / "data" / "siren_b_q26_normalization_v1.json"
    )
    normalization = Normalization.from_json(dataset_root / "training_statistics.json")
    with h5py.File(path, "r") as handle:
        directions = np.asarray(handle["direction_features"][:], dtype=np.float32)
        frequency = np.asarray(handle["frequency_hz"][:], dtype=np.float32).reshape(-1)
        interpolation = np.asarray(
            handle["interpolation_evaluation_mask"][:]
        ).reshape(-1).astype(bool)
    frequency_tensor = torch.from_numpy(
        frequency_coordinates(
            frequency,
            "dual",
            float(frequency.min()),
            float(frequency.max()),
        )
    ).cuda()
    magnitude = torch.from_numpy(
        q26_normalization.normalize(condition.binaural_magnitude_db)
    ).unsqueeze(0).cuda()
    condition_xyz = torch.from_numpy(condition.xyz).unsqueeze(0).cuda()
    mask = torch.from_numpy(condition.mask).unsqueeze(0).cuda()
    selected = np.flatnonzero(interpolation)[:16]
    with h5py.File(path, "r") as handle:
        target_db = np.asarray(
            handle["target_residual_db"][:, selected, :], dtype=np.float32
        )
        local_mca_db = np.asarray(
            handle["mca_logmag_db"][:, selected, :], dtype=np.float32
        )
    target = torch.from_numpy(target_db).cuda()
    target = (
        (target - normalization.target_mean) / normalization.target_std
    ).permute(1, 2, 0).reshape(-1, 2)
    xyz = torch.from_numpy(directions[selected, 2:5]).cuda()

    results: dict[str, dict[str, float | int]] = {}
    for scope in ("global_zero_local", "global_plus_local_mca"):
        set_seed(20260821)
        model = FilmSiren(
            FilmSirenConfig(
                coordinate_dimension=7,
                hidden_width=256,
                sine_layer_count=6,
                first_omega=20.0,
                hidden_omega=20.0,
                latent_dimension=128,
                modulation_variant="full",
                placement="all",
            ),
            ConditionEncoderConfig(latent_dimension=128),
        ).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        query = conditioned_coordinate_block(
            xyz,
            frequency_tensor,
            scope,
            local_mca_db if scope == "global_plus_local_mca" else None,
            normalization,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = model.forward_from_condition(
            query, magnitude, condition_xyz, mask
        )
        loss = nn.functional.mse_loss(prediction.float(), target.float())
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(f"Non-finite smoke result for {scope}")
        results[scope] = {
            "query_dimension": int(query.shape[1]),
            "loss": float(loss.detach().item()),
            "gradient_norm": float(gradient_norm.detach().item()),
        }
    print(
        json.dumps(
            {
                "status": "passed",
                "subject": "P0002",
                "split": "train",
                "results": results,
                "validation_targets_read": 0,
                "test_subjects_read": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
