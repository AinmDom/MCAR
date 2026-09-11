#!/usr/bin/env python3
"""Independent ARI FSC entry point; currently exposes a no-update smoke mode.

It deliberately has no test split option. Formal training is enabled only after
the smoke artifact is reviewed and the separate training loop is added.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from mcar.data import Normalization
from mcar.models.film_siren import ConditionEncoderConfig, FilmSiren, FilmSirenConfig
from mcar.models.film_siren_spectral_cnn import FilmSirenSpectralCNN, SpectralRefinerConfig
from mcar.q26_condition import Q26MagnitudeNormalization
from mcar.training.train_film_siren import conditioned_coordinate_block
from mcar.training.train_siren import frequency_coordinates


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load(path: Path, normalization: Q26MagnitudeNormalization, device: torch.device):
    with h5py.File(path, "r") as h:
        split = h.attrs["split"]
        split = split.decode() if isinstance(split, bytes) else str(split)
        if split not in {"train", "val"} or int(np.asarray(h.attrs["test_subjects_read"]).item()) != 0:
            raise PermissionError("ARI entry point rejects non-train/validation or test-accessed input")
        q = np.squeeze(h["sparse_direction_indices_zero_based"][:]).astype(np.int64)
        features = np.asarray(h["direction_features"][:], dtype=np.float32)
        frequency = np.squeeze(h["frequency_hz"][:]).astype(np.float32)
        mca = np.asarray(h["mca_logmag_db"][:, :4, :], dtype=np.float32)
        correction = np.asarray(h["correction_logmag_db"][:, :4, :], dtype=np.float32)
        target = np.asarray(h["target_residual_db"][:, :4, :], dtype=np.float32)
        observed = np.asarray(h["reference_logmag_db"][:], dtype=np.float32)[:, q, :]
    condition = torch.from_numpy(normalization.normalize(observed)[None]).to(device)
    xyz = torch.from_numpy(features[q, 2:5][None]).to(device)
    mask = torch.ones((1, 26), dtype=torch.bool, device=device)
    return condition, xyz, mask, features[:4], frequency, mca, correction, target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["status"] != "frozen_pre_training" or not torch.cuda.is_available():
        raise RuntimeError("frozen configuration and CUDA are required")
    stage_b_path = root / config["stage_b"]["source_config"]
    stage_d_path = root / config["stage_d"]["source_config"]
    if digest(stage_b_path) != config["stage_b"]["source_config_sha256"] or digest(stage_d_path) != config["stage_d"]["source_config_sha256"]:
        raise ValueError("frozen source config SHA mismatch")
    b = json.loads(stage_b_path.read_text(encoding="utf-8")); d = json.loads(stage_d_path.read_text(encoding="utf-8"))
    device = torch.device("cuda")
    normalization = Normalization.from_json(root / config["training_statistics"])
    qnorm = Q26MagnitudeNormalization.from_json(root / config["q26_normalization"])
    source = root / config["dataset_root"] / "train" / "nh2.h5"
    if not source.is_file():
        source = sorted((root / config["dataset_root"] / "train").glob("nh*.h5"))[0]
    condition, qxyz, mask, features, frequency, mca, correction, target = load(source, qnorm, device)
    film = FilmSiren(FilmSirenConfig(**b["model"]), ConditionEncoderConfig(**b["condition_encoder"])).to(device).eval()
    mapping = b["frequency_mapping"]
    fcoord = torch.from_numpy(frequency_coordinates(frequency, mapping["mode"], mapping["frequency_minimum_hz"], mapping["frequency_maximum_hz"])).to(device)
    query = conditioned_coordinate_block(torch.from_numpy(features[:, 2:5]).to(device), fcoord, "global_plus_local_mca", mca, normalization)
    with torch.no_grad():
        latent = film.encode_condition(condition, qxyz, mask)
        stage_b = film(query, latent).reshape(4, 463, 2).permute(2, 0, 1)
        refiner = FilmSirenSpectralCNN(film, SpectralRefinerConfig(**{**d["spectral_refiner"], "dilation_schedule": tuple(d["spectral_refiner"]["dilation_schedule"])}), freeze_film_siren=True).to(device).eval()
        final, base, delta = refiner.forward_grid(query, latent, torch.from_numpy((mca-normalization.mca_mean)/normalization.mca_std).to(device), torch.from_numpy((correction-normalization.correction_mean)/normalization.correction_std).to(device), torch.from_numpy((np.log10(frequency)-normalization.log_frequency_mean)/normalization.log_frequency_std).to(device), torch.from_numpy(features[:, 2:5]).to(device))
        mse = torch.mean((final - torch.from_numpy((target-normalization.target_mean)/normalization.target_std).to(device)) ** 2)
    result = {"status": "passed", "mode": "no_update_smoke", "stage_b_shape": list(stage_b.shape), "stage_d_shape": list(final.shape), "stage_d_base_shape": list(base.shape), "stage_d_delta_shape": list(delta.shape), "normalized_target_mse": float(mse.item()), "all_finite": bool(torch.isfinite(final).all().item()), "train_subjects_read": 1, "validation_subjects_read": 0, "test_subjects_read": 0, "parameter_updates": 0, "source_config_sha256_verified": True}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__": main()
