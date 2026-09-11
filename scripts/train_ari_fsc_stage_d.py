#!/usr/bin/env python3
"""Run frozen Stage-D spectral-CNN training through ARI-only adapters."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mcar.data import Normalization
from mcar.models.film_siren import ConditionEncoderConfig, FilmSiren, FilmSirenConfig
from mcar.models.film_siren_spectral_cnn import FilmSirenSpectralCNN, SpectralRefinerConfig
from mcar.q26_condition import Q26MagnitudeNormalization
from mcar.training import train_film_siren_stage_c as stage
from mcar.training.train_film_siren import conditioned_coordinate_block, condition_tensors
from mcar.training.train_siren import frequency_coordinates
from train_ari_fsc_stage_b import (
    horizontal_interpolation_indices,
    load_condition_cache,
    read_block,
    read_common_grid,
    split_subject_paths,
)


EXPECTED_STAGE_B_SHA256 = "74A27310D0E15F67FAB2823E14F915CF618EDB82F01745F030EA4A20F215B280"
EXPECTED_STAGE_D_SOURCE_SHA256 = "53585DF74C0CBE5A288DA8FC757B8AFD8DDE702E095E25A7B43EFFFCD567EB97"
FROZEN_SOURCE_FIELDS = (
    "model_version", "experiment_type", "search_stage", "frequency_mapping",
    "condition_encoder", "model", "spectral_refiner", "optimizer", "scheduler",
    "objective", "cycles", "steps_per_cycle", "global_directions_per_step",
    "horizontal_directions_per_step", "validation_interval_cycles",
    "validation_directions_per_block", "gradient_clip",
    "automatic_mixed_precision", "precision", "require_cuda",
    "conditioning_scope", "local_mca_policy", "formal_fixed_cycle",
    "checkpoint_policy", "freeze_film_siren",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def install_ari_adapters() -> None:
    stage.split_subject_paths = split_subject_paths
    stage.load_condition_cache = load_condition_cache
    stage.read_common_grid = read_common_grid
    stage.horizontal_interpolation_indices = horizontal_interpolation_indices
    stage.read_block = read_block


def validate_configuration(configuration: dict, root: Path) -> Path:
    if configuration.get("status") != "frozen_pre_stage_d":
        raise ValueError("Stage-D configuration is not frozen")
    if configuration.get("experiment_type") != "film_siren_spectral_cnn_stage_d":
        raise ValueError("configuration is not ARI Stage-D")
    if int(configuration["seed"]) != 20260911 or int(configuration["cycles"]) != 190:
        raise ValueError("frozen Stage-D seed/cycle mismatch")
    if configuration.get("checkpoint_policy") != "fixed_stop_cycle_last":
        raise ValueError("ARI Stage-D must use the E190 last checkpoint")
    if not bool(configuration.get("freeze_film_siren")):
        raise ValueError("FiLM-SIREN backbone must remain frozen")
    source = root / configuration["source_configuration"]
    if sha256(source) != EXPECTED_STAGE_D_SOURCE_SHA256:
        raise ValueError("frozen SONICOM Stage-D source SHA256 mismatch")
    source_configuration = json.loads(source.read_text(encoding="utf-8"))
    mismatches = [
        key for key in FROZEN_SOURCE_FIELDS
        if configuration.get(key) != source_configuration.get(key)
    ]
    if mismatches:
        raise ValueError(f"frozen Stage-D source-field mismatch: {mismatches}")
    adapter = root / configuration["ari_data_adapter"]
    if sha256(adapter) != str(configuration["ari_data_adapter_sha256"]).upper():
        raise ValueError("ARI data adapter SHA256 mismatch")
    checkpoint = root / configuration["initial_film_checkpoint"]
    expected = str(configuration["initial_film_checkpoint_sha256"]).upper()
    if expected != EXPECTED_STAGE_B_SHA256 or sha256(checkpoint) != expected:
        raise ValueError("authoritative ARI E130 checkpoint SHA256 mismatch")
    return checkpoint


def preflight(configuration: dict, root: Path, checkpoint_path: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    output = root / "artifacts" / "training" / configuration["run_name"]
    if output.exists():
        raise FileExistsError(f"formal output already exists: {output}")
    dataset = root / configuration["dataset_root"]
    split_csv = root / configuration["subject_split_csv"]
    train_paths = split_subject_paths(dataset, split_csv, "train")
    val_paths = split_subject_paths(dataset, split_csv, "val")
    try:
        split_subject_paths(dataset, split_csv, "test")
    except PermissionError:
        test_guard_verified = True
    else:
        raise PermissionError("ARI Stage-D test guard did not refuse test")
    if {p[0] for p in train_paths} & {p[0] for p in val_paths}:
        raise ValueError("train/validation leakage")
    qnorm = Q26MagnitudeNormalization.from_json(root / configuration["q26_normalization"])
    train = load_condition_cache(train_paths, None, None, None, qnorm, "train")
    val = load_condition_cache(val_paths, None, None, None, qnorm, "val")
    directions, frequency, mask, _ = read_common_grid(train[0].path)
    horizontal = horizontal_interpolation_indices(directions, mask)

    device = torch.device("cuda")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if int(checkpoint["cycle"]) != 130 or checkpoint.get("training_stage") != "film_siren_stage_c":
        raise ValueError("initial checkpoint is not the authoritative completed E130 model")
    if checkpoint["film_siren_configuration"] != configuration["model"]:
        raise ValueError("FiLM-SIREN model configuration mismatch")
    if checkpoint["condition_encoder_configuration"] != configuration["condition_encoder"]:
        raise ValueError("condition encoder configuration mismatch")
    film = FilmSiren(
        FilmSirenConfig(**configuration["model"]),
        ConditionEncoderConfig(**configuration["condition_encoder"]),
    ).to(device)
    film.load_state_dict(checkpoint["model_state"])
    refiner_values = dict(configuration["spectral_refiner"])
    refiner_values["dilation_schedule"] = tuple(refiner_values["dilation_schedule"])
    model = FilmSirenSpectralCNN(
        film, SpectralRefinerConfig(**refiner_values), freeze_film_siren=True
    ).to(device).eval()
    normalization = Normalization.from_json(dataset / "training_statistics.json")
    selected = np.flatnonzero(mask)[:4]
    target, mca, features, strict = read_block(train[0], selected, strict_ild=True)
    correction = stage.read_correction_block(train[0], selected)
    mapping = configuration["frequency_mapping"]
    fcoord = torch.from_numpy(frequency_coordinates(
        frequency, mapping["mode"], mapping["frequency_minimum_hz"],
        mapping["frequency_maximum_hz"]
    )).to(device)
    magnitude, xyz, condition_mask = condition_tensors(train[0], device)
    query_xyz = torch.from_numpy(features[:, 2:5]).to(device)
    query = conditioned_coordinate_block(
        query_xyz, fcoord, "global_plus_local_mca", mca, normalization
    )
    with torch.no_grad():
        latent = model.encode_condition(magnitude, xyz, condition_mask)
        final, base, delta = model.forward_grid(
            query,
            latent,
            torch.from_numpy((mca - normalization.mca_mean) / normalization.mca_std).to(device),
            torch.from_numpy((correction - normalization.correction_mean) / normalization.correction_std).to(device),
            torch.from_numpy(
                ((np.log10(frequency) - normalization.log_frequency_mean) /
                 normalization.log_frequency_std).astype(np.float32)
            ).to(device),
            query_xyz,
        )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.film_siren.parameters() if not p.requires_grad)
    all_finite = all(torch.isfinite(x).all().item() for x in (final, base, delta))
    if final.shape != (2, 4, 463) or not all_finite or frozen == 0 or trainable == 0:
        raise ValueError("Stage-D model/shape/finite/freeze preflight failed")
    return {
        "status": "passed",
        "train_subjects": len(train),
        "validation_subjects": len(val),
        "test_subjects_read": 0,
        "test_guard_verified": test_guard_verified,
        "directions": len(directions),
        "interpolation_directions": int(mask.sum()),
        "horizontal_interpolation_directions": len(horizontal),
        "frequencies": len(frequency),
        "stage_d_shape": list(final.shape),
        "stage_b_shape": list(base.shape),
        "zero_initial_delta": bool(torch.count_nonzero(delta).item() == 0),
        "all_finite": bool(all_finite and np.isfinite(target).all() and
                           np.isfinite(strict["reference_ild_db"]).all()),
        "film_siren_frozen_parameters": frozen,
        "spectral_cnn_trainable_parameters": trainable,
        "initial_checkpoint_cycle": int(checkpoint["cycle"]),
        "initial_checkpoint_sha256": sha256(checkpoint_path),
        "cycles": int(configuration["cycles"]),
        "optimizer_steps_per_epoch": len(train),
        "planned_optimizer_steps": int(configuration["cycles"]) * len(train),
        "parameter_updates": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = validate_configuration(configuration, root)
    install_ari_adapters()
    if args.preflight:
        result = preflight(configuration, root, checkpoint)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    if args.output is not None:
        raise ValueError("--output is only valid with --preflight")
    stage.run(configuration, root, config_path)


if __name__ == "__main__":
    main()
