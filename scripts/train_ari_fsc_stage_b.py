#!/usr/bin/env python3
"""Run frozen Stage-B FiLM-SIREN training through ARI-only adapters.

The upstream SONICOM training module is not modified.  This entry point swaps
only its dataset-facing functions and preserves its model, objective,
optimizer, scheduler, validation and checkpoint implementation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from mcar.q26_condition import Q26MagnitudeNormalization
from mcar.training import train_film_siren_stage_c as stage
from mcar.training.train_film_siren import SubjectCondition


def split_subject_paths(dataset_root: Path, split_csv: Path, split: str, direction_count: int = 26):
    if split == "test":
        raise PermissionError("ARI Stage-B never permits test access")
    if split not in {"train", "val"} or direction_count != 26:
        raise ValueError("invalid ARI training split or observation count")
    with split_csv.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == split]
    expected = {"train": 130, "val": 22}[split]
    if len(rows) != expected or len({row["subject_id"] for row in rows}) != expected:
        raise ValueError(f"frozen ARI {split} count/uniqueness mismatch")
    output = []
    for row in rows:
        label = row["subject_id"]
        if not label.startswith("nh") or not label[2:].isdigit():
            raise ValueError(f"invalid ARI subject {label}")
        path = dataset_root / split / f"{label}.h5"
        if not path.is_file():
            raise FileNotFoundError(path)
        output.append((int(label[2:]), label, path))
    return output


def load_condition_cache(subject_paths, dataset_root, split_csv, q26_csv,
                         normalization: Q26MagnitudeNormalization, split,
                         direction_count=26):
    del dataset_root, split_csv, q26_csv
    result = []
    for subject_id, label, path in subject_paths:
        with h5py.File(path, "r") as handle:
            actual = handle.attrs["split"]
            actual = actual.decode() if isinstance(actual, bytes) else str(actual)
            if actual != split or int(np.asarray(handle.attrs["test_subjects_read"]).item()) != 0:
                raise PermissionError(f"split/test-access violation: {path}")
            indices = np.squeeze(handle["sparse_direction_indices_zero_based"][:]).astype(np.int64)
            reference = np.asarray(handle["reference_logmag_db"][:], dtype=np.float32)[:, indices, :]
            xyz = np.asarray(handle["direction_features"][:], dtype=np.float32)[indices, 2:5]
        if reference.shape != (2, direction_count, 463) or xyz.shape != (direction_count, 3):
            raise ValueError(f"invalid ARI condition shape: {path}")
        result.append(SubjectCondition(subject_id, label, split, path,
            normalization.normalize(reference), xyz, np.ones(direction_count, dtype=bool)))
    return result


def read_common_grid(path: Path):
    with h5py.File(path, "r") as handle:
        directions = np.asarray(handle["direction_features"][:], dtype=np.float32)
        frequency = np.squeeze(handle["frequency_hz"][:]).astype(np.float32)
        mask = np.squeeze(handle["interpolation_evaluation_mask"][:]).astype(bool)
    if directions.shape != (1550, 6) or frequency.shape != (463,) or mask.shape != (1550,) or mask.sum() != 1524:
        raise ValueError("frozen ARI grid shape/count mismatch")
    weights = directions[:, 5].astype(np.float64)
    if np.any(weights[mask] <= 0) or np.any(weights[~mask] != 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("frozen ARI evaluation weights invalid")
    return directions, frequency, mask, weights


def horizontal_interpolation_indices(directions: np.ndarray, interpolation_mask: np.ndarray):
    indices = np.flatnonzero(interpolation_mask & (np.abs(directions[:, 1]) <= 1e-6))
    if indices.size < 16:
        raise ValueError(f"insufficient ARI horizontal interpolation directions: {indices.size}")
    return indices


def read_block(subject: SubjectCondition, indices: np.ndarray, *, strict_ild: bool):
    """Read one ARI block while normalizing its [1, direction] ILD layout."""
    with h5py.File(subject.path, "r") as handle:
        stage.validate_subject_identity(subject, handle)
        target = np.asarray(handle["target_residual_db"][:, indices, :], dtype=np.float32)
        mca = np.asarray(handle["mca_logmag_db"][:, indices, :], dtype=np.float32)
        direction_features = np.asarray(handle["direction_features"][indices, :], dtype=np.float32)
        if not strict_ild:
            return target, mca, direction_features, None
        strict = handle["strict_ild"]
        reference_ild = np.squeeze(strict["reference_ild_db"][:]).astype(np.float32)
        if reference_ild.shape != (1550,):
            raise ValueError(f"invalid ARI reference ILD shape: {reference_ild.shape}")
        metadata = {
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
            "reference_ild_db": reference_ild[indices],
            "single_sided_frequency_count": int(
                np.asarray(strict.attrs["single_sided_frequency_count"]).item()
            ),
            "hrir_length": int(np.asarray(strict.attrs["hrir_length"]).item()),
        }
    return target, mca, direction_features, metadata


def build_configuration(execution: dict, root: Path) -> dict:
    source_path = root / execution["stage_b"]["source_config"]
    actual_sha = hashlib.sha256(source_path.read_bytes()).hexdigest().upper()
    if actual_sha != execution["stage_b"]["source_config_sha256"]:
        raise ValueError("Stage-B source configuration SHA256 mismatch")
    configuration = json.loads(source_path.read_text(encoding="utf-8"))
    configuration.update({
        "dataset_root": execution["dataset_root"],
        "subject_split_csv": execution["subject_split_csv"],
        "q26_csv": execution["q26_csv"],
        "q26_normalization": execution["q26_normalization"],
        "seed": execution["seed"],
        "cycles": execution["stage_b"]["cycles"],
        "run_name": execution["run_names"]["stage_b"],
        "require_clean_git": False,
        "test_policy": execution["test_policy"],
    })
    configuration.pop("ensemble", None)
    configuration["external_database"] = "ARI"
    configuration["epoch_subject_count"] = 130
    configuration["source_steps_per_cycle_field"] = configuration.pop("steps_per_cycle", None)
    return configuration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    execution = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    configuration = build_configuration(execution, root)
    stage.split_subject_paths = split_subject_paths
    stage.load_condition_cache = load_condition_cache
    stage.read_common_grid = read_common_grid
    stage.horizontal_interpolation_indices = horizontal_interpolation_indices
    stage.read_block = read_block
    if args.preflight:
        train = split_subject_paths(root / configuration["dataset_root"], root / configuration["subject_split_csv"], "train")
        val = split_subject_paths(root / configuration["dataset_root"], root / configuration["subject_split_csv"], "val")
        normalization = Q26MagnitudeNormalization.from_json(root / configuration["q26_normalization"])
        train_cache = load_condition_cache(train, None, None, None, normalization, "train")
        val_cache = load_condition_cache(val, None, None, None, normalization, "val")
        directions, frequency, mask, _ = read_common_grid(train[0][2])
        horizontal = horizontal_interpolation_indices(directions, mask)
        print(json.dumps({"status":"passed","train":len(train_cache),"val":len(val_cache),"directions":len(directions),"frequencies":len(frequency),"interpolation":int(mask.sum()),"horizontal_interpolation":len(horizontal),"optimizer_steps_per_epoch":len(train),"cycles":configuration["cycles"],"condition_shape":list(train_cache[0].normalized_magnitude.shape),"test_subjects_read":0}))
        return
    stage.run(configuration, root, args.config.resolve())


if __name__ == "__main__":
    main()
