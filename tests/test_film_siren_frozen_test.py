from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from mcar.data import Normalization
from mcar.evaluation.predict_film_siren_ensemble import split_subject_paths
from mcar.frozen_test import (
    atomic_update_registry,
    file_sha256,
    identity_sha256,
    load_verified_manifest,
    load_verified_registry,
)
from mcar.predictors import FilmSirenPredictor
from mcar.q26_condition import Q26Condition


def test_manifest_hash_guard_and_registry_identity(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    checkpoint = tmp_path / "last.pt"
    dataset_resource = tmp_path / "split.csv"
    evaluator_resource = tmp_path / "predictor.py"
    for path, content in (
        (config, b"{}\n"),
        (checkpoint, b"checkpoint"),
        (dataset_resource, b"subject_id,split\n"),
        (evaluator_resource, b"# evaluator\n"),
    ):
        path.write_bytes(content)
    manifest = {
        "status": "frozen",
        "model_version": "test-v1",
        "e_final": 140,
        "scheduler_horizon_cycles": 150,
        "ensemble": {"type": "residual_db_mean", "weights": [1.0]},
        "members": [
            {
                "seed": 1,
                "config": config.name,
                "config_sha256": file_sha256(config),
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": file_sha256(checkpoint),
            }
        ],
        "dataset": {
            "resources": [
                {"path": dataset_resource.name, "sha256": file_sha256(dataset_resource)}
            ]
        },
        "evaluator": {
            "resources": [
                {
                    "path": evaluator_resource.name,
                    "sha256": file_sha256(evaluator_resource),
                }
            ]
        },
    }
    manifest["identity_sha256"] = identity_sha256(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    verified = load_verified_manifest(manifest_path, tmp_path)
    registry = {
        "state": "not_started",
        "manifest_identity_sha256": verified["identity_sha256"],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    loaded = load_verified_registry(registry_path, verified)
    atomic_update_registry(registry_path, loaded, "started", run_name="locked")
    assert json.loads(registry_path.read_text(encoding="utf-8"))["state"] == "started"
    with pytest.raises(ValueError, match="not one of"):
        load_verified_registry(registry_path, verified)
    checkpoint.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="Resource SHA-256 mismatch"):
        load_verified_manifest(manifest_path, tmp_path)


def test_split_subject_paths_constructs_only_requested_split(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    split_csv = tmp_path / "split.csv"
    rows = ["subject_id,split"]
    for index in range(1, 45):
        label = f"P{index:04d}"
        rows.append(f"{label},val")
        path = dataset_root / "subjects" / label / "q26.h5"
        path.parent.mkdir(parents=True)
        path.touch()
    for index in range(45, 89):
        rows.append(f"P{index:04d},test")
    split_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    paths = split_subject_paths(dataset_root, split_csv, "val")
    assert len(paths) == 44
    assert all("P0045" not in str(path) for path in paths)


class _IdentityConditionNormalization:
    def normalize(self, value: np.ndarray) -> np.ndarray:
        return value


class _LocalEchoModel:
    def encode_condition(self, *_: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, 1), dtype=torch.float32)

    def __call__(self, query: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        assert query.shape[1] == 7
        return query[:, -2:]


def test_global_plus_local_predictor_appends_normalized_mca(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "dataset" / "subjects" / "P0001" / "q26.h5"
    source.parent.mkdir(parents=True)
    directions = np.zeros((3, 6), dtype=np.float32)
    directions[:, 2:5] = np.eye(3, dtype=np.float32)
    frequency = np.asarray([100.0, 1000.0], dtype=np.float32)
    local_mca = np.arange(12, dtype=np.float32).reshape(2, 3, 2) + 10.0
    with h5py.File(source, "w") as handle:
        handle.attrs["subject_id"] = 1
        handle.create_dataset("direction_features", data=directions)
        handle.create_dataset("frequency_hz", data=frequency)
        handle.create_dataset("mca_logmag_db", data=local_mca)
    condition = Q26Condition(
        subject_label="P0001",
        source_indices=np.arange(26, dtype=np.int64),
        binaural_magnitude_db=np.zeros((2, 26, 2), dtype=np.float32),
        xyz=np.zeros((26, 3), dtype=np.float32),
        mask=np.ones(26, dtype=bool),
        frequency_hz=frequency,
    )
    monkeypatch.setattr("mcar.predictors.build_q26_condition", lambda *a, **k: condition)
    predictor = FilmSirenPredictor.__new__(FilmSirenPredictor)
    predictor.device = torch.device("cpu")
    predictor.model = _LocalEchoModel()
    predictor.target_mean = -1.0
    predictor.target_std = 2.0
    predictor.frequency_mode = "dual"
    predictor.frequency_minimum_hz = 100.0
    predictor.frequency_maximum_hz = 1000.0
    predictor.conditioning_scope = "global_plus_local_mca"
    predictor.normalization = Normalization(10.0, 2.0, 0.0, 1.0, -1.0, 2.0, 0.0, 1.0)
    predictor.split_csv = tmp_path / "split.csv"
    predictor.q26_csv = tmp_path / "q26.csv"
    predictor.condition_normalization = _IdentityConditionNormalization()
    predictor.directions_per_block = 2
    predictor.allow_test = False
    actual = predictor.predict_residual_db(source)
    expected = (local_mca - 10.0) / 2.0 * 2.0 - 1.0
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
