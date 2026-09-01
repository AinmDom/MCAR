from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from mcar.q26_condition import Q26Condition, SparseCondition


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "configs/experiments/sonicom_bounded_e25_input_direction_sensitivity_v1.json"
)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_direction_sensitivity_protocol_is_nested_and_validation_only() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    grid = rows(ROOT / config["dataset"]["sparse_grid_file"])
    by_count = {
        count: [
            int(row["source_index_zero_based"])
            for row in grid
            if int(row["direction_count"]) == count
        ]
        for count in config["dataset"]["direction_counts"]
    }
    assert config["dataset"]["direction_counts"] == [14, 26, 50]
    assert set(by_count[14]).issubset(by_count[26])
    assert set(by_count[26]).issubset(by_count[50])
    assert all(len(indices) == len(set(indices)) == count for count, indices in by_count.items())
    assert config["dataset"]["fixed_evaluation_direction_count"] == 793 - 50
    assert config["dataset"]["split"] == "val"
    assert config["inference"]["test_access_allowed"] is False


def test_q14_and_q26_are_inherited_exactly() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    grid = rows(ROOT / config["dataset"]["sparse_grid_file"])
    old = rows(ROOT / "configs/data/sonicom_nested_sparse_grid_q6_q14_q26_v1.csv")
    q26 = rows(ROOT / "configs/data/sonicom_sparse_grid_q26_v1.csv")
    current_q14 = [int(row["source_index_zero_based"]) for row in grid if row["sparsity_level"] == "Q14"]
    old_q14 = [int(row["source_index_zero_based"]) for row in old if row["sparsity_level"] == "Q14"]
    current_q26 = [int(row["source_index_zero_based"]) for row in grid if row["sparsity_level"] == "Q26"]
    frozen_q26 = [int(row["source_index_zero_based"]) for row in q26]
    assert current_q14 == old_q14
    assert current_q26 == frozen_q26


def test_q50_is_left_right_reflection_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    grid = rows(ROOT / config["dataset"]["sparse_grid_file"])
    q50 = {
        (round(float(row["azimuth_deg"]) % 360.0, 8), round(float(row["elevation_deg"]), 8))
        for row in grid
        if row["sparsity_level"] == "Q50"
    }
    assert len(q50) == 50
    for azimuth, elevation in q50:
        assert (round((-azimuth) % 360.0, 8), elevation) in q50


def test_variable_sparse_condition_keeps_q26_guard() -> None:
    frequency = np.asarray([100.0, 200.0], dtype=np.float32)
    sparse = SparseCondition(
        subject_label="P0001",
        source_indices=np.arange(14, dtype=np.int64),
        binaural_magnitude_db=np.zeros((2, 14, 2), dtype=np.float32),
        xyz=np.zeros((14, 3), dtype=np.float32),
        mask=np.ones(14, dtype=bool),
        frequency_hz=frequency,
    )
    sparse.validate()
    q26 = Q26Condition(**sparse.__dict__)
    try:
        q26.validate()
    except ValueError as error:
        assert "shape [26]" in str(error)
    else:
        raise AssertionError("Q26Condition accepted a 14-direction input")


def test_main_model_identity_is_the_formal_e25_ensemble() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    manifest = json.loads(
        (ROOT / config["paper_main_model"]["manifest"]).read_text(encoding="utf-8")
    )
    assert manifest["identity_sha256"] == config["paper_main_model"][
        "manifest_identity_sha256"
    ]
    assert manifest["e_final"] == 25
    assert manifest["ensemble"]["weights"] == [1.0 / 3.0] * 3
