from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/sonicom_learned_methods_sparsity_v1.json"


def load_config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_nested_sparse_grids_match_preregistered_indices() -> None:
    config = load_config()
    grid_path = ROOT / config["dataset"]["sparse_grid_file"]
    with grid_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    previous: set[int] = set()
    for count in config["dataset"]["direction_counts"]:
        indices = [
            int(row["source_index_zero_based"])
            for row in rows
            if int(row["direction_count"]) == count
        ]
        assert indices == config["sparse_grid_selection"][
            f"q{count}_source_indices_zero_based"
        ]
        assert len(indices) == len(set(indices)) == count
        assert previous.issubset(indices)
        previous = set(indices)


def test_protocol_is_learning_only_and_uses_one_fixed_target_mask() -> None:
    config = load_config()
    assert [method["id"] for method in config["methods"]] == [
        "MCARv32",
        "FSPAE",
        "RANF",
    ]
    assert config["dataset"]["subject_count"] == 44
    assert config["dataset"]["reference_direction_count"] == 793
    assert "exclude all 26 directions" in config["dataset"][
        "evaluation_mask_policy"
    ]
    assert config["methods"][0]["parameter_updates_allowed"] is False
    assert config["methods"][1]["parameter_updates_allowed"] is False
    assert config["methods"][2]["pretraining_updates_allowed"] is False
    assert config["methods"][2]["adaptation_updates_allowed"] is True


def test_q26_checkpoint_hashes_are_frozen() -> None:
    config = load_config()
    hashes = {method["id"]: method.get("checkpoint_sha256") for method in config["methods"]}
    assert hashes["MCARv32"] == (
        "1076EBA7EC24C25914E5569E1C14EDDB584A6649E7551194FBA38984FE11F10C"
    )
    assert hashes["FSPAE"] == (
        "25DB1EB83A1B647B2E0E12C6BE1FF9EE31BF2B216DC50B57C19ADC84D1E12F5F"
    )
    assert config["methods"][2]["checkpoint_sha256"].lower() == (
        "7b288f6f9198664e9ce75fee418e90ec3da2f5d0b7996a4edefe7d5f8da3e367"
    )
