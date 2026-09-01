from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

from mcar.predictors import FilmSirenSpectralCNNPredictor


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/sonicom_ten_method_direction_sensitivity_v1.json"


def test_registry_and_fixed_validation_boundary() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert [method["id"] for method in config["methods"]] == [
        "SHOnly", "SUpDEqSH", "SUpDEqNN", "SUpDEqBary", "MCA",
        "MCARv351", "FSPAE", "RANF", "HYBRID", "BOUNDED",
    ]
    assert config["dataset"]["split"] == "val"
    assert config["dataset"]["subject_count"] == 44
    assert config["dataset"]["direction_counts"] == [14, 26, 50]
    assert config["dataset"]["fixed_evaluation_direction_count"] == 743
    assert config["inference"]["test_access_allowed"] is False
    assert config["inference"]["parameter_updates_allowed"] is False


def test_grids_are_exactly_nested() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with (ROOT / config["dataset"]["sparse_grid_file"]).open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    values = {
        count: [
            int(row["source_index_zero_based"])
            for row in rows
            if int(row["direction_count"]) == count
        ]
        for count in (14, 26, 50)
    }
    assert len(values[14]) == 14 and len(values[26]) == 26 and len(values[50]) == 50
    assert set(values[14]).issubset(values[26])
    assert set(values[26]).issubset(values[50])


def test_frozen_hybrid_predictor_accepts_variable_sparse_condition() -> None:
    signature = inspect.signature(FilmSirenSpectralCNNPredictor)
    assert "condition_source_indices" in signature.parameters
    assert "condition_dataset_root" in signature.parameters


def test_frozen_model_identities_and_native_ranf_budget() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    methods = {method["id"]: method for method in config["methods"]}
    hybrid = json.loads((ROOT / methods["HYBRID"]["manifest"]).read_text(encoding="utf-8"))
    bounded = json.loads((ROOT / methods["BOUNDED"]["manifest"]).read_text(encoding="utf-8"))
    assert hybrid["identity_sha256"] == methods["HYBRID"]["manifest_identity_sha256"]
    assert bounded["identity_sha256"] == methods["BOUNDED"]["manifest_identity_sha256"]
    assert methods["RANF"]["adaptation_epochs"] == 1000
    assert methods["RANF"]["adaptation_batch_size"] == 3
