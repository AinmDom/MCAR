"""Derive a frozen secondary-test manifest for the selected FSC member."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/experiments/sonicom_complete_ten_method_secondary_deferred_test_v1_manifest.json"
OUTPUT = ROOT / "configs/experiments/sonicom_fsc_single_seed20260822_secondary_test_v1_manifest.json"
PREDICTION_ROOT = "artifacts/reconstruction/sonicom_fsc_q26_e190_single_seed20260822_test"
PRIMARY = "results/sonicom_fsc_single_seed20260822_q26_test_ten_method/metric_long.csv"


def digest_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest().upper()


def inventory() -> str:
    value = hashlib.sha256()
    for subject in sorted((ROOT / PREDICTION_ROOT / "subjects").iterdir()):
        prediction = subject / "prediction.h5"
        if not prediction.is_file():
            raise FileNotFoundError(prediction)
        value.update(subject.name.encode("utf-8") + b"\0")
        value.update(digest_file(prediction).encode("ascii") + b"\n")
    return value.hexdigest().upper()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    payload["created_on"] = "2026-09-08"
    payload["purpose"] = "Locked secondary/deferred characterization for the selected single FSC member"
    payload["methods"]["HYBRID"]["label"] = "FSC single member E190 (selected checkpoint)"
    payload["source_predictions"]["HYBRID"]["root"] = PREDICTION_ROOT
    payload["source_predictions"]["HYBRID"]["inventory_sha256"] = inventory()
    payload["primary_metrics"]["metric_long_csv"] = PRIMARY
    payload["primary_metrics"]["sha256"] = digest_file(ROOT / PRIMARY)
    payload["outputs"] = {
        "supplementary_results_root": "results/sonicom_fsc_single_seed20260822_secondary_deferred_test_v1",
        "complete_results_root": "results/sonicom_fsc_single_seed20260822_complete_test_v1",
        "delivery_root": "outputs/sonicom_fsc_single_seed20260822_complete_test_v1",
    }
    payload["commands"]["evaluate"] = (
        "D:/miniconda3/envs/ml/python.exe scripts/evaluate_complete_ten_method_test_metrics.py "
        "configs/experiments/sonicom_fsc_single_seed20260822_secondary_test_v1_manifest.json --allow-test"
    )
    payload.pop("identity_sha256", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    payload["identity_sha256"] = hashlib.sha256(canonical).hexdigest().upper()
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)), "identity": payload["identity_sha256"]}))


if __name__ == "__main__":
    main()
