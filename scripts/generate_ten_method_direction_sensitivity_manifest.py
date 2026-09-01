"""Freeze ten-method sensitivity dependencies after result-blind implementation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/sonicom_ten_method_direction_sensitivity_v1.json"
OUTPUT = ROOT / "configs/experiments/sonicom_ten_method_direction_sensitivity_v1_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def resource(path: str, purpose: str) -> dict[str, object]:
    absolute = ROOT / path
    if not absolute.is_file():
        raise FileNotFoundError(absolute)
    return {
        "path": path,
        "sha256": sha256(absolute),
        "size_bytes": absolute.stat().st_size,
        "purpose": purpose,
    }


def identity(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def main() -> None:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Generate the manifest only from a clean tracked worktree")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["status"] != "preregistered" or config["dataset"]["split"] != "val":
        raise ValueError("Expected a validation-only preregistered config")
    if config["inference"]["test_access_allowed"]:
        raise PermissionError("Test access is forbidden")
    ranf = next(method for method in config["methods"] if method["id"] == "RANF")
    ranf_head = subprocess.check_output(
        ["git", "-C", str(ROOT / ranf["adapter_repository"]), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if ranf_head != ranf["adapter_commit"]:
        raise ValueError(f"RANF adapter commit mismatch: {ranf_head}")
    if subprocess.check_output(
        ["git", "-C", str(ROOT / ranf["adapter_repository"]), "status", "--porcelain"],
        text=True,
    ).strip():
        raise RuntimeError("RANF adapter repository is dirty")
    mcar = next(method for method in config["methods"] if method["id"] == "MCARv351")
    resources = [
        resource(config["protocol"], "result-blind protocol"),
        resource(config["dataset"]["sparse_grid_file"], "nested Q14/Q26/Q50 grid"),
        resource(config["dataset"]["prepared_input_manifest"], "hash-locked current-Q inputs"),
        resource("src/mcar/predictors.py", "variable-Q frozen predictors"),
        resource("src/mcar/evaluation/predict_ten_method_direction_sensitivity.py", "frozen comparator inference"),
        resource("matlab/+mcar/evaluate_ten_method_direction_sensitivity.m", "strict ten-method evaluator"),
        resource("configs/experiments/sonicom_film_siren_spectral_cnn_final_e190_ensemble_manifest.json", "Hybrid E190 identity"),
        resource("configs/experiments/sonicom_bounded_mcar_film_correction_final_e25_ensemble_manifest.json", "Bounded E25 identity"),
        resource("artifacts/frozen/fsp_ae_q26_epoch40_25db1eb83a1b647b.pt", "frozen FSP-AE checkpoint"),
    ]
    resources.extend(
        resource(component["checkpoint"], f"frozen MCAR {component['role']} checkpoint")
        for component in mcar["components"]
    )
    manifest = {
        "schema_version": "1.0",
        "status": "frozen_before_inference",
        "created_on": "2026-09-02",
        "experiment_config": CONFIG.relative_to(ROOT).as_posix(),
        "experiment_config_sha256": sha256(CONFIG),
        "split": "val",
        "subject_count": 44,
        "direction_counts": [14, 26, 50],
        "fixed_evaluation_direction_count": 743,
        "method_count": 10,
        "resources": resources,
        "ranf_adapter_commit": ranf_head,
        "git": {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "dirty": False,
        },
        "test_subject_count_read": 0,
    }
    manifest["identity_sha256"] = identity(manifest)
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT.relative_to(ROOT).as_posix())
    print(manifest["identity_sha256"])


if __name__ == "__main__":
    main()
