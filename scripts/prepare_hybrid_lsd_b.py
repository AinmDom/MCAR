"""Freeze the authorized three LSD-B configs and hash existing validation controls."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "configs/experiments/sonicom_hybrid_lsd_b_e190_preparation_manifest.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def evidence(relative: str) -> dict:
    path = ROOT / relative
    return {"path": relative, "sha256": sha(path), "bytes": path.stat().st_size}


def main():
    if (ROOT / MANIFEST).exists():
        raise FileExistsError("Preparation is already frozen")
    old_manifest_path = "configs/experiments/sonicom_film_siren_spectral_cnn_final_e190_ensemble_manifest.json"
    old_manifest = json.loads((ROOT / old_manifest_path).read_text())
    members = []
    for member in old_manifest["members"]:
        seed = member["seed"]
        old = json.loads((ROOT / member["config"]).read_text())
        for key in ("config", "checkpoint"):
            assert sha(ROOT / member[key]) == member[f"{key}_sha256"]
        parent = evidence(old["initial_film_checkpoint"])
        assert parent["sha256"] == old["initial_film_checkpoint_sha256"]
        old_report = json.loads((ROOT / member["training_report"]).read_text())
        assert old_report["status"] == "completed" and old_report["cycles"] == 190
        assert old_report["test_subjects_read"] == 0 and not old_report["git"]["dirty"]
        new = json.loads(json.dumps(old))
        run_name = f"sonicom_film_siren_spectral_cnn_lsd_b_seed{seed}_e190"
        new.update(created_on="2026-09-04", experiment_id=f"HYBRID-LSD-B-SEED{seed}-E190",
                   model_version="film-siren-spectral-cnn-lsd-b-v1",
                   protocol="experiments/film_siren/STAGE_D_HYBRID_LSD_B_E190_PROTOCOL.md",
                   search_stage="lsd_b_fixed_cycle_e190", run_name=run_name)
        new["objective"].update(lsd_weight=1.0, lsd_epsilon_db=1e-6)
        config = f"configs/experiments/{run_name}.json"
        assert not (ROOT / "artifacts/training" / run_name).exists()
        with (ROOT / config).open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(new, indent=2) + "\n")
        members.append({"seed": seed, "config": evidence(config), "parent": parent,
                        "existing_control_config": evidence(member["config"]),
                        "existing_control_checkpoint": evidence(member["checkpoint"]),
                        "existing_control_report": evidence(member["training_report"]),
                        "output_directory": f"artifacts/training/{run_name}"})
    source_paths = [old_manifest_path,
        "configs/data/sonicom_subject_split_v1.csv", "configs/data/sonicom_sparse_grid_q26_v1.csv",
        "configs/data/siren_b_q26_normalization_v1.json",
        "data/processed/sonicom_residual_q26_v1/training_statistics.json"]
    for directory in ("results/sonicom_film_secondary_metrics_v1_validation",
                      "results/sonicom_film_siren_spectral_cnn_final_e190_validation"):
        source_paths.extend(p.relative_to(ROOT).as_posix() for p in sorted((ROOT / directory).glob("*"))
                            if p.suffix in (".json", ".csv"))
    with (ROOT / "configs/data/sonicom_subject_split_v1.csv").open() as stream:
        split_rows = list(csv.DictReader(stream))
    # Split metadata is permitted; only val labels are used to address predictions.
    val_labels = {row["subject_id"] for row in split_rows if row["split"] == "val"}
    prediction_root = ROOT / "artifacts/reconstruction/sonicom_film_siren_spectral_cnn_final_e190_ensemble_validation/subjects"
    predictions = sorted(prediction_root.glob("P*/prediction.h5"))
    assert {p.parent.name for p in predictions} == val_labels and len(predictions) == 44
    with (ROOT / "results/sonicom_film_secondary_metrics_v1_validation/per_subject_metrics.csv").open() as stream:
        control_rows = [r for r in csv.DictReader(stream) if r["Method"] == "HYBRID" and r["Endpoint"] == "FullSphereLSD"]
    assert len(control_rows) == 44 and {r["SubjectLabel"] for r in control_rows} == val_labels
    manifest = {"status": "prepared_not_started", "created_on": "2026-09-04",
        "protocol": "experiments/film_siren/STAGE_D_HYBRID_LSD_B_E190_PROTOCOL.md",
        "members": members, "sources": [evidence(p) for p in source_paths],
        "existing_validation_prediction_files": [evidence(p.relative_to(ROOT).as_posix()) for p in predictions],
        "existing_hybrid_val_lsd_mean_db": sum(float(r["Value"]) for r in control_rows) / 44,
        "test_subjects_read": 0, "control_training_runs_started": 0,
        "validation_subject_labels": sorted(val_labels),
        "integrity": "Existing config/checkpoint hashes and completed E190 reports verified; 44 validation prediction byte hashes captured. No prediction tensor re-evaluation."}
    (ROOT / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "prepared", "members": len(members), "control_predictions": len(predictions), "test_subjects_read": 0}))


if __name__ == "__main__":
    main()
