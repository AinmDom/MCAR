"""Freeze matched-density CNN configs from completed FiLM E130 checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260821, 20260822, 20260823)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


for q in (14, 50):
    for seed in SEEDS:
        film_cfg_path = ROOT / "configs/experiments" / f"sonicom_fsc_q{q}_film_seed{seed}_e130.json"
        film_cfg = json.loads(film_cfg_path.read_text(encoding="utf-8"))
        run_name = film_cfg["run_name"]
        report_path = ROOT / "artifacts/training" / run_name / "training_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "completed" or report.get("cycles") != 130:
            raise RuntimeError(f"FiLM run is not complete: {run_name}")
        if int(report.get("test_subjects_read", -1)) != 0:
            raise RuntimeError(f"FiLM run accessed test subjects: {run_name}")
        checkpoint = ROOT / "artifacts/training" / run_name / "last.pt"
        actual_hash = sha256(checkpoint)
        if actual_hash != str(report["authoritative_checkpoint_sha256"]).upper():
            raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
        cnn_path = ROOT / "configs/experiments" / f"sonicom_fsc_q{q}_cnn_seed{seed}_e190.json"
        cnn = json.loads(cnn_path.read_text(encoding="utf-8"))
        cnn["initial_film_checkpoint"] = f"artifacts/training/{run_name}/last.pt"
        cnn["initial_film_checkpoint_sha256"] = actual_hash
        cnn_path.write_text(json.dumps(cnn, indent=2) + "\n", encoding="utf-8")
        print(f"Q{q} seed{seed}: {run_name} {actual_hash}")
