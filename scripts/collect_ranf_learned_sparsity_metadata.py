"""Collect auditable RANF adaptation cost and checkpoint metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


PARAMETER_PATTERN = re.compile(r"Number of trainable parameters:\s*(\d+)")
TIME_PATTERN = re.compile(
    r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([^\r\n]+)"
)
MEMORY_PATTERN = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
PHASE_PATTERN = re.compile(r"^\[([^]]+)\] (Starting adaptation|adaptation exited with status 0)$", re.MULTILINE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def elapsed_seconds(value: str) -> float:
    parts = [float(part) for part in value.strip().split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"Unexpected elapsed-time value: {value}")


def collect(direction_count: int, exp_root: Path) -> dict[str, object]:
    required = [
        exp_root / "best.ckpt",
        exp_root / "adaptation.ckpt",
        exp_root / "config.yaml",
        exp_root / "log/adaptation/adaptation.log",
        exp_root / "adaptation_time.txt",
        exp_root / "formal_pipeline_report.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete RANF Q{direction_count}: {missing}")
    adaptation_log = required[3].read_text(encoding="utf-8", errors="replace")
    time_text = required[4].read_text(encoding="utf-8", errors="replace")
    parameters = PARAMETER_PATTERN.findall(adaptation_log)
    elapsed = TIME_PATTERN.search(time_text)
    memory = MEMORY_PATTERN.search(time_text)
    if not parameters or elapsed is None or memory is None:
        raise ValueError(f"Cannot parse RANF Q{direction_count} cost metadata")
    pipeline = json.loads(required[5].read_text(encoding="utf-8"))
    if int(pipeline["status"]) != 0:
        raise RuntimeError(f"RANF Q{direction_count} pipeline did not complete")
    return {
        "direction_count": direction_count,
        "adaptation_epochs": 1000,
        "adaptation_batch_size": 3,
        "trainable_parameter_count": int(parameters[-1]),
        "adaptation_elapsed_seconds": elapsed_seconds(elapsed.group(1)),
        "adaptation_maximum_resident_kb": int(memory.group(1)),
        "pipeline_elapsed_seconds_including_evaluation": int(
            pipeline["elapsed_seconds"]
        ),
        "pretrained_checkpoint_sha256": sha256(required[0]),
        "adapted_checkpoint_sha256": sha256(required[1]),
        "source_experiment_root": str(exp_root),
    }


def collect_q26(exp_root: Path) -> dict[str, object]:
    checkpoint = exp_root / "best.ckpt"
    adapted = exp_root / "adaptation.ckpt"
    adaptation_log = exp_root / "log/adaptation/adaptation.log"
    pipeline_log = exp_root / "final_pipeline_console.log"
    for path in (checkpoint, adapted, adaptation_log, pipeline_log):
        if not path.is_file():
            raise FileNotFoundError(path)
    log_text = adaptation_log.read_text(encoding="utf-8", errors="replace")
    parameters = PARAMETER_PATTERN.findall(log_text)
    phases = PHASE_PATTERN.findall(
        pipeline_log.read_text(encoding="utf-8", errors="replace")
    )
    if not parameters or len(phases) < 2:
        raise ValueError("Cannot parse formal RANF Q26 metadata")
    started = datetime.fromisoformat(phases[-2][0])
    completed = datetime.fromisoformat(phases[-1][0])
    return {
        "direction_count": 26,
        "adaptation_epochs": 1000,
        "adaptation_batch_size": 3,
        "trainable_parameter_count": int(parameters[-1]),
        "adaptation_elapsed_seconds": (completed - started).total_seconds(),
        "adaptation_maximum_resident_kb": None,
        "pipeline_elapsed_seconds_including_evaluation": None,
        "pretrained_checkpoint_sha256": sha256(checkpoint),
        "adapted_checkpoint_sha256": sha256(adapted),
        "source_experiment_root": str(exp_root),
        "timing_source": "formal pipeline phase timestamps",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q6-exp-root", type=Path, required=True)
    parser.add_argument("--q14-exp-root", type=Path, required=True)
    parser.add_argument("--q26-exp-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "runs": [
            collect(6, args.q6_exp_root),
            collect(14, args.q14_exp_root),
            collect_q26(args.q26_exp_root),
        ],
    }
    hashes = {run["pretrained_checkpoint_sha256"] for run in report["runs"]}
    if hashes != {
        "7b288f6f9198664e9ce75fee418e90ec3da2f5d0b7996a4edefe7d5f8da3e367"
    }:
        raise ValueError(f"RANF pretrained checkpoint mismatch: {hashes}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
