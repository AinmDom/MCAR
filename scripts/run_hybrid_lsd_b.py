"""Launch exactly three pre-registered B runs concurrently and record their exits.

The controller blocks on process completion; it never polls training metrics or
starts inference. Run once from a clean Git tree. All output is local/ignored.
"""
from __future__ import annotations

from datetime import datetime
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare_hybrid_lsd_b import MANIFEST, ROOT, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.attempt < 1:
        raise ValueError("attempt must be positive")
    manifest = json.loads((ROOT / MANIFEST).read_text())
    git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if git_status.strip():
        raise RuntimeError("Training requires a clean committed Git tree")
    for member in manifest["members"]:
        for key in ("config", "parent"):
            item = member[key]
            if sha(ROOT / item["path"]) != item["sha256"]:
                raise ValueError(f"Hash mismatch: {item['path']}")
        if (ROOT / member["output_directory"]).exists():
            raise FileExistsError(member["output_directory"])
    log_dir = ROOT / "outputs/hybrid_lsd_b_e190"
    if args.attempt > 1:
        log_dir = log_dir / f"attempt{args.attempt}"
    log_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = log_dir / "launch.json"
    receipt = {"status": "launching", "attempt": args.attempt, "started_at": datetime.now().astimezone().isoformat(),
               "controller_pid": os.getpid(), "python": sys.executable,
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "git_dirty": False, "manifest_sha256": sha(ROOT / MANIFEST),
               "members": [], "test_subjects_read": 0}
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)

    def save():
        temp = receipt_path.with_suffix(".tmp")
        temp.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        temp.replace(receipt_path)

    processes = []
    try:
        for member in manifest["members"]:
            seed = member["seed"]
            stdout_path = log_dir / f"seed{seed}.stdout.log"
            stderr_path = log_dir / f"seed{seed}.stderr.log"
            command = [sys.executable, "-u", "-m", "mcar.training.train_film_siren_stage_c", member["config"]["path"]]
            with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            entry = {"seed": seed, "pid": process.pid, "command": command,
                     "output_directory": member["output_directory"],
                     "stdout": stdout_path.relative_to(ROOT).as_posix(),
                     "stderr": stderr_path.relative_to(ROOT).as_posix(), "exit_code": None}
            processes.append((process, entry))
            receipt["members"].append(entry)
            save()
        receipt["status"] = "running"
        save()
        print(json.dumps(receipt, indent=2), flush=True)
    except Exception as error:
        # Preserve any already-started authorized runs; never kill them on a
        # partial launch error. The receipt exposes all known PIDs for handoff.
        receipt["status"] = "partial_launch_failed"
        receipt["launch_error"] = repr(error)
        save()
        raise
    for process, entry in processes:
        entry["exit_code"] = process.wait()
        entry["exit_observed_at"] = datetime.now().astimezone().isoformat()
        save()
    receipt["status"] = "processes_exited_successfully_pending_artifact_verification" if all(
        entry["exit_code"] == 0 for _, entry in processes
    ) else "process_failure"
    receipt["finished_at"] = datetime.now().astimezone().isoformat()
    save()
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
