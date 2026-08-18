"""Run three matched 40-epoch v3.5.1 scratch-component continuations."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "sonicom_mlp_cnn_q26_v351_continuation_e40x3.json"
)
RUNNER_ROOT = (
    ROOT / "artifacts" / "training" / "sonicom_mlp_cnn_q26_v351_e40x3_runner"
)
RUNNER_STATE = RUNNER_ROOT / "runner_state.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--branch", choices=("all", "a", "b", "c"), default="all")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def output_dir(branch: dict) -> Path:
    return ROOT / branch["output_root"]


def complete(branch: dict, epochs: int) -> bool:
    directory = output_dir(branch)
    report_path = directory / "training_report.json"
    if not report_path.exists() or not (directory / "best.pt").exists():
        return False
    report = read_json(report_path)
    return (
        report.get("status") == "completed"
        and report.get("completed_epochs") == epochs
        and report.get("checkpoint_files_saved") is True
        and report.get("training_stage") == branch["training_stage"]
    )


def command(config: dict, branch: dict) -> list[str]:
    training = config["common_training"]
    tracking = config["tracking"]
    source = config["source_scratch_checkpoint"]
    return [
        sys.executable,
        "-u",
        "-m",
        "mcar.training.train_mlp_cnn_v3",
        config["dataset_root"],
        source,
        "--initial-cnn-checkpoint",
        source,
        "--cnn-channels",
        "48",
        "--unfreeze-mlp",
        "--training-stage-label",
        branch["training_stage"],
        "--run-name",
        output_dir(branch).name,
        "--epochs",
        str(training["epochs"]),
        "--steps-per-epoch",
        str(training["steps_per_epoch"]),
        "--validation-steps",
        str(training["validation_steps"]),
        "--directions-per-batch",
        str(training["global_directions_per_batch"]),
        "--ild-directions-per-batch",
        str(training["horizontal_ild_directions_per_batch"]),
        "--dual-sampling-strict-ild",
        "--ild-loss-mode",
        training["ild_loss_mode"],
        "--interpolation-only",
        "--direction-weighted-residual",
        "--erb-weight",
        str(training["erb_weight"]),
        "--high-frequency-weight",
        str(training["high_frequency_weight"]),
        "--ild-weight",
        str(branch["strict_horizontal_hrir_ild_weight"]),
        "--spectral-band-ild-weight",
        str(branch["spectral_band_ild_weight"]),
        "--spectral-band-ild-minimum-center-hz",
        str(training["spectral_band_ild_minimum_center_hz"]),
        "--spectral-band-ild-maximum-center-hz",
        str(training["spectral_band_ild_maximum_center_hz"]),
        "--spectral-band-ild-beta-db",
        str(training["spectral_band_ild_beta_db"]),
        "--learning-rate",
        str(training["cnn_learning_rate"]),
        "--mlp-learning-rate",
        str(training["mlp_learning_rate"]),
        "--warmup-epochs",
        str(training["warmup_epochs"]),
        "--weight-decay",
        str(training["weight_decay"]),
        "--gradient-clip",
        str(training["gradient_clip"]),
        "--amp-initial-scale",
        str(training["amp_initial_scale"]),
        "--seed",
        str(training["training_seed"]),
        "--validation-seed",
        str(training["fixed_validation_sampler_seed"]),
        "--wandb-mode",
        tracking["wandb_mode"],
        "--wandb-project",
        tracking["wandb_project"],
        "--wandb-group",
        tracking["wandb_group"],
        "--wandb-job-type",
        tracking["wandb_job_type"],
        "--wandb-tags",
        "v351",
        f"branch-{branch['id']}",
        "scratch-continuation",
        "joint-mlp-cnn",
        "40-epoch",
        "validation-only",
    ]


def validate(config: dict) -> tuple[Path, str]:
    if config["test_subject_count_read"] != 0:
        raise RuntimeError("v3.5.1 training must not read test")
    if not config["safety"]["test_access_forbidden"]:
        raise RuntimeError("The consumed test split must remain forbidden")
    source = ROOT / config["source_scratch_checkpoint"]
    if not source.exists():
        raise FileNotFoundError(source)
    expected = config["source_scratch_checkpoint_sha256"]
    actual = sha256(source)
    if actual != expected:
        raise RuntimeError("The locked scratch e60 checkpoint hash changed")
    if len(config["branches"]) != 3:
        raise RuntimeError("Expected exactly three matched branches")
    return source, actual


def main() -> None:
    cli = parse_arguments()
    config = read_json(CONFIG_PATH)
    source, source_hash = validate(config)
    branches = [
        branch
        for branch in config["branches"]
        if cli.branch == "all" or branch["id"] == cli.branch
    ]
    commands = [command(config, branch) for branch in branches]
    if cli.dry_run:
        for branch, current in zip(branches, commands):
            print(f"[{branch['id']}] {subprocess.list2cmdline(current)}")
        return

    RUNNER_ROOT.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "running",
        "source_checkpoint": str(source.relative_to(ROOT)).replace("\\", "/"),
        "source_checkpoint_sha256": source_hash,
        "target_epochs_per_branch": config["common_training"]["epochs"],
        "selected_branches": [branch["id"] for branch in branches],
        "completed_branches": [],
        "active_branch": None,
        "test_subject_count_read": 0,
    }
    write_json_atomic(RUNNER_STATE, state)
    for branch, current in zip(branches, commands):
        directory = output_dir(branch)
        if complete(branch, config["common_training"]["epochs"]):
            state["completed_branches"].append(branch["id"])
            write_json_atomic(RUNNER_STATE, state)
            print(f"[{branch['id']}] completed run found; skipping", flush=True)
            continue
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(
                f"Incomplete non-empty output requires inspection: {directory}"
            )
        directory.mkdir(parents=True, exist_ok=True)
        branch_state = {
            "status": "running",
            "branch": branch["id"],
            "name": branch["name"],
            "training_stage": branch["training_stage"],
            "source_checkpoint_sha256_before": source_hash,
            "target_epochs": config["common_training"]["epochs"],
            "test_subject_count_read": 0,
        }
        write_json_atomic(directory / "runner_state.json", branch_state)
        state["active_branch"] = branch["id"]
        write_json_atomic(RUNNER_STATE, state)
        print(f"[{branch['id']}] starting {branch['name']}", flush=True)
        with (directory / "runner.log").open(
            "a", encoding="utf-8", buffering=1
        ) as log:
            log.write(f"COMMAND: {subprocess.list2cmdline(current)}\n")
            subprocess.run(
                current,
                cwd=ROOT,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if not complete(branch, config["common_training"]["epochs"]):
            raise RuntimeError(f"Branch {branch['id']} did not complete")
        source_hash_after = sha256(source)
        if source_hash_after != source_hash:
            raise RuntimeError("Source checkpoint changed during branch training")
        report = read_json(directory / "training_report.json")
        branch_state.update(
            {
                "status": "completed",
                "best_epoch": report["best_epoch"],
                "best_validation_total_loss": report[
                    "best_validation_total_loss"
                ],
                "best_checkpoint": str(
                    (directory / "best.pt").relative_to(ROOT)
                ).replace("\\", "/"),
                "best_checkpoint_sha256": sha256(directory / "best.pt"),
                "elapsed_seconds": report["elapsed_seconds"],
                "source_checkpoint_sha256_after": source_hash_after,
                "source_checkpoint_hash_unchanged": True,
            }
        )
        write_json_atomic(directory / "runner_state.json", branch_state)
        state["completed_branches"].append(branch["id"])
        state["active_branch"] = None
        write_json_atomic(RUNNER_STATE, state)
        print(f"[{branch['id']}] completed", flush=True)
    state["status"] = "completed"
    state["active_branch"] = None
    write_json_atomic(RUNNER_STATE, state)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
