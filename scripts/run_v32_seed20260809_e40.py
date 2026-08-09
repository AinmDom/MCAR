"""Run the predeclared 40-epoch v3.2 seed-20260809 training job."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "sonicom_mlp_cnn_q26_v32_seed20260809_e40.json"
)
OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "training"
    / "sonicom_mlp_cnn_q26_v32_seed20260809_e40"
)
LOG_PATH = OUTPUT_DIR / "runner.log"
STATE_PATH = OUTPUT_DIR / "runner_state.json"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _complete(config: dict) -> bool:
    report_path = OUTPUT_DIR / "training_report.json"
    if not report_path.exists():
        return False
    report = _read_json(report_path)
    return (
        report.get("status") == "completed"
        and report.get("completed_epochs") == config["training"]["epochs"]
        and report.get("checkpoint_files_saved") is True
        and (OUTPUT_DIR / "best.pt").exists()
    )


def _command(config: dict) -> list[str]:
    training = config["training"]
    tracking = config["tracking"]
    return [
        sys.executable,
        "-u",
        "-m",
        "mcar.training.train_mlp_cnn_v3",
        config["dataset_root"],
        config["base_mlp_checkpoint"],
        "--initial-cnn-checkpoint",
        config["initial_cnn_checkpoint"],
        "--run-name",
        OUTPUT_DIR.name,
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
        str(training["strict_horizontal_hrir_ild_weight"]),
        "--learning-rate",
        str(training["learning_rate"]),
        "--weight-decay",
        str(training["weight_decay"]),
        "--gradient-clip",
        str(training["gradient_clip"]),
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
        "v32",
        "sonicom",
        "q26",
        "seed-stability",
        "seed-20260809",
        "40-epoch",
        "validation-only",
    ]


def main() -> None:
    config = _read_json(CONFIG_PATH)
    if config["test_subject_count_read"] != 0:
        raise RuntimeError("The validation-only run must not read test")
    if config["seed_selection"]["selected_seed"] != 20260809:
        raise RuntimeError("Unexpected selected training seed")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if _complete(config):
        print("Completed 40-epoch run found; skipping.", flush=True)
        return
    state = {
        "status": "running",
        "training_seed": config["training"]["training_seed"],
        "validation_sampler_seed": config["training"][
            "fixed_validation_sampler_seed"
        ],
        "target_epochs": config["training"]["epochs"],
        "test_subject_count_read": 0,
    }
    _write_json_atomic(STATE_PATH, state)
    command = _command(config)
    with LOG_PATH.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"COMMAND: {subprocess.list2cmdline(command)}\n")
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if not _complete(config):
        raise RuntimeError("Training ended without a complete best checkpoint")
    report = _read_json(OUTPUT_DIR / "training_report.json")
    state.update(
        {
            "status": "completed",
            "best_epoch": report["best_epoch"],
            "best_validation_total_loss": report[
                "best_validation_total_loss"
            ],
            "best_checkpoint": str(
                (OUTPUT_DIR / "best.pt").relative_to(ROOT)
            ).replace("\\", "/"),
            "elapsed_seconds": report["elapsed_seconds"],
        }
    )
    _write_json_atomic(STATE_PATH, state)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
