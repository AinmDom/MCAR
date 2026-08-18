"""Run checkpoint-free joint MLP+CNN v3.2 training for 60 epochs."""

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
    / "sonicom_mlp_cnn_q26_v32_scratch_joint_e60.json"
)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _output_dir(config: dict) -> Path:
    return ROOT / config["output_root"]


def _complete(config: dict) -> bool:
    output_dir = _output_dir(config)
    report_path = output_dir / "training_report.json"
    if not report_path.exists() or not (output_dir / "best.pt").exists():
        return False
    report = _read_json(report_path)
    return (
        report.get("status") == "completed"
        and report.get("completed_epochs") == config["training"]["epochs"]
        and report.get("checkpoint_files_saved") is True
        and report.get("training_stage")
        == "scratch_joint_mlp_cnn_global_magnitude_horizontal_hrir_ild_v32"
    )


def _validate(config: dict) -> None:
    if config["test_subject_count_read"] != 0:
        raise RuntimeError("Scratch training must not read test")
    if not config["safety"]["pretrained_checkpoint_access_forbidden"]:
        raise RuntimeError("Pretrained checkpoint access must be forbidden")
    initialization = config["initialization"]
    if initialization["base_mlp_checkpoint"] is not None:
        raise RuntimeError("Scratch training must not load an MLP checkpoint")
    if initialization["initial_cnn_checkpoint"] is not None:
        raise RuntimeError("Scratch training must not load a CNN checkpoint")
    if not (ROOT / config["dataset_root"]).exists():
        raise FileNotFoundError(config["dataset_root"])


def _command(config: dict) -> list[str]:
    model = config["model"]
    training = config["training"]
    tracking = config["tracking"]
    return [
        sys.executable,
        "-u",
        "-m",
        "mcar.training.train_mlp_cnn_v3",
        config["dataset_root"],
        "--random-initialize-mlp",
        "--zero-initialize-mlp-output",
        "--mlp-width",
        str(model["mlp_width"]),
        "--mlp-block-count",
        str(model["mlp_block_count"]),
        "--cnn-channels",
        str(model["cnn_channels"]),
        "--unfreeze-mlp",
        "--run-name",
        _output_dir(config).name,
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
        "v32",
        "sonicom",
        "q26",
        "scratch",
        "joint-mlp-cnn",
        "zero-output",
        "60-epoch",
        "validation-only",
    ]


def main() -> None:
    arguments = _parse_arguments()
    config = _read_json(CONFIG_PATH)
    _validate(config)
    command = _command(config)
    if arguments.dry_run:
        print(subprocess.list2cmdline(command))
        return

    output_dir = _output_dir(config)
    if _complete(config):
        print("Completed scratch-training run found; skipping.", flush=True)
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Incomplete non-empty output directory requires inspection: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "running",
        "initialization_mode": "random_mlp_cnn_zero_output",
        "pretrained_checkpoint_accessed": False,
        "training_seed": config["training"]["training_seed"],
        "validation_sampler_seed": config["training"][
            "fixed_validation_sampler_seed"
        ],
        "target_epochs": config["training"]["epochs"],
        "test_subject_count_read": 0,
    }
    _write_json_atomic(output_dir / "runner_state.json", state)
    with (output_dir / "runner.log").open(
        "a", encoding="utf-8", buffering=1
    ) as log:
        log.write(f"COMMAND: {subprocess.list2cmdline(command)}\n")
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if not _complete(config):
        raise RuntimeError("Scratch training did not produce a complete run")
    report = _read_json(output_dir / "training_report.json")
    state.update(
        {
            "status": "completed",
            "best_epoch": report["best_epoch"],
            "best_validation_total_loss": report[
                "best_validation_total_loss"
            ],
            "best_checkpoint": str(
                (output_dir / "best.pt").relative_to(ROOT)
            ).replace("\\", "/"),
            "best_checkpoint_sha256": _sha256(output_dir / "best.pt"),
            "elapsed_seconds": report["elapsed_seconds"],
        }
    )
    _write_json_atomic(output_dir / "runner_state.json", state)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
