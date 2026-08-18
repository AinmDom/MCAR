"""Run the locked validation-only CNN reinitialization/joint-tuning protocol."""

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
    / "sonicom_mlp_cnn_q26_v32_cnn_reinit_joint_v1.json"
)
PROTOCOL_STATE_PATH = (
    ROOT
    / "artifacts"
    / "training"
    / "sonicom_mlp_cnn_q26_v32_cnn_reinit_joint_v1_runner_state.json"
)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("cnn-reinit", "cnn-only", "joint", "all"),
        default="all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the protocol and print commands without training.",
    )
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


def _output_dir(stage_config: dict) -> Path:
    return ROOT / stage_config["output_root"]


def _is_complete(stage_config: dict) -> bool:
    output_dir = _output_dir(stage_config)
    report_path = output_dir / "training_report.json"
    if not report_path.exists() or not (output_dir / "best.pt").exists():
        return False
    report = _read_json(report_path)
    return (
        report.get("status") == "completed"
        and report.get("completed_epochs") == stage_config["epochs"]
        and report.get("checkpoint_files_saved") is True
    )


def _common_command(config: dict, stage_config: dict) -> list[str]:
    common = config["common_training"]
    tracking = config["tracking"]
    output_dir = _output_dir(stage_config)
    return [
        sys.executable,
        "-u",
        "-m",
        "mcar.training.train_mlp_cnn_v3",
        config["dataset_root"],
        config["base_mlp_checkpoint"],
        "--run-name",
        output_dir.name,
        "--epochs",
        str(stage_config["epochs"]),
        "--steps-per-epoch",
        str(common["steps_per_epoch"]),
        "--validation-steps",
        str(common["validation_steps"]),
        "--directions-per-batch",
        str(common["global_directions_per_batch"]),
        "--ild-directions-per-batch",
        str(common["horizontal_ild_directions_per_batch"]),
        "--dual-sampling-strict-ild",
        "--ild-loss-mode",
        common["ild_loss_mode"],
        "--interpolation-only",
        "--direction-weighted-residual",
        "--erb-weight",
        str(common["erb_weight"]),
        "--high-frequency-weight",
        str(common["high_frequency_weight"]),
        "--ild-weight",
        str(common["strict_horizontal_hrir_ild_weight"]),
        "--learning-rate",
        str(stage_config["cnn_learning_rate"]),
        "--weight-decay",
        str(common["weight_decay"]),
        "--gradient-clip",
        str(common["gradient_clip"]),
        "--amp-initial-scale",
        str(common["amp_initial_scale"]),
        "--seed",
        str(common["training_seed"]),
        "--validation-seed",
        str(common["fixed_validation_sampler_seed"]),
        "--wandb-mode",
        tracking["wandb_mode"],
        "--wandb-project",
        tracking["wandb_project"],
        "--wandb-group",
        tracking["wandb_group"],
    ]


def _stage1_command(config: dict) -> list[str]:
    stage = config["stage1_cnn_reinitialization"]
    if stage["initial_cnn_checkpoint"] is not None:
        raise RuntimeError("Stage 1 must not load an initial CNN checkpoint")
    return _common_command(config, stage) + [
        "--wandb-job-type",
        "cnn-reinitialization",
        "--wandb-tags",
        "v32",
        "sonicom",
        "q26",
        "cnn-reinit",
        "frozen-mlp",
        "40-epoch",
        "validation-only",
    ]


def _stage2_command(config: dict, branch_name: str) -> list[str]:
    stage2 = config["stage2_branches"]
    branch = stage2[branch_name]
    command = _common_command(config, branch)
    command.extend(
        ["--initial-cnn-checkpoint", stage2["source_checkpoint"]]
    )
    if branch["unfreeze_mlp"]:
        command.extend(
            [
                "--unfreeze-mlp",
                "--mlp-learning-rate",
                str(branch["mlp_learning_rate"]),
            ]
        )
    job_type = (
        "joint-mlp-cnn-finetune"
        if branch["unfreeze_mlp"]
        else "cnn-only-continuation-control"
    )
    command.extend(
        [
            "--wandb-job-type",
            job_type,
            "--wandb-tags",
            "v321" if branch["unfreeze_mlp"] else "v32",
            "sonicom",
            "q26",
            "cnn-reinit-source",
            "joint-mlp-cnn" if branch["unfreeze_mlp"] else "cnn-only",
            "10-epoch",
            "validation-only",
        ]
    )
    return command


def _validate_protocol(config: dict) -> None:
    if config["test_subject_count_read"] != 0:
        raise RuntimeError("The validation-only protocol must not read test")
    if not config["safety"]["test_access_forbidden"]:
        raise RuntimeError("Test access must remain forbidden")
    required_paths = (
        ROOT / config["dataset_root"],
        ROOT / config["base_mlp_checkpoint"],
        ROOT / config["comparison_baseline"]["checkpoint"],
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing protocol inputs: {missing}")
    stage1 = config["stage1_cnn_reinitialization"]
    if stage1["initial_cnn_checkpoint"] is not None:
        raise RuntimeError("Stage 1 CNN checkpoint must be null")
    branches = config["stage2_branches"]
    if branches["cnn_only_control"]["unfreeze_mlp"]:
        raise RuntimeError("The CNN-only control must keep the MLP frozen")
    if not branches["joint_mlp_cnn"]["unfreeze_mlp"]:
        raise RuntimeError("The joint branch must unfreeze the MLP")
    if branches["joint_mlp_cnn"]["mlp_learning_rate"] is None:
        raise RuntimeError("The joint branch requires an explicit MLP LR")


def _run_stage(
    config: dict,
    stage_name: str,
    stage_config: dict,
    command: list[str],
    source_checkpoint: Path | None = None,
) -> dict:
    output_dir = _output_dir(stage_config)
    if _is_complete(stage_config):
        report = _read_json(output_dir / "training_report.json")
        return {
            "status": "completed_existing",
            "best_epoch": report["best_epoch"],
            "best_validation_total_loss": report[
                "best_validation_total_loss"
            ],
            "best_checkpoint": str(
                (output_dir / "best.pt").relative_to(ROOT)
            ).replace("\\", "/"),
        }
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Incomplete non-empty output directory requires inspection: {output_dir}"
        )
    source_hash_before = (
        None if source_checkpoint is None else _sha256(source_checkpoint)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "running",
        "stage": stage_name,
        "command": subprocess.list2cmdline(command),
        "source_checkpoint_sha256_before": source_hash_before,
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
    if not _is_complete(stage_config):
        raise RuntimeError(f"{stage_name} did not produce a complete checkpoint")
    source_hash_after = (
        None if source_checkpoint is None else _sha256(source_checkpoint)
    )
    if source_hash_before != source_hash_after:
        raise RuntimeError(f"{stage_name} modified its source checkpoint")
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
            "source_checkpoint_sha256_after": source_hash_after,
        }
    )
    _write_json_atomic(output_dir / "runner_state.json", state)
    return state


def main() -> None:
    arguments = _parse_arguments()
    config = _read_json(CONFIG_PATH)
    _validate_protocol(config)
    commands = {
        "cnn-reinit": _stage1_command(config),
        "cnn-only": _stage2_command(config, "cnn_only_control"),
        "joint": _stage2_command(config, "joint_mlp_cnn"),
    }
    if arguments.dry_run:
        print(
            json.dumps(
                {
                    key: subprocess.list2cmdline(value)
                    for key, value in commands.items()
                },
                indent=2,
            )
        )
        return

    requested = (
        ("cnn-reinit", "cnn-only", "joint")
        if arguments.stage == "all"
        else (arguments.stage,)
    )
    results: dict[str, dict] = {}
    for stage_name in requested:
        if stage_name == "cnn-reinit":
            stage_config = config["stage1_cnn_reinitialization"]
            source_checkpoint = None
        else:
            branch_name = (
                "cnn_only_control"
                if stage_name == "cnn-only"
                else "joint_mlp_cnn"
            )
            stage_config = config["stage2_branches"][branch_name]
            source_checkpoint = (
                ROOT / config["stage2_branches"]["source_checkpoint"]
            )
            if not source_checkpoint.exists():
                raise FileNotFoundError(
                    "Stage 1 best checkpoint is required before stage 2"
                )
        results[stage_name] = _run_stage(
            config,
            stage_name,
            stage_config,
            commands[stage_name],
            source_checkpoint,
        )
        _write_json_atomic(
            PROTOCOL_STATE_PATH,
            {
                "status": "running",
                "completed_results": results,
                "test_subject_count_read": 0,
            },
        )
    final_state = {
        "status": "completed",
        "completed_results": results,
        "test_subject_count_read": 0,
    }
    _write_json_atomic(PROTOCOL_STATE_PATH, final_state)
    print(json.dumps(final_state, indent=2), flush=True)


if __name__ == "__main__":
    main()
