"""Run the locked SONICOM v3.2 recipe for three training seeds.

The validation sampler is held fixed across runs and checkpoint serialization is
disabled.  Only histories, reports, logs, and aggregate statistics are retained.
"""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "sonicom_mlp_cnn_q26_v32_seed_stability.json"
)
RESULT_ROOT = (
    PROJECT_ROOT / "results" / "sonicom_mlp_cnn_q26_v32_seed_stability"
)
STATE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "training"
    / "sonicom_mlp_cnn_q26_v32_seed_stability_runner_state.json"
)
LOG_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "training"
    / "sonicom_mlp_cnn_q26_v32_seed_stability_logs"
)


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


def _read_history(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _is_complete(report_path: Path, epochs: int) -> bool:
    if not report_path.exists():
        return False
    report = _read_json(report_path)
    return (
        report.get("status") == "completed"
        and report.get("completed_epochs") == epochs
        and report.get("checkpoint_files_saved") is False
    )


def _assert_no_checkpoints(output_dir: Path) -> None:
    checkpoints = sorted(output_dir.glob("*.pt"))
    if checkpoints:
        names = ", ".join(path.name for path in checkpoints)
        raise RuntimeError(f"Checkpoint storage policy violated: {names}")


def _training_command(config: dict, seed: int, run_name: str) -> list[str]:
    training = config["training"]
    command = [
        sys.executable,
        "-u",
        "-m",
        "mcar.training.train_mlp_cnn_v3",
        config["dataset_root"],
        config["base_mlp_checkpoint"],
        "--initial-cnn-checkpoint",
        config["initial_cnn_checkpoint"],
        "--run-name",
        run_name,
        "--epochs",
        str(training["epochs_per_seed"]),
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
        str(seed),
        "--validation-seed",
        str(config["fixed_validation_sampler_seed"]),
        "--no-save-checkpoints",
        "--wandb-mode",
        "disabled",
    ]
    if not training["automatic_mixed_precision"]:
        command.append("--no-amp")
    return command


def _aggregate(config: dict) -> dict:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    per_seed_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    for seed in config["training_seeds"]:
        run_name = f"sonicom_mlp_cnn_q26_v32_seed_stability_{seed}"
        output_dir = PROJECT_ROOT / "artifacts" / "training" / run_name
        _assert_no_checkpoints(output_dir)
        report = _read_json(output_dir / "training_report.json")
        history = _read_history(output_dir / "history.csv")
        if len(history) != config["training"]["epochs_per_seed"]:
            raise RuntimeError(f"Unexpected history length for seed {seed}")
        best = report["best_epoch_metrics"]
        final = history[-1]
        per_seed_rows.append(
            {
                "seed": seed,
                "best_epoch": report["best_epoch"],
                "best_validation_total_loss": report[
                    "best_validation_total_loss"
                ],
                "best_validation_residual_mae_db": best[
                    "validation_residual_mae_db"
                ],
                "best_validation_erb_mae_db": best[
                    "validation_erb_mae_db"
                ],
                "best_validation_high_frequency_mae_db": best[
                    "validation_contralateral_high_frequency_mae_db"
                ],
                "best_validation_strict_ild_mae_db": best[
                    "validation_ild_mae_db"
                ],
                "final_validation_total_loss": float(
                    final["validation_total_loss"]
                ),
                "final_validation_residual_mae_db": float(
                    final["validation_residual_mae_db"]
                ),
                "final_validation_erb_mae_db": float(
                    final["validation_erb_mae_db"]
                ),
                "final_validation_high_frequency_mae_db": float(
                    final["validation_contralateral_high_frequency_mae_db"]
                ),
                "final_validation_strict_ild_mae_db": float(
                    final["validation_ild_mae_db"]
                ),
                "elapsed_seconds": report["elapsed_seconds"],
                "peak_cuda_allocated_mib": report[
                    "peak_cuda_allocated_mib"
                ],
                "skipped_optimizer_steps": report[
                    "total_skipped_optimizer_steps"
                ],
                "checkpoint_files_saved": report[
                    "checkpoint_files_saved"
                ],
            }
        )
        for row in history:
            history_rows.append({"seed": seed, **row})

    with (RESULT_ROOT / "per_seed_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed_rows[0]))
        writer.writeheader()
        writer.writerows(per_seed_rows)
    with (RESULT_ROOT / "history_long.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0]))
        writer.writeheader()
        writer.writerows(history_rows)

    metric_fields = [
        "best_validation_total_loss",
        "best_validation_residual_mae_db",
        "best_validation_erb_mae_db",
        "best_validation_high_frequency_mae_db",
        "best_validation_strict_ild_mae_db",
        "final_validation_total_loss",
        "final_validation_residual_mae_db",
        "final_validation_erb_mae_db",
        "final_validation_high_frequency_mae_db",
        "final_validation_strict_ild_mae_db",
    ]
    aggregate = {}
    for field in metric_fields:
        values = [float(row[field]) for row in per_seed_rows]
        aggregate[field] = {
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values),
            "minimum": min(values),
            "maximum": max(values),
            "range": max(values) - min(values),
        }
    summary = {
        "schema_version": "1.0",
        "status": "completed",
        "split": "validation",
        "training_seeds": config["training_seeds"],
        "fixed_validation_sampler_seed": config[
            "fixed_validation_sampler_seed"
        ],
        "epochs_per_seed": config["training"]["epochs_per_seed"],
        "completed_seed_count": len(per_seed_rows),
        "test_subject_count_read": 0,
        "model_checkpoints_saved": False,
        "best_epochs": [int(row["best_epoch"]) for row in per_seed_rows],
        "aggregate": aggregate,
        "interpretation_policy": (
            "Diagnostic seed variability only; the locked v3.2 paper model "
            "and its consumed final test result remain unchanged."
        ),
    }
    _write_json_atomic(RESULT_ROOT / "summary.json", summary)
    return summary


def main() -> None:
    config = _read_json(CONFIG_PATH)
    if config["test_subject_count_read"] != 0 or "test" not in config[
        "split_policy"
    ]:
        raise RuntimeError("Seed study must explicitly forbid test access")
    state = {
        "status": "running",
        "config": str(CONFIG_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "completed_seeds": [],
        "current_seed": None,
        "test_subject_count_read": 0,
    }
    if STATE_PATH.exists():
        previous = _read_json(STATE_PATH)
        state["completed_seeds"] = previous.get("completed_seeds", [])
    for seed in config["training_seeds"]:
        run_name = f"sonicom_mlp_cnn_q26_v32_seed_stability_{seed}"
        output_dir = PROJECT_ROOT / "artifacts" / "training" / run_name
        report_path = output_dir / "training_report.json"
        _assert_no_checkpoints(output_dir)
        if _is_complete(report_path, config["training"]["epochs_per_seed"]):
            print(f"seed={seed}: completed run found; skipping", flush=True)
        else:
            state["current_seed"] = seed
            _write_json_atomic(STATE_PATH, state)
            command = _training_command(config, seed, run_name)
            print(f"seed={seed}: starting 10-epoch run", flush=True)
            LOG_ROOT.mkdir(parents=True, exist_ok=True)
            seed_log_path = LOG_ROOT / f"seed_{seed}.log"
            with seed_log_path.open("a", encoding="utf-8", buffering=1) as log:
                log.write(f"COMMAND: {subprocess.list2cmdline(command)}\n")
                subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            _assert_no_checkpoints(output_dir)
            if not _is_complete(
                report_path, config["training"]["epochs_per_seed"]
            ):
                raise RuntimeError(f"Seed {seed} did not produce a complete report")
            print(f"seed={seed}: completed without checkpoints", flush=True)
        if seed not in state["completed_seeds"]:
            state["completed_seeds"].append(seed)
        state["current_seed"] = None
        _write_json_atomic(STATE_PATH, state)
    summary = _aggregate(config)
    state["status"] = "completed"
    state["result_summary"] = str(
        (RESULT_ROOT / "summary.json").relative_to(PROJECT_ROOT)
    ).replace("\\", "/")
    _write_json_atomic(STATE_PATH, state)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
