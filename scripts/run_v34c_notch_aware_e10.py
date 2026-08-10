"""Run the predeclared v3.4c notch-aware objective ablation."""

from __future__ import annotations

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
    / "sonicom_mlp_cnn_q26_v34c_notch_aware_e10.json"
)
OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "training"
    / "sonicom_mlp_cnn_q26_v34c_notch_aware_e10"
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


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
    objective = config["objective"]
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
        config["initial_mlp_cnn_checkpoint"],
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
        "--high-frequency-first-difference-weight",
        str(objective["spectral_difference_weights"][0]),
        "--high-frequency-second-difference-weight",
        str(objective["spectral_difference_weights"][1]),
        "--spectral-band-ild-weight",
        str(objective["spectral_band_ild_weight"]),
        "--notch-depth-weight",
        str(objective["notch_depth_weight"]),
        "--notch-minimum-frequency-hz",
        str(objective["minimum_frequency_hz"]),
        "--notch-maximum-frequency-hz",
        str(objective["maximum_frequency_hz"]),
        "--notch-radii-bins",
        *[str(value) for value in objective["radii_bins"]],
        "--notch-depth-threshold-db",
        str(objective["depth_threshold_db"]),
        "--notch-softplus-temperature-db",
        str(objective["softplus_temperature_db"]),
        "--ild-weight",
        str(training["strict_horizontal_hrir_ild_weight"]),
        "--learning-rate",
        str(training["learning_rate"]),
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
        "v34c",
        "sonicom",
        "q26",
        "notch-aware",
        "frozen-mlp",
        "validation-only",
    ]


def main() -> None:
    config = _read_json(CONFIG_PATH)
    objective = config["objective"]
    if config["test_subject_count_read"] != 0:
        raise RuntimeError("The validation-only run must not read test")
    if config["training"]["high_frequency_weight"] != 0.25:
        raise RuntimeError("v3.4c must not change the existing HF scalar weight")
    if objective["spectral_difference_weights"] != [0.0, 0.0]:
        raise RuntimeError("v3.4c must isolate notch loss from v3.4a")
    if objective["spectral_band_ild_weight"] != 0.0:
        raise RuntimeError("v3.4c must isolate notch loss from v3.4b")
    source_checkpoint = ROOT / config["initial_mlp_cnn_checkpoint"]
    expected_hash = config["initial_checkpoint_sha256"]
    source_hash_before = _sha256(source_checkpoint)
    if source_hash_before != expected_hash:
        raise RuntimeError("The frozen epoch-39 source checkpoint hash changed")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if _complete(config):
        print("Completed v3.4c run found; skipping.", flush=True)
        return
    state = {
        "status": "running",
        "target_epochs": config["training"]["epochs"],
        "training_seed": config["training"]["training_seed"],
        "validation_sampler_seed": config["training"][
            "fixed_validation_sampler_seed"
        ],
        "freeze_mlp": True,
        "train_local_cnn": True,
        "notch_depth_weight": objective["notch_depth_weight"],
        "notch_radii_bins": objective["radii_bins"],
        "source_checkpoint_sha256_before": source_hash_before,
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
    source_hash_after = _sha256(source_checkpoint)
    if source_hash_after != expected_hash:
        raise RuntimeError("The source epoch-39 checkpoint changed during training")
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
            "source_checkpoint_sha256_after": source_hash_after,
            "source_checkpoint_hash_unchanged": True,
        }
    )
    _write_json_atomic(STATE_PATH, state)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
