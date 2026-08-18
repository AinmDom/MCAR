"""Train or jointly fine-tune the v3 binaural spectral residual model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import h5py
import numpy as np
import torch
from torch import nn

from mcar.models.residual_mlp_cnn import (
    ResidualMLPCNN,
    total_parameter_count,
    trainable_parameter_count,
)
from mcar.losses import (
    spectral_band_ild_smooth_l1_and_mae,
    strict_hrir_ild_mae,
)
from mcar.data import (
    BinauralSpectrumSampler,
    Normalization,
    list_hdf5_files,
)
from mcar.training.train_mlp_v2 import (
    LossMetrics,
    add_metrics,
    average_metrics,
    calculate_losses,
    make_erb_center_frequencies_hz,
    make_erb_weights,
)
from mcar.paths import project_root


@dataclass
class EvaluationMetrics:
    total_loss: float
    residual_mae_db: float
    erb_mae_db: float
    contralateral_high_frequency_mae_db: float
    ild_mae_db: float
    ild_spectral_proxy_mae_db: float
    high_frequency_first_difference_mae_db_per_bin: float
    high_frequency_second_difference_mae_db_per_bin2: float
    spectral_band_ild_smooth_l1_db: float
    spectral_band_ild_mae_db: float
    notch_depth_mae_db: float
    cnn_delta_mean_absolute_db: float


@dataclass
class EpochMetrics:
    epoch: int
    train_total_loss: float
    train_residual_mae_db: float
    train_erb_mae_db: float
    train_contralateral_high_frequency_mae_db: float
    train_ild_mae_db: float
    train_ild_spectral_proxy_mae_db: float
    train_high_frequency_first_difference_mae_db_per_bin: float
    train_high_frequency_second_difference_mae_db_per_bin2: float
    train_spectral_band_ild_smooth_l1_db: float
    train_spectral_band_ild_mae_db: float
    train_notch_depth_mae_db: float
    train_cnn_delta_mean_absolute_db: float
    validation_total_loss: float
    validation_residual_mae_db: float
    validation_erb_mae_db: float
    validation_contralateral_high_frequency_mae_db: float
    validation_ild_mae_db: float
    validation_ild_spectral_proxy_mae_db: float
    validation_high_frequency_first_difference_mae_db_per_bin: float
    validation_high_frequency_second_difference_mae_db_per_bin2: float
    validation_spectral_band_ild_smooth_l1_db: float
    validation_spectral_band_ild_mae_db: float
    validation_notch_depth_mae_db: float
    validation_cnn_delta_mean_absolute_db: float
    learning_rate: float
    mlp_learning_rate: float | None
    epoch_seconds: float
    skipped_optimizer_steps: int


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "initial_checkpoint",
        type=Path,
        nargs="?",
        help=(
            "Pretrained MLP checkpoint. Omit only with "
            "--random-initialize-mlp."
        ),
    )
    parser.add_argument(
        "--random-initialize-mlp",
        action="store_true",
        help=(
            "Construct the MLP from explicit architecture arguments instead "
            "of loading a checkpoint. Requires --unfreeze-mlp."
        ),
    )
    parser.add_argument("--mlp-width", type=int, default=128)
    parser.add_argument("--mlp-block-count", type=int, default=3)
    parser.add_argument(
        "--zero-initialize-mlp-output",
        action="store_true",
        help=(
            "Set the randomly initialized MLP output projection to exact "
            "zero so scratch training starts from the MCA residual baseline."
        ),
    )
    parser.add_argument(
        "--initial-cnn-checkpoint",
        type=Path,
        help=(
            "Optional full MLP-CNN checkpoint for v3.1 fine-tuning. "
            "Without it, the CNN remains zero-initialized as in v3."
        ),
    )
    parser.add_argument("--run-name", default="mlp_cnn_n03_v3")
    parser.add_argument(
        "--training-stage-label",
        help=(
            "Optional explicit checkpoint/report stage label for a locked "
            "continuation experiment."
        ),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--validation-steps", type=int, default=96)
    parser.add_argument("--directions-per-batch", type=int, default=16)
    parser.add_argument("--cnn-channels", type=int, default=48)
    parser.add_argument(
        "--global-context-attention",
        action="store_true",
        help=(
            "Add a downsampled full-spectrum attention branch and gated "
            "residual fusion while retaining the local dilated CNN."
        ),
    )
    parser.add_argument("--global-attention-width", type=int, default=32)
    parser.add_argument("--global-attention-heads", type=int, default=4)
    parser.add_argument("--global-attention-blocks", type=int, default=2)
    parser.add_argument("--global-attention-stride", type=int, default=4)
    parser.add_argument(
        "--freeze-local-cnn",
        action="store_true",
        help=(
            "Freeze the source local CNN and train only the new global "
            "context branch/gate (plus MLP only if explicitly unfrozen)."
        ),
    )
    parser.add_argument(
        "--interpolation-only",
        action="store_true",
        help=(
            "Sample only directions marked by interpolation_evaluation_mask."
        ),
    )
    parser.add_argument(
        "--direction-weighted-residual",
        action="store_true",
        help=(
            "Apply direction_features[:, 5] weights to residual SmoothL1 "
            "and residual MAE."
        ),
    )
    parser.add_argument(
        "--horizontal-only",
        action="store_true",
        help=(
            "Sample only directions with zero elevation. Intended for "
            "strict horizontal-plane HRIR ILD fine-tuning."
        ),
    )
    parser.add_argument(
        "--dual-sampling-strict-ild",
        action="store_true",
        help=(
            "v3.2 mode: calculate residual/ERB/high-frequency losses on a "
            "global interpolation batch and strict HRIR ILD on a separate "
            "horizontal interpolation batch."
        ),
    )
    parser.add_argument(
        "--ild-directions-per-batch",
        type=int,
        help=(
            "Direction count for the separate horizontal strict-ILD batch. "
            "Defaults to --directions-per-batch."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help=(
            "Linear learning-rate warmup epochs before cosine annealing. "
            "Zero preserves the historical scheduler."
        ),
    )
    parser.add_argument(
        "--unfreeze-mlp",
        action="store_true",
        help=(
            "Jointly fine-tune the base MLP and CNN. The default keeps the "
            "MLP frozen for backward-compatible v3/v3.1/v3.2 training."
        ),
    )
    parser.add_argument(
        "--mlp-learning-rate",
        type=float,
        help=(
            "Learning rate for the MLP parameter group when --unfreeze-mlp "
            "is enabled. A separate explicit value is required to avoid "
            "accidental full-rate MLP updates."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--erb-weight", type=float, default=0.50)
    parser.add_argument("--high-frequency-weight", type=float, default=0.25)
    parser.add_argument(
        "--high-frequency-first-difference-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the direction-weighted first spectral-difference "
            "MAE above --spectral-difference-minimum-frequency-hz."
        ),
    )
    parser.add_argument(
        "--high-frequency-second-difference-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the direction-weighted second spectral-difference "
            "MAE above --spectral-difference-minimum-frequency-hz."
        ),
    )
    parser.add_argument(
        "--spectral-difference-minimum-frequency-hz",
        type=float,
        default=4000.0,
    )
    parser.add_argument(
        "--spectral-band-ild-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for horizontal auditory-band ILD SmoothL1. This "
            "auxiliary objective requires --dual-sampling-strict-ild."
        ),
    )
    parser.add_argument(
        "--spectral-band-ild-minimum-center-hz",
        type=float,
        default=200.0,
    )
    parser.add_argument(
        "--spectral-band-ild-maximum-center-hz",
        type=float,
        default=18000.0,
    )
    parser.add_argument(
        "--spectral-band-ild-beta-db",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--notch-depth-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for multi-scale notch-depth-map MAE on the global "
            "interpolation batch."
        ),
    )
    parser.add_argument(
        "--notch-minimum-frequency-hz", type=float, default=4000.0
    )
    parser.add_argument(
        "--notch-maximum-frequency-hz", type=float, default=18000.0
    )
    parser.add_argument(
        "--notch-radii-bins",
        type=int,
        nargs="+",
        default=[4, 8, 16],
    )
    parser.add_argument(
        "--notch-depth-threshold-db", type=float, default=1.0
    )
    parser.add_argument(
        "--notch-softplus-temperature-db", type=float, default=0.5
    )
    parser.add_argument("--ild-weight", type=float, default=0.25)
    parser.add_argument(
        "--ild-loss-mode",
        choices=("spectral_proxy", "strict_hrir"),
        default="spectral_proxy",
        help=(
            "spectral_proxy reproduces v3; strict_hrir reconstructs the "
            "original-phase HRIR and matches the final MATLAB ILD definition."
        ),
    )
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--amp-initial-scale", type=float, default=1024.0)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--validation-seed",
        type=int,
        help=(
            "Optional fixed validation sampler seed. By default it is the "
            "training seed plus one. Set this when comparing training seeds "
            "against identical validation batches."
        ),
    )
    parser.add_argument("--overfit-subject", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--no-save-checkpoints",
        action="store_true",
        help=(
            "Keep configuration, history, and reports but do not write "
            "last.pt, best.pt, or best_strict_ild.pt. Intended for "
            "diagnostic seed studies where trained weights are not needed."
        ),
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "online", "offline"),
        default="disabled",
        help="Enable online/offline W&B tracking; disabled preserves local-only runs.",
    )
    parser.add_argument("--wandb-project", default="mcar-mlp-cnn-v3")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-group", default="hutubs-n03")
    parser.add_argument("--wandb-job-type", default="cnn-only-train")
    parser.add_argument("--wandb-tags", nargs="*")
    parser.add_argument("--wandb-notes")
    parser.add_argument(
        "--wandb-watch",
        action="store_true",
        help="Also track CNN gradients/parameters; disabled by default to reduce overhead.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def serializable_arguments(
    arguments: argparse.Namespace,
) -> dict[str, object]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(arguments).items()
    }


def training_stage(arguments: argparse.Namespace) -> str:
    explicit_label = getattr(arguments, "training_stage_label", None)
    if explicit_label:
        return str(explicit_label)
    if getattr(arguments, "random_initialize_mlp", False):
        if arguments.dual_sampling_strict_ild:
            return (
                "scratch_joint_mlp_cnn_global_magnitude_"
                "horizontal_hrir_ild_v32"
            )
        if arguments.ild_loss_mode == "strict_hrir":
            return "scratch_joint_mlp_cnn_strict_hrir_ild_v32"
        return "scratch_joint_mlp_cnn"
    spectral_difference_training = (
        getattr(arguments, "high_frequency_first_difference_weight", 0.0)
        > 0.0
        or getattr(
            arguments, "high_frequency_second_difference_weight", 0.0
        )
        > 0.0
    )
    spectral_band_ild_training = (
        getattr(arguments, "spectral_band_ild_weight", 0.0) > 0.0
    )
    notch_training = getattr(arguments, "notch_depth_weight", 0.0) > 0.0
    if notch_training and not (
        spectral_difference_training or spectral_band_ild_training
    ):
        trainable_prefix = (
            "joint_mlp_cnn" if arguments.unfreeze_mlp else "cnn_only_frozen_mlp"
        )
        if arguments.dual_sampling_strict_ild:
            return (
                f"{trainable_prefix}_global_magnitude_horizontal_hrir_ild_"
                "notch_aware_v34c"
            )
        return f"{trainable_prefix}_notch_aware_v34c"
    if notch_training:
        trainable_prefix = (
            "joint_mlp_cnn" if arguments.unfreeze_mlp else "cnn_only_frozen_mlp"
        )
        return f"{trainable_prefix}_combined_targeted_losses_v34"
    if spectral_difference_training and spectral_band_ild_training:
        trainable_prefix = (
            "joint_mlp_cnn" if arguments.unfreeze_mlp else "cnn_only_frozen_mlp"
        )
        return (
            f"{trainable_prefix}_global_magnitude_horizontal_hrir_ild_"
            "hf_spectral_differences_spectral_band_ild_v34"
        )
    if spectral_band_ild_training:
        trainable_prefix = (
            "joint_mlp_cnn" if arguments.unfreeze_mlp else "cnn_only_frozen_mlp"
        )
        if arguments.dual_sampling_strict_ild:
            return (
                f"{trainable_prefix}_global_magnitude_horizontal_hrir_ild_"
                "spectral_band_ild_v34b"
            )
        return f"{trainable_prefix}_spectral_band_ild_v34b"
    if spectral_difference_training:
        trainable_prefix = (
            "joint_mlp_cnn" if arguments.unfreeze_mlp else "cnn_only_frozen_mlp"
        )
        if arguments.dual_sampling_strict_ild:
            return (
                f"{trainable_prefix}_global_magnitude_horizontal_hrir_ild_"
                "hf_spectral_differences_v34a"
            )
        return f"{trainable_prefix}_hf_spectral_differences_v34a"
    if getattr(arguments, "global_context_attention", False):
        trainable_prefix = (
            "global_attention_only_frozen_mlp_local_cnn"
            if getattr(arguments, "freeze_local_cnn", False)
            else "local_global_cnn_frozen_mlp"
        )
        if arguments.unfreeze_mlp:
            trainable_prefix += "_joint_mlp"
        if arguments.dual_sampling_strict_ild:
            return (
                f"{trainable_prefix}_global_magnitude_"
                "horizontal_hrir_ild_v33"
            )
        return f"{trainable_prefix}_v33"
    trainable_prefix = (
        "joint_mlp_cnn" if arguments.unfreeze_mlp else "cnn_only_frozen_mlp"
    )
    if arguments.dual_sampling_strict_ild:
        version = "v321" if arguments.unfreeze_mlp else "v32"
        return (
            f"{trainable_prefix}_global_magnitude_horizontal_hrir_ild_{version}"
        )
    if arguments.ild_loss_mode == "strict_hrir":
        if arguments.horizontal_only:
            return f"{trainable_prefix}_strict_horizontal_hrir_ild_v31"
        return f"{trainable_prefix}_strict_hrir_ild_v31"
    return trainable_prefix


def strict_ild_directions_per_batch(arguments: argparse.Namespace) -> int:
    """Return the horizontal batch size, defaulting to the global size."""
    return (
        arguments.directions_per_batch
        if arguments.ild_directions_per_batch is None
        else arguments.ild_directions_per_batch
    )


def initialize_wandb(
    arguments: argparse.Namespace,
    configuration: dict[str, object],
    output_dir: Path,
    model: ResidualMLPCNN,
) -> Optional[Any]:
    if arguments.wandb_mode == "disabled":
        return None
    wandb_state_root = output_dir / "wandb_state"
    wandb_directories = {
        "WANDB_DIR": output_dir,
        "WANDB_DATA_DIR": wandb_state_root / "data",
        "WANDB_CACHE_DIR": wandb_state_root / "cache",
        "WANDB_CONFIG_DIR": wandb_state_root / "config",
        "WANDB_ARTIFACT_DIR": wandb_state_root / "artifacts",
    }
    for environment_name, directory in wandb_directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(environment_name, str(directory.resolve()))
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B tracking was requested but wandb is not installed. "
            "Install requirements/tracking.txt."
        ) from error

    tracking_configuration = dict(serializable_arguments(arguments))
    tracking_configuration.update(
        {
            "device": configuration["device"],
            "torch_version": configuration["torch_version"],
            "cuda_version": configuration["cuda_version"],
            "base_mlp_checkpoint_epoch": configuration[
                "base_mlp_checkpoint_epoch"
            ],
            "initial_model_epoch": configuration["initial_model_epoch"],
            "initial_model_stage": configuration["initial_model_stage"],
            "total_parameter_count": configuration[
                "total_parameter_count"
            ],
            "trainable_parameter_count": configuration[
                "trainable_parameter_count"
            ],
            "frozen_mlp_parameter_count": configuration[
                "frozen_mlp_parameter_count"
            ],
            "cnn_parameter_count": configuration["cnn_parameter_count"],
            "mlp_parameter_count": configuration["mlp_parameter_count"],
            "mlp_trainable_parameter_count": configuration[
                "mlp_trainable_parameter_count"
            ],
            "cnn_learning_rate": configuration["cnn_learning_rate"],
            "mlp_learning_rate": configuration["mlp_learning_rate"],
            "batch_sample_count": configuration["batch_sample_count"],
            "frequency_count": configuration["frequency_count"],
            "train_subject_count": configuration["train_subject_count"],
            "validation_subject_count": configuration[
                "validation_subject_count"
            ],
            "validation_sampler_seed": configuration[
                "validation_sampler_seed"
            ],
        }
    )
    tags = arguments.wandb_tags
    if tags is None:
        tags = ["hutubs", "n03"]
        if arguments.unfreeze_mlp:
            tags.extend(["joint-mlp-cnn", "unfrozen-mlp"])
        else:
            tags.extend(["cnn-only", "frozen-mlp"])
        if arguments.overfit_subject is not None:
            tags.append("overfit-check")
    run = wandb.init(
        project=arguments.wandb_project,
        entity=arguments.wandb_entity,
        name=arguments.wandb_run_name or arguments.run_name,
        group=arguments.wandb_group,
        job_type=arguments.wandb_job_type,
        tags=tags,
        notes=arguments.wandb_notes,
        config=tracking_configuration,
        mode=arguments.wandb_mode,
        dir=str(output_dir),
    )
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("validation/*", step_metric="epoch")
    run.define_metric("optimizer/*", step_metric="epoch")
    run.define_metric("system/*", step_metric="epoch")
    run.define_metric("checkpoint/*", step_metric="epoch")
    if arguments.wandb_watch:
        run.watch(
            (
                model
                if arguments.unfreeze_mlp
                or arguments.global_context_attention
                else model.cnn
            ),
            log="gradients",
            log_freq=max(1, arguments.steps_per_epoch // 5),
            log_graph=False,
        )
    return run


def relative_improvement_percent(initial: float, current: float) -> float:
    if initial == 0.0:
        return 0.0
    return 100.0 * (initial - current) / initial


def log_initial_validation(
    run: Optional[Any],
    metrics: EvaluationMetrics,
    learning_rate: float,
    mlp_learning_rate: float | None,
    amp_scale: float,
) -> None:
    if run is None:
        return
    run.log(
        {
            "epoch": 0,
            "validation/total_loss": metrics.total_loss,
            "validation/residual_mae_db": metrics.residual_mae_db,
            "validation/erb_mae_db": metrics.erb_mae_db,
            "validation/contralateral_high_frequency_mae_db": (
                metrics.contralateral_high_frequency_mae_db
            ),
            "validation/ild_mae_db": metrics.ild_mae_db,
            "validation/ild_spectral_proxy_mae_db": (
                metrics.ild_spectral_proxy_mae_db
            ),
            "validation/high_frequency_first_difference_mae_db_per_bin": (
                metrics.high_frequency_first_difference_mae_db_per_bin
            ),
            "validation/high_frequency_second_difference_mae_db_per_bin2": (
                metrics.high_frequency_second_difference_mae_db_per_bin2
            ),
            "validation/spectral_band_ild_smooth_l1_db": (
                metrics.spectral_band_ild_smooth_l1_db
            ),
            "validation/spectral_band_ild_mae_db": (
                metrics.spectral_band_ild_mae_db
            ),
            "validation/notch_depth_mae_db": metrics.notch_depth_mae_db,
            "validation/cnn_delta_mean_absolute_db": (
                metrics.cnn_delta_mean_absolute_db
            ),
            "optimizer/learning_rate": learning_rate,
            "optimizer/cnn_learning_rate": learning_rate,
            "optimizer/mlp_learning_rate": mlp_learning_rate,
            "optimizer/amp_scale": amp_scale,
            "optimizer/skipped_steps_epoch": 0,
            "optimizer/skipped_steps_total": 0,
            "checkpoint/is_best": 0,
            "checkpoint/best_epoch_so_far": 0,
        },
        step=0,
    )


def log_epoch_to_wandb(
    run: Optional[Any],
    metrics: EpochMetrics,
    initial: EvaluationMetrics,
    amp_scale: float,
    total_skipped_optimizer_steps: int,
    peak_cuda_allocated_mib: float,
    is_best: bool,
    best_epoch: int,
) -> None:
    if run is None:
        return
    run.log(
        {
            "epoch": metrics.epoch,
            "train/total_loss": metrics.train_total_loss,
            "train/residual_mae_db": metrics.train_residual_mae_db,
            "train/erb_mae_db": metrics.train_erb_mae_db,
            "train/contralateral_high_frequency_mae_db": (
                metrics.train_contralateral_high_frequency_mae_db
            ),
            "train/ild_mae_db": metrics.train_ild_mae_db,
            "train/ild_spectral_proxy_mae_db": (
                metrics.train_ild_spectral_proxy_mae_db
            ),
            "train/high_frequency_first_difference_mae_db_per_bin": (
                metrics.train_high_frequency_first_difference_mae_db_per_bin
            ),
            "train/high_frequency_second_difference_mae_db_per_bin2": (
                metrics.train_high_frequency_second_difference_mae_db_per_bin2
            ),
            "train/spectral_band_ild_smooth_l1_db": (
                metrics.train_spectral_band_ild_smooth_l1_db
            ),
            "train/spectral_band_ild_mae_db": (
                metrics.train_spectral_band_ild_mae_db
            ),
            "train/notch_depth_mae_db": metrics.train_notch_depth_mae_db,
            "train/cnn_delta_mean_absolute_db": (
                metrics.train_cnn_delta_mean_absolute_db
            ),
            "validation/total_loss": metrics.validation_total_loss,
            "validation/residual_mae_db": (
                metrics.validation_residual_mae_db
            ),
            "validation/erb_mae_db": metrics.validation_erb_mae_db,
            "validation/contralateral_high_frequency_mae_db": (
                metrics.validation_contralateral_high_frequency_mae_db
            ),
            "validation/ild_mae_db": metrics.validation_ild_mae_db,
            "validation/ild_spectral_proxy_mae_db": (
                metrics.validation_ild_spectral_proxy_mae_db
            ),
            "validation/high_frequency_first_difference_mae_db_per_bin": (
                metrics.validation_high_frequency_first_difference_mae_db_per_bin
            ),
            "validation/high_frequency_second_difference_mae_db_per_bin2": (
                metrics.validation_high_frequency_second_difference_mae_db_per_bin2
            ),
            "validation/spectral_band_ild_smooth_l1_db": (
                metrics.validation_spectral_band_ild_smooth_l1_db
            ),
            "validation/spectral_band_ild_mae_db": (
                metrics.validation_spectral_band_ild_mae_db
            ),
            "validation/notch_depth_mae_db": (
                metrics.validation_notch_depth_mae_db
            ),
            "validation/cnn_delta_mean_absolute_db": (
                metrics.validation_cnn_delta_mean_absolute_db
            ),
            "validation/total_improvement_vs_initial_model_percent": (
                relative_improvement_percent(
                    initial.total_loss,
                    metrics.validation_total_loss,
                )
            ),
            "validation/residual_improvement_vs_initial_model_percent": (
                relative_improvement_percent(
                    initial.residual_mae_db,
                    metrics.validation_residual_mae_db,
                )
            ),
            "validation/erb_improvement_vs_initial_model_percent": (
                relative_improvement_percent(
                    initial.erb_mae_db,
                    metrics.validation_erb_mae_db,
                )
            ),
            "validation/high_frequency_improvement_vs_initial_model_percent": (
                relative_improvement_percent(
                    initial.contralateral_high_frequency_mae_db,
                    metrics.validation_contralateral_high_frequency_mae_db,
                )
            ),
            "validation/ild_improvement_vs_initial_model_percent": (
                relative_improvement_percent(
                    initial.ild_mae_db,
                    metrics.validation_ild_mae_db,
                )
            ),
            "optimizer/learning_rate": metrics.learning_rate,
            "optimizer/cnn_learning_rate": metrics.learning_rate,
            "optimizer/mlp_learning_rate": metrics.mlp_learning_rate,
            "optimizer/amp_scale": amp_scale,
            "optimizer/skipped_steps_epoch": (
                metrics.skipped_optimizer_steps
            ),
            "optimizer/skipped_steps_total": (
                total_skipped_optimizer_steps
            ),
            "system/epoch_seconds": metrics.epoch_seconds,
            "system/peak_cuda_allocated_mib": peak_cuda_allocated_mib,
            "checkpoint/is_best": int(is_best),
            "checkpoint/best_epoch_so_far": best_epoch,
        },
        step=metrics.epoch,
    )


def sample_to_device(
    sample: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, int | str],
    ],
    device: torch.device,
    strict_ild: bool,
) -> tuple[object, ...]:
    (
        features,
        target_normalized,
        target_db,
        mca_db,
        direction_features,
        frequency_hz,
        metadata,
    ) = sample
    point_features = torch.from_numpy(
        np.transpose(features, (1, 0, 2, 3))
    ).to(device, non_blocking=True)
    tensors: tuple[object, ...] = (
        point_features,
        torch.from_numpy(target_normalized).to(device, non_blocking=True),
        torch.from_numpy(target_db).to(device, non_blocking=True),
        torch.from_numpy(mca_db).to(device, non_blocking=True),
        torch.from_numpy(direction_features).to(device, non_blocking=True),
        torch.from_numpy(frequency_hz).to(device, non_blocking=True),
    )
    if not strict_ild:
        return tensors + (None,)
    raw_strict = metadata.get("strict_ild")
    if not isinstance(raw_strict, dict):
        raise ValueError("Strict HRIR ILD metadata is missing from sampled batch")
    strict_metadata: dict[str, torch.Tensor | int] = {}
    for name, value in raw_strict.items():
        strict_metadata[name] = (
            torch.from_numpy(value).to(device, non_blocking=True)
            if isinstance(value, np.ndarray)
            else int(value)
        )
    return tensors + (strict_metadata,)


def calculate_model_losses(
    model: ResidualMLPCNN,
    batch: tuple[object, ...],
    log_erb_weights: torch.Tensor,
    normalization: Normalization,
    arguments: argparse.Namespace,
    use_amp: bool,
    include_strict_ild: bool = True,
) -> tuple[torch.Tensor, LossMetrics, torch.Tensor]:
    (
        point_features,
        target_normalized,
        target_db,
        mca_db,
        direction_features,
        frequency_hz,
        strict_metadata,
    ) = batch
    if not all(
        isinstance(value, torch.Tensor)
        for value in (
            point_features,
            target_normalized,
            target_db,
            mca_db,
            direction_features,
            frequency_hz,
        )
    ):
        raise TypeError("Model batch tensors are incomplete")
    with torch.amp.autocast("cuda", enabled=use_amp):
        prediction, _, cnn_delta = model(point_features)
    loss, loss_metrics = calculate_losses(
        prediction.permute(1, 0, 2),
        target_normalized,
        target_db,
        mca_db,
        direction_features,
        frequency_hz,
        log_erb_weights,
        normalization.target_mean,
        normalization.target_std,
        arguments.erb_weight,
        arguments.high_frequency_weight,
        (
            arguments.ild_weight
            if arguments.ild_loss_mode == "spectral_proxy"
            else 0.0
        ),
        arguments.direction_weighted_residual,
        high_frequency_first_difference_weight=(
            getattr(arguments, "high_frequency_first_difference_weight", 0.0)
        ),
        high_frequency_second_difference_weight=(
            getattr(arguments, "high_frequency_second_difference_weight", 0.0)
        ),
        spectral_difference_minimum_frequency_hz=(
            getattr(
                arguments,
                "spectral_difference_minimum_frequency_hz",
                4000.0,
            )
        ),
        notch_depth_weight=getattr(arguments, "notch_depth_weight", 0.0),
        notch_minimum_frequency_hz=getattr(
            arguments, "notch_minimum_frequency_hz", 4000.0
        ),
        notch_maximum_frequency_hz=getattr(
            arguments, "notch_maximum_frequency_hz", 18000.0
        ),
        notch_radii_bins=tuple(
            getattr(arguments, "notch_radii_bins", (4, 8, 16))
        ),
        notch_depth_threshold_db=getattr(
            arguments, "notch_depth_threshold_db", 1.0
        ),
        notch_softplus_temperature_db=getattr(
            arguments, "notch_softplus_temperature_db", 0.5
        ),
    )
    if arguments.ild_loss_mode == "strict_hrir" and include_strict_ild:
        if not isinstance(strict_metadata, dict):
            raise ValueError("Strict HRIR ILD mode requires metadata")
        prediction_db = (
            prediction.permute(1, 0, 2).float()
            * normalization.target_std
            + normalization.target_mean
        )
        corrected_selected_db = mca_db.float() + prediction_db
        strict_ild_mae = strict_hrir_ild_mae(
            corrected_selected_db,
            direction_features[:, 5],
            strict_metadata,
        )
        loss = (
            loss
            + arguments.ild_weight
            * strict_ild_mae
            / normalization.target_std
        )
        loss_metrics.total = float(loss.detach().item())
        loss_metrics.ild_mae_db = float(strict_ild_mae.detach().item())
    return loss, loss_metrics, cnn_delta


def calculate_strict_ild_loss(
    model: ResidualMLPCNN,
    batch: tuple[object, ...],
    normalization: Normalization,
    arguments: argparse.Namespace,
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate only the strict HRIR ILD term for a horizontal batch."""
    (
        point_features,
        _target_normalized,
        _target_db,
        mca_db,
        direction_features,
        _frequency_hz,
        strict_metadata,
    ) = batch
    if not isinstance(point_features, torch.Tensor) or not isinstance(
        mca_db, torch.Tensor
    ) or not isinstance(direction_features, torch.Tensor):
        raise TypeError("Strict ILD batch tensors are incomplete")
    if not isinstance(strict_metadata, dict):
        raise ValueError("Strict HRIR ILD mode requires metadata")
    with torch.amp.autocast("cuda", enabled=use_amp):
        prediction, _, cnn_delta = model(point_features)
    prediction_db = (
        prediction.permute(1, 0, 2).float() * normalization.target_std
        + normalization.target_mean
    )
    strict_ild_mae = strict_hrir_ild_mae(
        mca_db.float() + prediction_db,
        direction_features[:, 5],
        strict_metadata,
    )
    return (
        arguments.ild_weight * strict_ild_mae / normalization.target_std,
        strict_ild_mae,
        cnn_delta,
    )


def calculate_horizontal_ild_losses(
    model: ResidualMLPCNN,
    batch: tuple[object, ...],
    log_erb_weights: torch.Tensor,
    band_center_frequency_hz: torch.Tensor,
    normalization: Normalization,
    arguments: argparse.Namespace,
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate strict broadband and spectral-band ILD on one batch."""
    (
        point_features,
        _target_normalized,
        target_db,
        mca_db,
        direction_features,
        _frequency_hz,
        strict_metadata,
    ) = batch
    if not all(
        isinstance(value, torch.Tensor)
        for value in (
            point_features,
            target_db,
            mca_db,
            direction_features,
        )
    ):
        raise TypeError("Horizontal ILD batch tensors are incomplete")
    if not isinstance(strict_metadata, dict):
        raise ValueError("Strict HRIR ILD mode requires metadata")
    with torch.amp.autocast("cuda", enabled=use_amp):
        prediction, _, cnn_delta = model(point_features)
    prediction_db = (
        prediction.permute(1, 0, 2).float() * normalization.target_std
        + normalization.target_mean
    )
    corrected_db = mca_db.float() + prediction_db
    reference_db = mca_db.float() + target_db.float()
    direction_weights = direction_features[:, 5]
    strict_ild_mae = strict_hrir_ild_mae(
        corrected_db,
        direction_weights,
        strict_metadata,
    )
    band_smooth_l1, band_mae = spectral_band_ild_smooth_l1_and_mae(
        corrected_db,
        reference_db,
        direction_weights,
        log_erb_weights,
        band_center_frequency_hz,
        getattr(
            arguments,
            "spectral_band_ild_minimum_center_hz",
            200.0,
        ),
        getattr(
            arguments,
            "spectral_band_ild_maximum_center_hz",
            18000.0,
        ),
        getattr(arguments, "spectral_band_ild_beta_db", 0.5),
    )
    total = (
        arguments.ild_weight * strict_ild_mae
        + getattr(arguments, "spectral_band_ild_weight", 0.0)
        * band_smooth_l1
    ) / normalization.target_std
    return total, strict_ild_mae, band_smooth_l1, band_mae, cnn_delta


def calculate_dual_sampling_losses(
    model: ResidualMLPCNN,
    global_batch: tuple[object, ...],
    horizontal_ild_batch: tuple[object, ...],
    log_erb_weights: torch.Tensor,
    normalization: Normalization,
    arguments: argparse.Namespace,
    use_amp: bool,
    band_center_frequency_hz: torch.Tensor | None = None,
) -> tuple[torch.Tensor, LossMetrics, torch.Tensor]:
    """Combine global spectral losses with horizontal strict HRIR ILD.

    The two batches are sampled independently to prevent the ILD objective
    from shrinking the spatial support of residual, ERB, and high-frequency
    supervision.
    """
    global_loss, metrics, global_delta = calculate_model_losses(
        model,
        global_batch,
        log_erb_weights,
        normalization,
        arguments,
        use_amp,
        include_strict_ild=False,
    )
    if getattr(arguments, "spectral_band_ild_weight", 0.0) > 0.0:
        if band_center_frequency_hz is None:
            raise ValueError(
                "Spectral-band ILD requires ERB-band center frequencies"
            )
        (
            horizontal_loss,
            strict_ild_mae,
            spectral_band_ild_smooth_l1,
            spectral_band_ild_mae,
            _,
        ) = calculate_horizontal_ild_losses(
            model,
            horizontal_ild_batch,
            log_erb_weights,
            band_center_frequency_hz,
            normalization,
            arguments,
            use_amp,
        )
    else:
        horizontal_loss, strict_ild_mae, _ = calculate_strict_ild_loss(
            model,
            horizontal_ild_batch,
            normalization,
            arguments,
            use_amp,
        )
        spectral_band_ild_smooth_l1 = global_loss.new_zeros(())
        spectral_band_ild_mae = global_loss.new_zeros(())
    total_loss = global_loss + horizontal_loss
    metrics.total = float(total_loss.detach().item())
    metrics.ild_mae_db = float(strict_ild_mae.detach().item())
    metrics.spectral_band_ild_smooth_l1_db = float(
        spectral_band_ild_smooth_l1.detach().item()
    )
    metrics.spectral_band_ild_mae_db = float(
        spectral_band_ild_mae.detach().item()
    )
    return total_loss, metrics, global_delta


def make_sampler(
    files: Sequence[Path],
    normalization: Normalization,
    directions_per_batch: int,
    seed: int,
    strict_ild: bool,
    interpolation_only: bool,
    horizontal_only: bool,
) -> BinauralSpectrumSampler:
    return BinauralSpectrumSampler(
        files,
        normalization,
        directions_per_batch=directions_per_batch,
        seed=seed,
        strict_ild=strict_ild,
        interpolation_only=interpolation_only,
        horizontal_only=horizontal_only,
    )


@torch.no_grad()
def evaluate(
    model: ResidualMLPCNN,
    files: Sequence[Path],
    normalization: Normalization,
    steps: int,
    directions_per_batch: int,
    sampler_seed: int,
    device: torch.device,
    use_amp: bool,
    log_erb_weights: torch.Tensor,
    band_center_frequency_hz: torch.Tensor,
    arguments: argparse.Namespace,
) -> EvaluationMetrics:
    model.eval()
    sampler = make_sampler(
        files,
        normalization,
        directions_per_batch,
        sampler_seed,
        (
            arguments.ild_loss_mode == "strict_hrir"
            and not arguments.dual_sampling_strict_ild
        ),
        arguments.interpolation_only,
        arguments.horizontal_only and not arguments.dual_sampling_strict_ild,
    )
    horizontal_ild_sampler: BinauralSpectrumSampler | None = None
    if arguments.dual_sampling_strict_ild:
        horizontal_ild_sampler = make_sampler(
            files,
            normalization,
            strict_ild_directions_per_batch(arguments),
            sampler_seed + 10_000,
            True,
            True,
            True,
        )
    accumulated = LossMetrics()
    delta_mean_absolute_db = 0.0
    for _ in range(steps):
        batch = sample_to_device(
            sampler.sample_batch(),
            device,
            (
                arguments.ild_loss_mode == "strict_hrir"
                and not arguments.dual_sampling_strict_ild
            ),
        )
        if horizontal_ild_sampler is None:
            _, current, cnn_delta = calculate_model_losses(
                model,
                batch,
                log_erb_weights,
                normalization,
                arguments,
                use_amp,
            )
        else:
            horizontal_ild_batch = sample_to_device(
                horizontal_ild_sampler.sample_batch(), device, True
            )
            _, current, cnn_delta = calculate_dual_sampling_losses(
                model,
                batch,
                horizontal_ild_batch,
                log_erb_weights,
                normalization,
                arguments,
                use_amp,
                band_center_frequency_hz,
            )
        add_metrics(accumulated, current)
        delta_mean_absolute_db += float(
            torch.mean(torch.abs(cnn_delta.float())).item()
            * normalization.target_std
        )
    averaged = average_metrics(accumulated, steps)
    return EvaluationMetrics(
        total_loss=averaged.total,
        residual_mae_db=averaged.residual_mae_db,
        erb_mae_db=averaged.erb_mae_db,
        contralateral_high_frequency_mae_db=(
            averaged.contralateral_high_frequency_mae_db
        ),
        ild_mae_db=averaged.ild_mae_db,
        ild_spectral_proxy_mae_db=(
            averaged.ild_spectral_proxy_mae_db
        ),
        high_frequency_first_difference_mae_db_per_bin=(
            averaged.high_frequency_first_difference_mae_db_per_bin
        ),
        high_frequency_second_difference_mae_db_per_bin2=(
            averaged.high_frequency_second_difference_mae_db_per_bin2
        ),
        spectral_band_ild_smooth_l1_db=(
            averaged.spectral_band_ild_smooth_l1_db
        ),
        spectral_band_ild_mae_db=averaged.spectral_band_ild_mae_db,
        notch_depth_mae_db=averaged.notch_depth_mae_db,
        cnn_delta_mean_absolute_db=delta_mean_absolute_db / steps,
    )


def save_checkpoint(
    path: Path,
    model: ResidualMLPCNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: EpochMetrics,
    arguments: argparse.Namespace,
    initial_checkpoint_epoch: int,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": asdict(metrics),
            "arguments": serializable_arguments(arguments),
            "initial_checkpoint_epoch": initial_checkpoint_epoch,
            "training_stage": training_stage(arguments),
        },
        path,
    )


def write_history(path: Path, history: list[EpochMetrics]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=asdict(history[0]).keys(),
        )
        writer.writeheader()
        writer.writerows(asdict(item) for item in history)


def gradients_are_finite(parameters: Sequence[nn.Parameter]) -> bool:
    gradients = [
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    ]
    return bool(gradients) and all(
        bool(torch.all(torch.isfinite(gradient)).item())
        for gradient in gradients
    )


def make_optimizer(
    model: ResidualMLPCNN,
    learning_rate: float,
    weight_decay: float,
    unfreeze_mlp: bool = False,
    mlp_learning_rate: float | None = None,
    global_context_only: bool = False,
) -> tuple[torch.optim.AdamW, list[nn.Parameter]]:
    """Configure frozen-CNN or discriminative-rate joint fine-tuning."""
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if global_context_only:
        if model.global_context is None or model.global_gate is None:
            raise ValueError(
                "global_context_only requires a global-context model"
            )
        for parameter in model.cnn.parameters():
            parameter.requires_grad_(False)
        spectral_parameters = [
            *model.global_context.parameters(),
            *model.global_gate.parameters(),
        ]
        spectral_group_name = "global_context"
    else:
        spectral_parameters = list(model.cnn.parameters())
        if model.global_context is not None and model.global_gate is not None:
            spectral_parameters.extend(model.global_context.parameters())
            spectral_parameters.extend(model.global_gate.parameters())
        spectral_group_name = "cnn"
    parameter_groups: list[dict[str, object]] = [
        {
            "params": spectral_parameters,
            "lr": learning_rate,
            "name": spectral_group_name,
        }
    ]
    if unfreeze_mlp:
        if mlp_learning_rate is None or mlp_learning_rate <= 0.0:
            raise ValueError(
                "--unfreeze-mlp requires a positive --mlp-learning-rate"
            )
        model.unfreeze_mlp()
        parameter_groups.append(
            {
                "params": list(model.mlp.parameters()),
                "lr": mlp_learning_rate,
                "name": "mlp",
            }
        )
    else:
        if mlp_learning_rate is not None:
            raise ValueError(
                "--mlp-learning-rate requires --unfreeze-mlp"
            )
        model.freeze_mlp()
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=weight_decay,
    )
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return optimizer, trainable_parameters


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int = 0,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build the historical cosine schedule with optional linear warmup."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError("warmup_epochs must satisfy 0 <= warmup < epochs")
    if warmup_epochs == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
        )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def infer_mlp_architecture_from_arguments(
    checkpoint_arguments: dict[str, object],
) -> tuple[int, int]:
    """Read MLP dimensions from either legacy MLP or MLP+CNN checkpoints."""
    if "width" in checkpoint_arguments:
        width = int(checkpoint_arguments["width"])
    elif "mlp_width" in checkpoint_arguments:
        width = int(checkpoint_arguments["mlp_width"])
    else:
        raise KeyError("Checkpoint arguments do not contain an MLP width")
    if "block_count" in checkpoint_arguments:
        block_count = int(checkpoint_arguments["block_count"])
    elif "mlp_block_count" in checkpoint_arguments:
        block_count = int(checkpoint_arguments["mlp_block_count"])
    else:
        raise KeyError("Checkpoint arguments do not contain an MLP block count")
    if width < 1 or block_count < 1:
        raise ValueError("Checkpoint MLP dimensions must be positive")
    return width, block_count


def main() -> None:
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v3 CNN training")
    if arguments.epochs < 1 or arguments.steps_per_epoch < 1:
        raise ValueError("epochs and steps_per_epoch must be positive")
    if arguments.validation_steps < 1:
        raise ValueError("validation_steps must be positive")
    if arguments.mlp_width < 1 or arguments.mlp_block_count < 1:
        raise ValueError("MLP width and block count must be positive")
    if arguments.warmup_epochs < 0 or arguments.warmup_epochs >= arguments.epochs:
        raise ValueError("--warmup-epochs must satisfy 0 <= warmup < epochs")
    if arguments.random_initialize_mlp:
        if arguments.initial_checkpoint is not None:
            raise ValueError(
                "--random-initialize-mlp conflicts with initial_checkpoint"
            )
        if arguments.initial_cnn_checkpoint is not None:
            raise ValueError(
                "--random-initialize-mlp conflicts with "
                "--initial-cnn-checkpoint"
            )
        if not arguments.unfreeze_mlp:
            raise ValueError(
                "--random-initialize-mlp requires --unfreeze-mlp"
            )
    else:
        if arguments.initial_checkpoint is None:
            raise ValueError(
                "initial_checkpoint is required unless "
                "--random-initialize-mlp is enabled"
            )
        if arguments.zero_initialize_mlp_output:
            raise ValueError(
                "--zero-initialize-mlp-output is only valid with "
                "--random-initialize-mlp"
            )
    if arguments.high_frequency_first_difference_weight < 0.0:
        raise ValueError(
            "--high-frequency-first-difference-weight must be non-negative"
        )
    if arguments.high_frequency_second_difference_weight < 0.0:
        raise ValueError(
            "--high-frequency-second-difference-weight must be non-negative"
        )
    if arguments.spectral_difference_minimum_frequency_hz < 0.0:
        raise ValueError(
            "--spectral-difference-minimum-frequency-hz must be non-negative"
        )
    if arguments.spectral_band_ild_weight < 0.0:
        raise ValueError("--spectral-band-ild-weight must be non-negative")
    if arguments.spectral_band_ild_minimum_center_hz < 0.0:
        raise ValueError(
            "--spectral-band-ild-minimum-center-hz must be non-negative"
        )
    if (
        arguments.spectral_band_ild_maximum_center_hz
        <= arguments.spectral_band_ild_minimum_center_hz
    ):
        raise ValueError(
            "--spectral-band-ild-maximum-center-hz must exceed the minimum"
        )
    if arguments.spectral_band_ild_beta_db <= 0.0:
        raise ValueError("--spectral-band-ild-beta-db must be positive")
    if (
        arguments.spectral_band_ild_weight > 0.0
        and not arguments.dual_sampling_strict_ild
    ):
        raise ValueError(
            "--spectral-band-ild-weight requires --dual-sampling-strict-ild"
        )
    if arguments.notch_depth_weight < 0.0:
        raise ValueError("--notch-depth-weight must be non-negative")
    if arguments.notch_minimum_frequency_hz < 0.0:
        raise ValueError(
            "--notch-minimum-frequency-hz must be non-negative"
        )
    if (
        arguments.notch_maximum_frequency_hz
        <= arguments.notch_minimum_frequency_hz
    ):
        raise ValueError(
            "--notch-maximum-frequency-hz must exceed the minimum"
        )
    if not arguments.notch_radii_bins or any(
        radius < 1 for radius in arguments.notch_radii_bins
    ):
        raise ValueError("--notch-radii-bins must contain positive integers")
    if len(set(arguments.notch_radii_bins)) != len(
        arguments.notch_radii_bins
    ):
        raise ValueError("--notch-radii-bins must not contain duplicates")
    if arguments.notch_depth_threshold_db < 0.0:
        raise ValueError(
            "--notch-depth-threshold-db must be non-negative"
        )
    if arguments.notch_softplus_temperature_db <= 0.0:
        raise ValueError(
            "--notch-softplus-temperature-db must be positive"
        )
    if arguments.freeze_local_cnn and not arguments.global_context_attention:
        raise ValueError(
            "--freeze-local-cnn requires --global-context-attention"
        )
    if arguments.global_attention_width < 1:
        raise ValueError("--global-attention-width must be positive")
    if arguments.global_attention_heads < 1:
        raise ValueError("--global-attention-heads must be positive")
    if arguments.global_attention_blocks < 1:
        raise ValueError("--global-attention-blocks must be positive")
    if arguments.global_attention_stride < 1:
        raise ValueError("--global-attention-stride must be positive")
    if (
        arguments.global_attention_width
        % arguments.global_attention_heads
        != 0
    ):
        raise ValueError(
            "--global-attention-width must be divisible by "
            "--global-attention-heads"
        )
    if arguments.unfreeze_mlp:
        if (
            arguments.mlp_learning_rate is None
            or arguments.mlp_learning_rate <= 0.0
        ):
            raise ValueError(
                "--unfreeze-mlp requires a positive --mlp-learning-rate"
            )
    elif arguments.mlp_learning_rate is not None:
        raise ValueError("--mlp-learning-rate requires --unfreeze-mlp")
    if arguments.dual_sampling_strict_ild:
        if arguments.ild_loss_mode != "strict_hrir":
            raise ValueError(
                "--dual-sampling-strict-ild requires --ild-loss-mode strict_hrir"
            )
        if arguments.horizontal_only:
            raise ValueError(
                "--horizontal-only conflicts with --dual-sampling-strict-ild; "
                "the global batch must cover all interpolation directions"
            )
        if not arguments.interpolation_only:
            raise ValueError(
                "--dual-sampling-strict-ild requires --interpolation-only"
            )
        if strict_ild_directions_per_batch(arguments) < 1:
            raise ValueError("--ild-directions-per-batch must be positive")
    set_seed(arguments.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    use_amp = not arguments.no_amp
    normalization = Normalization.from_json(
        arguments.dataset_root / "training_statistics.json"
    )

    if arguments.overfit_subject is None:
        train_files = list_hdf5_files(arguments.dataset_root, split="train")
        validation_files = list_hdf5_files(arguments.dataset_root, split="val")
    else:
        train_files = list_hdf5_files(
            arguments.dataset_root,
            subject_ids=[arguments.overfit_subject],
        )
        validation_files = train_files

    train_sampler = make_sampler(
        train_files,
        normalization,
        arguments.directions_per_batch,
        arguments.seed,
        (
            arguments.ild_loss_mode == "strict_hrir"
            and not arguments.dual_sampling_strict_ild
        ),
        arguments.interpolation_only,
        arguments.horizontal_only and not arguments.dual_sampling_strict_ild,
    )
    train_horizontal_ild_sampler: BinauralSpectrumSampler | None = None
    if arguments.dual_sampling_strict_ild:
        train_horizontal_ild_sampler = make_sampler(
            train_files,
            normalization,
            strict_ild_directions_per_batch(arguments),
            arguments.seed + 10_000,
            True,
            True,
            True,
        )
    initial = None
    if not arguments.random_initialize_mlp:
        initial = torch.load(
            arguments.initial_checkpoint,
            map_location=device,
            weights_only=False,
        )
        initial_arguments = initial["arguments"]
        mlp_width, mlp_block_count = infer_mlp_architecture_from_arguments(
            initial_arguments
        )
    else:
        mlp_width = arguments.mlp_width
        mlp_block_count = arguments.mlp_block_count
    model = ResidualMLPCNN(
        mlp_width=mlp_width,
        mlp_block_count=mlp_block_count,
        cnn_channels=arguments.cnn_channels,
        global_context_attention=arguments.global_context_attention,
        global_attention_width=arguments.global_attention_width,
        global_attention_heads=arguments.global_attention_heads,
        global_attention_blocks=arguments.global_attention_blocks,
        global_attention_stride=arguments.global_attention_stride,
    ).to(device)
    if arguments.random_initialize_mlp:
        if arguments.zero_initialize_mlp_output:
            model.mlp.zero_initialize_output()
        initial_model_epoch = 0
        initial_model_stage = (
            "random_mlp_cnn_zero_output"
            if arguments.zero_initialize_mlp_output
            else "random_mlp_cnn"
        )
    elif arguments.initial_cnn_checkpoint is None:
        assert initial is not None
        initial_model_epoch = int(initial["epoch"])
        initial_model_stage = initial.get("training_stage", "residual_mlp")
        model.load_mlp_state_dict(initial["model_state"])
    else:
        initial_cnn = torch.load(
            arguments.initial_cnn_checkpoint,
            map_location=device,
            weights_only=False,
        )
        if arguments.global_context_attention:
            incompatible = model.load_state_dict(
                initial_cnn["model_state"], strict=False
            )
            allowed_prefixes = ("global_context.", "global_gate.")
            unexpected = list(incompatible.unexpected_keys)
            disallowed_missing = [
                key
                for key in incompatible.missing_keys
                if not key.startswith(allowed_prefixes)
            ]
            if unexpected or disallowed_missing:
                raise RuntimeError(
                    "Incompatible source checkpoint for global-context "
                    f"initialization: missing={disallowed_missing}, "
                    f"unexpected={unexpected}"
                )
        else:
            model.load_state_dict(initial_cnn["model_state"])
        initial_model_epoch = int(initial_cnn["epoch"])
        initial_model_stage = initial_cnn.get(
            "training_stage", "mlp_cnn"
        )
    optimizer, trainable_parameters = make_optimizer(
        model,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        unfreeze_mlp=arguments.unfreeze_mlp,
        mlp_learning_rate=arguments.mlp_learning_rate,
        global_context_only=arguments.freeze_local_cnn,
    )
    scheduler = make_scheduler(
        optimizer,
        epochs=arguments.epochs,
        warmup_epochs=arguments.warmup_epochs,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
        init_scale=arguments.amp_initial_scale,
    )

    with h5py.File(train_files[0], "r") as sample_handle:
        sample_frequency = np.squeeze(sample_handle["frequency_hz"][:])
    erb_weights = torch.from_numpy(make_erb_weights(sample_frequency)).to(device)
    log_erb_weights = torch.log(torch.clamp(erb_weights, min=1e-12)).view(
        1,
        1,
        erb_weights.shape[0],
        erb_weights.shape[1],
    )
    band_center_frequency_hz = torch.from_numpy(
        make_erb_center_frequencies_hz(erb_weights.shape[0])
    ).to(device)

    output_dir = project_root() / "artifacts" / "training" / arguments.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_seed = (
        arguments.seed + 1
        if arguments.validation_seed is None
        else arguments.validation_seed
    )
    configuration = {
        "arguments": serializable_arguments(arguments),
        "training_stage": training_stage(arguments),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "amp": use_amp,
        "initialization_mode": (
            (
                "random_mlp_cnn_zero_output"
                if arguments.zero_initialize_mlp_output
                else "random_mlp_cnn"
            )
            if arguments.random_initialize_mlp
            else "checkpoint"
        ),
        "base_mlp_checkpoint_epoch": (
            None if initial is None else int(initial["epoch"])
        ),
        "initial_model_epoch": initial_model_epoch,
        "initial_model_stage": initial_model_stage,
        "initial_cnn_checkpoint": (
            None
            if arguments.initial_cnn_checkpoint is None
            else str(arguments.initial_cnn_checkpoint)
        ),
        "total_parameter_count": total_parameter_count(model),
        "trainable_parameter_count": trainable_parameter_count(model),
        "frozen_mlp_parameter_count": (
            0 if arguments.unfreeze_mlp else total_parameter_count(model.mlp)
        ),
        "mlp_parameter_count": total_parameter_count(model.mlp),
        "mlp_trainable_parameter_count": trainable_parameter_count(model.mlp),
        "cnn_parameter_count": total_parameter_count(model.cnn),
        "global_context_parameter_count": (
            0
            if model.global_context is None or model.global_gate is None
            else total_parameter_count(model.global_context)
            + total_parameter_count(model.global_gate)
        ),
        "global_context_attention": arguments.global_context_attention,
        "freeze_local_cnn": arguments.freeze_local_cnn,
        "cnn_learning_rate": arguments.learning_rate,
        "mlp_learning_rate": arguments.mlp_learning_rate,
        "optimizer_parameter_groups": [
            {
                "name": group.get("name"),
                "learning_rate": float(group["lr"]),
                "parameter_count": sum(
                    parameter.numel() for parameter in group["params"]
                ),
            }
            for group in optimizer.param_groups
        ],
        "batch_sample_count": train_sampler.batch_size,
        "frequency_count": train_sampler.frequency_count,
        "eligible_direction_count": int(
            train_sampler.eligible_direction_indices.size
        ),
        "interpolation_only": arguments.interpolation_only,
        "horizontal_only": arguments.horizontal_only,
        "dual_sampling_strict_ild": arguments.dual_sampling_strict_ild,
        "global_eligible_direction_count": int(
            train_sampler.eligible_direction_indices.size
        ),
        "strict_ild_directions_per_batch": (
            None
            if train_horizontal_ild_sampler is None
            else train_horizontal_ild_sampler.directions_per_batch
        ),
        "strict_ild_batch_sample_count": (
            None
            if train_horizontal_ild_sampler is None
            else train_horizontal_ild_sampler.batch_size
        ),
        "strict_ild_eligible_direction_count": (
            None
            if train_horizontal_ild_sampler is None
            else int(
                train_horizontal_ild_sampler.eligible_direction_indices.size
            )
        ),
        "direction_weighted_residual": (
            arguments.direction_weighted_residual
        ),
        "train_subject_count": len(train_files),
        "validation_subject_count": len(validation_files),
        "validation_sampler_seed": validation_seed,
        "validation_batches_are_fixed_each_epoch": True,
        "train_files": [str(path) for path in train_files],
        "validation_files": [str(path) for path in validation_files],
    }
    wandb_run = initialize_wandb(
        arguments,
        configuration,
        output_dir,
        model,
    )
    configuration["wandb"] = {
        "enabled": wandb_run is not None,
        "mode": arguments.wandb_mode,
        "project": arguments.wandb_project,
        "entity": arguments.wandb_entity,
        "run_name": arguments.wandb_run_name or arguments.run_name,
        "run_id": None if wandb_run is None else wandb_run.id,
        "run_url": None if wandb_run is None else wandb_run.url,
    }
    (output_dir / "configuration.json").write_text(
        json.dumps(configuration, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"device={configuration['device']}, "
        f"parameters={configuration['total_parameter_count']}, "
        f"trainable={configuration['trainable_parameter_count']}, "
        f"batch_samples={configuration['batch_sample_count']}, "
        f"train_files={len(train_files)}, val_files={len(validation_files)}"
    )

    initial_validation = evaluate(
        model,
        validation_files,
        normalization,
        arguments.validation_steps,
        arguments.directions_per_batch,
        validation_seed,
        device,
        use_amp,
        log_erb_weights,
        band_center_frequency_hz,
        arguments,
    )
    (output_dir / "initial_validation.json").write_text(
        json.dumps(asdict(initial_validation), indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "initial_model "
        f"total={initial_validation.total_loss:.5f} "
        f"res={initial_validation.residual_mae_db:.3f} "
        f"erb={initial_validation.erb_mae_db:.3f} "
        f"hf={initial_validation.contralateral_high_frequency_mae_db:.3f} "
        f"d1={initial_validation.high_frequency_first_difference_mae_db_per_bin:.3f} "
        f"d2={initial_validation.high_frequency_second_difference_mae_db_per_bin2:.3f} "
        f"bandild={initial_validation.spectral_band_ild_mae_db:.3f} "
        f"notch={initial_validation.notch_depth_mae_db:.3f} "
        f"ild={initial_validation.ild_mae_db:.3f}"
    )
    log_initial_validation(
        wandb_run,
        initial_validation,
        float(optimizer.param_groups[0]["lr"]),
        (
            None
            if len(optimizer.param_groups) == 1
            else float(optimizer.param_groups[1]["lr"])
        ),
        float(scaler.get_scale()),
    )

    history: list[EpochMetrics] = []
    best_validation_total = float("inf")
    best_epoch = 0
    best_validation_strict_ild = float("inf")
    best_strict_ild_epoch = 0
    total_skipped_optimizer_steps = 0
    run_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, arguments.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        accumulated = LossMetrics()
        delta_mean_absolute_db = 0.0
        skipped_optimizer_steps = 0
        for _ in range(arguments.steps_per_epoch):
            batch = sample_to_device(
                train_sampler.sample_batch(),
                device,
                (
                    arguments.ild_loss_mode == "strict_hrir"
                    and not arguments.dual_sampling_strict_ild
                ),
            )
            optimizer.zero_grad(set_to_none=True)
            if train_horizontal_ild_sampler is None:
                loss, current, cnn_delta = calculate_model_losses(
                    model,
                    batch,
                    log_erb_weights,
                    normalization,
                    arguments,
                    use_amp,
                )
            else:
                horizontal_ild_batch = sample_to_device(
                    train_horizontal_ild_sampler.sample_batch(), device, True
                )
                loss, current, cnn_delta = calculate_dual_sampling_losses(
                    model,
                    batch,
                    horizontal_ild_batch,
                    log_erb_weights,
                    normalization,
                    arguments,
                    use_amp,
                    band_center_frequency_hz,
                )
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Encountered a non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            finite_gradients = gradients_are_finite(trainable_parameters)
            if finite_gradients:
                nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=arguments.gradient_clip,
                )
                scaler.step(optimizer)
            else:
                skipped_optimizer_steps += 1
                total_skipped_optimizer_steps += 1
                if use_amp:
                    scaler.step(optimizer)
            scaler.update()
            add_metrics(accumulated, current)
            delta_mean_absolute_db += float(
                torch.mean(torch.abs(cnn_delta.detach().float())).item()
                * normalization.target_std
            )

        train_metrics = average_metrics(
            accumulated,
            arguments.steps_per_epoch,
        )
        validation = evaluate(
            model,
            validation_files,
            normalization,
            arguments.validation_steps,
            arguments.directions_per_batch,
            validation_seed,
            device,
            use_amp,
            log_erb_weights,
            band_center_frequency_hz,
            arguments,
        )
        metrics = EpochMetrics(
            epoch=epoch,
            train_total_loss=train_metrics.total,
            train_residual_mae_db=train_metrics.residual_mae_db,
            train_erb_mae_db=train_metrics.erb_mae_db,
            train_contralateral_high_frequency_mae_db=(
                train_metrics.contralateral_high_frequency_mae_db
            ),
            train_ild_mae_db=train_metrics.ild_mae_db,
            train_ild_spectral_proxy_mae_db=(
                train_metrics.ild_spectral_proxy_mae_db
            ),
            train_high_frequency_first_difference_mae_db_per_bin=(
                train_metrics.high_frequency_first_difference_mae_db_per_bin
            ),
            train_high_frequency_second_difference_mae_db_per_bin2=(
                train_metrics.high_frequency_second_difference_mae_db_per_bin2
            ),
            train_spectral_band_ild_smooth_l1_db=(
                train_metrics.spectral_band_ild_smooth_l1_db
            ),
            train_spectral_band_ild_mae_db=(
                train_metrics.spectral_band_ild_mae_db
            ),
            train_notch_depth_mae_db=train_metrics.notch_depth_mae_db,
            train_cnn_delta_mean_absolute_db=(
                delta_mean_absolute_db / arguments.steps_per_epoch
            ),
            validation_total_loss=validation.total_loss,
            validation_residual_mae_db=validation.residual_mae_db,
            validation_erb_mae_db=validation.erb_mae_db,
            validation_contralateral_high_frequency_mae_db=(
                validation.contralateral_high_frequency_mae_db
            ),
            validation_ild_mae_db=validation.ild_mae_db,
            validation_ild_spectral_proxy_mae_db=(
                validation.ild_spectral_proxy_mae_db
            ),
            validation_high_frequency_first_difference_mae_db_per_bin=(
                validation.high_frequency_first_difference_mae_db_per_bin
            ),
            validation_high_frequency_second_difference_mae_db_per_bin2=(
                validation.high_frequency_second_difference_mae_db_per_bin2
            ),
            validation_spectral_band_ild_smooth_l1_db=(
                validation.spectral_band_ild_smooth_l1_db
            ),
            validation_spectral_band_ild_mae_db=(
                validation.spectral_band_ild_mae_db
            ),
            validation_notch_depth_mae_db=(
                validation.notch_depth_mae_db
            ),
            validation_cnn_delta_mean_absolute_db=(
                validation.cnn_delta_mean_absolute_db
            ),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            mlp_learning_rate=(
                None
                if len(optimizer.param_groups) == 1
                else float(optimizer.param_groups[1]["lr"])
            ),
            epoch_seconds=time.perf_counter() - epoch_started,
            skipped_optimizer_steps=skipped_optimizer_steps,
        )
        history.append(metrics)
        print(
            f"epoch={epoch:03d} "
            f"train(total={metrics.train_total_loss:.5f},"
            f"res={metrics.train_residual_mae_db:.3f},"
            f"erb={metrics.train_erb_mae_db:.3f},"
            f"hf={metrics.train_contralateral_high_frequency_mae_db:.3f},"
            f"d1={metrics.train_high_frequency_first_difference_mae_db_per_bin:.3f},"
            f"d2={metrics.train_high_frequency_second_difference_mae_db_per_bin2:.3f},"
            f"bandild={metrics.train_spectral_band_ild_mae_db:.3f},"
            f"notch={metrics.train_notch_depth_mae_db:.3f},"
            f"ild={metrics.train_ild_mae_db:.3f},"
            f"delta={metrics.train_cnn_delta_mean_absolute_db:.3f}) "
            f"val(total={metrics.validation_total_loss:.5f},"
            f"res={metrics.validation_residual_mae_db:.3f},"
            f"erb={metrics.validation_erb_mae_db:.3f},"
            f"hf={metrics.validation_contralateral_high_frequency_mae_db:.3f},"
            f"d1={metrics.validation_high_frequency_first_difference_mae_db_per_bin:.3f},"
            f"d2={metrics.validation_high_frequency_second_difference_mae_db_per_bin2:.3f},"
            f"bandild={metrics.validation_spectral_band_ild_mae_db:.3f},"
            f"notch={metrics.validation_notch_depth_mae_db:.3f},"
            f"ild={metrics.validation_ild_mae_db:.3f},"
            f"delta={metrics.validation_cnn_delta_mean_absolute_db:.3f}) "
            f"skipped={metrics.skipped_optimizer_steps} "
            f"time={metrics.epoch_seconds:.1f}s"
        )
        if not arguments.no_save_checkpoints:
            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer,
                epoch,
                metrics,
                arguments,
                initial_model_epoch,
            )
        is_best = validation.total_loss < best_validation_total
        if is_best:
            best_validation_total = validation.total_loss
            best_epoch = epoch
            if not arguments.no_save_checkpoints:
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    metrics,
                    arguments,
                    initial_model_epoch,
                )
        if (
            arguments.ild_loss_mode == "strict_hrir"
            and validation.ild_mae_db < best_validation_strict_ild
        ):
            best_validation_strict_ild = validation.ild_mae_db
            best_strict_ild_epoch = epoch
            if not arguments.no_save_checkpoints:
                save_checkpoint(
                    output_dir / "best_strict_ild.pt",
                    model,
                    optimizer,
                    epoch,
                    metrics,
                    arguments,
                    initial_model_epoch,
                )
        scheduler.step()
        write_history(output_dir / "history.csv", history)
        log_epoch_to_wandb(
            wandb_run,
            metrics,
            initial_validation,
            float(scaler.get_scale()),
            total_skipped_optimizer_steps,
            torch.cuda.max_memory_allocated() / (1024.0**2),
            is_best,
            best_epoch,
        )

    elapsed_seconds = time.perf_counter() - run_started
    best_metrics = history[best_epoch - 1]
    report = {
        "status": "completed",
        "training_stage": training_stage(arguments),
        "elapsed_seconds": elapsed_seconds,
        "completed_epochs": arguments.epochs,
        "best_epoch": best_epoch,
        "best_validation_total_loss": best_validation_total,
        "best_strict_ild_epoch": best_strict_ild_epoch,
        "best_validation_strict_ild_mae_db": (
            best_validation_strict_ild
            if arguments.ild_loss_mode == "strict_hrir"
            else None
        ),
        "total_skipped_optimizer_steps": total_skipped_optimizer_steps,
        "final_amp_scale": float(scaler.get_scale()),
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / (1024.0**2)
        ),
        "checkpoint_files_saved": not arguments.no_save_checkpoints,
        "wandb": configuration["wandb"],
        "initial_validation": asdict(initial_validation),
        "best_epoch_metrics": asdict(best_metrics),
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "best_epoch": best_epoch,
                "best_validation_total_loss": best_validation_total,
                "best_validation_residual_mae_db": (
                    best_metrics.validation_residual_mae_db
                ),
                "best_validation_erb_mae_db": (
                    best_metrics.validation_erb_mae_db
                ),
                "best_validation_contralateral_high_frequency_mae_db": (
                    best_metrics.validation_contralateral_high_frequency_mae_db
                ),
                "best_validation_high_frequency_first_difference_mae_db_per_bin": (
                    best_metrics.validation_high_frequency_first_difference_mae_db_per_bin
                ),
                "best_validation_high_frequency_second_difference_mae_db_per_bin2": (
                    best_metrics.validation_high_frequency_second_difference_mae_db_per_bin2
                ),
                "best_validation_spectral_band_ild_smooth_l1_db": (
                    best_metrics.validation_spectral_band_ild_smooth_l1_db
                ),
                "best_validation_spectral_band_ild_mae_db": (
                    best_metrics.validation_spectral_band_ild_mae_db
                ),
                "best_validation_notch_depth_mae_db": (
                    best_metrics.validation_notch_depth_mae_db
                ),
                "best_validation_ild_mae_db": (
                    best_metrics.validation_ild_mae_db
                ),
                "best_strict_ild_epoch": best_strict_ild_epoch,
                "best_validation_strict_ild_mae_db": (
                    best_validation_strict_ild
                    if arguments.ild_loss_mode == "strict_hrir"
                    else None
                ),
                "total_skipped_optimizer_steps": (
                    total_skipped_optimizer_steps
                ),
                "elapsed_seconds": elapsed_seconds,
                "peak_cuda_allocated_mib": report[
                    "peak_cuda_allocated_mib"
                ],
            }
        )
        wandb_run.finish(exit_code=0)


if __name__ == "__main__":
    main()
