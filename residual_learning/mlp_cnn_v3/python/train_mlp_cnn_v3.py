"""Train the v3 binaural spectral CNN while keeping the v2 MLP frozen."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import h5py
import numpy as np
import torch
from torch import nn


SCRIPT_ROOT = Path(__file__).resolve().parent
V3_ROOT = SCRIPT_ROOT.parent
RESIDUAL_LEARNING_ROOT = V3_ROOT.parent
SHARED_PYTHON_ROOT = RESIDUAL_LEARNING_ROOT / "python"
for import_root in (SCRIPT_ROOT, SHARED_PYTHON_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from mlp_cnn_model import (  # noqa: E402
    ResidualMLPCNN,
    total_parameter_count,
    trainable_parameter_count,
)
from residual_data import (  # noqa: E402
    BinauralSpectrumSampler,
    Normalization,
    list_hdf5_files,
)
from train_residual_mlp_v2 import (  # noqa: E402
    LossMetrics,
    add_metrics,
    average_metrics,
    calculate_losses,
    make_erb_weights,
)


@dataclass
class EvaluationMetrics:
    total_loss: float
    residual_mae_db: float
    erb_mae_db: float
    contralateral_high_frequency_mae_db: float
    ild_mae_db: float
    cnn_delta_mean_absolute_db: float


@dataclass
class EpochMetrics:
    epoch: int
    train_total_loss: float
    train_residual_mae_db: float
    train_erb_mae_db: float
    train_contralateral_high_frequency_mae_db: float
    train_ild_mae_db: float
    train_cnn_delta_mean_absolute_db: float
    validation_total_loss: float
    validation_residual_mae_db: float
    validation_erb_mae_db: float
    validation_contralateral_high_frequency_mae_db: float
    validation_ild_mae_db: float
    validation_cnn_delta_mean_absolute_db: float
    learning_rate: float
    epoch_seconds: float
    skipped_optimizer_steps: int


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("initial_checkpoint", type=Path)
    parser.add_argument("--run-name", default="mlp_cnn_n03_v3")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--validation-steps", type=int, default=96)
    parser.add_argument("--directions-per-batch", type=int, default=16)
    parser.add_argument("--cnn-channels", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--erb-weight", type=float, default=0.50)
    parser.add_argument("--high-frequency-weight", type=float, default=0.25)
    parser.add_argument("--ild-weight", type=float, default=0.25)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--amp-initial-scale", type=float, default=1024.0)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--overfit-subject", type=int)
    parser.add_argument("--no-amp", action="store_true")
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
            "Install residual_learning/mlp_cnn_v3/requirements.txt."
        ) from error

    tracking_configuration = dict(serializable_arguments(arguments))
    tracking_configuration.update(
        {
            "device": configuration["device"],
            "torch_version": configuration["torch_version"],
            "cuda_version": configuration["cuda_version"],
            "initial_checkpoint_epoch": configuration[
                "initial_checkpoint_epoch"
            ],
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
        tags = ["cnn-only", "hutubs", "n03", "frozen-mlp"]
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
            model.cnn,
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
            "validation/cnn_delta_mean_absolute_db": (
                metrics.cnn_delta_mean_absolute_db
            ),
            "optimizer/learning_rate": learning_rate,
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
            "validation/cnn_delta_mean_absolute_db": (
                metrics.validation_cnn_delta_mean_absolute_db
            ),
            "validation/total_improvement_vs_v2_percent": (
                relative_improvement_percent(
                    initial.total_loss,
                    metrics.validation_total_loss,
                )
            ),
            "validation/residual_improvement_vs_v2_percent": (
                relative_improvement_percent(
                    initial.residual_mae_db,
                    metrics.validation_residual_mae_db,
                )
            ),
            "validation/erb_improvement_vs_v2_percent": (
                relative_improvement_percent(
                    initial.erb_mae_db,
                    metrics.validation_erb_mae_db,
                )
            ),
            "validation/high_frequency_improvement_vs_v2_percent": (
                relative_improvement_percent(
                    initial.contralateral_high_frequency_mae_db,
                    metrics.validation_contralateral_high_frequency_mae_db,
                )
            ),
            "validation/ild_improvement_vs_v2_percent": (
                relative_improvement_percent(
                    initial.ild_mae_db,
                    metrics.validation_ild_mae_db,
                )
            ),
            "optimizer/learning_rate": metrics.learning_rate,
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
) -> tuple[torch.Tensor, ...]:
    (
        features,
        target_normalized,
        target_db,
        mca_db,
        direction_features,
        frequency_hz,
        _,
    ) = sample
    point_features = torch.from_numpy(
        np.transpose(features, (1, 0, 2, 3))
    ).to(device, non_blocking=True)
    return (
        point_features,
        torch.from_numpy(target_normalized).to(device, non_blocking=True),
        torch.from_numpy(target_db).to(device, non_blocking=True),
        torch.from_numpy(mca_db).to(device, non_blocking=True),
        torch.from_numpy(direction_features).to(device, non_blocking=True),
        torch.from_numpy(frequency_hz).to(device, non_blocking=True),
    )


def calculate_model_losses(
    model: ResidualMLPCNN,
    batch: tuple[torch.Tensor, ...],
    log_erb_weights: torch.Tensor,
    normalization: Normalization,
    arguments: argparse.Namespace,
    use_amp: bool,
) -> tuple[torch.Tensor, LossMetrics, torch.Tensor]:
    (
        point_features,
        target_normalized,
        target_db,
        mca_db,
        direction_features,
        frequency_hz,
    ) = batch
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
        arguments.ild_weight,
    )
    return loss, loss_metrics, cnn_delta


def make_sampler(
    files: Sequence[Path],
    normalization: Normalization,
    directions_per_batch: int,
    seed: int,
) -> BinauralSpectrumSampler:
    return BinauralSpectrumSampler(
        files,
        normalization,
        directions_per_batch=directions_per_batch,
        seed=seed,
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
    arguments: argparse.Namespace,
) -> EvaluationMetrics:
    model.eval()
    sampler = make_sampler(
        files,
        normalization,
        directions_per_batch,
        sampler_seed,
    )
    accumulated = LossMetrics()
    delta_mean_absolute_db = 0.0
    for _ in range(steps):
        batch = sample_to_device(sampler.sample_batch(), device)
        _, current, cnn_delta = calculate_model_losses(
            model,
            batch,
            log_erb_weights,
            normalization,
            arguments,
            use_amp,
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
            "training_stage": "cnn_only_frozen_mlp",
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


def main() -> None:
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v3 CNN training")
    if arguments.epochs < 1 or arguments.steps_per_epoch < 1:
        raise ValueError("epochs and steps_per_epoch must be positive")
    if arguments.validation_steps < 1:
        raise ValueError("validation_steps must be positive")
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
    )
    initial = torch.load(
        arguments.initial_checkpoint,
        map_location=device,
        weights_only=False,
    )
    initial_arguments = initial["arguments"]
    model = ResidualMLPCNN(
        mlp_width=int(initial_arguments["width"]),
        mlp_block_count=int(initial_arguments["block_count"]),
        cnn_channels=arguments.cnn_channels,
    ).to(device)
    model.load_mlp_state_dict(initial["model_state"])
    model.freeze_mlp()
    optimizer = torch.optim.AdamW(
        model.cnn.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=arguments.epochs,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
        init_scale=arguments.amp_initial_scale,
    )
    cnn_parameters = list(model.cnn.parameters())

    with h5py.File(train_files[0], "r") as sample_handle:
        sample_frequency = np.squeeze(sample_handle["frequency_hz"][:])
    erb_weights = torch.from_numpy(make_erb_weights(sample_frequency)).to(device)
    log_erb_weights = torch.log(torch.clamp(erb_weights, min=1e-12)).view(
        1,
        1,
        erb_weights.shape[0],
        erb_weights.shape[1],
    )

    output_dir = V3_ROOT / "runs" / arguments.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_seed = arguments.seed + 1
    configuration = {
        "arguments": serializable_arguments(arguments),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "amp": use_amp,
        "initial_checkpoint_epoch": int(initial["epoch"]),
        "total_parameter_count": total_parameter_count(model),
        "trainable_parameter_count": trainable_parameter_count(model),
        "frozen_mlp_parameter_count": total_parameter_count(model.mlp),
        "cnn_parameter_count": total_parameter_count(model.cnn),
        "batch_sample_count": train_sampler.batch_size,
        "frequency_count": train_sampler.frequency_count,
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
        arguments,
    )
    (output_dir / "initial_validation.json").write_text(
        json.dumps(asdict(initial_validation), indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "initial_v2 "
        f"total={initial_validation.total_loss:.5f} "
        f"res={initial_validation.residual_mae_db:.3f} "
        f"erb={initial_validation.erb_mae_db:.3f} "
        f"hf={initial_validation.contralateral_high_frequency_mae_db:.3f} "
        f"ild={initial_validation.ild_mae_db:.3f}"
    )
    log_initial_validation(
        wandb_run,
        initial_validation,
        float(optimizer.param_groups[0]["lr"]),
        float(scaler.get_scale()),
    )

    history: list[EpochMetrics] = []
    best_validation_total = float("inf")
    best_epoch = 0
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
            batch = sample_to_device(train_sampler.sample_batch(), device)
            optimizer.zero_grad(set_to_none=True)
            loss, current, cnn_delta = calculate_model_losses(
                model,
                batch,
                log_erb_weights,
                normalization,
                arguments,
                use_amp,
            )
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Encountered a non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            finite_gradients = gradients_are_finite(cnn_parameters)
            if finite_gradients:
                nn.utils.clip_grad_norm_(
                    cnn_parameters,
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
            validation_cnn_delta_mean_absolute_db=(
                validation.cnn_delta_mean_absolute_db
            ),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
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
            f"ild={metrics.train_ild_mae_db:.3f},"
            f"delta={metrics.train_cnn_delta_mean_absolute_db:.3f}) "
            f"val(total={metrics.validation_total_loss:.5f},"
            f"res={metrics.validation_residual_mae_db:.3f},"
            f"erb={metrics.validation_erb_mae_db:.3f},"
            f"hf={metrics.validation_contralateral_high_frequency_mae_db:.3f},"
            f"ild={metrics.validation_ild_mae_db:.3f},"
            f"delta={metrics.validation_cnn_delta_mean_absolute_db:.3f}) "
            f"skipped={metrics.skipped_optimizer_steps} "
            f"time={metrics.epoch_seconds:.1f}s"
        )
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            metrics,
            arguments,
            int(initial["epoch"]),
        )
        is_best = validation.total_loss < best_validation_total
        if is_best:
            best_validation_total = validation.total_loss
            best_epoch = epoch
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                metrics,
                arguments,
                int(initial["epoch"]),
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
        "training_stage": "cnn_only_frozen_mlp",
        "elapsed_seconds": elapsed_seconds,
        "completed_epochs": arguments.epochs,
        "best_epoch": best_epoch,
        "best_validation_total_loss": best_validation_total,
        "total_skipped_optimizer_steps": total_skipped_optimizer_steps,
        "final_amp_scale": float(scaler.get_scale()),
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / (1024.0**2)
        ),
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
                "best_validation_ild_mae_db": (
                    best_metrics.validation_ild_mae_db
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
