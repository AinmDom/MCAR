"""Fine-tune ResidualMLP with paired-ear auditory-aware losses."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import h5py
from torch import nn

from mcar.data import (
    BinauralSpectrumSampler,
    Normalization,
    list_hdf5_files,
)
from mcar.models.residual_mlp import ResidualMLP, parameter_count
from mcar.paths import project_root


@dataclass
class LossMetrics:
    total: float = 0.0
    residual_smooth_l1: float = 0.0
    residual_mae_db: float = 0.0
    erb_mae_db: float = 0.0
    contralateral_high_frequency_mae_db: float = 0.0
    ild_mae_db: float = 0.0
    ild_spectral_proxy_mae_db: float = 0.0


@dataclass
class EpochMetrics:
    epoch: int
    train_total_loss: float
    train_residual_mae_db: float
    train_erb_mae_db: float
    train_contralateral_high_frequency_mae_db: float
    train_ild_mae_db: float
    validation_total_loss: float
    validation_residual_mae_db: float
    validation_erb_mae_db: float
    validation_contralateral_high_frequency_mae_db: float
    validation_ild_mae_db: float
    learning_rate: float
    elapsed_seconds: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("initial_checkpoint", type=Path)
    parser.add_argument("--run-name", default="mlp_n03_v2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--validation-steps", type=int, default=96)
    parser.add_argument("--directions-per-batch", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--erb-weight", type=float, default=0.50)
    parser.add_argument("--high-frequency-weight", type=float, default=0.25)
    parser.add_argument("--ild-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--interpolation-only",
        action="store_true",
        help="Exclude sparse input directions from paired-ear sampling.",
    )
    parser.add_argument(
        "--direction-weighted-residual",
        action="store_true",
        help=(
            "Apply direction_features[:, 5] weights to residual SmoothL1 "
            "and MAE, matching the perceptual loss spatial quadrature."
        ),
    )
    parser.add_argument("--overfit-subject", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "online", "offline"),
        default="disabled",
    )
    parser.add_argument("--wandb-project", default="mcar-residual-mlp-v2")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-job-type", default="mlp-v2-train")
    parser.add_argument("--wandb-tags", nargs="*")
    parser.add_argument("--wandb-notes")
    parser.add_argument("--wandb-watch", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def serializable_arguments(arguments: argparse.Namespace) -> dict[str, object]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(arguments).items()
    }


def initialize_wandb(
    arguments: argparse.Namespace,
    configuration: dict[str, object],
    output_dir: Path,
    model: nn.Module,
) -> Optional[Any]:
    if arguments.wandb_mode == "disabled":
        return None
    wandb_state_root = output_dir / "wandb_state"
    directories = {
        "WANDB_DIR": output_dir,
        "WANDB_DATA_DIR": wandb_state_root / "data",
        "WANDB_CACHE_DIR": wandb_state_root / "cache",
        "WANDB_CONFIG_DIR": wandb_state_root / "config",
        "WANDB_ARTIFACT_DIR": wandb_state_root / "artifacts",
    }
    for environment_name, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(environment_name, str(directory.resolve()))
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B tracking was requested but wandb is not installed."
        ) from error
    run = wandb.init(
        project=arguments.wandb_project,
        entity=arguments.wandb_entity,
        name=arguments.wandb_run_name or arguments.run_name,
        group=arguments.wandb_group,
        job_type=arguments.wandb_job_type,
        tags=arguments.wandb_tags or ["mlp-v2"],
        notes=arguments.wandb_notes,
        config={
            **serializable_arguments(arguments),
            "device": configuration["device"],
            "parameter_count": configuration["parameter_count"],
            "paired_batch_samples": configuration["batch_size"],
            "train_subject_count": configuration["train_subject_count"],
            "validation_subject_count": configuration[
                "validation_subject_count"
            ],
            "eligible_direction_count": configuration[
                "eligible_direction_count"
            ],
        },
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
            model,
            log="gradients",
            log_freq=max(1, arguments.steps_per_epoch // 5),
            log_graph=False,
        )
    return run


def erb_rate(frequency_hz: np.ndarray) -> np.ndarray:
    return 21.4 * np.log10(1.0 + 0.004367 * frequency_hz)


def make_erb_weights(frequency_hz: np.ndarray, band_count: int = 41) -> np.ndarray:
    """Create a smooth ERB-rate triangular proxy for AKerbError."""
    rates = erb_rate(np.asarray(frequency_hz, dtype=np.float64))
    centers = np.linspace(erb_rate(np.array([50.0]))[0], erb_rate(np.array([20000.0]))[0], band_count)
    spacing = float(np.mean(np.diff(centers)))
    distance = np.abs(rates[None, :] - centers[:, None])
    weights = np.maximum(0.0, 1.0 - distance / (1.5 * spacing))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return weights.astype(np.float32)


def weighted_direction_mean(
    values: torch.Tensor, direction_weights: torch.Tensor
) -> torch.Tensor:
    """Mean over ears, directions, and any trailing metric dimensions."""
    normalized = direction_weights / torch.sum(direction_weights)
    weight_shape = [1, normalized.numel()] + [1] * (values.ndim - 2)
    return torch.mean(
        torch.sum(values * normalized.reshape(weight_shape), dim=1)
    )


def band_energy_db(
    magnitude_db: torch.Tensor, log_erb_weights: torch.Tensor
) -> torch.Tensor:
    scale = math.log(10.0) / 10.0
    log_power = scale * magnitude_db.unsqueeze(-2) + log_erb_weights
    return (10.0 / math.log(10.0)) * torch.logsumexp(log_power, dim=-1)


def broadband_energy_db(magnitude_db: torch.Tensor) -> torch.Tensor:
    scale = math.log(10.0) / 10.0
    return (10.0 / math.log(10.0)) * torch.logsumexp(
        scale * magnitude_db, dim=-1
    )


def calculate_losses(
    prediction_normalized: torch.Tensor,
    target_normalized: torch.Tensor,
    target_db: torch.Tensor,
    mca_db: torch.Tensor,
    direction_features: torch.Tensor,
    frequency_hz: torch.Tensor,
    log_erb_weights: torch.Tensor,
    target_mean: float,
    target_std: float,
    erb_weight: float,
    high_frequency_weight: float,
    ild_weight: float,
    direction_weighted_residual: bool = False,
) -> tuple[torch.Tensor, LossMetrics]:
    prediction_db = prediction_normalized.float() * target_std + target_mean
    corrected_db = mca_db.float() + prediction_db
    reference_db = mca_db.float() + target_db.float()
    direction_weights = direction_features[:, 5].float()
    if direction_weighted_residual:
        residual_smooth_l1 = weighted_direction_mean(
            nn.functional.smooth_l1_loss(
                prediction_normalized.float(),
                target_normalized.float(),
                beta=1.0,
                reduction="none",
            ),
            direction_weights,
        )
        residual_mae = weighted_direction_mean(
            torch.abs(prediction_db - target_db),
            direction_weights,
        )
    else:
        residual_smooth_l1 = nn.functional.smooth_l1_loss(
            prediction_normalized.float(),
            target_normalized.float(),
            beta=1.0,
        )
        residual_mae = torch.mean(torch.abs(prediction_db - target_db))
    predicted_erb_db = band_energy_db(corrected_db, log_erb_weights)
    reference_erb_db = band_energy_db(reference_db, log_erb_weights)
    erb_mae = weighted_direction_mean(
        torch.abs(predicted_erb_db - reference_erb_db), direction_weights
    )

    frequency_mask = frequency_hz > 10000.0
    lateral_y = direction_features[:, 3]
    contra_mask = torch.stack((lateral_y < 0.0, lateral_y > 0.0), dim=0)
    high_error = torch.abs(prediction_db - target_db)[:, :, frequency_mask]
    contra_weights = (
        direction_weights.unsqueeze(0) * contra_mask.float()
    )
    contra_weights = contra_weights / torch.clamp(
        torch.sum(contra_weights, dim=1, keepdim=True), min=1e-12
    )
    high_frequency_mae = torch.mean(
        torch.sum(high_error * contra_weights.unsqueeze(-1), dim=1)
    )

    predicted_ear_energy_db = broadband_energy_db(corrected_db)
    reference_ear_energy_db = broadband_energy_db(reference_db)
    predicted_ild_db = (
        predicted_ear_energy_db[0] - predicted_ear_energy_db[1]
    )
    reference_ild_db = (
        reference_ear_energy_db[0] - reference_ear_energy_db[1]
    )
    ild_errors = torch.abs(predicted_ild_db - reference_ild_db)
    ild_mae = torch.sum(
        ild_errors * direction_weights / torch.sum(direction_weights)
    )

    total = (
        residual_smooth_l1
        + erb_weight * erb_mae / target_std
        + high_frequency_weight * high_frequency_mae / target_std
        + ild_weight * ild_mae / target_std
    )
    metrics = LossMetrics(
        total=float(total.detach().item()),
        residual_smooth_l1=float(residual_smooth_l1.detach().item()),
        residual_mae_db=float(residual_mae.detach().item()),
        erb_mae_db=float(erb_mae.detach().item()),
        contralateral_high_frequency_mae_db=float(
            high_frequency_mae.detach().item()
        ),
        ild_mae_db=float(ild_mae.detach().item()),
        ild_spectral_proxy_mae_db=float(ild_mae.detach().item()),
    )
    return total, metrics


def add_metrics(total: LossMetrics, current: LossMetrics) -> None:
    for field_name in asdict(total):
        setattr(
            total,
            field_name,
            getattr(total, field_name) + getattr(current, field_name),
        )


def average_metrics(total: LossMetrics, count: int) -> LossMetrics:
    return LossMetrics(
        **{
            field_name: value / count
            for field_name, value in asdict(total).items()
        }
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
    return tuple(
        torch.from_numpy(value).to(device, non_blocking=True)
        for value in sample[:6]
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    sampler: BinauralSpectrumSampler,
    steps: int,
    device: torch.device,
    use_amp: bool,
    log_erb_weights: torch.Tensor,
    normalization: Normalization,
    arguments: argparse.Namespace,
) -> LossMetrics:
    model.eval()
    accumulated = LossMetrics()
    for _ in range(steps):
        (
            features,
            target_normalized,
            target_db,
            mca_db,
            direction_features,
            frequency_hz,
        ) = sample_to_device(sampler.sample_batch(), device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            predicted = model(features.reshape(-1, features.shape[-1]))
        predicted = predicted.reshape(target_normalized.shape)
        _, metrics = calculate_losses(
            predicted,
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
            arguments.direction_weighted_residual,
        )
        add_metrics(accumulated, metrics)
    return average_metrics(accumulated, steps)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: EpochMetrics,
    arguments: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": asdict(metrics),
            "arguments": serializable_arguments(arguments),
            "training_stage": "paired-ear auditory-aware v2",
        },
        path,
    )


def main() -> None:
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v2 training.")
    set_seed(arguments.seed)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    normalization = Normalization.from_json(
        arguments.dataset_root / "training_statistics.json"
    )

    if arguments.overfit_subject is None:
        train_files = list_hdf5_files(arguments.dataset_root, split="train")
        validation_files = list_hdf5_files(arguments.dataset_root, split="val")
    else:
        train_files = list_hdf5_files(
            arguments.dataset_root, subject_ids=[arguments.overfit_subject]
        )
        validation_files = train_files
    train_sampler = BinauralSpectrumSampler(
        train_files,
        normalization,
        directions_per_batch=arguments.directions_per_batch,
        seed=arguments.seed,
        interpolation_only=arguments.interpolation_only,
    )

    def make_validation_sampler() -> BinauralSpectrumSampler:
        return BinauralSpectrumSampler(
            validation_files,
            normalization,
            directions_per_batch=arguments.directions_per_batch,
            seed=arguments.seed + 1,
            interpolation_only=arguments.interpolation_only,
        )

    initial = torch.load(
        arguments.initial_checkpoint, map_location=device, weights_only=False
    )
    initial_arguments = initial["arguments"]
    arguments.width = int(initial_arguments["width"])
    arguments.block_count = int(initial_arguments["block_count"])
    model = ResidualMLP(
        width=arguments.width,
        block_count=arguments.block_count,
    ).to(device)
    model.load_state_dict(initial["model_state"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=arguments.epochs
    )
    use_amp = not arguments.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    with h5py.File(train_files[0], "r") as sample_handle:
        sample_frequency = np.squeeze(sample_handle["frequency_hz"][:])
    erb_weights = torch.from_numpy(make_erb_weights(sample_frequency)).to(device)
    log_erb_weights = torch.log(torch.clamp(erb_weights, min=1e-12)).view(
        1, 1, erb_weights.shape[0], erb_weights.shape[1]
    )

    output_dir = project_root() / "artifacts" / "training" / arguments.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "arguments": serializable_arguments(arguments),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "parameter_count": parameter_count(model),
        "batch_size": train_sampler.batch_size,
        "frequency_count": train_sampler.frequency_count,
        "erb_proxy_band_count": int(erb_weights.shape[0]),
        "initial_checkpoint_epoch": int(initial["epoch"]),
        "initial_checkpoint": str(arguments.initial_checkpoint),
        "eligible_direction_count": int(
            train_sampler.eligible_direction_indices.size
        ),
        "interpolation_only": arguments.interpolation_only,
        "direction_weighted_residual": (
            arguments.direction_weighted_residual
        ),
        "train_subject_count": len(train_files),
        "validation_subject_count": len(validation_files),
        "validation_sampler_seed": arguments.seed + 1,
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
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"device={configuration['device']}, "
        f"parameters={configuration['parameter_count']}, "
        f"paired_batch_samples={configuration['batch_size']}, "
        f"train_files={len(train_files)}, val_files={len(validation_files)}, "
        f"eligible_directions={configuration['eligible_direction_count']}"
    )

    initial_validation = evaluate(
        model,
        make_validation_sampler(),
        arguments.validation_steps,
        device,
        use_amp,
        log_erb_weights,
        normalization,
        arguments,
    )
    (output_dir / "initial_validation.json").write_text(
        json.dumps(asdict(initial_validation), indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "initial_v1 "
        f"total={initial_validation.total:.5f} "
        f"res={initial_validation.residual_mae_db:.3f} "
        f"erb={initial_validation.erb_mae_db:.3f} "
        f"hf={initial_validation.contralateral_high_frequency_mae_db:.3f} "
        f"ild={initial_validation.ild_mae_db:.3f}"
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "epoch": 0,
                "validation/total_loss": initial_validation.total,
                "validation/residual_mae_db": (
                    initial_validation.residual_mae_db
                ),
                "validation/erb_mae_db": initial_validation.erb_mae_db,
                "validation/contralateral_high_frequency_mae_db": (
                    initial_validation.contralateral_high_frequency_mae_db
                ),
                "validation/ild_spectral_proxy_mae_db": (
                    initial_validation.ild_mae_db
                ),
                "optimizer/learning_rate": float(
                    optimizer.param_groups[0]["lr"]
                ),
                "optimizer/amp_scale": float(scaler.get_scale()),
                "checkpoint/is_best": 0,
                "checkpoint/best_epoch_so_far": 0,
            },
            step=0,
        )

    history: list[EpochMetrics] = []
    best_validation_total = float("inf")
    best_epoch = 0
    run_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, arguments.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        accumulated = LossMetrics()
        for _ in range(arguments.steps_per_epoch):
            (
                features,
                target_normalized,
                target_db,
                mca_db,
                direction_features,
                frequency_hz,
            ) = sample_to_device(train_sampler.sample_batch(), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                predicted = model(features.reshape(-1, features.shape[-1]))
            predicted = predicted.reshape(target_normalized.shape)
            loss, step_metrics = calculate_losses(
                predicted,
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
                arguments.direction_weighted_residual,
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            add_metrics(accumulated, step_metrics)

        train_metrics = average_metrics(accumulated, arguments.steps_per_epoch)
        validation_metrics = evaluate(
            model,
            make_validation_sampler(),
            arguments.validation_steps,
            device,
            use_amp,
            log_erb_weights,
            normalization,
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
            validation_total_loss=validation_metrics.total,
            validation_residual_mae_db=validation_metrics.residual_mae_db,
            validation_erb_mae_db=validation_metrics.erb_mae_db,
            validation_contralateral_high_frequency_mae_db=(
                validation_metrics.contralateral_high_frequency_mae_db
            ),
            validation_ild_mae_db=validation_metrics.ild_mae_db,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            elapsed_seconds=time.perf_counter() - epoch_started,
        )
        history.append(metrics)
        print(
            f"epoch={epoch:03d} total={metrics.train_total_loss:.5f} "
            f"train(res={metrics.train_residual_mae_db:.3f},"
            f"erb={metrics.train_erb_mae_db:.3f},"
            f"hf={metrics.train_contralateral_high_frequency_mae_db:.3f},"
            f"ild={metrics.train_ild_mae_db:.3f}) "
            f"val(total={metrics.validation_total_loss:.5f},"
            f"res={metrics.validation_residual_mae_db:.3f},"
            f"erb={metrics.validation_erb_mae_db:.3f},"
            f"hf={metrics.validation_contralateral_high_frequency_mae_db:.3f},"
            f"ild={metrics.validation_ild_mae_db:.3f}) "
            f"time={metrics.elapsed_seconds:.1f}s"
        )
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            metrics,
            arguments,
        )
        is_best = validation_metrics.total < best_validation_total
        if is_best:
            best_validation_total = validation_metrics.total
            best_epoch = epoch
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                metrics,
                arguments,
            )
        scheduler.step()
        with (output_dir / "history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=asdict(metrics).keys())
            writer.writeheader()
            writer.writerows(asdict(item) for item in history)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/total_loss": metrics.train_total_loss,
                    "train/residual_mae_db": metrics.train_residual_mae_db,
                    "train/erb_mae_db": metrics.train_erb_mae_db,
                    "train/contralateral_high_frequency_mae_db": (
                        metrics.train_contralateral_high_frequency_mae_db
                    ),
                    "train/ild_spectral_proxy_mae_db": (
                        metrics.train_ild_mae_db
                    ),
                    "validation/total_loss": metrics.validation_total_loss,
                    "validation/residual_mae_db": (
                        metrics.validation_residual_mae_db
                    ),
                    "validation/erb_mae_db": metrics.validation_erb_mae_db,
                    "validation/contralateral_high_frequency_mae_db": (
                        metrics.validation_contralateral_high_frequency_mae_db
                    ),
                    "validation/ild_spectral_proxy_mae_db": (
                        metrics.validation_ild_mae_db
                    ),
                    "optimizer/learning_rate": metrics.learning_rate,
                    "optimizer/amp_scale": float(scaler.get_scale()),
                    "system/epoch_seconds": metrics.elapsed_seconds,
                    "system/peak_cuda_allocated_mib": (
                        torch.cuda.max_memory_allocated() / (1024.0**2)
                    ),
                    "checkpoint/is_best": int(is_best),
                    "checkpoint/best_epoch_so_far": best_epoch,
                },
                step=epoch,
            )

    best_metrics = history[best_epoch - 1]
    report = {
        "status": "completed",
        "elapsed_seconds": time.perf_counter() - run_started,
        "best_validation_total_loss": best_validation_total,
        "best_epoch": best_epoch,
        "completed_epochs": arguments.epochs,
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / (1024.0**2)
        ),
        "wandb": configuration["wandb"],
        "initial_validation": asdict(initial_validation),
        "best_epoch_metrics": asdict(best_metrics),
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
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
                "best_validation_ild_spectral_proxy_mae_db": (
                    best_metrics.validation_ild_mae_db
                ),
                "elapsed_seconds": report["elapsed_seconds"],
                "peak_cuda_allocated_mib": report[
                    "peak_cuda_allocated_mib"
                ],
            }
        )
        wandb_run.finish(exit_code=0)


if __name__ == "__main__":
    main()
