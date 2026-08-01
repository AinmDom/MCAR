"""Train a lightweight MLP on an MCA residual data set."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from mcar.data import Normalization, ResidualBlockSampler, list_hdf5_files
from mcar.models.residual_mlp import ResidualMLP, parameter_count
from mcar.paths import project_root


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_mae_db: float
    validation_mae_db: float
    baseline_validation_mae_db: float
    learning_rate: float
    epoch_seconds: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="For example data/processed/hutubs_residual_v1_n03",
    )
    parser.add_argument("--run-name", default="mlp_n03")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=1200)
    parser.add_argument("--validation-steps", type=int, default=128)
    parser.add_argument("--directions-per-batch", type=int, default=64)
    parser.add_argument("--frequencies-per-batch", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--block-count", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--direction-sampling",
        choices=("uniform", "solid_angle"),
        default="uniform",
        help=(
            "Sample direction indices uniformly or according to the positive "
            "weights in direction_features[:, 5]."
        ),
    )
    parser.add_argument(
        "--interpolation-only",
        action="store_true",
        help="Exclude sparse input directions using interpolation_evaluation_mask.",
    )
    parser.add_argument(
        "--overfit-subject",
        type=int,
        help="Use one subject for both train and validation as a pipeline sanity check.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "online", "offline"),
        default="disabled",
    )
    parser.add_argument("--wandb-project", default="mcar-residual-mlp")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-job-type", default="mlp-v1-train")
    parser.add_argument("--wandb-tags", nargs="*")
    parser.add_argument("--wandb-notes")
    parser.add_argument("--wandb-watch", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(values).to(device, non_blocking=True)


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
            "W&B tracking was requested but wandb is not installed."
        ) from error

    tags = arguments.wandb_tags or ["mlp-v1"]
    run = wandb.init(
        project=arguments.wandb_project,
        entity=arguments.wandb_entity,
        name=arguments.wandb_run_name or arguments.run_name,
        group=arguments.wandb_group,
        job_type=arguments.wandb_job_type,
        tags=tags,
        notes=arguments.wandb_notes,
        config={
            **serializable_arguments(arguments),
            "device": configuration["device"],
            "torch_version": configuration["torch_version"],
            "cuda_version": configuration["cuda_version"],
            "parameter_count": configuration["parameter_count"],
            "batch_size": configuration["batch_size"],
            "train_subject_count": configuration["train_subject_count"],
            "validation_subject_count": configuration[
                "validation_subject_count"
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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    sampler: ResidualBlockSampler,
    steps: int,
    device: torch.device,
    use_amp: bool,
    target_mean: float,
    target_std: float,
) -> tuple[float, float]:
    model.eval()
    absolute_error = 0.0
    baseline_absolute_error = 0.0
    sample_count = 0
    for _ in range(steps):
        features, target_normalized, _ = sampler.sample_batch()
        feature_tensor = to_device(features, device)
        target_tensor = to_device(target_normalized, device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            predicted_normalized = model(feature_tensor)
        predicted_db = predicted_normalized.float() * target_std + target_mean
        target_db = target_tensor.float() * target_std + target_mean
        absolute_error += float(torch.sum(torch.abs(target_db - predicted_db)).item())
        baseline_absolute_error += float(torch.sum(torch.abs(target_db)).item())
        sample_count += target_db.numel()
    return absolute_error / sample_count, baseline_absolute_error / sample_count


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
        },
        path,
    )


def main() -> None:
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training entry point.")
    set_seed(arguments.seed)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    output_dir = project_root() / "artifacts" / "training" / arguments.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    normalization = Normalization.from_json(arguments.dataset_root / "training_statistics.json")

    if arguments.overfit_subject is None:
        train_files = list_hdf5_files(arguments.dataset_root, split="train")
        validation_files = list_hdf5_files(arguments.dataset_root, split="val")
    else:
        train_files = list_hdf5_files(
            arguments.dataset_root, subject_ids=[arguments.overfit_subject]
        )
        validation_files = train_files

    train_sampler = ResidualBlockSampler(
        train_files,
        normalization,
        arguments.directions_per_batch,
        arguments.frequencies_per_batch,
        arguments.seed,
        direction_sampling=arguments.direction_sampling,
        interpolation_only=arguments.interpolation_only,
    )

    def make_validation_sampler() -> ResidualBlockSampler:
        return ResidualBlockSampler(
            validation_files,
            normalization,
            arguments.directions_per_batch,
            arguments.frequencies_per_batch,
            arguments.seed + 1,
            direction_sampling=arguments.direction_sampling,
            interpolation_only=arguments.interpolation_only,
        )

    model = ResidualMLP(width=arguments.width, block_count=arguments.block_count).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=arguments.learning_rate, weight_decay=arguments.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=arguments.epochs
    )
    loss_function = nn.SmoothL1Loss(beta=1.0)
    use_amp = not arguments.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_configuration = {
        "arguments": serializable_arguments(arguments),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "parameter_count": parameter_count(model),
        "batch_size": train_sampler.batch_size,
        "eligible_direction_count": int(
            train_sampler.eligible_direction_indices.size
        ),
        "direction_sampling": arguments.direction_sampling,
        "interpolation_only": arguments.interpolation_only,
        "train_subject_count": len(train_files),
        "validation_subject_count": len(validation_files),
        "validation_sampler_seed": arguments.seed + 1,
        "validation_batches_are_fixed_each_epoch": True,
        "train_files": [str(path) for path in train_files],
        "validation_files": [str(path) for path in validation_files],
    }
    wandb_run = initialize_wandb(
        arguments,
        run_configuration,
        output_dir,
        model,
    )
    run_configuration["wandb"] = {
        "enabled": wandb_run is not None,
        "mode": arguments.wandb_mode,
        "project": arguments.wandb_project,
        "entity": arguments.wandb_entity,
        "run_name": arguments.wandb_run_name or arguments.run_name,
        "run_id": None if wandb_run is None else wandb_run.id,
        "run_url": None if wandb_run is None else wandb_run.url,
    }
    (output_dir / "configuration.json").write_text(
        json.dumps(run_configuration, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"device={run_configuration['device']}, parameters={run_configuration['parameter_count']}, "
        f"batch_size={train_sampler.batch_size}, train_files={len(train_files)}, "
        f"validation_files={len(validation_files)}, "
        f"eligible_directions={run_configuration['eligible_direction_count']}, "
        f"direction_sampling={arguments.direction_sampling}"
    )

    history: list[EpochMetrics] = []
    best_validation_mae = float("inf")
    best_epoch = 0
    run_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, arguments.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        sample_count = 0
        for _ in range(arguments.steps_per_epoch):
            features, target_normalized, _ = train_sampler.sample_batch()
            feature_tensor = to_device(features, device)
            target_tensor = to_device(target_normalized, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                predicted_normalized = model(feature_tensor)
                loss = loss_function(predicted_normalized, target_tensor)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            predicted_db = predicted_normalized.detach().float() * normalization.target_std + normalization.target_mean
            target_db = target_tensor.float() * normalization.target_std + normalization.target_mean
            running_loss += float(loss.item()) * target_tensor.numel()
            running_mae += float(torch.sum(torch.abs(target_db - predicted_db)).item())
            sample_count += target_tensor.numel()

        validation_mae, baseline_validation_mae = evaluate(
            model,
            make_validation_sampler(),
            arguments.validation_steps,
            device,
            use_amp,
            normalization.target_mean,
            normalization.target_std,
        )
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=running_loss / sample_count,
            train_mae_db=running_mae / sample_count,
            validation_mae_db=validation_mae,
            baseline_validation_mae_db=baseline_validation_mae,
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            epoch_seconds=time.perf_counter() - epoch_started,
        )
        history.append(metrics)
        print(
            f"epoch={epoch:03d} loss={metrics.train_loss:.6f} "
            f"train_mae={metrics.train_mae_db:.4f} dB "
            f"val_mae={metrics.validation_mae_db:.4f} dB "
            f"zero_val_mae={metrics.baseline_validation_mae_db:.4f} dB "
            f"time={metrics.epoch_seconds:.1f}s"
        )
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, metrics, arguments)
        is_best = validation_mae < best_validation_mae
        if is_best:
            best_validation_mae = validation_mae
            best_epoch = epoch
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, metrics, arguments)
        scheduler.step()

        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=asdict(metrics).keys())
            writer.writeheader()
            writer.writerows(asdict(item) for item in history)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/smooth_l1_loss": metrics.train_loss,
                    "train/residual_mae_db": metrics.train_mae_db,
                    "validation/residual_mae_db": (
                        metrics.validation_mae_db
                    ),
                    "validation/mca_zero_residual_mae_db": (
                        metrics.baseline_validation_mae_db
                    ),
                    "validation/improvement_percent": 100.0
                    * (
                        1.0
                        - metrics.validation_mae_db
                        / metrics.baseline_validation_mae_db
                    ),
                    "optimizer/learning_rate": metrics.learning_rate,
                    "optimizer/amp_scale": float(scaler.get_scale()),
                    "system/epoch_seconds": metrics.epoch_seconds,
                    "system/peak_cuda_allocated_mib": (
                        torch.cuda.max_memory_allocated() / (1024.0**2)
                    ),
                    "checkpoint/is_best": int(is_best),
                    "checkpoint/best_epoch_so_far": best_epoch,
                },
                step=epoch,
            )

    elapsed_seconds = time.perf_counter() - run_started
    best_metrics = history[best_epoch - 1]
    report = {
        "status": "completed",
        "elapsed_seconds": elapsed_seconds,
        "completed_epochs": arguments.epochs,
        "best_epoch": best_epoch,
        "best_validation_mae_db": best_validation_mae,
        "best_validation_mca_zero_residual_mae_db": (
            best_metrics.baseline_validation_mae_db
        ),
        "best_validation_improvement_percent": 100.0
        * (
            1.0
            - best_validation_mae
            / best_metrics.baseline_validation_mae_db
        ),
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / (1024.0**2)
        ),
        "wandb": run_configuration["wandb"],
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
                "best_validation_mae_db": best_validation_mae,
                "best_validation_improvement_percent": report[
                    "best_validation_improvement_percent"
                ],
                "elapsed_seconds": elapsed_seconds,
                "peak_cuda_allocated_mib": report[
                    "peak_cuda_allocated_mib"
                ],
            }
        )
        wandb_run.finish(exit_code=0)


if __name__ == "__main__":
    main()
