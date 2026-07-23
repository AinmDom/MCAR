"""Train a lightweight MLP on the HUTUBS MCA residual data set."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from residual_data import Normalization, ResidualBlockSampler, list_hdf5_files
from residual_model import ResidualMLP, parameter_count


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_mae_db: float
    validation_mae_db: float
    baseline_validation_mae_db: float
    learning_rate: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="For example residual_learning/data/hutubs_residual_v1_n03",
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
        "--overfit-subject",
        type=int,
        help="Use one subject for both train and validation as a pipeline sanity check.",
    )
    parser.add_argument("--no-amp", action="store_true")
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

    stage_root = Path(__file__).resolve().parents[1]
    output_dir = stage_root / "runs" / arguments.run_name
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
    )
    validation_sampler = ResidualBlockSampler(
        validation_files,
        normalization,
        arguments.directions_per_batch,
        arguments.frequencies_per_batch,
        arguments.seed + 1,
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
        "train_files": [str(path) for path in train_files],
        "validation_files": [str(path) for path in validation_files],
    }
    (output_dir / "configuration.json").write_text(
        json.dumps(run_configuration, indent=2), encoding="utf-8"
    )
    print(
        f"device={run_configuration['device']}, parameters={run_configuration['parameter_count']}, "
        f"batch_size={train_sampler.batch_size}, train_files={len(train_files)}, "
        f"validation_files={len(validation_files)}"
    )

    history: list[EpochMetrics] = []
    best_validation_mae = float("inf")
    for epoch in range(1, arguments.epochs + 1):
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
            validation_sampler,
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
        )
        history.append(metrics)
        print(
            f"epoch={epoch:03d} loss={metrics.train_loss:.6f} "
            f"train_mae={metrics.train_mae_db:.4f} dB "
            f"val_mae={metrics.validation_mae_db:.4f} dB "
            f"zero_val_mae={metrics.baseline_validation_mae_db:.4f} dB"
        )
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, metrics, arguments)
        if validation_mae < best_validation_mae:
            best_validation_mae = validation_mae
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, metrics, arguments)
        scheduler.step()

        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=asdict(metrics).keys())
            writer.writeheader()
            writer.writerows(asdict(item) for item in history)


if __name__ == "__main__":
    main()
