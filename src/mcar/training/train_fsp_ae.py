"""Train the SONICOM-Q26 adaptation of the published FSP-AE baseline."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from mcar.fsp_ae_data import (
    apply_normalization_to_model,
    list_cache_files,
    q26_source_indices,
    read_subject_cache,
)
from mcar.fsp_ae_signal import lsd_loss
from mcar.models.fsp_ae import FreqSrcPosCondAutoEncoder, parameter_count
from mcar.paths import project_root


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def target_indices(
    direction_count: int,
    limit: int | None,
    generator: torch.Generator,
) -> torch.Tensor:
    if limit is None or limit >= direction_count:
        return torch.arange(direction_count)
    if limit <= 0:
        raise ValueError("target direction limit must be positive")
    return torch.randperm(direction_count, generator=generator)[:limit].sort().values


def chunk_indices(indices: torch.Tensor, chunk_size: int) -> list[torch.Tensor]:
    if chunk_size <= 0:
        raise ValueError("target_directions_per_chunk must be positive")
    return list(torch.split(indices, chunk_size))


def subject_tensors(
    cache_file: Path,
    q26: torch.Tensor,
    device: torch.device,
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    subject = read_subject_cache(cache_file)
    magnitude = subject.hrtf_magnitude_db.to(device)
    itd = subject.itd_seconds.to(device)
    frequency = subject.frequency_hz.unsqueeze(0).to(device)
    positions = subject.source_positions_cartesian_m.to(device)
    q26_device = q26.to(device)
    prototype_inputs = (
        magnitude[q26_device].unsqueeze(0),
        itd[q26_device].unsqueeze(0),
        frequency,
        positions[q26_device].unsqueeze(0),
    )
    return subject, magnitude, itd, positions, prototype_inputs


def train_subject(
    model: FreqSrcPosCondAutoEncoder,
    optimizer: torch.optim.Optimizer,
    cache_file: Path,
    q26: torch.Tensor,
    selected_targets: torch.Tensor,
    chunk_size: int,
    itd_loss_weight: float,
    gradient_clip: float,
    device: torch.device,
) -> dict[str, float]:
    _, magnitude, itd, positions, inputs = subject_tensors(
        cache_file, q26, device
    )
    optimizer.zero_grad(set_to_none=True)
    prototype = model.encode(*inputs, dataset_name="sonicom")
    # Decode many direction chunks without retaining every decoder graph. The
    # leaf collects dL/dprototype; one final backward call propagates the sum
    # through the shared Q26 encoder graph.
    prototype_leaf = prototype.detach().requires_grad_(True)
    chunks = chunk_indices(selected_targets, chunk_size)
    metrics = {"lsd": 0.0, "itd_l1": 0.0, "loss": 0.0}
    target_count = selected_targets.numel()
    for chunk in chunks:
        chunk_device = chunk.to(device)
        predicted_magnitude, predicted_itd = model.decode(
            prototype_leaf,
            inputs[2],
            positions[chunk_device].unsqueeze(0),
            dataset_name="sonicom",
        )
        lsd = lsd_loss(
            predicted_magnitude, magnitude[chunk_device].unsqueeze(0)
        )
        itd_l1 = torch.mean(
            torch.abs(predicted_itd - itd[chunk_device].unsqueeze(0))
        )
        scale = chunk.numel() / target_count
        loss = scale * (lsd + itd_loss_weight * itd_l1)
        loss.backward()
        metrics["lsd"] += scale * float(lsd.detach())
        metrics["itd_l1"] += scale * float(itd_l1.detach())
        metrics["loss"] += float(loss.detach())
    if prototype_leaf.grad is None:
        raise RuntimeError("decoder did not produce a prototype gradient")
    prototype.backward(prototype_leaf.grad)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=gradient_clip
    )
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("non-finite FSP-AE gradient norm")
    optimizer.step()
    metrics["gradient_norm"] = float(gradient_norm)
    return metrics


@torch.no_grad()
def evaluate(
    model: FreqSrcPosCondAutoEncoder,
    files: Sequence[Path],
    q26: torch.Tensor,
    target_limit: int | None,
    chunk_size: int,
    itd_loss_weight: float,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    totals = {"lsd": 0.0, "itd_l1": 0.0, "loss": 0.0}
    for cache_file in files:
        _, magnitude, itd, positions, inputs = subject_tensors(
            cache_file, q26, device
        )
        selected = target_indices(magnitude.shape[0], target_limit, generator)
        prototype = model.encode(*inputs, dataset_name="sonicom")
        subject_metrics = {"lsd": 0.0, "itd_l1": 0.0, "loss": 0.0}
        for chunk in chunk_indices(selected, chunk_size):
            chunk_device = chunk.to(device)
            predicted_magnitude, predicted_itd = model.decode(
                prototype,
                inputs[2],
                positions[chunk_device].unsqueeze(0),
                dataset_name="sonicom",
            )
            lsd = lsd_loss(
                predicted_magnitude, magnitude[chunk_device].unsqueeze(0)
            )
            itd_l1 = torch.mean(
                torch.abs(predicted_itd - itd[chunk_device].unsqueeze(0))
            )
            scale = chunk.numel() / selected.numel()
            subject_metrics["lsd"] += scale * float(lsd)
            subject_metrics["itd_l1"] += scale * float(itd_l1)
        subject_metrics["loss"] = (
            subject_metrics["lsd"]
            + itd_loss_weight * subject_metrics["itd_l1"]
        )
        for key in totals:
            totals[key] += subject_metrics[key]
    model.train()
    return {key: value / len(files) for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: FreqSrcPosCondAutoEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_validation_loss: float,
    best_epoch: int,
    config: dict[str, Any],
    generator: torch.Generator,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(
        {
            "model": model.state_dict(),
            "stats": model.stats,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "training_generator_state": generator.get_state(),
            "config": config,
            "method": "FSP-AE-Q26 adaptation",
            "upstream": "https://github.com/ikets/FSP-AE",
            "upstream_license": "CC BY 4.0",
        },
        partial,
    )
    os.replace(partial, path)


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=Path,
        default=(
            root
            / "configs"
            / "experiments"
            / "sonicom_fsp_ae_q26_smoke.json"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = load_json(args.config)
    seed = int(config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)

    cache_root = root / config["data"]["cache_root"]
    train_files = list_cache_files(cache_root, "train")
    validation_files = list_cache_files(cache_root, "val")
    train_limit = config["data"].get("train_subject_limit")
    validation_limit = config["data"].get("validation_subject_limit")
    if train_limit is not None:
        train_files = train_files[: int(train_limit)]
    if validation_limit is not None:
        validation_files = validation_files[: int(validation_limit)]
    normalization = load_json(cache_root / "normalization.json")
    if not normalization.get("complete", False) and not config["data"].get(
        "allow_incomplete_normalization", False
    ):
        raise RuntimeError("formal training requires complete train normalization")
    q26 = torch.from_numpy(q26_source_indices(root / config["data"]["q26_csv"]))

    model = FreqSrcPosCondAutoEncoder().to(device)
    apply_normalization_to_model(model, normalization)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"]["learning_rate"])
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[
            int(value)
            for value in config["training"].get("learning_rate_milestones", [])
        ],
        gamma=float(config["training"].get("learning_rate_gamma", 0.1)),
    )
    run_root = root / "artifacts" / "training" / config["run_name"]
    if args.resume is None and run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(
            f"training output already exists; use --resume: {run_root}"
        )
    run_root.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    generator = torch.Generator().manual_seed(seed)
    start_epoch = 1
    if args.resume is not None:
        checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        if checkpoint.get("config", {}).get("run_name") != config["run_name"]:
            raise ValueError("resume checkpoint run_name differs from config")
        model.load_state_dict(checkpoint["model"])
        model.stats = checkpoint["stats"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        best_epoch = int(checkpoint.get("best_epoch", checkpoint["epoch"]))
        start_epoch = int(checkpoint["epoch"]) + 1
        if "training_generator_state" in checkpoint:
            generator.set_state(checkpoint["training_generator_state"])
        history_path = run_root / "history.json"
        if history_path.is_file():
            history = [
                record
                for record in load_json(history_path)
                if int(record["epoch"]) < start_epoch
            ]
    write_json_atomic(run_root / "config.json", config)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    epochs = int(config["training"]["epochs"])
    steps_per_epoch = config["training"].get("steps_per_epoch")
    target_limit = config["training"].get("target_direction_limit")
    validation_target_limit = config["validation"].get("target_direction_limit")
    chunk_size = int(config["training"]["target_directions_per_chunk"])
    itd_weight = float(config["training"]["itd_loss_weight"])
    gradient_clip = float(config["training"]["gradient_clip"])

    if start_epoch > epochs:
        raise ValueError("resume checkpoint already reached configured epochs")
    checkpoint_epochs = {
        int(value) for value in config["training"].get("checkpoint_epochs", [])
    }

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        order = torch.randperm(len(train_files), generator=generator).tolist()
        if steps_per_epoch is not None:
            order = order[: int(steps_per_epoch)]
        epoch_totals = {"lsd": 0.0, "itd_l1": 0.0, "loss": 0.0, "gradient_norm": 0.0}
        for file_index in order:
            subject = read_subject_cache(train_files[file_index])
            selected = target_indices(
                subject.hrtf_magnitude_db.shape[0], target_limit, generator
            )
            metrics = train_subject(
                model,
                optimizer,
                train_files[file_index],
                q26,
                selected,
                chunk_size,
                itd_weight,
                gradient_clip,
                device,
            )
            for key in epoch_totals:
                epoch_totals[key] += metrics[key]
        train_metrics = {
            key: value / len(order) for key, value in epoch_totals.items()
        }
        validation_metrics = evaluate(
            model,
            validation_files,
            q26,
            validation_target_limit,
            int(config["validation"]["target_directions_per_chunk"]),
            itd_weight,
            device,
            int(config["validation"].get("fixed_seed", seed + 10_000)),
        )
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(record)
        improved = validation_metrics["loss"] < best_validation_loss
        if improved:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
        scheduler.step()
        if improved:
            save_checkpoint(
                run_root / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_validation_loss,
                best_epoch,
                config,
                generator,
            )
        if epoch in checkpoint_epochs or epoch == epochs:
            save_checkpoint(
                run_root / f"checkpoint_epoch_{epoch:04d}.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_validation_loss,
                best_epoch,
                config,
                generator,
            )
        write_json_atomic(run_root / "history.json", history)

    elapsed = time.perf_counter() - started
    summary = {
        "status": "completed",
        "run_name": config["run_name"],
        "parameter_count": parameter_count(model),
        "train_subject_count_read": len(train_files),
        "validation_subject_count_read": len(validation_files),
        "test_subject_count_read": 0,
        "best_validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
    }
    write_json_atomic(run_root / "history.json", history)
    write_json_atomic(run_root / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
