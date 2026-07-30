"""Real-data gradient smoke test for the strict-HRIR-ILD v3.1 loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from mcar.data import BinauralSpectrumSampler, Normalization, list_hdf5_files
from mcar.models.residual_mlp_cnn import ResidualMLPCNN
from mcar.training.train_mlp_cnn_v3 import (
    calculate_model_losses,
    sample_to_device,
)
from mcar.training.train_mlp_v2 import make_erb_weights


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--subject", type=int, default=91)
    parser.add_argument("--directions", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the v3.1 smoke test")
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    device = torch.device("cuda")
    use_amp = not arguments.no_amp

    normalization = Normalization.from_json(
        arguments.dataset_root / "training_statistics.json"
    )
    files = list_hdf5_files(
        arguments.dataset_root, subject_ids=[arguments.subject]
    )
    sampler = BinauralSpectrumSampler(
        files,
        normalization,
        directions_per_batch=arguments.directions,
        seed=arguments.seed,
        strict_ild=True,
    )
    checkpoint = torch.load(
        arguments.checkpoint, map_location=device, weights_only=False
    )
    checkpoint_arguments = checkpoint["arguments"]
    model = ResidualMLPCNN(
        mlp_width=int(checkpoint_arguments["width"]),
        mlp_block_count=int(checkpoint_arguments["block_count"]),
    ).to(device)
    model.load_mlp_state_dict(checkpoint["model_state"])
    model.freeze_mlp()
    model.train()

    with h5py.File(files[0], "r") as handle:
        frequency_hz = np.squeeze(handle["frequency_hz"][:])
    erb_weights = torch.from_numpy(make_erb_weights(frequency_hz)).to(device)
    log_erb_weights = torch.log(torch.clamp(erb_weights, min=1e-12)).view(
        1, 1, erb_weights.shape[0], erb_weights.shape[1]
    )
    loss_arguments = SimpleNamespace(
        erb_weight=0.50,
        high_frequency_weight=0.25,
        ild_weight=0.25,
        ild_loss_mode="strict_hrir",
    )
    batch = sample_to_device(sampler.sample_batch(), device, strict_ild=True)
    point_features = batch[0]
    if not isinstance(point_features, torch.Tensor):
        raise TypeError("Point features were not transferred to CUDA")
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        initial, base, _ = model(point_features)
    identity_error = float(torch.max(torch.abs(initial - base)).item())
    if identity_error != 0.0:
        raise AssertionError(
            f"Zero-initialized v3.1 does not reproduce v2: {identity_error}"
        )

    optimizer = torch.optim.AdamW(model.cnn.parameters(), lr=3e-4)
    step_reports: list[dict[str, float | int]] = []
    for step in range(1, 3):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics, delta = calculate_model_losses(
            model,
            batch,
            log_erb_weights,
            normalization,
            loss_arguments,
            use_amp,
        )
        if not bool(torch.isfinite(loss).item()):
            raise AssertionError(f"Non-finite v3.1 loss at step {step}")
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.cnn.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not all(
            bool(torch.all(torch.isfinite(gradient)).item())
            for gradient in gradients
        ):
            raise AssertionError("Missing or non-finite CNN gradients")
        nonzero = sum(
            int(bool(torch.any(gradient != 0).item()))
            for gradient in gradients
        )
        optimizer.step()
        step_reports.append(
            {
                "step": step,
                "total_loss": metrics.total,
                "strict_hrir_ild_mae_db": metrics.ild_mae_db,
                "spectral_proxy_ild_mae_db": (
                    metrics.ild_spectral_proxy_mae_db
                ),
                "nonzero_cnn_gradient_tensor_count": nonzero,
                "delta_abs_max_normalized": float(
                    torch.max(torch.abs(delta.detach())).item()
                ),
            }
        )

    print(
        json.dumps(
            {
                "status": "passed",
                "subject_id": arguments.subject,
                "direction_count": arguments.directions,
                "frequency_count": sampler.frequency_count,
                "zero_initialization_identity_error": identity_error,
                "device": torch.cuda.get_device_name(0),
                "amp": use_amp,
                "steps": step_reports,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
