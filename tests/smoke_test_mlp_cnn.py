"""Run a real-data forward/backward smoke test for the v3 MLP + CNN model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from mcar.models.residual_mlp_cnn import (
    ResidualMLPCNN,
    total_parameter_count,
    trainable_parameter_count,
)
from mcar.data import (
    BinauralSpectrumSampler,
    Normalization,
    list_hdf5_files,
)
from mcar.training.train_mlp_v2 import (
    calculate_losses,
    make_erb_weights,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--subject", type=int, default=91)
    parser.add_argument("--directions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def finite_gradients(parameters: list[torch.nn.Parameter]) -> bool:
    gradients = [
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    ]
    return bool(gradients) and all(
        bool(torch.all(torch.isfinite(gradient)).item())
        for gradient in gradients
    )


def nonzero_gradient_tensor_count(
    parameters: list[torch.nn.Parameter],
) -> int:
    return sum(
        int(
            parameter.grad is not None
            and bool(torch.any(parameter.grad != 0).item())
        )
        for parameter in parameters
    )


def main() -> None:
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the v3 smoke test")
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    device = torch.device("cuda")
    use_amp = not arguments.no_amp
    normalization = Normalization.from_json(
        arguments.dataset_root / "training_statistics.json"
    )
    subject_files = list_hdf5_files(
        arguments.dataset_root,
        subject_ids=[arguments.subject],
    )
    sampler = BinauralSpectrumSampler(
        subject_files,
        normalization,
        directions_per_batch=arguments.directions,
        seed=arguments.seed,
    )
    (
        features,
        target_normalized,
        target_db,
        mca_db,
        direction_features,
        frequency_hz,
        metadata,
    ) = sampler.sample_batch()
    checkpoint = torch.load(
        arguments.checkpoint,
        map_location=device,
        weights_only=False,
    )
    checkpoint_arguments = checkpoint["arguments"]
    model = ResidualMLPCNN(
        mlp_width=int(checkpoint_arguments["width"]),
        mlp_block_count=int(checkpoint_arguments["block_count"]),
    ).to(device)
    model.load_mlp_state_dict(checkpoint["model_state"])
    model.freeze_mlp()
    model.train()

    point_features = torch.from_numpy(
        np.transpose(features, (1, 0, 2, 3))
    ).to(device)
    target_normalized_tensor = torch.from_numpy(target_normalized).to(device)
    target_db_tensor = torch.from_numpy(target_db).to(device)
    mca_db_tensor = torch.from_numpy(mca_db).to(device)
    direction_tensor = torch.from_numpy(direction_features).to(device)
    frequency_tensor = torch.from_numpy(frequency_hz).to(device)
    erb_weights = torch.from_numpy(make_erb_weights(frequency_hz)).to(device)
    log_erb_weights = torch.log(torch.clamp(erb_weights, min=1e-12)).view(
        1, 1, erb_weights.shape[0], erb_weights.shape[1]
    )

    torch.cuda.reset_peak_memory_stats()
    with torch.amp.autocast("cuda", enabled=use_amp):
        final_prediction, base_prediction, delta_prediction = model(
            point_features
        )
    expected_shape = (
        arguments.directions,
        2,
        sampler.frequency_count,
    )
    if tuple(final_prediction.shape) != expected_shape:
        raise AssertionError(
            f"Unexpected prediction shape {final_prediction.shape}"
        )
    identity_error = float(
        torch.max(torch.abs(final_prediction - base_prediction)).item()
    )
    if identity_error != 0.0:
        raise AssertionError(
            f"Zero-initialized CNN changed v2 output by {identity_error}"
        )

    optimizer = torch.optim.AdamW(
        model.cnn.parameters(),
        lr=3e-4,
        weight_decay=1e-5,
    )
    cnn_parameters = list(model.cnn.parameters())
    mlp_parameters = list(model.mlp.parameters())
    step_reports: list[dict[str, float | int | bool]] = []
    for step in range(1, 3):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            final_prediction, _, delta_prediction = model(point_features)
        prediction_for_loss = final_prediction.permute(1, 0, 2)
        loss, metrics = calculate_losses(
            prediction_for_loss,
            target_normalized_tensor,
            target_db_tensor,
            mca_db_tensor,
            direction_tensor,
            frequency_tensor,
            log_erb_weights,
            normalization.target_mean,
            normalization.target_std,
            erb_weight=0.50,
            high_frequency_weight=0.25,
            ild_weight=0.25,
        )
        if not bool(torch.isfinite(loss).item()):
            raise AssertionError(f"Non-finite loss at step {step}")
        loss.backward()
        if not finite_gradients(cnn_parameters):
            raise AssertionError(f"Non-finite CNN gradients at step {step}")
        if any(parameter.grad is not None for parameter in mlp_parameters):
            raise AssertionError("Frozen MLP unexpectedly received gradients")
        gradient_count = nonzero_gradient_tensor_count(cnn_parameters)
        optimizer.step()
        step_reports.append(
            {
                "step": step,
                "total_loss": metrics.total,
                "residual_mae_db": metrics.residual_mae_db,
                "erb_proxy_mae_db": metrics.erb_mae_db,
                "contralateral_high_frequency_mae_db": (
                    metrics.contralateral_high_frequency_mae_db
                ),
                "ild_proxy_mae_db": metrics.ild_mae_db,
                "nonzero_cnn_gradient_tensor_count": gradient_count,
                "delta_abs_max_normalized": float(
                    torch.max(torch.abs(delta_prediction.detach())).item()
                ),
            }
        )

    result = {
        "status": "passed",
        "device": torch.cuda.get_device_name(0),
        "amp": use_amp,
        "subject_id": int(metadata["subject_id"]),
        "direction_count": arguments.directions,
        "frequency_count": sampler.frequency_count,
        "input_shape": list(point_features.shape),
        "output_shape": list(final_prediction.shape),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "zero_initialization_identity_error": identity_error,
        "mlp_parameter_count": total_parameter_count(model.mlp),
        "cnn_parameter_count": total_parameter_count(model.cnn),
        "total_parameter_count": total_parameter_count(model),
        "trainable_parameter_count_with_frozen_mlp": (
            trainable_parameter_count(model)
        ),
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / (1024.0**2)
        ),
        "steps": step_reports,
    }
    numeric_values = [
        report[key]
        for report in step_reports
        for key in (
            "total_loss",
            "residual_mae_db",
            "erb_proxy_mae_db",
            "contralateral_high_frequency_mae_db",
            "ild_proxy_mae_db",
            "delta_abs_max_normalized",
        )
    ]
    if not all(math.isfinite(float(value)) for value in numeric_values):
        raise AssertionError("Smoke-test report contains non-finite values")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
