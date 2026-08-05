"""Offline regression test for v3.2 global-plus-horizontal loss mixing."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from mcar.data import BinauralSpectrumSampler, Normalization
from mcar.models.residual_mlp_cnn import ResidualMLPCNN
from mcar.training.train_mlp_cnn_v3 import (
    calculate_dual_sampling_losses,
    sample_to_device,
)
from mcar.training.train_mlp_v2 import make_erb_weights


def main() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "sample.h5"
        rng = np.random.default_rng(20260803)
        direction_count = 6
        # The last CNN block uses 24 reflected samples on each side, so retain
        # a frequency axis longer than that padding width.
        frequency_count = 65
        with h5py.File(path, "w") as handle:
            shape = (2, direction_count, frequency_count)
            mca = rng.normal(size=shape).astype(np.float32)
            handle.create_dataset("mca_logmag_db", data=mca)
            handle.create_dataset(
                "correction_logmag_db",
                data=rng.normal(size=shape).astype(np.float32),
            )
            handle.create_dataset(
                "target_residual_db",
                data=rng.normal(size=shape).astype(np.float32),
            )
            directions = np.zeros((direction_count, 6), dtype=np.float32)
            directions[:, 1] = np.array([0, 5, 0, -5, 0, 10])
            directions[:, 2] = np.linspace(-1.0, 1.0, direction_count)
            directions[:, 3] = np.array([-1, 1, -1, 1, -1, 1])
            directions[:, 5] = 1.0 / direction_count
            handle.create_dataset("direction_features", data=directions)
            handle.create_dataset(
                "interpolation_evaluation_mask",
                data=np.ones(direction_count, dtype=np.uint8),
            )
            frequency_hz = np.linspace(
                100.0, 20_000.0, frequency_count, dtype=np.float32
            )
            handle.create_dataset("frequency_hz", data=frequency_hz)
            handle.attrs["subject_id"] = 1
            handle.attrs["sparse_order"] = 3
            handle.attrs["split"] = "train"
            strict = handle.create_group("strict_ild")
            strict.create_dataset(
                "mca_selected_phase_rad",
                data=np.zeros(shape, dtype=np.float32),
            )
            strict.create_dataset(
                "mca_outside_real",
                data=np.empty((2, direction_count, 0), dtype=np.float32),
            )
            strict.create_dataset(
                "mca_outside_imag",
                data=np.empty((2, direction_count, 0), dtype=np.float32),
            )
            strict.create_dataset(
                "selected_bin_indices_zero_based",
                data=np.arange(frequency_count, dtype=np.int64),
            )
            strict.create_dataset(
                "outside_bin_indices_zero_based",
                data=np.empty(0, dtype=np.int64),
            )
            strict.create_dataset(
                "reference_ild_db",
                data=np.zeros(direction_count, dtype=np.float32),
            )
            strict.attrs["single_sided_frequency_count"] = frequency_count
            strict.attrs["hrir_length"] = 128

        normalization = Normalization(
            mca_mean=0.0,
            mca_std=1.0,
            correction_mean=0.0,
            correction_std=1.0,
            target_mean=0.0,
            target_std=1.0,
            log_frequency_mean=3.0,
            log_frequency_std=1.0,
        )
        global_sampler = BinauralSpectrumSampler(
            [path],
            normalization,
            directions_per_batch=4,
            seed=1,
            interpolation_only=True,
        )
        horizontal_sampler = BinauralSpectrumSampler(
            [path],
            normalization,
            directions_per_batch=3,
            seed=2,
            strict_ild=True,
            interpolation_only=True,
            horizontal_only=True,
        )
        assert global_sampler.eligible_direction_indices.size == 6
        assert horizontal_sampler.eligible_direction_indices.size == 3
        device = torch.device("cpu")
        global_batch = sample_to_device(
            global_sampler.sample_batch(), device, strict_ild=False
        )
        horizontal_batch = sample_to_device(
            horizontal_sampler.sample_batch(), device, strict_ild=True
        )
        model = ResidualMLPCNN(
            mlp_width=8, mlp_block_count=1, cnn_channels=4
        ).to(device)
        model.freeze_mlp()
        arguments = SimpleNamespace(
            erb_weight=0.75,
            high_frequency_weight=0.25,
            ild_weight=1.0,
            ild_loss_mode="strict_hrir",
            direction_weighted_residual=True,
        )
        erb_weights = torch.from_numpy(make_erb_weights(frequency_hz))
        log_erb_weights = torch.log(torch.clamp(erb_weights, min=1e-12)).view(
            1, 1, erb_weights.shape[0], erb_weights.shape[1]
        )
        loss, metrics, _ = calculate_dual_sampling_losses(
            model,
            global_batch,
            horizontal_batch,
            log_erb_weights,
            normalization,
            arguments,
            use_amp=False,
        )
        if not bool(torch.isfinite(loss).item()) or metrics.ild_mae_db < 0.0:
            raise AssertionError("Dual-sampling loss is not finite")
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
            raise AssertionError("Dual-sampling CNN gradients are invalid")
        print(
            {
                "status": "passed",
                "global_directions": global_sampler.eligible_direction_indices.size,
                "horizontal_directions": horizontal_sampler.eligible_direction_indices.size,
                "total_loss": float(loss.detach()),
                "strict_ild_mae_db": metrics.ild_mae_db,
            }
        )


if __name__ == "__main__":
    main()
