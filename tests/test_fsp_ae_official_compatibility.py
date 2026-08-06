"""Compare the MCAR FSP-AE port with the official CC BY 4.0 implementation.

This integration test intentionally requires a separately obtained official
repository and checkpoint. It does not vendor either artifact into MCAR.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from mcar.models.fsp_ae import FreqSrcPosCondAutoEncoder
from mcar.fsp_ae_signal import reconstruct_hrir_with_itd


def official_config() -> SimpleNamespace:
    return SimpleNamespace(
        fourier_feature_mapping=SimpleNamespace(
            num_features=SimpleNamespace(source_position=16, frequency=8),
            trainable=True,
        ),
        radius_norm=1.5,
        freq_norm=16_000.0,
        num_mes_norm=865.0,
        encoder=SimpleNamespace(
            num_layers=2,
            mid_dim=16,
            out_dim=64,
            use_freq=None,
            nonlinear=None,
        ),
        decoder=SimpleNamespace(
            num_layers=2,
            in_dim=64,
            mid_dim=16,
            use_freq=None,
            nonlinear=None,
        ),
        weight_bias_generator=SimpleNamespace(num_layers=2, mid_dim=64),
    )


def normalized_positions(count: int) -> torch.Tensor:
    positions = torch.randn(1, count, 3)
    return 1.5 * positions / torch.linalg.vector_norm(
        positions, dim=-1, keepdim=True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("official_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    if not (args.official_root / "model" / "models.py").is_file():
        raise FileNotFoundError("official FSP-AE model package was not found")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    sys.path.insert(0, str(args.official_root))
    try:
        from model import FreqSrcPosCondAutoEncoder as OfficialFSPAE
        from utils import get_hrir_with_itd as official_reconstruct_hrir
    finally:
        sys.path.pop(0)

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    official = OfficialFSPAE(official_config())
    port = FreqSrcPosCondAutoEncoder()
    official.load_state_dict(checkpoint["model"])
    port.load_state_dict(checkpoint["model"])
    official.stats = checkpoint["stats"]
    port.stats = checkpoint["stats"]
    official.eval()
    port.eval()

    torch.manual_seed(20260806)
    magnitude = torch.randn(1, 4, 2, 512) * 10.0 - 6.0
    itd = torch.randn(1, 4) * 3.5e-4
    frequency = torch.linspace(31.25, 16_000.0, 512).unsqueeze(0)
    measurement_positions = normalized_positions(4)
    target_positions = normalized_positions(5)
    with torch.no_grad():
        official_magnitude, official_itd = official(
            magnitude,
            itd,
            frequency,
            measurement_positions,
            target_positions,
            "hutubs",
            "cpu",
        )
        port_magnitude, port_itd = port(
            magnitude,
            itd,
            frequency,
            measurement_positions,
            target_positions,
            "hutubs",
        )
    magnitude_error = torch.max(
        torch.abs(official_magnitude - port_magnitude)
    ).item()
    itd_error = torch.max(torch.abs(official_itd - port_itd)).item()
    official_hrir = official_reconstruct_hrir(
        official_magnitude,
        official_itd,
        input_kind="hrtf_mag",
        fs=32_000.0,
        fs_up=384_000.0,
    )
    port_hrir = reconstruct_hrir_with_itd(
        port_magnitude,
        port_itd,
        sampling_rate_hz=32_000.0,
        upsampled_rate_hz=384_000.0,
    )
    hrir_error = torch.max(torch.abs(official_hrir - port_hrir)).item()
    if magnitude_error > 1e-6 or itd_error > 1e-10 or hrir_error > 1e-7:
        raise AssertionError(
            f"official compatibility failed: magnitude={magnitude_error}, "
            f"itd={itd_error}, hrir={hrir_error}"
        )
    print(
        {
            "status": "passed",
            "official_commit_checkpoint": str(args.checkpoint),
            "magnitude_max_abs_error_db": magnitude_error,
            "itd_max_abs_error_seconds": itd_error,
            "hrir_max_abs_error": hrir_error,
        }
    )


if __name__ == "__main__":
    main()
