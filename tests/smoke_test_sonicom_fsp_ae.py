"""Run one SONICOM-Q26 subject through the FSP-AE encoder and decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mcar.fsp_ae_data import (
    apply_normalization_to_model,
    q26_source_indices,
    read_subject_cache,
)
from mcar.fsp_ae_signal import lsd_loss
from mcar.models.fsp_ae import FreqSrcPosCondAutoEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("subject_cache", type=Path)
    parser.add_argument("normalization_json", type=Path)
    parser.add_argument("q26_csv", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--target-limit", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.target_limit <= 0:
        raise ValueError("target-limit must be positive")

    subject = read_subject_cache(args.subject_cache)
    if subject.split != "train":
        raise PermissionError("this smoke test intentionally requires a train subject")
    statistics = json.loads(
        args.normalization_json.read_text(encoding="utf-8")
    )
    model = FreqSrcPosCondAutoEncoder()
    if args.checkpoint is not None:
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
    apply_normalization_to_model(model, statistics)
    model.to(args.device).eval()

    q26 = torch.from_numpy(q26_source_indices(args.q26_csv))
    target_count = min(args.target_limit, subject.hrtf_magnitude_db.shape[0])
    target_indices = torch.arange(target_count)
    magnitude = subject.hrtf_magnitude_db
    positions = subject.source_positions_cartesian_m
    with torch.no_grad():
        prediction, predicted_itd = model(
            magnitude[q26].unsqueeze(0).to(args.device),
            subject.itd_seconds[q26].unsqueeze(0).to(args.device),
            subject.frequency_hz.unsqueeze(0).to(args.device),
            positions[q26].unsqueeze(0).to(args.device),
            positions[target_indices].unsqueeze(0).to(args.device),
            "sonicom",
        )
        loss = lsd_loss(
            prediction,
            magnitude[target_indices].unsqueeze(0).to(args.device),
        )
    assert prediction.shape == (1, target_count, 2, 512)
    assert predicted_itd.shape == (1, target_count)
    assert torch.all(torch.isfinite(prediction))
    assert torch.all(torch.isfinite(predicted_itd))
    print(
        {
            "status": "passed",
            "subject_id": subject.subject_id,
            "split": subject.split,
            "q26_measurement_count": int(q26.numel()),
            "target_count": target_count,
            "frequency_count": int(subject.frequency_hz.numel()),
            "frequency_maximum_hz": float(subject.frequency_hz[-1]),
            "initial_lsd_db": float(loss),
            "test_subject_count_read": 0,
        }
    )


if __name__ == "__main__":
    main()
