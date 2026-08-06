"""Generate SONICOM FSP-AE magnitude, ITD, and strict-HRIR predictions."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from mcar.fsp_ae_data import (
    SONICOM_HRIR_LENGTH,
    apply_normalization_to_model,
    list_cache_files,
    q26_source_indices,
    read_subject_cache,
)
from mcar.fsp_ae_signal import reconstruct_hrir_with_itd
from mcar.models.fsp_ae import FreqSrcPosCondAutoEncoder
from mcar.paths import project_root


def write_prediction(
    path: Path,
    subject_id: str,
    split: str,
    magnitude_db: torch.Tensor,
    itd_seconds: torch.Tensor,
    frequency_hz: torch.Tensor,
    hrir: torch.Tensor,
    checkpoint_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    try:
        with h5py.File(partial, "w") as handle:
            handle.create_dataset(
                "predicted_magnitude_db",
                data=magnitude_db.numpy(),
                chunks=(64, 2, magnitude_db.shape[-1]),
                compression="gzip",
                compression_opts=4,
            )
            handle.create_dataset(
                "predicted_itd_seconds", data=itd_seconds.numpy()
            )
            handle.create_dataset(
                "predicted_hrir", data=hrir.numpy(), compression="gzip"
            )
            handle.create_dataset("frequency_hz", data=frequency_hz.numpy())
            handle.attrs["subject_id"] = subject_id
            handle.attrs["split"] = split
            handle.attrs["method"] = "FSP-AE-Q26 adaptation"
            handle.attrs["checkpoint_path"] = str(checkpoint_path)
            handle.attrs["sampling_rate_hz"] = 44_100.0
            handle.attrs["nfft"] = 1024
            handle.attrs["hrir_length"] = SONICOM_HRIR_LENGTH
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("run_name")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=root / "data" / "processed" / "sonicom_fsp_ae_q26_v1",
    )
    parser.add_argument(
        "--q26-csv",
        type=Path,
        default=root / "configs" / "data" / "sonicom_sparse_grid_q26_v1.csv",
    )
    parser.add_argument("--subject-limit", type=int)
    parser.add_argument("--target-directions-per-chunk", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        raise PermissionError("test inference requires --allow-test")
    if args.subject_limit is not None and args.subject_limit <= 0:
        raise ValueError("subject-limit must be positive")
    if args.target_directions_per_chunk <= 0:
        raise ValueError("target-directions-per-chunk must be positive")
    device = torch.device(args.device)
    files = list_cache_files(
        args.cache_root, args.split, allow_test=args.allow_test
    )
    if args.subject_limit is not None:
        files = files[: args.subject_limit]
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model = FreqSrcPosCondAutoEncoder()
    model.load_state_dict(checkpoint["model"])
    model.stats = checkpoint["stats"]
    if "sonicom" not in model.stats:
        normalization = json.loads(
            (args.cache_root / "normalization.json").read_text(encoding="utf-8")
        )
        apply_normalization_to_model(model, normalization)
    model.to(device).eval()
    q26 = torch.from_numpy(q26_source_indices(args.q26_csv)).to(device)
    output_root = root / "artifacts" / "reconstruction" / args.run_name
    started = time.perf_counter()

    for file_index, cache_file in enumerate(files, start=1):
        subject = read_subject_cache(cache_file)
        magnitude = subject.hrtf_magnitude_db.to(device)
        itd = subject.itd_seconds.to(device)
        frequency = subject.frequency_hz.unsqueeze(0).to(device)
        positions = subject.source_positions_cartesian_m.to(device)
        with torch.no_grad():
            prototype = model.encode(
                magnitude[q26].unsqueeze(0),
                itd[q26].unsqueeze(0),
                frequency,
                positions[q26].unsqueeze(0),
                dataset_name="sonicom",
            )
            magnitude_chunks: list[torch.Tensor] = []
            itd_chunks: list[torch.Tensor] = []
            for start in range(
                0, positions.shape[0], args.target_directions_per_chunk
            ):
                stop = min(
                    start + args.target_directions_per_chunk, positions.shape[0]
                )
                predicted_magnitude, predicted_itd = model.decode(
                    prototype,
                    frequency,
                    positions[start:stop].unsqueeze(0),
                    dataset_name="sonicom",
                )
                magnitude_chunks.append(predicted_magnitude[0].cpu())
                itd_chunks.append(predicted_itd[0].cpu())
        predicted_magnitude = torch.cat(magnitude_chunks, dim=0)
        predicted_itd = torch.cat(itd_chunks, dim=0)
        with torch.no_grad():
            oversized_hrir = reconstruct_hrir_with_itd(
                predicted_magnitude.unsqueeze(0),
                predicted_itd.unsqueeze(0),
                sampling_rate_hz=subject.sampling_rate_hz,
            )
        predicted_hrir = oversized_hrir[0, :, :, :SONICOM_HRIR_LENGTH].cpu()
        if not all(
            torch.all(torch.isfinite(values))
            for values in (predicted_magnitude, predicted_itd, predicted_hrir)
        ):
            raise FloatingPointError(f"non-finite prediction for {subject.subject_id}")
        output_path = (
            output_root / "subjects" / subject.subject_id / "prediction.h5"
        )
        write_prediction(
            output_path,
            subject.subject_id,
            subject.split,
            predicted_magnitude,
            predicted_itd,
            subject.frequency_hz,
            predicted_hrir,
            args.checkpoint,
        )
        print(f"[{file_index}/{len(files)}] {subject.subject_id} -> {output_path}")

    summary = {
        "status": "completed",
        "run_name": args.run_name,
        "split": args.split,
        "subject_count": len(files),
        "test_subject_count_read": len(files) if args.split == "test" else 0,
        "frequency_count": 512,
        "target_direction_count": 793,
        "hrir_length": SONICOM_HRIR_LENGTH,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(args.checkpoint),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(summary)


if __name__ == "__main__":
    main()
