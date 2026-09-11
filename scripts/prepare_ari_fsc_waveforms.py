#!/usr/bin/env python3
"""Prepare the frozen ARI train/validation waveform inputs for FSC.

This adapter deliberately refuses the test split.  It performs only the
pre-registered common resampling and records sufficient provenance for the
separate MCA/FSC feature-generation step.  It never changes a SONICOM file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import resample_poly
from scipy.spatial import SphericalVoronoi


REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPO / "configs/experiments/ari_fsc_adapted_q26_external_replication_v2.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def direction_features(azimuth_deg: np.ndarray, elevation_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return [azimuth,elevation,x,y,z,eval_voronoi_weight] and eval weights.

    The Q26 directions are assigned zero evaluation weight.  Areas are
    calculated on the non-observed measured directions and normalized to one.
    """
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    xyz = np.column_stack((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)))
    return xyz, np.column_stack((azimuth_deg, elevation_deg, xyz))


def eval_voronoi_weights(xyz: np.ndarray, q26_indices: np.ndarray) -> np.ndarray:
    mask = np.ones(xyz.shape[0], dtype=bool)
    mask[q26_indices] = False
    sphere = SphericalVoronoi(xyz[mask], radius=1.0, center=np.zeros(3))
    areas = sphere.calculate_areas()
    if not np.isfinite(areas).all() or (areas <= 0).any():
        raise ValueError("non-finite/non-positive ARI evaluation Voronoi areas")
    weights = np.zeros(xyz.shape[0], dtype=np.float64)
    weights[mask] = areas / areas.sum()
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output-root", type=Path, default=REPO / "data/processed/ari_fsc_adapted_q26_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split == "test":
        raise SystemExit("TEST ACCESS REFUSED: this pre-training adapter accepts only train or val.")
    protocol_path = args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_pre_training":
        raise ValueError("protocol must be frozen_pre_training")
    if protocol["test_subjects_read"] != 0:
        raise ValueError("pre-training adapter requires a zero test-access counter")
    split_path = REPO / protocol["cohort"]["split_csv"]
    q26_path = REPO / protocol["q26"]["csv"]
    inventory_path = REPO / protocol["source_inventory"]
    if sha256(split_path).upper() != protocol["cohort"]["split_csv_sha256"]:
        raise ValueError("split manifest SHA256 does not match frozen protocol")
    if sha256(q26_path).upper() != protocol["q26"]["csv_sha256"]:
        raise ValueError("Q26 CSV SHA256 does not match frozen protocol")
    if sha256(inventory_path).upper() != protocol["source_inventory_sha256"]:
        raise ValueError("source inventory SHA256 does not match frozen protocol")

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    raw_root = Path(inventory["destination"])
    rows = [row for row in read_csv(split_path) if row["split"] == args.split]
    if len(rows) != protocol["cohort"]["split_counts"][args.split]:
        raise ValueError("split subject count differs from frozen protocol")
    q26_rows = read_csv(q26_path)
    q26_indices = np.asarray([int(row["ari_source_index_zero_based"]) for row in q26_rows], dtype=np.int64)
    if len(q26_indices) != 26 or len(np.unique(q26_indices)) != 26:
        raise ValueError("frozen Q26 must comprise 26 unique source indices")

    output_dir = args.output_root.resolve() / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    common_dirs: np.ndarray | None = None
    common_features: np.ndarray | None = None
    common_weights: np.ndarray | None = None
    prepared: list[dict[str, object]] = []
    for row in rows:
        source_path = raw_root / row["source_filename"]
        if not source_path.is_file() or sha256(source_path).lower() != row["sha256"].lower():
            raise ValueError(f"source file missing or checksum mismatch: {source_path}")
        destination = output_dir / f"{row['subject_id']}.h5"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite existing prepared input: {destination}")
        with h5py.File(source_path, "r") as source:
            ir = np.asarray(source["Data.IR"], dtype=np.float64)
            directions = np.asarray(source["SourcePosition"], dtype=np.float64)
            sampling_rate = float(np.asarray(source["Data.SamplingRate"]).squeeze())
        if ir.shape != (1550, 2, 256) or directions.shape != (1550, 3) or sampling_rate != 48000.0:
            raise ValueError(f"unexpected ARI metadata/shape for {row['subject_id']}")
        if not np.isfinite(ir).all() or not np.isfinite(directions).all():
            raise ValueError(f"non-finite source content for {row['subject_id']}")
        if common_dirs is None:
            common_dirs = directions
            xyz, base_features = direction_features(directions[:, 0], directions[:, 1])
            common_weights = eval_voronoi_weights(xyz, q26_indices)
            common_features = np.column_stack((base_features, common_weights))
        elif not np.array_equal(directions, common_dirs):
            raise ValueError(f"direction grid drift for {row['subject_id']}")
        resampled = resample_poly(ir, up=147, down=160, axis=-1)
        if resampled.shape != (1550, 2, 236):
            raise ValueError(f"unexpected resample length {resampled.shape} for {row['subject_id']}")
        padded = np.pad(resampled, ((0, 0), (0, 0), (0, 20)), mode="constant").astype(np.float32)
        if padded.shape != (1550, 2, 256) or not np.isfinite(padded).all():
            raise ValueError(f"invalid prepared waveform for {row['subject_id']}")
        temp = destination.with_suffix(".h5.tmp")
        with h5py.File(temp, "w") as target:
            target.create_dataset("hrir", data=padded, compression="gzip", compression_opts=1)
            target.create_dataset("source_position_spherical_deg_m", data=common_dirs)
            target.create_dataset("direction_features", data=common_features.astype(np.float64))
            target.create_dataset("q26_source_indices_zero_based", data=q26_indices)
            target.attrs["subject_id"] = row["subject_id"]
            target.attrs["split"] = args.split
            target.attrs["source_sha256"] = row["sha256"]
            target.attrs["source_sampling_rate_hz"] = 48000.0
            target.attrs["sampling_rate_hz"] = 44100.0
            target.attrs["resampling"] = "scipy.signal.resample_poly(up=147,down=160)"
            target.attrs["right_zero_pad_samples"] = 20
            target.attrs["protocol_sha256"] = sha256(protocol_path)
            target.attrs["test_subjects_read"] = 0
        os.replace(temp, destination)
        prepared.append({"subject_id": row["subject_id"], "path": str(destination), "sha256": sha256(destination)})

    summary = {
        "protocol": str(protocol_path), "protocol_sha256": sha256(protocol_path), "split": args.split,
        "subject_count": len(prepared), "test_subjects_read": 0, "all_finite": True,
        "direction_shape": [1550, 6], "waveform_shape_per_subject": [1550, 2, 256],
        "q26_count": 26, "evaluation_direction_count": 1524,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "subjects": prepared,
    }
    (output_dir / "preparation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("split", "subject_count", "test_subjects_read", "all_finite")}, indent=2))


if __name__ == "__main__":
    main()
