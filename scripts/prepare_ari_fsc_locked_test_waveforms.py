#!/usr/bin/env python3
"""One-shot raw-SOFA access for the frozen ARI FSC test split."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import resample_poly

from prepare_ari_fsc_waveforms import direction_features, eval_voronoi_weights


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_state(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_pre_test" or not config.get("test_access_authorized"):
        raise PermissionError("frozen authorized ARI test config required")
    for resource in config["resources"]:
        path = ROOT / resource["path"]
        if sha256(path) != resource["sha256"]:
            raise ValueError(f"frozen pipeline resource hash mismatch: {path}")
    access = ROOT / config["test_access_state"]
    output = ROOT / config["prepared_waveform_root"]
    if access.exists() or output.exists():
        raise FileExistsError("one-shot test access already claimed or output exists")

    split_path = ROOT / config["split_csv"]
    q26_path = ROOT / config["q26_csv"]
    inventory_path = ROOT / config["source_inventory"]
    for path, expected in (
        (split_path, config["split_csv_sha256"]),
        (q26_path, config["q26_csv_sha256"]),
        (inventory_path, config["source_inventory_sha256"]),
    ):
        if sha256(path) != expected:
            raise ValueError(f"frozen resource hash mismatch: {path}")
    test_rows = [row for row in rows(split_path) if row["split"] == "test"]
    if len(test_rows) != 22 or len({row["subject_id"] for row in test_rows}) != 22:
        raise ValueError("expected 22 unique frozen test subjects")
    q26 = np.asarray(
        [int(row["ari_source_index_zero_based"]) for row in rows(q26_path)],
        dtype=np.int64,
    )
    if q26.shape != (26,) or np.unique(q26).size != 26:
        raise ValueError("invalid frozen ARI Q26")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    raw_root = Path(inventory["destination"])

    access.parent.mkdir(parents=True, exist_ok=True)
    claimed = {
        "status": "claimed",
        "claimed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "planned_test_subjects": 22,
        "test_subjects_read": 0,
    }
    with access.open("x", encoding="utf-8") as stream:
        json.dump(claimed, stream, indent=2)
        stream.write("\n")
    output.mkdir(parents=True)
    completed: list[dict[str, object]] = []
    common_position = None
    try:
        for row in test_rows:
            source_path = raw_root / row["source_filename"]
            if not source_path.is_file() or sha256(source_path) != row["sha256"].upper():
                raise ValueError(f"missing or changed raw SOFA: {source_path}")
            with h5py.File(source_path, "r") as source:
                hrir = np.asarray(source["Data.IR"], dtype=np.float64)
                position = np.asarray(source["SourcePosition"], dtype=np.float64)
                sampling_rate = float(np.asarray(source["Data.SamplingRate"]).squeeze())
            if hrir.shape != (1550, 2, 256) or position.shape != (1550, 3):
                raise ValueError(f"unexpected ARI test shape: {row['subject_id']}")
            if sampling_rate != 48000.0 or not np.isfinite(hrir).all():
                raise ValueError(f"invalid ARI test waveform: {row['subject_id']}")
            if common_position is None:
                common_position = position
                xyz, base = direction_features(position[:, 0], position[:, 1])
                weights = eval_voronoi_weights(xyz, q26)
                features = np.column_stack((base, weights))
            elif not np.array_equal(position, common_position):
                raise ValueError(f"test direction grid drift: {row['subject_id']}")
            resampled = resample_poly(hrir, up=147, down=160, axis=-1)
            prepared = np.pad(resampled, ((0, 0), (0, 0), (0, 20))).astype(np.float32)
            if prepared.shape != (1550, 2, 256) or not np.isfinite(prepared).all():
                raise ValueError(f"invalid prepared test HRIR: {row['subject_id']}")
            destination = output / f"{row['subject_id']}.h5"
            temporary = destination.with_suffix(".h5.partial")
            with h5py.File(temporary, "w") as target:
                target.create_dataset("hrir", data=prepared, compression="gzip", compression_opts=1)
                target.create_dataset("source_position_spherical_deg_m", data=common_position)
                target.create_dataset("direction_features", data=features)
                target.create_dataset("q26_source_indices_zero_based", data=q26)
                target.attrs["subject_id"] = row["subject_id"]
                target.attrs["split"] = "test"
                target.attrs["source_sha256"] = row["sha256"]
                target.attrs["source_sampling_rate_hz"] = 48000.0
                target.attrs["sampling_rate_hz"] = 44100.0
                target.attrs["resampling"] = "scipy.signal.resample_poly(up=147,down=160)"
                target.attrs["right_zero_pad_samples"] = 20
                target.attrs["test_subjects_read"] = 1
                target.attrs["complete"] = 1
            os.replace(temporary, destination)
            completed.append({"subject_id": row["subject_id"], "sha256": sha256(destination)})
            claimed["test_subjects_read"] = len(completed)
            claimed["last_subject"] = row["subject_id"]
            write_state(access, claimed)
            print(f"ARI locked test raw access [{len(completed)}/22] {row['subject_id']}", flush=True)
    except BaseException as error:
        claimed["status"] = "failed"
        claimed["failed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        claimed["error"] = repr(error)
        write_state(access, claimed)
        raise
    claimed.update({
        "status": "completed",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "test_subjects_read": 22,
        "all_finite": True,
        "subjects": completed,
    })
    write_state(access, claimed)
    (output / "preparation_summary.json").write_text(
        json.dumps(claimed, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
