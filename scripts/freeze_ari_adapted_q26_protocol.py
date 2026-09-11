"""Freeze the approved ARI-specific, mirror-symmetric Q26 protocol.

This creates a new protocol only; it does not alter SONICOM Q26 or read a
single Data.IR value.  Q26 is selected by deterministic maximin sampling from
the common measured ARI grid, constrained to two median-plane anchors plus 12
exact left/right mirror pairs.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260911


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def xyz(position: np.ndarray) -> np.ndarray:
    az, el = np.deg2rad(position[:, 0]), np.deg2rad(position[:, 1])
    return np.column_stack((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)))


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.arccos(np.clip(a @ b.T, -1.0, 1.0)))


def write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def coordinate_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest().upper()


def main() -> None:
    inventory_path = ROOT / "configs/data/ari_hrtf_b_nh_source_inventory_v1.json"
    split_path = ROOT / "configs/data/ari_hrtf_b_nh_adapted_q26_subject_split_v2.csv"
    q26_path = ROOT / "configs/data/ari_hrtf_b_nh_adapted_q26_v2.csv"
    protocol_path = ROOT / "configs/experiments/ari_fsc_adapted_q26_external_replication_v2.json"
    for path in (split_path, q26_path, protocol_path):
        if path.exists(): raise FileExistsError(path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    eligible = sorted((r for r in inventory["files"] if r["eligible_metadata"]), key=lambda r: int(r["subject_id"][2:]))
    if len(eligible) != 174: raise ValueError("Unexpected eligible cohort count")
    root = Path(inventory["destination"])
    grid: np.ndarray | None = None
    grid_hashes: set[str] = set()
    sample_rate: float | None = None
    for row in eligible:
        with h5py.File(root / row["filename"], "r") as handle:
            pos = np.asarray(handle["SourcePosition"][:], dtype=np.float64)
            rate = float(np.asarray(handle["Data.SamplingRate"][:]).reshape(-1)[0])
        if pos.shape != (1550, 3) or rate != 48000.0: raise ValueError(f"non-common geometry/rate: {row['subject_id']}")
        grid_hashes.add(coordinate_digest(pos)); grid = pos if grid is None else grid
        sample_rate = rate
    if len(grid_hashes) != 1 or grid is None or sample_rate is None: raise ValueError("Grid differs across qualified subjects")
    grid_xyz = xyz(grid)
    # The anchors make the selected set explicitly retain ARI's low and high
    # elevation extent. They are self-mirrored, so 2 + 12*2 == 26.
    anchors = []
    for elevation in (-30.0, 80.0):
        found = np.flatnonzero((np.abs(grid[:, 0]) < 1e-9) & (np.abs(grid[:, 1] - elevation) < 1e-9))
        if found.size != 1: raise ValueError(f"missing required median anchor {elevation}")
        anchors.append(int(found[0]))
    pairs: list[tuple[int, int]] = []
    for index, (azimuth, elevation, _) in enumerate(grid):
        mirror_azimuth = (-azimuth) % 360.0
        mirror = np.flatnonzero((np.abs(((grid[:, 0] - mirror_azimuth + 180) % 360) - 180) < 1e-9) & (np.abs(grid[:, 1] - elevation) < 1e-9))
        if mirror.size == 1 and int(mirror[0]) != index and index < int(mirror[0]): pairs.append((index, int(mirror[0])))
    chosen = list(anchors)
    selected_pairs: list[tuple[int, int]] = []
    while len(selected_pairs) < 12:
        best: tuple[float, tuple[int, int]] | None = None
        for pair in pairs:
            if pair in selected_pairs or pair[0] in chosen or pair[1] in chosen: continue
            # A mirror pair is admitted only if both its distance to the
            # existing set *and its own internal separation* are maximin.
            score = min(float(distance(grid_xyz[list(pair)], grid_xyz[chosen]).min()), float(distance(grid_xyz[[pair[0]]], grid_xyz[[pair[1]]])[0, 0]))
            candidate = (score, pair)
            if best is None or candidate[0] > best[0] + 1e-12 or (abs(candidate[0] - best[0]) <= 1e-12 and candidate[1] < best[1]): best = candidate
        if best is None: raise RuntimeError("insufficient mirror pairs")
        selected_pairs.append(best[1]); chosen.extend(best[1])
    if len(chosen) != 26 or len(set(chosen)) != 26: raise AssertionError("Q26 uniqueness failure")
    qdist = distance(grid_xyz[chosen], grid_xyz[chosen]); np.fill_diagonal(qdist, np.inf)
    for pair in selected_pairs:
        if abs(((grid[pair[0], 0] + grid[pair[1], 0] + 180) % 360) - 180) > 1e-9 or grid[pair[0], 1] != grid[pair[1], 1]: raise AssertionError("mirror failure")
    qrows = []
    mirror_index = {anchors[0]: 0, anchors[1]: 1}
    for offset, pair in enumerate(selected_pairs): mirror_index[pair[0]] = 3 + 2 * offset; mirror_index[pair[1]] = 2 + 2 * offset
    for qindex, source_index in enumerate(chosen):
        qrows.append({"q26_index_zero_based": qindex, "ari_source_index_zero_based": source_index, "azimuth_deg": format(grid[source_index, 0], ".10f"), "elevation_deg": format(grid[source_index, 1], ".10f"), "radius_m": format(grid[source_index, 2], ".10f"), "mirror_q26_index_zero_based": mirror_index[source_index], "mapping_angular_error_deg": "0", "selection_role": "median_anchor" if qindex < 2 else "mirror_pair_maximin"})
    write(q26_path, qrows)
    ids = [r["subject_id"] for r in eligible]; order = [ids[i] for i in np.random.default_rng(SEED).permutation(len(ids))]
    # Nearest integer quotas: 130/22/22 of 174 (74.71/12.64/12.64%).
    allocation = {sid: "train" for sid in order[:130]}; allocation.update({sid: "val" for sid in order[130:152]}); allocation.update({sid: "test" for sid in order[152:]})
    srows = [{"subject_id": r["subject_id"], "source_filename": r["filename"], "sha256": r["sha256"], "metadata_eligible": str(bool(r["eligible_metadata"])).lower(), "exclusion_reason": r["exclusion_reason"], "split": allocation.get(r["subject_id"], "excluded")} for r in sorted(inventory["files"], key=lambda x: int(x["subject_id"][2:]))]
    write(split_path, srows)
    protocol = {"schema_version": "1.0", "status": "frozen_pre_training", "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"), "experiment_id": "ARI-FSC-ADAPTED-Q26-EXTERNAL-REPLICATION-V2", "authorization": "Author approved an ARI-coverage-adapted, mirror-symmetric Q26 after the original SONICOM Q26 mapping was rejected.", "source_inventory": str(inventory_path.relative_to(ROOT)).replace("\\", "/"), "source_inventory_sha256": sha(inventory_path), "cohort": {"eligible_subject_count": 174, "excluded_subjects": ["nh10", "nh22", "nh826"], "split_seed": SEED, "split_counts": {"train": 130, "val": 22, "test": 22}, "split_csv": str(split_path.relative_to(ROOT)).replace("\\", "/"), "split_csv_sha256": sha(split_path)}, "q26": {"selection": "two median-plane anchors (0,-30),(0,80) plus deterministic greedy maximin selection of 12 exact mirror pairs from the common ARI measured grid; pair-internal separation included; lexicographic source-index tie-break", "csv": str(q26_path.relative_to(ROOT)).replace("\\", "/"), "csv_sha256": sha(q26_path), "unique_direction_count": 26, "mirror_symmetric": True, "mapping_max_error_deg": 0.0, "mapping_mean_error_deg": 0.0, "minimum_pairwise_separation_deg": float(qdist.min()), "common_grid_sha256": next(iter(grid_hashes))}, "signal": {"source_sampling_rate_hz": sample_rate, "target_sampling_rate_hz": 44100.0, "resampling": "scipy.signal.resample_poly(up=147,down=160); right-zero-pad 236 samples to 256; no time alignment", "amplitude_compensation": "retain official ARI hrtf-b equalization", "itd": "retain delay through common polyphase resampling"}, "evaluation": {"directions": "all non-Q26 measured ARI directions", "weights": "spherical Voronoi areas normalized over the evaluation subset", "horizontal_ild": "ARI measured elevation==0 directions", "metrics": ["FullSphereERB", "Contralateral25ERB", "ERBBandILDMean", "ITDWeightedMAE", "FullSphereLSD"], "statistics": "subject-level mean/sample SD; FSC-MCA paired bootstrap 10000, seed 20260911"}, "training": {"seed": SEED, "film_siren_budget": "E130", "spectral_cnn_budget": "E190", "checkpoint_rule": "E190 last.pt only", "validation": "integrity-only; no selection"}, "test_policy": "No test Data.IR reads until adapter/config are frozen and E190 is complete; one authorized test pass only.", "test_subjects_read": 0}
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"split_counts": protocol["cohort"]["split_counts"], "q26_minimum_pairwise_separation_deg": float(qdist.min()), "grid_sha256": next(iter(grid_hashes))}, indent=2))


if __name__ == "__main__": main()
