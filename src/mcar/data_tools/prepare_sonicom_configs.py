"""Prepare reproducible geometry and subject-split configs for SONICOM.

The measured SONICOM grid covers elevations from -45 to 90 degrees and
therefore cannot directly provide the complete Lebedev N=3 grid used by
the HUTUBS experiment.  This module creates:

* a 26-point, measurement-native, left/right-symmetric sparse grid;
* solid-angle weights for all 793 measured directions;
* a deterministic 262/44/44 subject split stratified by free-field EQ;
* a JSON report containing geometry, conditioning, and split diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import h5py
import numpy as np

from mcar.paths import project_root


DEFAULT_DATASET_ROOT = (
    project_root()
    / "data"
    / "HRTF"
    / "sonicom_measured_ffcmp_minphase_44k1"
)
DEFAULT_OUTPUT_DIR = project_root() / "configs" / "data"
DEFAULT_SPLIT_SEED = 20260731
SPLIT_TARGETS = {"train": 262, "val": 44, "test": 44}
SPLIT_ORDER = ("train", "val", "test")

# SUpDEq's Lebedev N=3 grid, expressed as [azimuth, elevation] in degrees.
LEBEDEV_N3_AZIMUTH_ELEVATION_DEG = np.asarray(
    [
        [0.0, 90.0],
        [0.0, 45.0],
        [0.0, 0.0],
        [0.0, -45.0],
        [0.0, -90.0],
        [45.0, -35.264389682754654],
        [45.0, 0.0],
        [45.0, 35.264389682754654],
        [90.0, 45.0],
        [90.0, 0.0],
        [90.0, -45.0],
        [135.0, -35.264389682754654],
        [135.0, 0.0],
        [135.0, 35.264389682754654],
        [180.0, 45.0],
        [180.0, 0.0],
        [180.0, -45.0],
        [225.0, -35.264389682754654],
        [225.0, 0.0],
        [225.0, 35.264389682754654],
        [270.0, 45.0],
        [270.0, 0.0],
        [270.0, -45.0],
        [315.0, -35.264389682754654],
        [315.0, 0.0],
        [315.0, 35.264389682754654],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class SparseGridSelection:
    """Selected source indices and geometry diagnostics."""

    source_indices: tuple[int, ...]
    north_seed_index: int
    median_seed_index: int
    minimum_pairwise_angle_deg: float
    covering_radius_deg: float
    mean_covering_distance_deg: float
    p95_covering_distance_deg: float
    sh_order3_condition_number: float
    sh_order3_rank: int


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate SONICOM-Q26-v1 geometry and 262/44/44 subject "
            "split configs from the downloaded clean cohort."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Downloaded SONICOM clean-cohort root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Tracked config output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Deterministic within-EQ subject shuffle seed.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def canonicalize_positions(source_position: np.ndarray) -> np.ndarray:
    positions = np.asarray(source_position, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2:
        raise ValueError(
            f"Expected SourcePosition [direction, >=2], got {positions.shape}"
        )
    result = positions[:, :2].copy()
    result[:, 0] = np.mod(result[:, 0], 360.0)
    if not np.all(np.isfinite(result)):
        raise ValueError("SourcePosition contains non-finite values")
    return result


def spherical_xyz(positions_deg: np.ndarray) -> np.ndarray:
    positions = canonicalize_positions(positions_deg)
    azimuth = np.deg2rad(positions[:, 0])
    elevation = np.deg2rad(positions[:, 1])
    cos_elevation = np.cos(elevation)
    return np.column_stack(
        (
            cos_elevation * np.cos(azimuth),
            cos_elevation * np.sin(azimuth),
            np.sin(elevation),
        )
    )


def real_sh_basis_order3(positions_deg: np.ndarray) -> np.ndarray:
    """Return a normalized real spherical-harmonic basis through order 3."""

    x, y, z = spherical_xyz(positions_deg).T
    return np.column_stack(
        (
            np.full_like(x, 0.28209479177387814),
            0.4886025119029199 * y,
            0.4886025119029199 * z,
            0.4886025119029199 * x,
            1.0925484305920792 * x * y,
            1.0925484305920792 * y * z,
            0.31539156525252005 * (3.0 * z * z - 1.0),
            1.0925484305920792 * x * z,
            0.5462742152960396 * (x * x - y * y),
            0.5900435899266435 * y * (3.0 * x * x - y * y),
            2.890611442640554 * x * y * z,
            0.4570457994644658 * y * (5.0 * z * z - 1.0),
            0.3731763325901154 * z * (5.0 * z * z - 3.0),
            0.4570457994644658 * x * (5.0 * z * z - 1.0),
            1.445305721320277 * z * (x * x - y * y),
            0.5900435899266435 * x * (x * x - 3.0 * y * y),
        )
    )


def minimum_pairwise_angle_deg(
    xyz: np.ndarray, indices: Sequence[int]
) -> float:
    selected = xyz[np.asarray(indices, dtype=np.int64)]
    dot_products = np.clip(selected @ selected.T, -1.0, 1.0)
    np.fill_diagonal(dot_products, -1.0)
    return float(np.rad2deg(np.arccos(np.max(dot_products))))


def regularized_log_determinant(
    positions_deg: np.ndarray,
    indices: Sequence[int],
) -> float:
    design = real_sh_basis_order3(
        positions_deg[np.asarray(indices, dtype=np.int64)]
    )
    gram = design.T @ design + 1e-9 * np.eye(design.shape[1])
    sign, log_determinant = np.linalg.slogdet(gram)
    if sign <= 0:
        raise ValueError("Regularized spherical-harmonic Gram matrix failed")
    return float(log_determinant)


def build_lateral_reflection_orbits(
    positions_deg: np.ndarray,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    positions = canonicalize_positions(positions_deg)
    lookup = {
        (round(float(azimuth), 8), round(float(elevation), 8)): index
        for index, (azimuth, elevation) in enumerate(positions)
    }
    visited: set[int] = set()
    singleton_orbits: list[tuple[int, ...]] = []
    pair_orbits: list[tuple[int, ...]] = []
    for index, (azimuth, elevation) in enumerate(positions):
        if index in visited:
            continue
        mirror_key = (
            round(float((-azimuth) % 360.0), 8),
            round(float(elevation), 8),
        )
        if mirror_key not in lookup:
            raise ValueError(
                "SONICOM grid is not left/right symmetric at "
                f"azimuth={azimuth}, elevation={elevation}"
            )
        mirror_index = lookup[mirror_key]
        orbit = tuple(sorted({index, mirror_index}))
        visited.update(orbit)
        if len(orbit) == 1:
            singleton_orbits.append(orbit)
        else:
            pair_orbits.append(orbit)
    return singleton_orbits, pair_orbits


def select_sonicom_q26(
    positions_deg: np.ndarray,
) -> SparseGridSelection:
    """Select a deterministic symmetric maximin Q=26 measurement grid.

    The design always includes the measured north pole, searches every
    other median-plane singleton as the second seed, and greedily adds
    twelve left/right reflection pairs.  Great-circle separation is the
    primary objective; a regularized order-3 SH determinant breaks ties.
    """

    positions = canonicalize_positions(positions_deg)
    if positions.shape[0] != 793:
        raise ValueError(
            f"Expected 793 SONICOM directions, got {positions.shape[0]}"
        )
    xyz = spherical_xyz(positions)
    singleton_orbits, pair_orbits = build_lateral_reflection_orbits(
        positions
    )
    north_matches = np.flatnonzero(
        np.isclose(positions[:, 1], 90.0, atol=1e-9)
    )
    if north_matches.size != 1:
        raise ValueError("Expected exactly one measured north-pole direction")
    north_index = int(north_matches[0])

    candidates: list[tuple[float, float, int, tuple[int, ...]]] = []
    for singleton in singleton_orbits:
        median_index = singleton[0]
        if median_index == north_index:
            continue
        selected = [north_index, median_index]
        available_pairs = list(pair_orbits)
        for _ in range(12):
            scored_pairs = []
            for orbit in available_pairs:
                trial = selected + list(orbit)
                scored_pairs.append(
                    (
                        round(
                            minimum_pairwise_angle_deg(xyz, trial), 12
                        ),
                        regularized_log_determinant(positions, trial),
                        -max(orbit),
                        orbit,
                    )
                )
            best_orbit = max(scored_pairs)[-1]
            selected.extend(best_orbit)
            available_pairs.remove(best_orbit)

        design = real_sh_basis_order3(positions[selected])
        condition_number = float(np.linalg.cond(design))
        minimum_angle = minimum_pairwise_angle_deg(xyz, selected)
        candidates.append(
            (
                round(minimum_angle, 12),
                -condition_number,
                -median_index,
                tuple(selected),
            )
        )

    _, negative_condition, _, selected_tuple = max(candidates)
    selected = list(selected_tuple)
    if len(selected) != 26 or len(set(selected)) != 26:
        raise AssertionError("SONICOM-Q26 selection is not 26 unique points")
    design = real_sh_basis_order3(positions[selected])
    rank = int(np.linalg.matrix_rank(design))
    condition_number = -negative_condition
    if rank != 16 or condition_number >= 5.0:
        raise RuntimeError(
            "SONICOM-Q26 has an unstable order-3 SH design: "
            f"rank={rank}, condition={condition_number:.6f}"
        )

    selected_xyz = xyz[selected]
    nearest_angles = np.rad2deg(
        np.arccos(
            np.clip(np.max(xyz @ selected_xyz.T, axis=1), -1.0, 1.0)
        )
    )
    median_index = selected[1]
    return SparseGridSelection(
        source_indices=tuple(sorted(selected)),
        north_seed_index=north_index,
        median_seed_index=median_index,
        minimum_pairwise_angle_deg=minimum_pairwise_angle_deg(
            xyz, selected
        ),
        covering_radius_deg=float(np.max(nearest_angles)),
        mean_covering_distance_deg=float(np.mean(nearest_angles)),
        p95_covering_distance_deg=float(
            np.percentile(nearest_angles, 95.0)
        ),
        sh_order3_condition_number=condition_number,
        sh_order3_rank=rank,
    )


def lebedev_n3_compatibility(
    positions_deg: np.ndarray,
) -> dict[str, object]:
    measured_xyz = spherical_xyz(positions_deg)
    target_xyz = spherical_xyz(LEBEDEV_N3_AZIMUTH_ELEVATION_DEG)
    dot_products = np.clip(target_xyz @ measured_xyz.T, -1.0, 1.0)
    nearest_indices = np.argmax(dot_products, axis=1)
    errors = np.rad2deg(
        np.arccos(dot_products[np.arange(dot_products.shape[0]), nearest_indices])
    )
    return {
        "target_count": int(len(nearest_indices)),
        "unique_nearest_measured_count": int(
            len(np.unique(nearest_indices))
        ),
        "exact_match_count": int(np.sum(errors <= 1e-6)),
        "mean_nearest_error_deg": float(np.mean(errors)),
        "maximum_nearest_error_deg": float(np.max(errors)),
        "maximum_error_target_azimuth_elevation_deg": (
            LEBEDEV_N3_AZIMUTH_ELEVATION_DEG[
                int(np.argmax(errors))
            ].tolist()
        ),
    }


def solid_angle_weights(
    positions_deg: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute normalized ring-cell weights over the measured spherical cap."""

    positions = canonicalize_positions(positions_deg)
    elevations = np.unique(positions[:, 1])
    if elevations[0] != -45.0 or elevations[-1] != 90.0:
        raise ValueError(
            "Expected SONICOM elevation coverage from -45 to 90 degrees"
        )
    boundaries = np.empty(elevations.size + 1, dtype=np.float64)
    boundaries[0] = elevations[0]
    boundaries[-1] = elevations[-1]
    boundaries[1:-1] = 0.5 * (elevations[:-1] + elevations[1:])
    weights = np.zeros(positions.shape[0], dtype=np.float64)
    for elevation_index, elevation in enumerate(elevations):
        mask = np.isclose(positions[:, 1], elevation, atol=1e-9)
        count = int(np.sum(mask))
        lower = math.radians(float(boundaries[elevation_index]))
        upper = math.radians(float(boundaries[elevation_index + 1]))
        ring_area = 2.0 * math.pi * (math.sin(upper) - math.sin(lower))
        weights[mask] = ring_area / count
    domain_solid_angle = float(np.sum(weights))
    weights /= domain_solid_angle
    if not np.all(weights > 0.0) or not np.isclose(
        np.sum(weights), 1.0, atol=1e-12
    ):
        raise AssertionError("Invalid SONICOM solid-angle weights")
    return weights, domain_solid_angle


def load_and_validate_source_grid(
    dataset_root: Path,
) -> tuple[np.ndarray, list[dict[str, str]], str]:
    manifest = read_csv(dataset_root / "manifests" / "clean_subjects.csv")
    if len(manifest) != 350:
        raise ValueError(
            f"Expected 350 clean subjects, found {len(manifest)}"
        )
    reference_grid: np.ndarray | None = None
    reference_file = ""
    for row in manifest:
        sofa_path = dataset_root / row["local_relative_path"]
        if not sofa_path.is_file():
            raise FileNotFoundError(sofa_path)
        with h5py.File(sofa_path, "r") as sofa:
            positions = canonicalize_positions(sofa["SourcePosition"][:])
        if reference_grid is None:
            reference_grid = positions
            reference_file = row["local_relative_path"]
        elif not np.array_equal(positions, reference_grid):
            raise ValueError(
                f"SourcePosition differs for subject {row['subject_id']}"
            )
    assert reference_grid is not None
    return reference_grid, manifest, reference_file


def allocate_stratum_counts(
    stratum_sizes: Mapping[str, int],
    split_targets: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    """Integer apportionment preserving row and column totals."""

    total = sum(stratum_sizes.values())
    if total != sum(split_targets.values()):
        raise ValueError("Stratum and split totals differ")
    strata = sorted(stratum_sizes)
    raw = {
        stratum: {
            split: stratum_sizes[stratum] * split_targets[split] / total
            for split in SPLIT_ORDER
        }
        for stratum in strata
    }
    allocation = {
        stratum: {
            split: math.floor(raw[stratum][split])
            for split in SPLIT_ORDER
        }
        for stratum in strata
    }
    while True:
        column_counts = {
            split: sum(
                allocation[stratum][split] for stratum in strata
            )
            for split in SPLIT_ORDER
        }
        if all(
            column_counts[split] == split_targets[split]
            for split in SPLIT_ORDER
        ):
            break
        candidates = []
        for stratum_index, stratum in enumerate(strata):
            if sum(allocation[stratum].values()) >= stratum_sizes[stratum]:
                continue
            for split_index, split in enumerate(SPLIT_ORDER):
                if column_counts[split] >= split_targets[split]:
                    continue
                if allocation[stratum][split] >= math.ceil(
                    raw[stratum][split]
                ):
                    continue
                candidates.append(
                    (
                        raw[stratum][split]
                        - allocation[stratum][split],
                        -split_index,
                        -stratum_index,
                        stratum,
                        split,
                    )
                )
        if not candidates:
            raise RuntimeError("Could not complete stratified apportionment")
        _, _, _, stratum, split = max(candidates)
        allocation[stratum][split] += 1
    for stratum in strata:
        if sum(allocation[stratum].values()) != stratum_sizes[stratum]:
            raise AssertionError(f"Allocation failed for {stratum}")
    return allocation


def deterministic_group_seed(seed: int, group: str) -> int:
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def build_subject_split(
    manifest: Sequence[dict[str, str]],
    metadata_rows: Sequence[dict[str, str]],
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    clean_subjects = {row["subject_id"] for row in manifest}
    metadata = {
        row["Subject"].strip().upper(): row for row in metadata_rows
    }
    if clean_subjects - metadata.keys():
        raise ValueError("Clean manifest contains subjects absent from metadata")

    by_eq: dict[str, list[str]] = defaultdict(list)
    for subject in sorted(clean_subjects):
        eq_file = metadata[subject]["Free Field EQ File"].strip()
        if not eq_file:
            raise ValueError(f"{subject} has no Free Field EQ File")
        by_eq[eq_file].append(subject)
    allocation = allocate_stratum_counts(
        {eq_file: len(subjects) for eq_file, subjects in by_eq.items()},
        SPLIT_TARGETS,
    )

    assignments: dict[str, str] = {}
    for eq_file in sorted(by_eq):
        subjects = sorted(by_eq[eq_file])
        random.Random(
            deterministic_group_seed(seed, eq_file)
        ).shuffle(subjects)
        start = 0
        for split in SPLIT_ORDER:
            end = start + allocation[eq_file][split]
            for subject in subjects[start:end]:
                assignments[subject] = split
            start = end
        if start != len(subjects):
            raise AssertionError(f"Split assignment failed for {eq_file}")

    split_indices = Counter()
    rows: list[dict[str, object]] = []
    for subject in sorted(
        clean_subjects, key=lambda value: int(value[1:])
    ):
        split = assignments[subject]
        split_indices[split] += 1
        rows.append(
            {
                "subject_id": subject,
                "split": split,
                "split_index": split_indices[split],
                "seed": seed,
                "free_field_eq_file": metadata[subject][
                    "Free Field EQ File"
                ].strip(),
            }
        )

    split_counts = Counter(row["split"] for row in rows)
    if dict(split_counts) != SPLIT_TARGETS:
        raise AssertionError(
            f"Unexpected split counts: {dict(split_counts)}"
        )
    if len(assignments) != len(clean_subjects):
        raise AssertionError("Subject split contains overlap or omissions")

    eq_by_split: dict[str, dict[str, int]] = {}
    sex_by_split: dict[str, dict[str, int]] = {}
    known_age_by_split: dict[str, int] = {}
    for split in SPLIT_ORDER:
        split_subjects = [
            row["subject_id"] for row in rows if row["split"] == split
        ]
        eq_by_split[split] = dict(
            sorted(
                Counter(
                    metadata[str(subject)][
                        "Free Field EQ File"
                    ].strip()
                    for subject in split_subjects
                ).items()
            )
        )
        sex_by_split[split] = dict(
            sorted(
                Counter(
                    metadata[str(subject)]["Sex at birth"].strip()
                    or "missing"
                    for subject in split_subjects
                ).items()
            )
        )
        known_age_by_split[split] = sum(
            bool(metadata[str(subject)]["Age"].strip())
            for subject in split_subjects
        )
    report = {
        "seed": seed,
        "split_targets": SPLIT_TARGETS,
        "stratification_key": "Free Field EQ File",
        "stratum_allocation": allocation,
        "free_field_eq_distribution_by_split": eq_by_split,
        "sex_at_birth_distribution_by_split": sex_by_split,
        "known_age_count_by_split": known_age_by_split,
    }
    return rows, report


def source_grid_sha256(positions_deg: np.ndarray) -> str:
    canonical = np.asarray(
        canonicalize_positions(positions_deg), dtype="<f8"
    )
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def main() -> None:
    arguments = parse_arguments()
    dataset_root = arguments.dataset_root.resolve()
    output_dir = arguments.output_dir.resolve()

    print("Validating the shared SONICOM SourcePosition grid...")
    positions, manifest, reference_file = load_and_validate_source_grid(
        dataset_root
    )
    q26 = select_sonicom_q26(positions)
    weights, domain_solid_angle = solid_angle_weights(positions)
    lebedev_diagnostic = lebedev_n3_compatibility(positions)

    metadata_rows = read_csv(
        dataset_root / "metadata_and_readme" / "metadata.csv"
    )
    split_rows, split_report = build_subject_split(
        manifest, metadata_rows, arguments.seed
    )

    q26_index_set = set(q26.source_indices)
    q26_rows = []
    for q26_index, source_index in enumerate(q26.source_indices):
        azimuth, elevation = positions[source_index]
        if source_index == q26.north_seed_index:
            role = "north_seed"
        elif source_index == q26.median_seed_index:
            role = "median_seed"
        else:
            role = "maximin_mirror_pair"
        mirror_azimuth = (-azimuth) % 360.0
        mirror_matches = np.flatnonzero(
            np.isclose(positions[:, 0], mirror_azimuth, atol=1e-9)
            & np.isclose(positions[:, 1], elevation, atol=1e-9)
        )
        if mirror_matches.size != 1:
            raise AssertionError("Could not resolve Q26 mirror direction")
        mirror_index = int(mirror_matches[0])
        if mirror_index not in q26_index_set:
            raise AssertionError("Q26 lost left/right reflection symmetry")
        q26_rows.append(
            {
                "q26_index_zero_based": q26_index,
                "source_index_zero_based": source_index,
                "azimuth_deg": f"{azimuth:.10f}",
                "elevation_deg": f"{elevation:.10f}",
                "colatitude_deg": f"{90.0 - elevation:.10f}",
                "mirror_source_index_zero_based": mirror_index,
                "selection_role": role,
            }
        )

    reference_rows = []
    for source_index, ((azimuth, elevation), weight) in enumerate(
        zip(positions, weights)
    ):
        reference_rows.append(
            {
                "source_index_zero_based": source_index,
                "azimuth_deg": f"{azimuth:.10f}",
                "elevation_deg": f"{elevation:.10f}",
                "colatitude_deg": f"{90.0 - elevation:.10f}",
                "solid_angle_weight": f"{weight:.16g}",
                "is_q26_sparse_input": int(source_index in q26_index_set),
                "is_interpolation_evaluation": int(
                    source_index not in q26_index_set
                ),
            }
        )

    write_csv_atomic(
        output_dir / "sonicom_sparse_grid_q26_v1.csv",
        tuple(q26_rows[0].keys()),
        q26_rows,
    )
    write_csv_atomic(
        output_dir / "sonicom_reference_grid_v1.csv",
        tuple(reference_rows[0].keys()),
        reference_rows,
    )
    write_csv_atomic(
        output_dir / "sonicom_subject_split_v1.csv",
        tuple(split_rows[0].keys()),
        split_rows,
    )

    unique_elevations, elevation_counts = np.unique(
        positions[:, 1], return_counts=True
    )
    report = {
        "dataset": {
            "name": "SONICOM measured clean cohort",
            "variant": "FreeFieldCompMinPhase_44kHz",
            "subject_count": len(manifest),
            "reference_sofa": reference_file,
            "source_grid_sha256": source_grid_sha256(positions),
        },
        "reference_grid": {
            "direction_count": int(positions.shape[0]),
            "elevation_minimum_deg": float(np.min(positions[:, 1])),
            "elevation_maximum_deg": float(np.max(positions[:, 1])),
            "elevation_ring_counts": {
                f"{elevation:g}": int(count)
                for elevation, count in zip(
                    unique_elevations, elevation_counts
                )
            },
            "measured_domain_solid_angle_sr": domain_solid_angle,
            "normalized_weight_sum": float(np.sum(weights)),
        },
        "lebedev_n3_direct_mapping": lebedev_diagnostic,
        "sonicom_q26_v1": {
            "direction_count": len(q26.source_indices),
            "source_indices_zero_based": list(q26.source_indices),
            "north_seed_source_index_zero_based": q26.north_seed_index,
            "median_seed_source_index_zero_based": q26.median_seed_index,
            "left_right_symmetric": True,
            "selection_objective": (
                "north pole + best median-plane seed + 12 mirrored "
                "pairs; maximize minimum great-circle separation, "
                "break ties by regularized order-3 SH log determinant"
            ),
            "minimum_pairwise_angle_deg": (
                q26.minimum_pairwise_angle_deg
            ),
            "covering_radius_deg": q26.covering_radius_deg,
            "mean_covering_distance_deg": (
                q26.mean_covering_distance_deg
            ),
            "p95_covering_distance_deg": (
                q26.p95_covering_distance_deg
            ),
            "real_sh_order": 3,
            "real_sh_design_rank": q26.sh_order3_rank,
            "real_sh_design_condition_number": (
                q26.sh_order3_condition_number
            ),
        },
        "subject_split": split_report,
    }
    write_json_atomic(
        output_dir / "sonicom_preparation_report_v1.json", report
    )

    print(
        "SONICOM-Q26-v1: "
        f"min angle={q26.minimum_pairwise_angle_deg:.4f} deg, "
        f"covering radius={q26.covering_radius_deg:.4f} deg, "
        f"SH condition={q26.sh_order3_condition_number:.4f}."
    )
    print(
        "Subject split: "
        + ", ".join(
            f"{split}={SPLIT_TARGETS[split]}" for split in SPLIT_ORDER
        )
    )
    print(f"Configs written to {output_dir}")


if __name__ == "__main__":
    main()
