"""Evaluate the non-test SONICOM Q26 Tikhonov pilot sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch

from mcar.losses import strict_hrir_ild_errors
from mcar.paths import project_root
from mcar.training.train_mlp_v2 import make_erb_weights


DEFAULT_RUN_NAMES = (
    "sonicom_pilot_q26_tikh0",
    "sonicom_pilot_q26_tikh1e8",
    "sonicom_pilot_q26_tikh1e6",
    "sonicom_pilot_q26_tikh1e4",
    "sonicom_pilot_q26_tikh1e2",
    "sonicom_pilot_q26_tikh3e2",
    "sonicom_pilot_q26_tikh1e1",
    "sonicom_pilot_q26_tikh3e1",
    "sonicom_pilot_q26_tikh1",
)
EXPECTED_SUBJECTS = {
    "P0001",
    "P0002",
    "P0238",
    "P0242",
    "P0243",
    "P0277",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=project_root() / "data" / "processed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            project_root()
            / "results"
            / "sonicom_data_preparation"
            / "tikhonov_pilot_v1"
        ),
    )
    parser.add_argument(
        "--run-name",
        action="append",
        dest="run_names",
        help="Repeat to override the five default pilot run names.",
    )
    return parser.parse_args()


def decode(value: object) -> str:
    scalar = np.asarray(value).item()
    return scalar.decode() if isinstance(scalar, bytes) else str(scalar)


def normalized_scope_weights(
    direction_features: np.ndarray,
    scope_mask: np.ndarray,
) -> np.ndarray:
    weights = np.asarray(direction_features[:, 5], dtype=np.float64)
    weights = weights * np.asarray(scope_mask, dtype=np.float64)
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("Direction scope has zero solid-angle weight")
    return weights / total


def weighted_direction_average(
    direction_values: np.ndarray,
    weights: np.ndarray,
) -> float:
    values = np.asarray(direction_values, dtype=np.float64)
    if values.shape != weights.shape:
        raise ValueError(
            f"Direction values {values.shape} != weights {weights.shape}"
        )
    return float(np.sum(values * weights))


def band_energy_db(
    magnitude_db: np.ndarray,
    erb_weights: np.ndarray,
) -> np.ndarray:
    power = np.power(10.0, np.asarray(magnitude_db) / 10.0)
    band_power = np.einsum(
        "edf,bf->edb", power, erb_weights, optimize=True
    )
    return 10.0 * np.log10(np.maximum(band_power, 1e-30))


def strict_ild_error_by_direction(
    handle: h5py.File,
    mca_db: np.ndarray,
) -> np.ndarray:
    group = handle["strict_ild"]
    metadata = {
        "mca_selected_phase_rad": torch.from_numpy(
            group["mca_selected_phase_rad"][:]
        ),
        "mca_outside_real": torch.from_numpy(
            group["mca_outside_real"][:]
        ),
        "mca_outside_imag": torch.from_numpy(
            group["mca_outside_imag"][:]
        ),
        "selected_bin_indices_zero_based": torch.from_numpy(
            np.squeeze(
                group["selected_bin_indices_zero_based"][:]
            ).astype(np.int64)
        ),
        "outside_bin_indices_zero_based": torch.from_numpy(
            np.squeeze(
                group["outside_bin_indices_zero_based"][:]
            ).astype(np.int64)
        ),
        "reference_ild_db": torch.from_numpy(
            np.squeeze(group["reference_ild_db"][:])
        ),
        "single_sided_frequency_count": int(
            np.asarray(
                group.attrs["single_sided_frequency_count"]
            ).item()
        ),
        "hrir_length": int(
            np.asarray(group.attrs["hrir_length"]).item()
        ),
    }
    with torch.no_grad():
        errors = strict_hrir_ild_errors(
            torch.from_numpy(mca_db), metadata
        )
    return errors.numpy().astype(np.float64)


def evaluate_scope(
    *,
    mca_db: np.ndarray,
    residual_db: np.ndarray,
    direction_features: np.ndarray,
    frequency_hz: np.ndarray,
    strict_ild_error: np.ndarray,
    scope_mask: np.ndarray,
) -> dict[str, float]:
    reference_db = mca_db + residual_db
    weights = normalized_scope_weights(direction_features, scope_mask)
    absolute_error = np.abs(residual_db)
    direction_mae = np.mean(absolute_error, axis=(0, 2))
    direction_mse = np.mean(np.square(residual_db), axis=(0, 2))

    erb_weights = make_erb_weights(frequency_hz)
    mca_erb = band_energy_db(mca_db, erb_weights)
    reference_erb = band_energy_db(reference_db, erb_weights)
    direction_erb_mae = np.mean(
        np.abs(mca_erb - reference_erb), axis=(0, 2)
    )

    high_frequency_mask = frequency_hz > 10_000.0
    lateral_y = direction_features[:, 3]
    contra_values = []
    for ear_index, contra_mask in enumerate(
        (lateral_y < 0.0, lateral_y > 0.0)
    ):
        ear_mask = scope_mask & contra_mask
        ear_weights = normalized_scope_weights(
            direction_features, ear_mask
        )
        direction_high_error = np.mean(
            absolute_error[ear_index][:, high_frequency_mask],
            axis=-1,
        )
        contra_values.append(
            weighted_direction_average(
                direction_high_error, ear_weights
            )
        )

    return {
        "residual_mae_db": weighted_direction_average(
            direction_mae, weights
        ),
        "residual_rmse_db": math.sqrt(
            weighted_direction_average(direction_mse, weights)
        ),
        "erb_proxy_mae_db": weighted_direction_average(
            direction_erb_mae, weights
        ),
        "contralateral_high_frequency_mae_db": float(
            np.mean(contra_values)
        ),
        "strict_hrir_ild_mae_db": weighted_direction_average(
            strict_ild_error, weights
        ),
    }


def evaluate_file(path: Path, run_name: str) -> list[dict[str, object]]:
    with h5py.File(path, "r") as handle:
        subject = decode(handle.attrs["subject_label"])
        split = decode(handle.attrs["split"])
        if split == "test":
            raise ValueError(f"Locked test subject found in pilot: {path}")
        mca_db = np.asarray(handle["mca_logmag_db"][:])
        residual_db = np.asarray(handle["target_residual_db"][:])
        direction_features = np.asarray(handle["direction_features"][:])
        frequency_hz = np.squeeze(handle["frequency_hz"][:])
        interpolation_mask = np.squeeze(
            handle["interpolation_evaluation_mask"][:]
        ).astype(bool)
        strict_ild_error = strict_ild_error_by_direction(
            handle, mca_db
        )
        tikhonov_epsilon = float(
            np.asarray(handle.attrs["tikhonov_epsilon"]).item()
        )

    rows = []
    for scope, mask in (
        ("all_793", np.ones(interpolation_mask.shape, dtype=bool)),
        ("interpolation_only_767", interpolation_mask),
    ):
        rows.append(
            {
                "run_name": run_name,
                "tikhonov_epsilon": tikhonov_epsilon,
                "subject_id": subject,
                "split": split,
                "scope": scope,
                **evaluate_scope(
                    mca_db=mca_db,
                    residual_db=residual_db,
                    direction_features=direction_features,
                    frequency_hz=frequency_hz,
                    strict_ild_error=strict_ild_error,
                    scope_mask=mask,
                ),
            }
        )
    return rows


def aggregate_rows(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    metric_names = (
        "residual_mae_db",
        "residual_rmse_db",
        "erb_proxy_mae_db",
        "contralateral_high_frequency_mae_db",
        "strict_hrir_ild_mae_db",
    )
    groups: dict[
        tuple[str, float, str, str], list[dict[str, object]]
    ] = {}
    for row in rows:
        key = (
            str(row["run_name"]),
            float(row["tikhonov_epsilon"]),
            str(row["split"]),
            str(row["scope"]),
        )
        groups.setdefault(key, []).append(row)
    result = []
    for (run_name, epsilon, split, scope), group in sorted(
        groups.items(), key=lambda item: (item[0][1], item[0][2], item[0][3])
    ):
        result.append(
            {
                "run_name": run_name,
                "tikhonov_epsilon": epsilon,
                "split": split,
                "scope": scope,
                "subject_count": len(group),
                **{
                    metric_name: float(
                        np.mean(
                            [
                                float(row[metric_name])
                                for row in group
                            ]
                        )
                    )
                    for metric_name in metric_names
                },
            }
        )
    return result


def write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def main() -> None:
    arguments = parse_arguments()
    run_names = tuple(arguments.run_names or DEFAULT_RUN_NAMES)
    rows: list[dict[str, object]] = []
    run_subjects: dict[str, set[str]] = {}
    for run_name in run_names:
        run_root = arguments.processed_root / run_name
        files = sorted((run_root / "subjects").glob("*/*.h5"))
        if len(files) != 6:
            raise ValueError(
                f"Expected six pilot HDF5 files in {run_root}, "
                f"found {len(files)}"
            )
        current_rows = []
        for path in files:
            current_rows.extend(evaluate_file(path, run_name))
        subjects = {
            str(row["subject_id"]) for row in current_rows
        }
        if subjects != EXPECTED_SUBJECTS:
            raise ValueError(
                f"{run_name} subject set differs: {sorted(subjects)}"
            )
        run_subjects[run_name] = subjects
        rows.extend(current_rows)

    split_counts = Counter(
        (str(row["run_name"]), str(row["split"]))
        for row in rows
        if row["scope"] == "all_793"
    )
    for run_name in run_names:
        if split_counts[(run_name, "train")] != 3:
            raise ValueError(f"{run_name} does not contain three train subjects")
        if split_counts[(run_name, "val")] != 3:
            raise ValueError(f"{run_name} does not contain three val subjects")

    aggregate = aggregate_rows(rows)
    validation_candidates = [
        row
        for row in aggregate
        if row["split"] == "val"
        and row["scope"] == "interpolation_only_767"
    ]
    selected = min(
        validation_candidates,
        key=lambda row: (
            float(row["erb_proxy_mae_db"]),
            float(row["residual_mae_db"]),
            float(row["strict_hrir_ild_mae_db"]),
            float(row["tikhonov_epsilon"]),
        ),
    )
    output_dir = arguments.output_dir.resolve()
    write_csv_atomic(output_dir / "per_subject_metrics.csv", rows)
    write_csv_atomic(output_dir / "aggregate_metrics.csv", aggregate)
    summary = {
        "pilot_subjects": sorted(EXPECTED_SUBJECTS),
        "split_counts_per_run": {"train": 3, "val": 3, "test": 0},
        "scopes": {
            "primary": "interpolation_only_767",
            "secondary": "all_793",
        },
        "selection_rule": (
            "lowest validation interpolation-only ERB proxy MAE; "
            "tie-break by residual MAE, strict HRIR ILD MAE, then epsilon"
        ),
        "selected_run_name": selected["run_name"],
        "selected_tikhonov_epsilon": selected["tikhonov_epsilon"],
        "selected_validation_metrics": selected,
        "aggregate_metrics": aggregate,
    }
    write_json_atomic(output_dir / "summary.json", summary)
    print(
        "Selected "
        f"{selected['run_name']} "
        f"(epsilon={selected['tikhonov_epsilon']:.8g}): "
        f"val interpolation ERB={selected['erb_proxy_mae_db']:.6f} dB, "
        f"residual MAE={selected['residual_mae_db']:.6f} dB, "
        f"strict ILD={selected['strict_hrir_ild_mae_db']:.6f} dB."
    )
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
