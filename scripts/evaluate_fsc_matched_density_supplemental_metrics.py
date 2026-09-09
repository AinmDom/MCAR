"""Validation-only supplemental metrics for FSC matched-density diagonals.

This evaluator is deliberately separate from the frozen four-primary-metric
result.  It reuses the same 44 validation listeners, Q50-excluded common
743-direction mask, and diagonal FSC predictions, and adds the already
registered frequency-band ILD, ITD weighted MAE, and LAP 2024 LSD endpoints.
No test files are opened.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcar.evaluation.secondary_metrics import normalized_direction_weights, spectral_band_ild_profile
from mcar.paths import project_root
from scripts.evaluate_lap2024_metrics import lap2024_lsd
from scripts.evaluate_ten_method_direction_sensitivity_secondary import itd
from mcar.evaluation.deferred_secondary_metrics import enable_external_torchaudio


COUNTS = (14, 26, 50)
UNITS = {"ERBBandILDMean": "dB", "ITDWeightedMAE_us": "us", "LAP2024LSD_dB": "dB"}
LABELS = {
    14: "FSC-Q14@Q14",
    26: "FSC-Q26@Q26",
    50: "FSC-Q50@Q50",
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"Inconsistent columns: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def complex_dataset(dataset: h5py.Dataset) -> np.ndarray:
    values = np.asarray(dataset[:])
    return np.asarray(values["real"], dtype=np.float64) + 1j * np.asarray(
        values["imag"], dtype=np.float64
    )


def load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return grid, selected mask, reference DB/HRIR, MCA DB/full complex, frequency."""
    with h5py.File(path, "r") as handle:
        cache = handle["cache"]
        grid = np.asarray(cache["referenceGrid"][:], dtype=np.float64).T
        frequency_mask = np.asarray(cache["frequencyMask"][:], dtype=bool).reshape(-1)
        frequency = np.asarray(cache["frequencyHz"][:], dtype=np.float64).reshape(-1)
        reference = np.stack(
            [complex_dataset(cache["referenceLeft"]), complex_dataset(cache["referenceRight"])]
        )
        mca = np.stack(
            [complex_dataset(cache["mcaLeft"]), complex_dataset(cache["mcaRight"])]
        )
        reference_db = 20.0 * np.log10(
            np.maximum(np.abs(reference[:, frequency_mask, :].transpose(0, 2, 1)), 1e-10)
        )
        mca_db = 20.0 * np.log10(
            np.maximum(np.abs(mca[:, frequency_mask, :].transpose(0, 2, 1)), 1e-10)
        )
        reference_hrir = np.asarray(cache["referenceHrir"][:], dtype=np.float32).transpose(1, 0, 2)
        direction_count = int(np.asarray(cache["directionCount"][:]).reshape(-1)[0])
        if direction_count not in COUNTS or grid.shape != (793, 3):
            raise ValueError(f"Invalid cache provenance: {path}")
        if reference_hrir.shape != (793, 2, 256):
            raise ValueError(f"Invalid reference HRIR shape: {path}")
    return grid, frequency_mask, frequency, reference_db, reference_hrir, mca_db, mca.transpose(0, 2, 1)


def reconstruct_hrir(predicted_db: np.ndarray, mca_full: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply predicted selected-bin magnitudes with inherited MCA phase."""
    spectrum = np.asarray(mca_full, dtype=np.complex128).copy()
    selected = np.asarray(predicted_db, dtype=np.float64)
    if selected.shape != (2, 793, int(mask.sum())):
        raise ValueError(f"Unexpected selected spectrum shape: {selected.shape}")
    phase = np.angle(spectrum[:, :, mask])
    spectrum[:, :, mask] = 10.0 ** (selected / 20.0) * np.exp(1j * phase)
    hrir = np.fft.irfft(spectrum, n=1024, axis=2)[..., :256]
    return hrir.transpose(1, 0, 2).astype(np.float32)


def xyz_from_grid(grid: np.ndarray) -> np.ndarray:
    az = np.deg2rad(grid[:, 0])
    colat = np.deg2rad(90.0 - grid[:, 1])
    return np.column_stack((np.sin(colat) * np.cos(az), np.sin(colat) * np.sin(az), np.cos(colat)))


def read_subjects(root: Path) -> list[tuple[int, str]]:
    with (root / "configs/data/sonicom_subject_split_v1.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "val"]
    if len(rows) != 44:
        raise ValueError(f"Expected 44 validation listeners, found {len(rows)}")
    return [(int(row["subject_id"][1:]), row["subject_id"]) for row in rows]


def prediction_path(root: Path, count: int, subject: str) -> Path:
    if count == 26:
        return root / "artifacts/reconstruction/sonicom_film_siren_spectral_cnn_final_e190_ensemble_validation" / "subjects" / subject / "prediction.h5"
    return root / f"artifacts/reconstruction/sonicom_fsc_q{count}_e190_ensemble_validation/subjects/{subject}/prediction.h5"


def read_residual(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        split = np.asarray(handle.attrs.get("split", "")).item()
        if isinstance(split, bytes):
            split = split.decode()
        if str(split) != "val":
            raise PermissionError(f"Non-validation prediction: {path}")
        residual = np.asarray(handle["predicted_residual_db"][:], dtype=np.float64)
    if residual.shape != (2, 793, 463) or not np.all(np.isfinite(residual)):
        raise FloatingPointError(f"Invalid prediction: {path}")
    return residual


def bootstrap(values: Iterable[float], replicates: int, seed: int) -> tuple[float, float, float, float]:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size != 44 or not np.all(np.isfinite(data)):
        raise ValueError("Bootstrap requires 44 finite subject values")
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(replicates, data.size))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975], method="linear")
    return float(data.mean()), float(np.std(data, ddof=1)), float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    root = project_root()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["status"] != "frozen" or config["split"] != "val" or config["test_access_allowed"]:
        raise PermissionError("Frozen validation-only config required")
    enable_external_torchaudio(Path("D:/miniconda3/envs/asd/Lib/site-packages"))
    output = root / config["output_root"]
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(output)
    partial.mkdir(parents=True)
    subjects = read_subjects(root)

    grid_rows = list(csv.DictReader((root / config["sparse_grid_file"]).open(encoding="utf-8-sig")))
    q_indices = {
        count: np.asarray(
            [int(row["source_index_zero_based"]) for row in grid_rows if int(row["direction_count"]) == count],
            dtype=np.int64,
        )
        for count in COUNTS
    }
    common_mask = np.ones(793, dtype=bool)
    common_mask[q_indices[50]] = False
    if int(common_mask.sum()) != 743:
        raise ValueError("Common Q50-excluded mask must contain 743 directions")

    rows: list[dict[str, object]] = []
    values: dict[tuple[str, int], list[float]] = {(endpoint, count): [] for endpoint in UNITS for count in COUNTS}
    for subject_id, subject in subjects:
        for count in COUNTS:
            cache_path = root / f"artifacts/sparsity/sonicom_bounded_e25_input_direction_sensitivity_v1/subjects/{subject}/q{count}/cache.mat"
            grid, frequency_mask, frequency, reference_db, reference_hrir, mca_db, mca_full = load_cache(cache_path)
            if not np.array_equal(np.flatnonzero(frequency_mask), np.arange(513)[frequency_mask]):
                raise AssertionError("Unexpected frequency mask indexing")
            residual = read_residual(prediction_path(root, count, subject))
            predicted_db = mca_db + residual
            predicted_hrir = reconstruct_hrir(predicted_db, mca_full, frequency_mask)
            weights = grid[:, 2]
            # Grid column 1 is colatitude; elevation=0 is therefore 90 degrees.
            horizontal = common_mask & (np.abs(90.0 - grid[:, 1]) <= 1e-9)
            if not np.any(horizontal):
                raise ValueError(f"No common horizontal directions for {subject} Q{count}")
            _, _, band_mean = spectral_band_ild_profile(
                predicted_db[:, horizontal, :], reference_db[:, horizontal, :], weights[horizontal], frequency[frequency_mask]
            )
            reference_itd = itd(reference_hrir)
            predicted_itd = itd(predicted_hrir)
            itd_error = np.abs(predicted_itd - reference_itd) * 1e6
            itd_mean = float(np.sum(itd_error[common_mask] * normalized_direction_weights(weights[common_mask])))
            lap_lsd = lap2024_lsd(reference_hrir[common_mask], predicted_hrir[common_mask], 44100.0)
            metrics = {"ERBBandILDMean": float(band_mean), "ITDWeightedMAE_us": itd_mean, "LAP2024LSD_dB": lap_lsd}
            for endpoint, value in metrics.items():
                if not np.isfinite(value):
                    raise FloatingPointError(f"Non-finite {endpoint}: {subject} Q{count}")
                values[(endpoint, count)].append(value)
                rows.append(
                    {
                        "SubjectLabel": subject,
                        "SubjectID": subject_id,
                        "Split": "val",
                        "Method": "HYBRID",
                        "MethodLabel": LABELS[count],
                        "DirectionCount": count,
                        "Endpoint": endpoint,
                        "EvidenceTier": config["evidence_tiers"][endpoint],
                        "Unit": UNITS[endpoint],
                        "EvaluationDirectionCount": int(common_mask.sum()),
                        "Value": format(value, ".17g"),
                    }
                )
        print(f"FSC supplemental [{subjects.index((subject_id, subject)) + 1}/44] {subject}", flush=True)

    aggregate: list[dict[str, object]] = []
    table: list[dict[str, object]] = []
    for endpoint in UNITS:
        for count in COUNTS:
            mean, std, low, high = bootstrap(values[(endpoint, count)], int(config["bootstrap_replicates"]), int(config["bootstrap_seed"]) + count)
            aggregate.append(
                {
                    "Method": "HYBRID",
                    "MethodLabel": LABELS[count],
                    "DirectionCount": count,
                    "Endpoint": endpoint,
                    "EvidenceTier": config["evidence_tiers"][endpoint],
                    "Unit": UNITS[endpoint],
                    "SubjectCount": 44,
                    "Mean": format(mean, ".17g"),
                    "SampleStd": format(std, ".17g"),
                    "Bootstrap95Lower": format(low, ".17g"),
                    "Bootstrap95Upper": format(high, ".17g"),
                }
            )
        table.append(
            {
                "Metric": endpoint,
                "Unit": UNITS[endpoint],
                "EvidenceTier": config["evidence_tiers"][endpoint],
                **{
                    f"Q{count}_Mean": format(float(aggregate[-3 + i]["Mean"]), ".17g")
                    for i, count in enumerate(COUNTS)
                },
                **{
                    f"Q{count}_SD": format(float(aggregate[-3 + i]["SampleStd"]), ".17g")
                    for i, count in enumerate(COUNTS)
                },
            }
        )

    write_csv(partial / "supplemental_subject_level.csv", rows)
    write_csv(partial / "supplemental_summary_mean_std.csv", aggregate)
    write_csv(partial / "paper_observation_density_secondary_table.csv", table)

    # A seven-endpoint convenience table for manuscript plotting, retaining the
    # primary four-endpoint file unchanged.
    primary_table = root / "results/sonicom_fsc_matched_density_v1/paper_observation_density_table.csv"
    primary_rows = list(csv.DictReader(primary_table.open(encoding="utf-8-sig")))
    combined_rows = [
        {
            "Metric": row["Metric"],
            "Unit": "dB",
            "EvidenceTier": "Primary frozen engineering test",
            "Q14_Mean": row["Q14_Mean_dB"],
            "Q14_SD": row["Q14_SD_dB"],
            "Q26_Mean": row["Q26_Mean_dB"],
            "Q26_SD": row["Q26_SD_dB"],
            "Q50_Mean": row["Q50_Mean_dB"],
            "Q50_SD": row["Q50_SD_dB"],
        }
        for row in primary_rows
    ]
    combined_rows.extend(table)
    write_csv(partial / "paper_observation_density_all_metrics.csv", combined_rows)
    summary = {
        "status": "completed",
        "split": "val",
        "subject_count": 44,
        "method": "HYBRID",
        "diagonal_counts": list(COUNTS),
        "metrics": list(UNITS),
        "fixed_evaluation_direction_count": int(common_mask.sum()),
        "band_ild_direction_definition": "common_mask AND horizontal plane",
        "itd_direction_definition": "common_mask with direction-area weights",
        "lap2024_lsd_direction_definition": "common_mask, official 20-20000 Hz formula",
        "bootstrap_replicates": int(config["bootstrap_replicates"]),
        "bootstrap_seed_base": int(config["bootstrap_seed"]),
        "subject_row_count": len(rows),
        "aggregate_row_count": len(aggregate),
        "all_finite": True,
        "test_subject_count_read": 0,
        "evidence_tiers": config["evidence_tiers"],
    }
    (partial / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (partial / "README.md").write_text(
        "# FSC matched-density supplemental metrics\n\n"
        "Validation-only diagonal results for FSC-Q14@Q14, FSC-Q26@Q26, and FSC-Q50@Q50. "
        "The common Q50-excluded 743-direction mask is used for all endpoints; ERB-band ILD "
        "is additionally restricted to the horizontal plane. The primary four-metric table is "
        "not overwritten. See `supplemental_summary_mean_std.csv` and "
        "`paper_observation_density_all_metrics.csv`. Test subjects were not read.\n",
        encoding="utf-8",
    )
    partial.replace(output)


if __name__ == "__main__":
    main()
