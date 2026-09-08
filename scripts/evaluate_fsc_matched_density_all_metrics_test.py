"""Locked-test supplemental metrics and seven-endpoint FSC diagonal table."""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path
import h5py, numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mcar.evaluation.deferred_secondary_metrics import enable_external_torchaudio, strict_hrir_from_selected_db
from mcar.evaluation.secondary_metrics import normalized_direction_weights, spectral_band_ild_profile
from scripts.evaluate_lap2024_metrics import lap2024_lsd
from scripts.evaluate_ten_method_direction_sensitivity_secondary import itd

COUNTS = (14, 26, 50)
LABELS = {14: "FSC-Q14@Q14", 26: "FSC-Q26@Q26", 50: "FSC-Q50@Q50"}
SUPPLEMENTAL = {"ERBBandILDMean": "dB", "ITDWeightedMAE_us": "us", "LAP2024LSD_dB": "dB"}

def csv_write(path: Path, rows: list[dict]) -> None:
    if not rows: raise ValueError(f"empty output: {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def attr_text(value: object) -> str:
    value = np.asarray(value).item()
    return value.decode() if isinstance(value, bytes) else str(value)

def metadata(h: h5py.File) -> dict:
    g = h["strict_ild"]
    return {"mca_selected_phase_rad": np.asarray(g["mca_selected_phase_rad"][:]),
            "mca_outside_real": np.asarray(g["mca_outside_real"][:]),
            "mca_outside_imag": np.asarray(g["mca_outside_imag"][:]),
            "selected_bin_indices_zero_based": np.asarray(g["selected_bin_indices_zero_based"][:]),
            "outside_bin_indices_zero_based": np.asarray(g["outside_bin_indices_zero_based"][:]),
            "single_sided_frequency_count": int(np.asarray(g.attrs["single_sided_frequency_count"]).reshape(-1)[0]),
            "hrir_length": int(np.asarray(g.attrs["hrir_length"]).reshape(-1)[0])}

def bootstrap(x: list[float], reps: int, seed: int) -> tuple[float, float, float, float]:
    a = np.asarray(x, dtype=float); rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(reps, len(a)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975], method="linear")
    return float(a.mean()), float(a.std(ddof=1)), float(lo), float(hi)

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("config", type=Path); args = p.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if cfg["status"] != "frozen" or cfg["split"] != "test" or not cfg["test_access_allowed"]:
        raise PermissionError("frozen authorized-test config required")
    enable_external_torchaudio(Path(cfg["torchaudio_site_packages"]))
    out = ROOT / cfg["output_root"]; partial = out.with_name(out.name + ".partial")
    if partial.exists(): raise FileExistsError(partial)
    if out.exists():
        required = {"primary_subject_level.csv", "primary_summary_mean_std.csv", "primary_summary.json"}
        if not required.issubset({p.name for p in out.iterdir()}):
            raise FileExistsError(out)
    partial.mkdir(parents=True)
    with (ROOT / "configs/data/sonicom_subject_split_v1.csv").open(encoding="utf-8-sig", newline="") as f:
        subjects = [r["subject_id"] for r in csv.DictReader(f) if r["split"] == "test"]
    if len(subjects) != 44: raise ValueError("Expected 44 frozen test subjects")
    grid_rows = list(csv.DictReader((ROOT / "configs/data/sonicom_nested_sparse_grid_q14_q26_q50_v1.csv").open(encoding="utf-8-sig")))
    q50 = np.array([int(r["source_index_zero_based"]) for r in grid_rows if int(r["direction_count"]) == 50])
    common = np.ones(793, dtype=bool); common[q50] = False
    if common.sum() != 743: raise ValueError("Common mask must contain 743 directions")
    roots = {14: ROOT / "data/processed/sonicom_fsc_q14_residual_test_v1", 26: ROOT / "data/processed/sonicom_residual_q26_v1", 50: ROOT / "data/processed/sonicom_fsc_q50_residual_test_v1"}
    default_prediction_roots = {14: "sonicom_fsc_q14_e190_ensemble_test", 26: "sonicom_film_siren_spectral_cnn_final_e190_ensemble_test", 50: "sonicom_fsc_q50_e190_ensemble_test"}
    configured_prediction_roots = cfg.get("prediction_roots", default_prediction_roots)
    pred_roots = {q: ROOT / "artifacts" / "reconstruction" / configured_prediction_roots[str(q)] if str(q) in configured_prediction_roots else ROOT / "artifacts" / "reconstruction" / configured_prediction_roots[q] for q in COUNTS}
    sofa_root = ROOT / "data/HRTF/sonicom_measured_ffcmp_minphase_44k1/subjects"
    rows, values, bands = [], {(e, q): [] for e in SUPPLEMENTAL for q in COUNTS}, []
    for si, subject in enumerate(subjects, 1):
        for q in COUNTS:
            source_path = roots[q] / "subjects" / subject / f"q{q}.h5"
            pred_path = pred_roots[q] / "subjects" / subject / "prediction.h5"
            with h5py.File(source_path, "r") as h:
                if attr_text(h.attrs["split"]) != "test": raise PermissionError(source_path)
                ref_db = np.asarray(h["reference_logmag_db"][:], dtype=np.float64)
                mca_db = np.asarray(h["mca_logmag_db"][:], dtype=np.float64)
                directions = np.asarray(h["direction_features"][:], dtype=np.float64)
                freq = np.asarray(h["frequency_hz"][:], dtype=np.float64).reshape(-1)
                interp = np.asarray(h["interpolation_evaluation_mask"][:], dtype=bool).reshape(-1)
                meta = metadata(h)
            with h5py.File(pred_path, "r") as h:
                if attr_text(h.attrs["split"]) != "test": raise PermissionError(pred_path)
                residual = np.asarray(h["predicted_residual_db"][:], dtype=np.float64)
            pred_db = mca_db + residual
            pred_hrir = strict_hrir_from_selected_db(pred_db, meta).permute(1, 0, 2).cpu().numpy().astype(np.float32)
            with h5py.File(sofa_root / f"{subject}_FreeFieldCompMinPhase_44kHz.sofa", "r") as h:
                ref_hrir = np.asarray(h["Data.IR"][:], dtype=np.float32)
            if ref_hrir.shape != (793, 2, 256) or pred_hrir.shape != ref_hrir.shape: raise ValueError("HRIR shape mismatch")
            horizontal = common & (np.abs(directions[:, 1]) <= 1e-9)
            centers, profile, band_mean = spectral_band_ild_profile(pred_db[:, horizontal, :], ref_db[:, horizontal, :], directions[horizontal, 5], freq)
            ref_itd, pred_itd = itd(ref_hrir), itd(pred_hrir)
            itd_value = float(np.sum(np.abs(pred_itd - ref_itd)[common] * normalized_direction_weights(directions[common, 5]) * 1e6))
            lap = float(lap2024_lsd(ref_hrir[common], pred_hrir[common], 44100.0))
            metrics = {"ERBBandILDMean": float(band_mean), "ITDWeightedMAE_us": itd_value, "LAP2024LSD_dB": lap}
            for endpoint, value in metrics.items():
                if not np.isfinite(value): raise FloatingPointError(f"non-finite {subject} Q{q} {endpoint}")
                values[(endpoint, q)].append(value)
                rows.append({"SubjectLabel": subject, "SubjectID": int(subject[1:]), "Split": "test", "MethodLabel": LABELS[q], "DirectionCount": q, "Endpoint": endpoint, "Unit": SUPPLEMENTAL[endpoint], "EvaluationDirectionCount": 743, "Value": format(value, ".17g")})
            for bi, (center, value) in enumerate(zip(centers, profile)):
                bands.append({"SubjectLabel": subject, "DirectionCount": q, "MethodLabel": LABELS[q], "BandIndex": bi, "BandCenter_Hz": format(float(center), ".17g"), "ILDAbsoluteError_dB": format(float(value), ".17g")})
        print(f"FSC test supplemental [{si}/44] {subject}", flush=True)
    aggregate, table = [], []
    for endpoint in SUPPLEMENTAL:
        qstats = {}
        for q in COUNTS:
            mean, sd, lo, hi = bootstrap(values[(endpoint, q)], int(cfg["bootstrap_replicates"]), int(cfg["bootstrap_seed"]) + q)
            qstats[q] = (mean, sd)
            aggregate.append({"MethodLabel": LABELS[q], "DirectionCount": q, "Endpoint": endpoint, "Unit": SUPPLEMENTAL[endpoint], "SubjectCount": 44, "Mean": format(mean, ".17g"), "SampleStd": format(sd, ".17g"), "Bootstrap95Lower": format(lo, ".17g"), "Bootstrap95Upper": format(hi, ".17g")})
        table.append({"Metric": endpoint, "Unit": SUPPLEMENTAL[endpoint], "EvidenceTier": cfg["evidence_tiers"][endpoint], **{f"Q{q}_Mean": format(qstats[q][0], ".17g") for q in COUNTS}, **{f"Q{q}_SD": format(qstats[q][1], ".17g") for q in COUNTS}})
    csv_write(partial / "supplemental_subject_level.csv", rows); csv_write(partial / "supplemental_summary_mean_std.csv", aggregate); csv_write(partial / "band_ild_profile.csv", bands); csv_write(partial / "paper_observation_density_supplemental_test.csv", table)
    summary = {"status": "completed", "split": "test", "subject_count": 44, "test_subject_count_read": 44, "diagonal_counts": list(COUNTS), "fixed_evaluation_direction_count": 743, "horizontal_common_direction_count": int(np.count_nonzero(horizontal)), "supplemental_metric_row_count": len(rows), "aggregate_row_count": len(aggregate), "all_finite": True, "cross_density_evaluation": False, "evidence_tiers": cfg["evidence_tiers"]}
    (partial / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    # Primary MATLAB artifacts already occupy ``out``; add only these new
    # supplemental files without replacing the directory on Windows.
    for path in partial.iterdir():
        target = out / path.name
        if target.exists():
            raise FileExistsError(target)
        path.replace(target)
    partial.rmdir()

if __name__ == "__main__": main()
