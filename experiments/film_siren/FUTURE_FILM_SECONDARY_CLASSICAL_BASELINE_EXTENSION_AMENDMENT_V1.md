# FiLM secondary/deferred classical-baseline extension amendment v1

Status: **FROZEN BEFORE NEW METRIC COMPUTATION**  
Date: 2026-08-31  
Evidence tier: validation-only secondary/deferred comparison

## Scope and data boundary

This amendment adds five already-registered comparators to the frozen FiLM
secondary and deferred validation endpoints:

1. `SHOnly` — SH interpolation without SUpDEq;
2. `SUpDEqSH` — SUpDEq followed by SH interpolation;
3. `SUpDEqNN` — SUpDEq followed by natural-neighbour interpolation;
4. `SUpDEqBary` — SUpDEq followed by barycentric interpolation;
5. `MCA` — the frozen MCA reconstruction supplied by the Q26 dataset.

Only the 44 locked SONICOM validation subjects in
`configs/data/sonicom_subject_split_v1.csv` may be read. The exporter is
hard-coded to reject any split other than `val` and any `allowTest=true`
request. No test path may be constructed, and every output must report
`test_subject_count_read=0`. Existing Hybrid/Bounded test evidence is not an
input to reconstruction, metric computation, or selection.

## Frozen reconstruction

The five reconstructions must use the same numerical definitions already used
by `mcar.evaluate_sonicom_interpolation_baselines`:

- SONICOM Q26 observations, 793-direction reference grid, 44.1 kHz, 1024-point
  spectrum, head radius 0.09 m, and SH Tikhonov epsilon `1e-2`;
- SUpDEq NN/Bary fixed operators built once from the frozen Q26/reference grids;
- first-subject parity against the upstream `supdeq_interpHRTF` entry point with
  maximum absolute complex-spectrum error below `1e-9`;
- MCA phase and outside-band complex bins inherited exactly from the processed
  Q26 HDF5; strict selected bins are the source `strict_ild` indices;
- HRIRs are the first 256 samples obtained from the 1024-point conjugate-symmetric
  inverse FFT used by the existing strict evaluator.

For each subject/method the exporter writes one immutable HDF5 containing
`predicted_magnitude_db` with Python-visible shape `[2,793,463]`,
`predicted_hrir` with shape `[793,2,256]`, `frequency_hz`, and split/subject/
method attributes. Existing output paths must never be overwritten.

## Frozen endpoints

The formulas, masks, weighting, units, bootstrap procedure, and Bounded-minus-
baseline sign convention are inherited without modification from
`FUTURE_FILM_SECONDARY_METRICS_PROTOCOL.md`,
`FUTURE_FILM_DEFERRED_SECONDARY_METRICS_PROTOCOL.md`, and the already frozen
RANF/FSP-AE extension amendment. The added results are:

- five scalar secondary endpoints: `FullSphereLSD`,
  `HFFirstDifferenceMAE`, `HFSecondDifferenceMAE`,
  `MultiScaleNotchDepthMAE`, and `ERBBandILDMean`;
- the 35-band horizontal ERB ILD-error profile;
- four fixed distance-to-Q26 spatial LSD bins and the 767-direction mean map;
- seven deferred endpoints: penalized/matched dominant-notch MAE, notch miss
  and spurious rates, descriptive reference-notch fraction, weighted ITD MAE,
  and maximum absolute ITD error.

All aggregates use all 44 validation subjects. Confidence intervals use the
existing 10,000 subject-paired bootstrap replicates with seed `20260829`.
Worst-decile summaries use the worst five subjects. `ReferenceNotchFraction`
is descriptive and is excluded from superiority claims.

## Execution and interpretation

The sequence is fixed: commit this amendment and implementation; export all
five validation reconstructions; freeze their inventory and all evaluator
resource hashes in a manifest; only then compute the new metrics. Any missing,
non-finite, mis-shaped, non-validation, or identity-mismatched artifact fails
the run. Partial directories are not publishable.

These metrics cannot change the frozen Hybrid or Bounded candidates and cannot
justify any test-driven tuning. Gate/correction mechanism, same-hardware
efficiency, and localization remain structurally unavailable for these five
methods and must be reported as such rather than imputed.
