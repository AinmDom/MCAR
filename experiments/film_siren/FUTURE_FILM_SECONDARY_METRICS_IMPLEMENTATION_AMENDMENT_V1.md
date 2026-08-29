# Future FiLM Secondary Metrics Implementation Amendment v1

**Status:** frozen before any secondary endpoint was evaluated on SONICOM validation
data.

**Freeze date:** 2026-08-30 (Asia/Hong_Kong)

**Parent protocol:**
`FUTURE_FILM_CANDIDATE_SECONDARY_METRICS_PREREGISTRATION.md`, SHA-256
`E5C8C231BA297ED395D0AFC170DF734CD88CC641CEA0B29F3434D1EA7FB829E0`.

**Frozen manifest:**
`configs/experiments/sonicom_film_secondary_metrics_v1_validation_manifest.json`,
identity
`4F70628ED6EC6525EE8B7275B423120DF49F75FA18B5555E6F2E824663A45548`.

## 1. Result-blind declaration

At this freeze point no secondary metric defined by the parent protocol had been
computed on, aggregated over, or inspected for any SONICOM validation or test subject.
Only source schemas, file identities, array counts, existing committed primary-metric
rows, and synthetic numerical fixtures were inspected. The synthetic qualification
suite passed 6/6 tests. No training or inference was run.

## 2. Result-blind correction to the parent protocol

Section 3.1 of the parent protocol named
`data/processed/sonicom_residual_q26_v1/dataset_definition.json`; that file does not
exist. The frozen dataset definition is the following bundle, whose hashes are recorded
in the manifest:

- `configs/data/sonicom_preparation_report_v1.json`;
- `configs/data/sonicom_subject_split_v1.csv`;
- `configs/data/sonicom_sparse_grid_q26_v1.csv`; and
- `data/processed/sonicom_residual_q26_v1/training_statistics.json`.

This correction changes no subject, split, direction, frequency, mask, reconstruction,
endpoint, comparator, or statistical rule. It is made before evaluating any secondary
result.

## 3. Frozen runnable tranche

The first validation-only tranche implements:

1. full-sphere LSD, including the unmasked subject/method/ear/direction map and a
   767-direction solid-angle aggregate;
2. first- and second-order high-frequency spectral-difference MAE above 4 kHz;
3. multi-scale notch-depth MAE at 4--18 kHz, radii `(4,8,16)`, 1 dB threshold, and
   0.5 dB softplus temperature;
4. the 200--18000 Hz horizontal ERB-band ILD absolute-error profile and its arithmetic
   band mean;
5. LSD in the fixed `[0,10)`, `[10,20)`, `[20,30)`, and `[30,180]` degree
   distance-to-Q26 bins; and
6. paired central/tail-risk statistics for the five scalar endpoints above, the four
   already committed primary endpoints, and the four spatial-distance endpoints.

The primary values are read without recomputation from the frozen Stage-E validation
`metric_long.csv`. Its historical method IDs are deterministically mapped as
`HYBRID -> BOUNDED`, `PARENT -> HYBRID`, `FILMENS -> FILMENS`, and `MCAR -> MCAR`.

## 4. Frozen implementation and numerical conventions

- Implementation commits: `90ab7d0`, the result-blind dataset-resource guard
  correction `afdf10b`, and the entry-import correction
  `6d1b47be3b495fc32e28edbbcce615728b872b8b`.
- Entry point: `scripts/evaluate_film_secondary_metrics_validation.py`.
- Metric module: `src/mcar/evaluation/secondary_metrics.py`.
- Existing spectral definitions are invoked from `src/mcar/losses.py`; ERB weights and
  centers are invoked from `src/mcar/training/train_mlp_v2.py`.
- Runtime: `D:/miniconda3/envs/ml/python.exe`; Python 3.9.23, NumPy 1.26.4,
  h5py 3.14.0, PyTorch 2.8.0+cu128.
- Bootstrap: 10,000 subject resamples, seed `20260829`, NumPy
  `random.default_rng` with PCG64, two-sided percentile interval using linear quantile
  interpolation.
- Every scalar is computed per subject before aggregation. Standard deviations use
  sample `ddof=1`. Worst decile is the largest five of 44 paired differences.
- Candidate-minus-baseline is used throughout; negative favors BOUNDED.
- All four prediction inventories, inference reports, dataset resources, code files,
  and the primary-metric source are SHA-256 guarded by the manifest.

The exact frozen command is:

```powershell
D:\miniconda3\envs\ml\python.exe scripts/evaluate_film_secondary_metrics_validation.py configs/experiments/sonicom_film_secondary_metrics_v1_validation_manifest.json
```

The entry point refuses an existing final or `.partial` output directory and writes to
`results/sonicom_film_secondary_metrics_v1_validation` only after all checks complete.

## 5. Frozen comparators

The order and identities are fixed in the manifest:

1. BOUNDED: formal E25 bounded-correction ensemble;
2. HYBRID: formal E190 FiLM-SIREN plus spectral-CNN ensemble;
3. FILMENS: corrected D1/D2+notch E130 FiLM-SIREN ensemble; and
4. MCAR: v3.5.1 output-level ensemble.

No method may be added, removed, relabelled, or substituted after execution.

## 6. Required integrity checks

The run must abort unless all of the following hold:

- split is exactly `val`, with 44 unique subjects;
- every source and prediction declares `split=val` and the correct subject ID;
- shapes are `[2,793,463]` and all predictions are finite;
- Q26/interpolation masks are exact complements with counts 26/767;
- the horizontal interpolation count is 72;
- all file, inventory, implementation, and manifest hashes match;
- the per-band profile mean agrees numerically with the registered repository loss;
- all output values are finite; and
- `test_subject_count_read=0` is written to quality and summary artifacts.

## 7. Deferred endpoints

This tranche does not compute localization, ITD, gate/correction behavior, efficiency,
or discrete notch location. Their omission cannot be presented as a favorable result.

- Localization and ITD still require immutable external package/model versions,
  coordinate-convention fixtures, stimulus/lag parameters, and reference-self tests.
- Gate/correction behavior requires a separately hash-guarded diagnostic replay of all
  three BOUNDED members; averaged gates remain prohibited.
- Efficiency requires a frozen hardware/driver/power/precision/thread manifest and a
  separate warm-up/timing run.
- Discrete notch location remains outside v1 until smoothing, detection, matching, and
  missing/extra-notch handling are prospectively frozen.

Any later tranche requires another result-blind amendment and must remain
validation-only. None of these endpoints is authorized on the already consumed test
set.
