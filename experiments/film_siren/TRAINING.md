# FiLM-SIREN training

## Stage A1: single-subject representation baseline

The first runnable experiment fits a plain coordinate SIREN to the complete
residual field of train subject P0002. It is a representation-ceiling test, not
a direction-generalization result.

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.training.train_siren `
  configs/experiments/sonicom_siren_single_subject_baseline_v1.json
```

The model predicts normalized residuals from `[x,y,z,f_linear]` and is
converted back to dB for reporting. Outputs are written below
`artifacts/training/<run_name>/`; no validation or test subject is read.

The v1 run completed successfully. It reduced full-field residual RMSE from
5.0429 dB to 1.8520 dB with no NaN/Inf, but the best result occurred at the
final epoch. Its decision is therefore `RETEST`; see
[`../../reports/film_siren_siren_a1_p0002_baseline_v1.md`](../../reports/film_siren_siren_a1_p0002_baseline_v1.md).

## Stage A2: five-subject backbone matrix

The Stage A2 joint coarse screening (`frequency {linear,erb,dual}` ×
`first-omega {20,30,50}` = 9 configurations × 5 fixed train subjects) is
pre-registered in
[`STAGE_A2_PROTOCOL.md`](STAGE_A2_PROTOCOL.md). The frozen lock files live at:

- `configs/data/siren_a2_subjects_v1.csv` — five train subjects stratified by
  `Avg RMS dB (Free Field)` (seed `20260819`);
- `configs/data/siren_a2_holdout_v1.csv` — 64 fixed interpolation-only
  directions (solid-angle stratified, seed `20260819`), hash-verified at load.

Regenerate the lock files (deterministic) with:

```powershell
D:\miniconda3\envs\ml\python.exe scripts\prepare_siren_a2_matrix.py
```

Regenerate the nine frozen configs with:

```powershell
D:\miniconda3\envs\ml\python.exe scripts\generate_siren_a2_configs.py
```

Run one configuration (five independent plain SIRENs, one per subject,
train = 793 − 64 holdout directions, holdout evaluated every epoch):

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.training.train_siren `
  configs/experiments/sonicom_siren_a2_dual_w256_d6_o30.json
```

Run the whole matrix serially:

```powershell
Get-ChildItem configs/experiments/sonicom_siren_a2_*_w256_d6_o*.json |
  Sort-Object Name | ForEach-Object {
    D:\miniconda3\envs\ml\python.exe -m mcar.training.train_siren $_.FullName
  }
```

Aggregate the nine matrix summaries and apply the frozen Top-2 rule:

```powershell
D:\miniconda3\envs\ml\python.exe scripts\analyze_siren_a2_matrix.py
```

Per-subject artifacts are written below
`artifacts/training/<run_name>/<PXXXX>/`; the aggregate lives in
`artifacts/siren_a2_matrix/summary.csv` and `top2.json`.
Validation/test HDF5 files are never read (`test_subjects_read` is 0 in every
report).
