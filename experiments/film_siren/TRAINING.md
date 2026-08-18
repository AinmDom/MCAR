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
