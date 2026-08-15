# HUTUBS external sparsity experiment

> Archived/deprecated on 2026-08-13. This experiment is retained only as a
> historical audit record and must not be used as an active sparsity result.
> The active learned-method sparsity experiment compares MCAR, FSP-AE, and
> RANF on SONICOM.
> This directory is stored under `废弃实验/HUTUBS稀疏度实验/` so that it is
> physically separated from active experiment outputs.

- External split: HUTUBS frozen test (12 subjects)
- Observed directions: `6, 14, 26`
- Models: frozen SONICOM-trained MCAR v3.2 epoch 39 and FSP-AE epoch 40
- Lower error and lower positive Q6-Q26 degradation are better.

Complete aggregates are in `aggregate_metrics.csv`; paired robustness statistics are in `robustness_statistics.csv`.
Direct MCAR-versus-baseline endpoint and degradation contrasts are in
`paired_robustness_contrasts.csv`; their interpretation is summarized in
`robustness_conclusion.md`.

## MCAR v3.2 Q6-Q26 degradation

| Metric | Mean degradation (dB) | 95% paired bootstrap CI |
|---|---:|---:|
| FullSphereERB | 0.3249 | [0.2720, 0.3783] |
| Contralateral25ERB | -0.0957 | [-0.1906, -0.0097] |
| ContralateralHighFrequency | 0.5915 | [0.4029, 0.7681] |
| HorizontalILDMAE | 0.7998 | [0.5713, 1.0299] |
