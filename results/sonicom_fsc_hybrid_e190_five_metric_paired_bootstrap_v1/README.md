# FSC/Hybrid E190 unified five-metric paired statistics

This directory is a result-level derivation from frozen test CSV files. It does
not retrain, re-infer, or read raw SOFA/HDF5 data.

## Definition

For each metric and baseline, the statistic is computed from the same 44
listener-level values:

`difference_subject = FSC/Hybrid E190_subject - baseline_subject`

The reported mean is the mean of those paired differences. The 95% interval is
the percentile interval of 10,000 bootstrap means, resampling listeners with
replacement. Bootstrap uses the project's
`mcar.evaluation.secondary_metrics.paired_tail_statistics` implementation
(NumPy `default_rng`, linear quantiles), with fixed seed `20260906`. All metrics
are lower-is-better, so negative values favor FSC.

## Metrics and evidence tiers

| Metric | Endpoint | Evidence tier |
|---|---|---|
| Full-sphere ERB | `FullSphereERB` | Primary frozen engineering test |
| Contralateral-25 ERB | `Contralateral25ERB` | Primary frozen engineering test |
| ERB-band ILD mean | `ERBBandILDMean` | Secondary test |
| ITD weighted MAE | `ITDWeightedMAE_us` | Deferred test |
| LAP 2024 LSD | `LAP2024LSD_dB` | LAP2024 locked-test descriptive evaluation |

The source evidence tiers are retained; no metric is relabeled as preregistered
Primary. `paired_bootstrap.csv` contains all 15 requested pair-by-metric rows,
and `paper_summary_table.csv` is the compact paper-facing table. The LAP source
is the formally completed `results/lap2024_test_metrics_v1/` result with
`LAP2024LSD_dB`, 10 methods, and the same 44 listeners.
