# FSC matched-density locked-test all metrics

Diagonal-only locked-test evaluation for `FSC-Q14@Q14`, `FSC-Q26@Q26`, and
`FSC-Q50@Q50`. Q14/Q50 use independently exported current-Q residual data and
cycle-190 three-seed residual-dB ensembles; Q26 reuses the existing frozen
test ensemble. All seven endpoints use the common Q50-excluded 743-direction
mask. The three supplemental endpoints additionally preserve their registered
band/ITD/LAP definitions. No cross-density input evaluation is included.

Primary MATLAB outputs are in `primary_subject_level.csv` and
`primary_summary_mean_std.csv`. Supplemental outputs are in
`supplemental_subject_level.csv` and `supplemental_summary_mean_std.csv`.
`paper_observation_density_all_metrics_test.csv` is the manuscript-ready
mean/SD table; `all_metrics_summary_mean_std.csv` includes bootstrap 95% CIs.
All 44 test subjects per diagonal were read exactly once for evaluation;
`summary.json` records the test authorization and integrity checks.
