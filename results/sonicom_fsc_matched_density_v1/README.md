# FSC matched-density summary

Diagonal validation only: FSC-Q14@Q14, FSC-Q26@Q26 (existing frozen validation), and FSC-Q50@Q50. Metrics and the common 743-direction mask are inherited from the strict ten-method evaluator.

Files: `subject_level.csv`, `summary_mean_std.csv`, and `paper_observation_density_table.csv`. Test subjects were not read.

The three additional registered endpoints (frequency-band ILD, ITD weighted MAE,
and LAP 2024 LSD) are in the independent validation-only supplement
`results/sonicom_fsc_matched_density_supplemental_metrics_v1/`; its
`paper_observation_density_all_metrics.csv` combines all seven endpoints without
overwriting the primary four-metric evidence.
