# Stage E complete ten-method horizontal comparison protocol

## Scope

This protocol registers the paper-wide comparison set requested on 2026-08-30. The canonical
order is SH only, SUpDEq SH, SUpDEq NN, SUpDEq Barycentric, MCA, MCAR v3.5.1,
FSP-AE, RANF, Hybrid E190, and Bounded E25. The machine-readable registry is
`configs/experiments/sonicom_complete_horizontal_comparison_methods_v1.json`.

This is a consolidation protocol, not a new result-blind model-selection protocol. Some upstream
results already exist and have already been inspected. The prospective part is limited to fixed
source mapping, row construction, paired descriptive statistics, availability labels, and workbook
layout. No model, checkpoint, hyperparameter, endpoint definition, or split may be changed here.

## Evidence tiers

1. **Primary validation:** all ten methods, 44 frozen validation subjects, four strict endpoints.
2. **Primary frozen engineering test:** nine registered methods. Hybrid E190 is `NOT RUN` because it
   was not part of the frozen nine-method test evaluation. No retrospective test run is authorized.
3. **Secondary validation:** spectral/ILD supplementary endpoints and spatial profiles. Include only
   methods with an existing frozen same-protocol result.
4. **Deferred validation:** ITD and discrete dominant-notch endpoints; exploratory only. Include only
   methods with an existing frozen same-protocol result.
5. **Mechanism, efficiency, localization:** report the measured quantity only where a common frozen
   definition exists. Otherwise use `NOT APPLICABLE`, `NOT AVAILABLE`, or `NOT RUN` with a reason.

Validation, test, supplementary, and exploratory values must remain in separate columns or tables.
They must not be pooled into a single rank or averaged across evidence tiers.

## Canonical source mapping

- SH only, SUpDEq SH/NN/Barycentric, MCA validation primary:
  `results/sonicom_fsp_ae_q26_formal_validation/metric_long.csv`.
- MCAR v3.5.1, Hybrid E190, Bounded E25 validation primary:
  `results/sonicom_bounded_mcar_film_correction_final_e25_validation/metric_long.csv`, mapping
  `MCAR -> MCARv351`, `PARENT -> HYBRID`, and `HYBRID -> BOUNDED`.
- FSP-AE and RANF validation primary:
  `results/sonicom_film_siren_gl_final_d1d2_notch_vs_ranf_fsp_v351_validation/metric_long.csv`.
- Nine-method primary test:
  `results/sonicom_bounded_mcar_film_correction_final_e25_frozen_test_nine_method/metric_long.csv`.
- Internal secondary/deferred validation:
  `results/sonicom_film_secondary_metrics_v1_validation/` and
  `results/sonicom_film_deferred_secondary_metrics_v1_validation/`.
- RANF/FSP-AE secondary/deferred validation:
  `results/sonicom_film_secondary_baseline_extension_ranf_fsp_v1_validation/`.
- Efficiency:
  `results/sonicom_bounded_mcar_film_efficiency_v1_validation/`.

## Fixed calculations

- Per-method aggregate: arithmetic mean and sample standard deviation across the 44 aligned subjects.
- Paired difference: Bounded E25 minus comparator; lower is better.
- Paired interval: 10,000 subject-level bootstrap resamples using PCG64 seed `20260830`; percentile
  2.5% and 97.5% limits. These are descriptive for already-consumed evidence, not new confirmation.
- Wins/ties/losses: counts from per-subject differences using exact floating-point equality for ties.
- Ranks: ascending mean within one endpoint, one split, and one evidence tier; missing methods receive
  no rank. `ReferenceNotchFraction` is a descriptive reference prevalence and receives no rank.
- Every numeric input must be finite, subject IDs must align exactly, and each complete method-endpoint
  cell must contain 44 subjects.

## Missingness and claim boundary

Missing cells are structural, not zero and not estimated. `NOT RUN` means no frozen evaluation exists;
`NOT AVAILABLE` means the required measurement was not recorded under a common protocol;
`NOT APPLICABLE` means the method lacks the internal quantity being measured. Model-based localization
remains `NOT RUN` because the frozen official dependency is unavailable.

Generating the consolidation must read no raw test HDF5/SOFA and perform no new test inference.
The output summary must record `new_test_subject_count_read = 0`.
