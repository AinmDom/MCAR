# Stage E bounded-correction frozen test protocol

## Authorization and scope

- User authorization: `可以先做test评价`.
- Recorded at `2026-08-29T23:31:38+08:00` (`2026-08-29T15:31:38.3013460Z`).
- This authorizes one locked inference over the 44 SONICOM test subjects for the
  already frozen Stage-E bounded-correction ensemble only. It does not authorize
  test-driven tuning, member selection, weighting, retraining, or evaluation of
  Hybrid/FiLM candidates that do not already have frozen test predictions.

## Candidate frozen before test

- Identity: `72B7319F8334307664A02BC6369FBD9EECE1D22383C402EC9683D1F6F05F2705`.
- Three seeds `20260821/20260822/20260823`, each using cycle-25 `last.pt`.
- Prediction is the equal `1/3` residual-dB mean. No post-test fallback or
  alternative checkpoint is permitted.

## Evaluation freeze

- Exactly 44 test subjects from `sonicom_subject_split_v1.csv` are predicted once.
- Strict reconstruction uses `evaluate_sonicom_interpolation_baselines` with the
  candidate supplied through its legacy MCAR prediction slot. Its raw method key
  `MCARv32` is deterministically relabelled to `BOUNDED`; it is not MCAR v3.2.
- The final table contains `BOUNDED` plus the already frozen eight-method test table:
  `MCARv351`, `FSPAE`, `RANF`, `MCA`, `SUpDEqBary`, `SUpDEqNN`, `SUpDEqSH`, `SHOnly`.
- Existing eight-method source is
  `results/sonicom_eight_method_q26_v351_main_test/metric_long.csv`, frozen SHA-256
  `3D74AF135CAD510FC3F9166076ECEDF41ABFAB11ADD5231219227D7592E97D63`.
- Metrics are FullSphereERB, Contralateral25ERB,
  ContralateralHighFrequency, and HorizontalILDMAE; lower is better.
- Candidate-minus-baseline comparisons against MCAR v3.5.1, RANF, FSP-AE and MCA
  use 10,000 subject-paired percentile bootstrap replicates with seed `20260829`.
- All observed results are final and reportable regardless of direction. A retry is
  allowed only for a proven incomplete failure under the unchanged manifest identity.

## Frozen outputs

- Manifest/registry: `configs/experiments/sonicom_bounded_mcar_film_correction_final_e25_test_{manifest,registry}.json`.
- Predictions: `artifacts/reconstruction/sonicom_bounded_mcar_film_correction_final_e25_ensemble_test/`.
- Raw evaluation: `results/sonicom_bounded_mcar_film_correction_final_e25_frozen_test_raw/`.
- Final nine-method evaluation: `results/sonicom_bounded_mcar_film_correction_final_e25_frozen_test_nine_method/`.

