# Stage D Hybrid E190 frozen test protocol

Status: `AUTHORIZED / FROZEN BEFORE TEST`

## Authorization and scope

- User authorization: `目前FiLM-SIREN + CNN（Hybrid E190）是不是没有在test集上跑过，如果没有的话可以跑一次`.
- Recorded at `2026-08-31T11:49:13+08:00`.
- Repository evidence confirms Hybrid E190 has validation predictions only and
  is marked `NOT RUN` in the existing ten-method test availability table.
- This authorizes one inference over the 44 SONICOM test subjects for the
  already frozen Hybrid E190 ensemble. It does not authorize training,
  checkpoint/member/weight selection, post-test tuning, or a second run.

## Frozen candidate

- Source validation manifest identity:
  `A3CFAC9C206E0A53FD2FA130817673AAFE07B66855322B5D34824E9173BDCCFE`.
- Seeds `20260821/20260822/20260823`, each using its cycle-190 `last.pt`.
- Prediction is the equal `1/3` mean of residual-dB tensors.
- The three checkpoint SHA-256 values are copied into the test manifest and
  verified immediately before inference.

## Frozen evaluation

- Exactly 44 test subjects from `sonicom_subject_split_v1.csv` are predicted once.
- Strict reconstruction uses `evaluate_sonicom_interpolation_baselines`; Hybrid
  occupies its legacy `MCARv32` prediction slot and is deterministically
  relabelled `HYBRID` during the result-blind merge.
- The frozen nine-method source is
  `results/sonicom_bounded_mcar_film_correction_final_e25_frozen_test_nine_method/metric_long.csv`,
  SHA-256 `C8212C0B17175EE2A7447BA2DBDBCBF6C70172C14A2151879D79D807E0396EBE`.
- Final method count is ten: existing nine methods plus Hybrid E190.
- Metrics are FullSphereERB, Contralateral25ERB,
  ContralateralHighFrequency, and HorizontalILDMAE; lower is better.
- Hybrid-minus-baseline comparisons against Bounded E25, MCAR v3.5.1, RANF,
  FSP-AE, and MCA use 10,000 paired percentile bootstrap replicates with seed
  `20260831` and subject as the resampling unit.
- Results are final regardless of direction. Retry is permitted only after a
  proven incomplete technical failure under the unchanged manifest identity.

## Frozen outputs

- Manifest/registry: `configs/experiments/sonicom_film_siren_spectral_cnn_final_e190_test_{manifest,registry}.json`.
- Prediction: `artifacts/reconstruction/sonicom_film_siren_spectral_cnn_final_e190_ensemble_test/`.
- Raw strict evaluation: `results/sonicom_film_siren_spectral_cnn_final_e190_frozen_test_raw/`.
- Final ten-method result: `results/sonicom_film_siren_spectral_cnn_final_e190_frozen_test_ten_method/`.

