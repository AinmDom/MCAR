# Stage D Hybrid E190 vs MCAR/RANF/FSP-AE validation comparison

Status: `FROZEN BEFORE RESULTS`

## Scope

- Split: the existing 44-subject SONICOM validation split only.
- Primary candidate: frozen Hybrid E190 1/3 residual-dB ensemble, manifest
  identity `A3CFAC9C206E0A53FD2FA130817673AAFE07B66855322B5D34824E9173BDCCFE`.
- Baselines: MCAR v3.5.1, corrected RANF validation export, and formal FSP-AE.
- No prediction, checkpoint, ensemble weight, reconstruction rule, or baseline
  artifact may be changed after results are observed.
- Test access is not authorized; `test_subject_count_read` must remain zero.

## Frozen evaluation

- Metrics (dB, lower is better): FullSphereERB, Contralateral25ERB,
  ContralateralHighFrequency, and HorizontalILDMAE.
- Reconstruction and regions are identical to the corrected four-method
  evaluation at commit `c9e1aa5`.
- RANF HRIR extraction must use the complete left/right responses:
  `squeeze(ranfHrir(:, ear, :)).'`.
- FSP-AE HDF5 HRIR dimensions must be interpreted as MATLAB
  `[time, ear, direction]` after `h5read`.
- Paired differences are `HYBRID - baseline`; negative favors HYBRID.
- Bootstrap: subject-level paired percentile interval, 10,000 replicates,
  seed `20260819`, two-sided 95% CI.

## Inputs and outputs

- Hybrid prediction: `artifacts/reconstruction/sonicom_film_siren_spectral_cnn_final_e190_ensemble_validation/`.
- MCAR prediction: `artifacts/reconstruction/sonicom_q26_validation_v351_previous30_b70/`.
- RANF prediction: `artifacts/reconstruction/sonicom_ranf_q26_validation_frozen/`.
- FSP-AE prediction: `artifacts/reconstruction/sonicom_fsp_ae_q26_formal_validation/`.
- Output: `results/sonicom_film_siren_spectral_cnn_final_e190_vs_ranf_fsp_v351_validation/`.

The comparison is descriptive validation evidence for paper positioning. It
must not be used to retune the frozen Hybrid ensemble.
