# FSP-AE-Q26 frozen final test report

Date: 2026-08-12

## Freeze and data-isolation checks

- Selected checkpoint: epoch 40, selected using validation loss only.
- Frozen checkpoint: `artifacts/frozen/fsp_ae_q26_epoch40_25db1eb83a1b647b.pt`.
- Checkpoint SHA-256 before and after inference: `25DB1EB83A1B647B2E0E12C6BE1FF9EE31BF2B216DC50B57C19ADC84D1E12F5F`.
- Frozen copy is read-only.
- Model selection read 262 train, 44 validation, and 0 test subjects.
- Test inference read 44 test subjects only after explicit authorization.
- Training-only normalization SHA-256 remained unchanged before and after test access: `D25121E1954802C79593B14F4A0FC1E7BD26E57D0ADC1CA70CC5DFBF7F49FF95`.
- No training, hyperparameter selection, checkpoint selection, or model update was performed after test access.

## Final test results

All values are subject mean +/- sample standard deviation in dB; lower is better. The evaluation excludes the 26 observed input directions and scores 767 interpolation directions for all 44 test subjects.

| Method | Full ERB | Contra25 ERB | Contra HF | Horizontal ILD |
|---|---:|---:|---:|---:|
| MCA | 1.082 +/- 0.096 | 1.746 +/- 0.155 | 4.699 +/- 0.266 | 0.829 +/- 0.158 |
| RANF | 1.063 +/- 0.126 | 1.557 +/- 0.164 | 3.470 +/- 0.254 | 0.775 +/- 0.189 |
| MCAR v3.2 | **0.868 +/- 0.162** | **1.365 +/- 0.167** | 3.612 +/- 0.277 | **0.687 +/- 0.274** |
| FSP-AE-Q26 | 1.184 +/- 0.349 | 1.909 +/- 0.701 | **3.164 +/- 0.685** | 0.759 +/- 1.040 |

Relative to MCAR v3.2, FSP-AE is 36.49% worse on Full ERB and 39.81% worse on Contra25 ERB, 12.39% better on contralateral high-frequency error, and 10.43% worse on mean horizontal ILD MAE. Subject-level FSP-AE wins against MCAR are 0/44, 0/44, 43/44, and 28/44 respectively.

Relative to RANF, FSP-AE is 11.40% worse on Full ERB, 22.57% worse on Contra25 ERB, 8.80% better on contralateral high-frequency error, and 2.17% better on mean horizontal ILD MAE. Subject-level wins are 10/44, 3/44, 43/44, and 35/44.

## Interpretation and failure case

The result confirms the validation pattern: the adapted FSP-AE is a strong high-frequency contralateral baseline, but it does not match MCAR's broadband or contralateral aggregate accuracy. Its median test metrics are close to validation, while one genuine failure case, P0339, raises all four test means and especially the ILD variance. P0339 scores 3.343 / 6.389 / 7.271 / 7.434 dB. Its prediction file is structurally complete and contains only finite values, so it remains in the official 44-subject result.

For diagnostic context only, excluding P0339 gives 43-subject means of 1.134 / 1.805 / 3.069 / 0.603 dB. These values are not used as the official test result.

## Artifacts

- Lock manifest: `configs/experiments/sonicom_fsp_ae_q26_locked_test.json`
- Inference summary: `artifacts/reconstruction/sonicom_fsp_ae_q26_locked_final_test/summary.json`
- Strict evaluation: `results/sonicom_fsp_ae_q26_locked_final_test/`
- Aggregate figure: `results/sonicom_fsp_ae_q26_locked_final_test/figures/test44_aggregate_baselines.png`
