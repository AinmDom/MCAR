# SONICOM learned-method sparsity experiment

Completed on 44 fixed test subjects. All methods use the same 767-target mask.

## Main finding

RANF is the most robust learned method in this experiment. At Q6 it has the
lowest mean error for all four metrics, and its Q6-minus-Q26 degradation is
also the smallest for all four metrics. The paired 10,000-replicate bootstrap
favors RANF over both MCAR v3.2 and FSP-AE for every Q6 endpoint and every
Q6-minus-Q26 degradation contrast (two-sided bootstrap p = 0.0002).

FSP-AE is the most sparsity-sensitive method: its measured-domain ERB error
increases by 2.290 dB from Q26 to Q6, versus 1.037 dB for MCAR v3.2 and
0.147 dB for RANF. For contralateral high-frequency error, MCAR and FSP-AE
are statistically indistinguishable at the Q6 endpoint (95% CI for MCAR minus
FSP-AE: -0.209 to 0.310 dB; p = 0.424), but MCAR degrades less from Q26 to Q6
(difference -0.319 dB, 95% CI -0.473 to -0.193 dB; p = 0.0002).

## Robustness summary

| Method | Metric | Q6 | Q26 | Q6-Q26 | Slope / halving |
|---|---|---:|---:|---:|---:|
| MCAR v3.2 | MeasuredDomainERB | 1.893 | 0.856 | 1.037 | 0.495 |
| FSP-AE | MeasuredDomainERB | 3.475 | 1.184 | 2.290 | 1.083 |
| RANF | MeasuredDomainERB | 1.210 | 1.063 | 0.147 | 0.070 |
| MCAR v3.2 | Contralateral25ERB | 1.693 | 1.350 | 0.343 | 0.139 |
| FSP-AE | Contralateral25ERB | 2.876 | 1.909 | 0.967 | 0.445 |
| RANF | Contralateral25ERB | 1.575 | 1.557 | 0.017 | 0.008 |
| MCAR v3.2 | ContralateralHemisphereHighFrequency | 5.489 | 3.590 | 1.898 | 0.902 |
| FSP-AE | ContralateralHemisphereHighFrequency | 5.382 | 3.164 | 2.218 | 1.051 |
| RANF | ContralateralHemisphereHighFrequency | 3.997 | 3.470 | 0.527 | 0.250 |
| MCAR v3.2 | HorizontalILDMAE | 1.533 | 0.660 | 0.873 | 0.420 |
| FSP-AE | HorizontalILDMAE | 2.385 | 0.759 | 1.627 | 0.738 |
| RANF | HorizontalILDMAE | 0.998 | 0.775 | 0.222 | 0.109 |

## Paired bootstrap

Differences are Method A minus Method B; negative favors A. See `paired_bootstrap_comparisons.csv` for all 24 contrasts.

## Validation and cost

- Expected table sizes were obtained: 1,584 listener-level metric rows, 36
  aggregate rows, 12 robustness rows, and 24 paired contrasts.
- All 132 listener/Q quality rows use exactly 767 evaluation targets. FSP-AE's
  frequency grid has zero recorded mismatch; RANF's grid mismatch is zero and
  its maximum observed-direction HRIR discrepancy is 1.49e-8.
- The Q26 RANF and FSP-AE metric rows reproduce their earlier formal-test rows
  to within 8.5e-8 dB. MCAR input MCA values are exactly identical and the
  frozen residual output differs by at most 0.00968 dB under mixed-precision
  inference; downstream strict metrics are therefore treated as a fresh
  audited evaluation rather than claimed as bitwise reproduction.
- RANF adapts 537,600 parameters for 1,000 epochs at each Q. Recorded
  adaptation times were 3,312 s (Q6), 7,198 s (Q14), and 12,335 s (Q26).

Machine-readable details are in `summary.json`,
`artifact_validation_report.json`, and `ranf_adaptation_cost.json`.
