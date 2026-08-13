# SONICOM eight-method sparsity experiment

Five horizontal baselines are combined with the unchanged MCAR v3.2, FSP-AE, and RANF listener-level results.
All methods use the same 44 test listeners, Q6/Q14/Q26 grids, and 767-target mask.

## Main findings

- RANF ranks first on all four Q6 endpoints. Its Q6 means are 1.210 / 1.575 / 3.997 / 0.998 dB in the metric order used below.
- MCAR v3.2 is significantly better than the strongest Q6 horizontal baseline for every metric. MCAR-minus-baseline differences are -0.330 / -0.450 / -0.576 / -0.281 dB (all paired-bootstrap p=0.0002).
- FSP-AE beats the strongest Q6 baseline only for contralateral-hemisphere high-frequency error; its FSP-AE-minus-baseline differences are 1.252 / 0.733 / -0.683 / 0.571 dB.
- RANF also degrades less from Q26 to Q6 than MCAR on all four metrics. A flat or negative curve is not treated as robust by itself when its absolute endpoint error remains high (notably SH only).

## Robustness summary

| Method | Metric | Q6 | Q26 | Q6-Q26 | Slope / halving |
|---|---|---:|---:|---:|---:|
| SH only | MeasuredDomainERB | 3.268 | 2.685 | 0.583 | 0.283 |
| SH only | Contralateral25ERB | 2.686 | 3.835 | -1.149 | -0.636 |
| SH only | ContralateralHemisphereHighFrequency | 6.691 | 9.067 | -2.376 | -1.225 |
| SH only | HorizontalILDMAE | 2.292 | 3.551 | -1.259 | -0.646 |
| SUpDEq + SH | MeasuredDomainERB | 2.662 | 1.882 | 0.780 | 0.384 |
| SUpDEq + SH | Contralateral25ERB | 2.480 | 2.228 | 0.252 | 0.102 |
| SUpDEq + SH | ContralateralHemisphereHighFrequency | 6.142 | 5.994 | 0.149 | 0.057 |
| SUpDEq + SH | HorizontalILDMAE | 2.391 | 2.018 | 0.373 | 0.195 |
| SUpDEq + Natural Neighbor | MeasuredDomainERB | 2.486 | 1.852 | 0.634 | 0.303 |
| SUpDEq + Natural Neighbor | Contralateral25ERB | 2.686 | 2.241 | 0.445 | 0.205 |
| SUpDEq + Natural Neighbor | ContralateralHemisphereHighFrequency | 6.065 | 5.595 | 0.470 | 0.220 |
| SUpDEq + Natural Neighbor | HorizontalILDMAE | 2.276 | 1.637 | 0.639 | 0.320 |
| SUpDEq + Barycentric | MeasuredDomainERB | 2.592 | 1.752 | 0.839 | 0.402 |
| SUpDEq + Barycentric | Contralateral25ERB | 2.808 | 2.182 | 0.626 | 0.292 |
| SUpDEq + Barycentric | ContralateralHemisphereHighFrequency | 6.072 | 5.476 | 0.597 | 0.282 |
| SUpDEq + Barycentric | HorizontalILDMAE | 2.100 | 1.615 | 0.485 | 0.246 |
| MCA | MeasuredDomainERB | 2.223 | 1.082 | 1.141 | 0.548 |
| MCA | Contralateral25ERB | 2.143 | 1.746 | 0.397 | 0.156 |
| MCA | ContralateralHemisphereHighFrequency | 6.769 | 4.699 | 2.070 | 0.995 |
| MCA | HorizontalILDMAE | 1.814 | 0.829 | 0.985 | 0.475 |
| MCAR v3.2 | MeasuredDomainERB | 1.893 | 0.856 | 1.037 | 0.495 |
| MCAR v3.2 | Contralateral25ERB | 1.693 | 1.350 | 0.343 | 0.139 |
| MCAR v3.2 | ContralateralHemisphereHighFrequency | 5.489 | 3.590 | 1.898 | 0.902 |
| MCAR v3.2 | HorizontalILDMAE | 1.533 | 0.660 | 0.873 | 0.420 |
| FSP-AE | MeasuredDomainERB | 3.475 | 1.184 | 2.290 | 1.083 |
| FSP-AE | Contralateral25ERB | 2.876 | 1.909 | 0.967 | 0.445 |
| FSP-AE | ContralateralHemisphereHighFrequency | 5.382 | 3.164 | 2.218 | 1.051 |
| FSP-AE | HorizontalILDMAE | 2.385 | 0.759 | 1.627 | 0.738 |
| RANF | MeasuredDomainERB | 1.210 | 1.063 | 0.147 | 0.070 |
| RANF | Contralateral25ERB | 1.575 | 1.557 | 0.017 | 0.008 |
| RANF | ContralateralHemisphereHighFrequency | 3.997 | 3.470 | 0.527 | 0.250 |
| RANF | HorizontalILDMAE | 0.998 | 0.775 | 0.222 | 0.109 |

## Paired comparisons

`paired_bootstrap_all_pairs.csv` contains 224 contrasts. `paired_bootstrap_learned_vs_baselines.csv` contains the 120 learned-versus-baseline contrasts.
Differences are Method A minus Method B; negative favors Method A.
