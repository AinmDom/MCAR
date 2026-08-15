# Paired sparsity robustness conclusion

Negative contrasts favor MCAR v3.2. Confidence intervals are paired-subject
bootstrap intervals (10000 replicates, seed 20260812).

| Metric | MCAR Q6 | MCA Q6 | MCAR-MCA at Q6 | MCAR-MCA degradation contrast |
|---|---:|---:|---:|---:|
| FullSphereERB | 2.106 | 1.623 | 0.483 [0.444, 0.523] | -0.495 [-0.536, -0.453] |
| Contralateral25ERB | 2.319 | 2.382 | -0.064 [-0.112, -0.004] | -0.617 [-0.678, -0.552] |
| ContralateralHighFrequency | 5.207 | 5.690 | -0.483 [-0.655, -0.292] | -1.042 [-1.239, -0.850] |
| HorizontalILDMAE | 1.896 | 1.663 | 0.233 [0.057, 0.400] | 0.031 [-0.123, 0.181] |

MCAR has significantly less Q26-to-Q6 degradation than MCA on the three
spectral metrics. Its horizontal-ILD degradation is statistically
indistinguishable from MCA, while its absolute Q6 horizontal-ILD error is
higher. MCAR beats SH at Q6 and in degradation on all four metrics. The
direct generator has much higher Q6 error on all four metrics; its negative
ILD degradation is not evidence of useful robustness because its absolute
error is already high and decreases when fewer observations are supplied.
