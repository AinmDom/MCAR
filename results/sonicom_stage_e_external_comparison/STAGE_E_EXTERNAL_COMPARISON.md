# Stage E external horizontal comparison

Candidate: frozen Stage E Bounded MCAR + FiLM-SIREN correction E25 1/3 ensemble.

## Evidence boundary

- Primary four metrics use the 44-subject frozen engineering test. The project test has historical consumption, so this is not presented as a study-wide untouched confirmation.
- Secondary spectral/ILD metrics use the 44-subject validation split and are supplementary.
- Deferred ITD and dominant-notch endpoints use validation; dominant-notch results are exploratory.
- Every difference is Stage E minus baseline; lower is better. No data HDF5 were read while generating this report.

## Primary frozen-test comparison

| Baseline | Metric | Stage E | Baseline | Difference | 95% CI | Wins | Decision | Rank/4 |
|---|---|---:|---:|---:|---:|---:|---|---:|
| MCAR v3.5.1 | FullSphereERB | 0.794131 | 0.817937 | -0.0238056 | [-0.0362775, -0.00162666] | 42/44 | Significantly better | 1 |
| MCAR v3.5.1 | Contralateral25ERB | 1.20152 | 1.27449 | -0.0729715 | [-0.0905437, -0.0594868] | 44/44 | Significantly better | 1 |
| MCAR v3.5.1 | ContralateralHighFrequency | 3.47658 | 3.50848 | -0.0319009 | [-0.0470894, -0.0170815] | 34/44 | Significantly better | 3 |
| MCAR v3.5.1 | HorizontalILDMAE | 0.674382 | 0.644709 | +0.0296724 | [-0.0248789, +0.121017] | 25/44 | Comparable | 2 |
| RANF | FullSphereERB | 0.794131 | 1.06324 | -0.269104 | [-0.305559, -0.211412] | 43/44 | Significantly better | 1 |
| RANF | Contralateral25ERB | 1.20152 | 1.55732 | -0.355805 | [-0.408422, -0.29752] | 43/44 | Significantly better | 1 |
| RANF | ContralateralHighFrequency | 3.47658 | 3.46968 | +0.00689733 | [-0.0465485, +0.0684963] | 21/44 | Comparable | 3 |
| RANF | HorizontalILDMAE | 0.674382 | 0.775463 | -0.101081 | [-0.203766, +0.0464121] | 38/44 | Comparable | 2 |
| FSP-AE | FullSphereERB | 0.794131 | 1.18449 | -0.390361 | [-0.433227, -0.358287] | 44/44 | Significantly better | 1 |
| FSP-AE | Contralateral25ERB | 1.20152 | 1.90887 | -0.707355 | [-0.907968, -0.593464] | 44/44 | Significantly better | 1 |
| FSP-AE | ContralateralHighFrequency | 3.47658 | 3.16435 | +0.312225 | [+0.130787, +0.421416] | 1/44 | Significantly worse | 3 |
| FSP-AE | HorizontalILDMAE | 0.674382 | 0.758668 | -0.0842864 | [-0.27853, +0.0304075] | 20/44 | Comparable | 2 |

## Validation supplementary comparison

| Baseline | Metric | Stage E | Baseline | Difference | 95% CI | Wins | Decision | Rank/4 |
|---|---|---:|---:|---:|---:|---:|---|---:|
| MCAR v3.5.1 | FullSphereLSD | 3.58744 | 3.60976 | -0.0223163 | [-0.0372003, -0.00721029] | 31/44 | Significantly better | 3 |
| MCAR v3.5.1 | HFFirstDifferenceMAE | 0.427257 | 0.429587 | -0.00233004 | [-0.00373847, -0.000647351] | 34/44 | Significantly better | 3 |
| MCAR v3.5.1 | HFSecondDifferenceMAE | 0.239463 | 0.25381 | -0.0143468 | [-0.0162102, -0.0125567] | 43/44 | Significantly better | 3 |
| MCAR v3.5.1 | MultiScaleNotchDepthMAE | 0.507 | 0.500104 | +0.00689621 | [+0.00542739, +0.00863487] | 1/44 | Significantly worse | 4 |
| MCAR v3.5.1 | ERBBandILDMean | 1.49182 | 1.56254 | -0.0707167 | [-0.0850488, -0.0563518] | 40/44 | Significantly better | 1 |
| RANF | FullSphereLSD | 3.58744 | 3.36145 | +0.225994 | [+0.196767, +0.257059] | 0/44 | Significantly worse | 3 |
| RANF | HFFirstDifferenceMAE | 0.427257 | 0.408586 | +0.0186709 | [+0.0156679, +0.0217798] | 2/44 | Significantly worse | 3 |
| RANF | HFSecondDifferenceMAE | 0.239463 | 0.219457 | +0.0200061 | [+0.0162744, +0.023889] | 3/44 | Significantly worse | 3 |
| RANF | MultiScaleNotchDepthMAE | 0.507 | 0.491396 | +0.0156038 | [+0.0129434, +0.0183744] | 2/44 | Significantly worse | 4 |
| RANF | ERBBandILDMean | 1.49182 | 1.72281 | -0.230985 | [-0.274218, -0.187488] | 41/44 | Significantly better | 1 |
| FSP-AE | FullSphereLSD | 3.58744 | 3.07003 | +0.51741 | [+0.485385, +0.550795] | 0/44 | Significantly worse | 3 |
| FSP-AE | HFFirstDifferenceMAE | 0.427257 | 0.410575 | +0.0166823 | [+0.0144715, +0.0189773] | 0/44 | Significantly worse | 3 |
| FSP-AE | HFSecondDifferenceMAE | 0.239463 | 0.218813 | +0.0206505 | [+0.0189567, +0.0224485] | 0/44 | Significantly worse | 3 |
| FSP-AE | MultiScaleNotchDepthMAE | 0.507 | 0.503355 | +0.00364514 | [-0.000242088, +0.00778298] | 19/44 | Comparable | 4 |
| FSP-AE | ERBBandILDMean | 1.49182 | 1.59911 | -0.10729 | [-0.136359, -0.0777584] | 38/44 | Significantly better | 1 |

## Exploratory/deferred validation comparison

| Baseline | Metric | Stage E | Baseline | Difference | 95% CI | Wins | Decision | Rank/4 |
|---|---|---:|---:|---:|---:|---:|---|---:|
| MCAR v3.5.1 | DominantNotchPenalizedMAE_Hz | 962.414 | 949.249 | +13.1644 | [+6.87731, +19.6131] | 12/44 | Significantly worse | 4 |
| MCAR v3.5.1 | DominantNotchMatchedMAE_Hz | 509.514 | 502.035 | +7.479 | [+0.0787834, +14.9961] | 19/44 | Significantly worse | 4 |
| MCAR v3.5.1 | DominantNotchMissRate | 0.458053 | 0.448618 | +0.00943537 | [+0.0049568, +0.0139021] | 9/44 | Significantly worse | 4 |
| MCAR v3.5.1 | DominantNotchSpuriousRate | 0.0227273 | 0.0227273 | +0 | [+0, +0] | 0/44 | Comparable | 1 |
| MCAR v3.5.1 | ITDWeightedMAE_us | 15.5208 | 15.5203 | +0.000571201 | [-0.0258628, +0.0278494] | 20/44 | Comparable | 2 |
| MCAR v3.5.1 | ITDMaximumAbsoluteError_us | 130.208 | 129.972 | +0.236743 | [-0.355113, +0.887785] | 10/44 | Comparable | 3 |
| RANF | DominantNotchPenalizedMAE_Hz | 962.414 | 859.001 | +103.413 | [+88.2553, +118.552] | 0/44 | Significantly worse | 4 |
| RANF | DominantNotchMatchedMAE_Hz | 509.514 | 432.688 | +76.826 | [+63.5014, +90.1103] | 3/44 | Significantly worse | 4 |
| RANF | DominantNotchMissRate | 0.458053 | 0.400214 | +0.0578387 | [+0.0479993, +0.0673258] | 3/44 | Significantly worse | 4 |
| RANF | DominantNotchSpuriousRate | 0.0227273 | 0.0227273 | +0 | [+0, +0] | 0/44 | Comparable | 1 |
| RANF | ITDWeightedMAE_us | 15.5208 | 17.5894 | -2.06852 | [-2.90975, -1.23145] | 33/44 | Significantly better | 2 |
| RANF | ITDMaximumAbsoluteError_us | 130.208 | 155.599 | -25.3906 | [-39.8926, -12.429] | 30/44 | Significantly better | 3 |
| FSP-AE | DominantNotchPenalizedMAE_Hz | 962.414 | 829.641 | +132.772 | [+119.63, +146.516] | 0/44 | Significantly worse | 4 |
| FSP-AE | DominantNotchMatchedMAE_Hz | 509.514 | 417.504 | +92.01 | [+78.1358, +106.472] | 1/44 | Significantly worse | 4 |
| FSP-AE | DominantNotchMissRate | 0.458053 | 0.380177 | +0.0778757 | [+0.0684452, +0.0877555] | 0/44 | Significantly worse | 4 |
| FSP-AE | DominantNotchSpuriousRate | 0.0227273 | 0.0227273 | +0 | [+0, +0] | 0/44 | Comparable | 1 |
| FSP-AE | ITDWeightedMAE_us | 15.5208 | 15.5439 | -0.0230783 | [-1.07764, +0.949463] | 20/44 | Comparable | 2 |
| FSP-AE | ITDMaximumAbsoluteError_us | 130.208 | 104.403 | +25.8049 | [+13.7311, +37.1686] | 6/44 | Significantly worse | 3 |

## Comparability limitations

| Baseline | Category | Status |
|---|---|---|
| MCAR v3.5.1 | Mechanism | Not comparable: no FiLM gate |
| RANF | Mechanism | Not applicable: no common internal quantity |
| FSP-AE | Mechanism | Not applicable: no common internal quantity |
| MCAR v3.5.1 | Efficiency | Not available under the frozen same-hardware protocol |
| RANF | Efficiency | Not available under the frozen same-hardware protocol |
| FSP-AE | Efficiency | Not available under the frozen same-hardware protocol |
| MCAR v3.5.1 | Localization | Not run: immutable official dependency unavailable |
| RANF | Localization | Not run: immutable official dependency unavailable |
| FSP-AE | Localization | Not run: immutable official dependency unavailable |

## Paper-facing conclusion

Stage E is strongest on the two broad ERB endpoints and improves MCAR on the three primary spectral endpoints. Its primary horizontal ILD is statistically comparable to MCAR, RANF, and FSP-AE, with a slightly worse mean than MCAR. RANF and especially FSP-AE retain advantages in high-frequency spectral-detail metrics; FSP-AE is decisively better on primary contralateral HF. Stage E improves ERB-band ILD over all three baselines. ITD is comparable to MCAR and FSP-AE and better than RANF. Dominant-notch location metrics are exploratory and favor the external baselines over Stage E.
