# SONICOM Q26 eight-method test comparison

- Subjects: 44 locked test subjects
- Input/evaluation directions: 26 / 767 (inputs excluded)
- MCAR paper model: v3.2 epoch 39
- FSP-AE: frozen Q26 adaptation, epoch 40
- Merge-time test/model-prediction reads: 0
- Common-method maximum source difference: 0 dB

| Method | Measured ERB | Contra25 ERB | Contra HF | Horizontal ILD |
|---|---:|---:|---:|---:|
| SH only | 2.685 +/- 0.107 | 3.835 +/- 0.448 | 9.067 +/- 0.824 | 3.551 +/- 0.664 |
| SUpDEq + SH | 1.882 +/- 0.506 | 2.228 +/- 0.195 | 5.994 +/- 0.344 | 2.018 +/- 1.564 |
| SUpDEq + Natural Neighbor | 1.852 +/- 0.290 | 2.241 +/- 0.212 | 5.595 +/- 0.258 | 1.637 +/- 0.469 |
| SUpDEq + Barycentric | 1.752 +/- 0.257 | 2.182 +/- 0.193 | 5.476 +/- 0.217 | 1.615 +/- 0.442 |
| MCA | 1.082 +/- 0.096 | 1.746 +/- 0.155 | 4.699 +/- 0.266 | 0.829 +/- 0.158 |
| RANF | 1.063 +/- 0.126 | 1.557 +/- 0.164 | 3.470 +/- 0.254 | 0.775 +/- 0.189 |
| MCAR v3.2 epoch 39 | 0.856 +/- 0.160 | 1.350 +/- 0.179 | 3.590 +/- 0.275 | 0.660 +/- 0.250 |
| FSP-AE | 1.184 +/- 0.349 | 1.909 +/- 0.701 | 3.164 +/- 0.685 | 0.759 +/- 1.040 |

Values are mean +/- subject standard deviation in dB; lower is better.

## FSP-AE versus MCAR v3.2 epoch 39

- MeasuredDomainERB: FSP-AE minus MCAR 0.3288 dB, 95% bootstrap CI [0.2843, 0.3979] dB; MCAR better for 44/44 subjects.
- Contralateral25ERB: FSP-AE minus MCAR 0.5589 dB, 95% bootstrap CI [0.4574, 0.7350] dB; MCAR better for 44/44 subjects.
- ContralateralHemisphereHF: FSP-AE minus MCAR -0.4261 dB, 95% bootstrap CI [-0.5396, -0.2412] dB; MCAR better for 1/44 subjects.
- HorizontalILDMAE: FSP-AE minus MCAR 0.0984 dB, 95% bootstrap CI [-0.0579, 0.3704] dB; MCAR better for 17/44 subjects.

FSP-AE is the best method on contralateral-hemisphere high-frequency error. MCAR is best on the other three reported metrics.
