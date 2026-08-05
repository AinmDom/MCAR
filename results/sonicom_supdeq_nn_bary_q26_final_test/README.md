# SONICOM Q26 interpolation baselines

- Split: `test`
- Subjects: 44
- Sparse grid: `SONICOM-Q26-v1`
- Evaluation directions: 767 (Q26 inputs excluded)
- SH Tikhonov epsilon: 0.01
- MCAR prediction: `sonicom_q26_test_mlp_cnn_v32_locked_ild075`

| Method | Full ERB | Contra25 ERB | Contra HF | Horizontal ILD |
|---|---:|---:|---:|---:|
| SH only | 2.685 +/- 0.107 | 3.835 +/- 0.448 | 9.067 +/- 0.824 | 3.551 +/- 0.664 |
| SUpDEq + SH | 1.882 +/- 0.506 | 2.228 +/- 0.195 | 5.994 +/- 0.344 | 2.018 +/- 1.564 |
| SUpDEq + Natural Neighbor | 1.852 +/- 0.290 | 2.241 +/- 0.212 | 5.595 +/- 0.258 | 1.637 +/- 0.469 |
| SUpDEq + Barycentric | 1.752 +/- 0.257 | 2.182 +/- 0.193 | 5.476 +/- 0.217 | 1.615 +/- 0.442 |
| MCA | 1.082 +/- 0.096 | 1.746 +/- 0.155 | 4.699 +/- 0.266 | 0.829 +/- 0.158 |
| MCAR v3.2 | 0.868 +/- 0.162 | 1.365 +/- 0.167 | 3.612 +/- 0.277 | 0.687 +/- 0.274 |

All entries are mean +/- subject standard deviation in dB; lower is better.

## Reproducibility checks

Natural Neighbor and Barycentric use the upstream SUpDEq geometry routines. The first subject was also evaluated through the native `supdeq_interpHRTF` entry point.

- Natural Neighbor maximum complex-spectrum parity error: 4.97e-16
- Barycentric maximum complex-spectrum parity error: 4.58e-16
