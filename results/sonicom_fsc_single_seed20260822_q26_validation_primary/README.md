# SONICOM Q26 interpolation baselines

- Split: `val`
- Subjects: 44
- Sparse grid: `SONICOM-Q26-v1`
- Evaluation directions: 767 (Q26 inputs excluded)
- SH Tikhonov epsilon: 0.01
- MCAR prediction: `sonicom_fsc_q26_e190_single_seed20260822_validation`

| Method | Full ERB | Contra25 ERB | Contra HF | Horizontal ILD |
|---|---:|---:|---:|---:|
| SH only | 2.666 +/- 0.155 | 3.830 +/- 0.361 | 9.016 +/- 0.621 | 3.629 +/- 0.636 |
| SUpDEq + SH | 1.819 +/- 0.237 | 2.246 +/- 0.152 | 6.006 +/- 0.366 | 1.739 +/- 0.446 |
| SUpDEq + Natural Neighbor | 1.849 +/- 0.250 | 2.219 +/- 0.166 | 5.662 +/- 0.286 | 1.563 +/- 0.474 |
| SUpDEq + Barycentric | 1.742 +/- 0.220 | 2.162 +/- 0.159 | 5.527 +/- 0.271 | 1.533 +/- 0.440 |
| MCA | 1.096 +/- 0.145 | 1.764 +/- 0.177 | 4.749 +/- 0.234 | 0.830 +/- 0.181 |
| MCAR v3.2 | 0.826 +/- 0.161 | 1.241 +/- 0.188 | 3.550 +/- 0.288 | 0.641 +/- 0.178 |

All entries are mean +/- subject standard deviation in dB; lower is better.

## Reproducibility checks

Natural Neighbor and Barycentric use the upstream SUpDEq geometry routines. The first subject was also evaluated through the native `supdeq_interpHRTF` entry point.

- Natural Neighbor maximum complex-spectrum parity error: 4.97e-16
- Barycentric maximum complex-spectrum parity error: 4.97e-16
