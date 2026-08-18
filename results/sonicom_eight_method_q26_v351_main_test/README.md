# SONICOM Q26 eight-method comparison with MCAR v3.5.1

- Current engineering main model: MCAR v3.5.1
- Frozen test subjects: 44
- Result merge reads HRTFs/predictions/test subjects: 0
- The seven non-MCAR baseline rows are byte-for-value unchanged.

| Method | Full-sphere ERB | Contra-25 ERB | Contra HF | Horizontal ILD |
|---|---:|---:|---:|---:|
| MCAR v3.5.1 | 0.818 +/- 0.175 | 1.274 +/- 0.148 | 3.508 +/- 0.288 | 0.645 +/- 0.248 |
| FSP-AE | 1.184 +/- 0.349 | 1.909 +/- 0.701 | 3.164 +/- 0.685 | 0.759 +/- 1.040 |
| RANF | 1.063 +/- 0.126 | 1.557 +/- 0.164 | 3.470 +/- 0.254 | 0.775 +/- 0.189 |
| MCA | 1.082 +/- 0.096 | 1.746 +/- 0.155 | 4.699 +/- 0.266 | 0.829 +/- 0.158 |
| SUpDEq + Barycentric | 1.752 +/- 0.257 | 2.182 +/- 0.193 | 5.476 +/- 0.217 | 1.615 +/- 0.442 |
| SUpDEq + Natural Neighbor | 1.852 +/- 0.290 | 2.241 +/- 0.212 | 5.595 +/- 0.258 | 1.637 +/- 0.469 |
| SUpDEq + SH | 1.882 +/- 0.506 | 2.228 +/- 0.195 | 5.994 +/- 0.344 | 2.018 +/- 1.564 |
| SH only | 2.685 +/- 0.107 | 3.835 +/- 0.448 | 9.067 +/- 0.824 | 3.551 +/- 0.664 |

Values are mean +/- subject standard deviation in dB; lower is better.

MCAR v3.5.1 ranks 1/1/3/1 across the four metrics.
