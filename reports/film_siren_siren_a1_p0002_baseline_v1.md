# SIREN A1 P0002 baseline v1

## Outcome

The plain coordinate SIREN trained successfully on the complete residual field
of SONICOM train subject P0002. This run is an all-direction representation
smoke baseline, not a held-out-direction or cross-subject result.

## Frozen setup

- Experiment: `SIREN-A1-P0002-LINEAR-W256-D6-O30-V1`
- Subject: P0002, train split
- Query: `[x,y,z,f_linear]`
- Frequency mapping: 86.1328125-19982.8125 Hz to `[-1,1]`
- Backbone: width 256, six sine layers, first/hidden omega 30
- Output: two normalized residual channels, converted to dB for metrics
- Parameters: 330754
- Optimizer: Adam, learning rate 1e-4, no weight decay
- Budget: 20 epochs x 100 steps, 16 directions per step, full 463-bin spectra
- Precision: FP32
- Seed: 20260818
- Test subjects read: 0

## Results

| Metric | Initialization | Best epoch 20 |
|---|---:|---:|
| Residual MAE | 3.3205 dB | 1.1645 dB |
| Residual RMSE | 5.0429 dB | 1.8520 dB |
| MAE above 8 kHz | 4.1844 dB | 1.3100 dB |
| MAE above 10 kHz | 4.2974 dB | 1.3366 dB |
| First spectral difference | 0.5305 dB/bin | 0.4203 dB/bin |
| Second spectral difference | 0.2551 dB/bin^2 | 0.2647 dB/bin^2 |
| Notch-depth metric | 0.7317 dB | 0.3894 dB |

- Elapsed training/evaluation time: 9.39 s
- Peak CUDA allocated memory: 121.95 MiB
- Full 793x463 subject inference: 30.67 ms
- NaN/Inf: no
- Best checkpoint SHA-256:
  `E3DB9F8F80684436848733D45F596C4F1C9759500436599220FFB63FF14072F1C`

## Decision

`RETEST`

The implementation is numerically healthy and clearly learns the residual
field, but the best checkpoint is the final epoch and the RMSE curve is still
decreasing. A longer fixed-budget run is needed before this can be treated as
the single-subject representation ceiling. The slight worsening of the second
spectral-difference metric also shows that residual MSE alone does not guarantee
every spectral-shape metric improves.
