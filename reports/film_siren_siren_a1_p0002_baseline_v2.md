# SIREN A1 P0002 baseline v2 — 100-epoch representation ceiling

## Outcome

The plain coordinate SIREN was retrained from scratch for 100 epochs on the
complete residual field of SONICOM train subject P0002.

This experiment extends the 20-epoch A1-v1 smoke baseline without changing
the architecture, frequency representation, optimizer, sampling protocol, or
random seed. Its purpose is to determine whether the single-subject
representation error has reached a practical optimization plateau.

The run remained numerically stable and reached its best full-field residual
RMSE at epoch 98. Because the best checkpoint occurred before the end of the
training budget and the final ten epochs showed only a 0.9248% RMSE reduction,
Stage A1 is considered complete.

## Frozen setup

- Experiment: `SIREN-A1-P0002-LINEAR-W256-D6-O30-V2`
- Model version: `sonicom-siren-a1-v2`
- Subject: P0002, train split
- Query: `[x,y,z,f_linear]`
- Frequency mapping: 86.1328125–19982.8125 Hz to `[-1,1]`
- Backbone: width 256, six sine layers
- First omega: 30
- Hidden omega: 30
- Output: two normalized binaural residual channels
- Parameters: 330754
- Optimizer: Adam
- Learning rate: 1e-4
- Weight decay: 0
- Budget: 100 epochs × 100 steps
- Directions per step: 16
- Frequency bins: 463
- Precision: FP32
- Seed: 20260818
- Test subjects read: 0

## Results

| Metric | Initialization | A1-v1 epoch 20 | A1-v2 best |
|---|---:|---:|---:|
| Residual MAE | 3.3205 dB | 1.1645 dB | **0.9365 dB** |
| Residual RMSE | 5.0429 dB | 1.8520 dB | **1.4987 dB** |
| MAE above 8 kHz | 4.1844 dB | 1.3100 dB | **1.0415 dB** |
| MAE above 10 kHz | 4.2974 dB | 1.3366 dB | **1.0632 dB** |
| First spectral difference | 0.5305 dB/bin | 0.4203 dB/bin | **0.3827 dB/bin** |
| Second spectral difference | 0.2551 dB/bin² | 0.2647 dB/bin² | **0.2709 dB/bin²** |
| Notch-depth metric | 0.7317 dB | 0.3894 dB | **0.3181 dB** |

Relative to the 20-epoch A1-v1 result, the 100-epoch run improved:

- residual MAE by approximately 19.6%;
- residual RMSE by approximately 19.1%;
- MAE above 8 kHz by approximately 20.5%;
- MAE above 10 kHz by approximately 20.5%;
- first spectral-difference MAE by approximately 8.9%;
- notch-depth MAE by approximately 18.3%.

The second spectral-difference metric worsened by approximately 2.4% relative
to A1-v1, despite improvement in the main residual and notch metrics.

## Convergence assessment

- Best epoch: `98`
- Best RMSE: `1.498697 dB`
- Epoch 91 RMSE: `1.525809 dB`
- Epoch 100 RMSE: `1.511698 dB`
- Relative RMSE decrease from epoch 91 to epoch 100: `0.9248%`
- NaN/Inf: `no`

The final ten epochs fluctuate around approximately 1.50–1.53 dB rather than
showing a sustained steep downward trend. Together with the best checkpoint
occurring at epoch 98 rather than at the final epoch, this indicates that the
current optimization has reached a practical plateau.

The remaining second-order spectral-difference degradation suggests that
minimizing residual MSE alone does not guarantee simultaneous improvement of
all high-order spectral-shape metrics. This observation will be retained as a
diagnostic for later frequency-coordinate, omega, and spectral-loss
experiments rather than addressed during Stage A1.

## Efficiency

- Elapsed training/evaluation time: `48.04 s`
- Peak CUDA allocated memory: `121.95 MiB`
- Full 793×463 subject inference: `31.28 ms`
- Best checkpoint SHA-256:
  `FCFF8B909D3040E44720F59C98EEC8AD5EEE6094F92899E2EE41C00317EBEA20`

## Decision

`KEEP — A1 COMPLETE`

The plain SIREN implementation is numerically stable and demonstrates strong
capacity to represent a single SONICOM subject's complete MCA residual field.
Further optimization of P0002 is unlikely to be the most informative use of
experimental budget.

The next stage should therefore move from single-subject capacity testing to
multi-subject backbone selection. Frequency coordinate and first-layer omega
should be selected jointly across a fixed set of train subjects rather than
further tuned on P0002.

## Limitations

A1 is an all-direction single-subject representation experiment. It does not
measure cross-subject generalization, sparse-condition generalization, or
held-out-direction interpolation. The current linear-frequency / omega-30
configuration must therefore not be treated as the final SIREN backbone solely
from this result.