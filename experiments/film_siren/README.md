# FiLM-SIREN experiments

This directory contains the canonical protocol and run notes for replacing the
MCAR residual model with a coordinate-based SIREN and a Q26-conditioned
FiLM-SIREN.

- [`EXPERIMENT_CHECKLIST.md`](EXPERIMENT_CHECKLIST.md): frozen experiment and
  data-access rules.
- [`TRAINING.md`](TRAINING.md): current runnable entry points and experiment
  status.

Strict reconstruction metrics remain owned by the shared MCAR evaluation
pipeline. Model-specific code produces residual predictions only.

## Current status

- Stage A1 single-subject representation testing is complete on train subject
  P0002.
- The 100-epoch plain-SIREN run reached its best full-field residual RMSE of
  `1.4987 dB` at epoch 98, compared with `1.8520 dB` in the earlier 20-epoch
  smoke run.
- The final ten epochs reduced RMSE by only `0.9248%`, indicating a practical
  optimization plateau.
- Residual MAE, high-frequency MAE, first spectral difference and notch-depth
  error all improved substantially; second spectral difference remained a
  diagnostic weakness.
- Decision: `KEEP — A1 COMPLETE`.
- Next stage: joint frequency-coordinate × first-omega screening across a fixed
  multi-subject train set.