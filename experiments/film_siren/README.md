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

- Stage A1 v1 completed on train subject P0002.
- The 20-epoch smoke baseline was numerically stable and reduced full-field
  residual RMSE from 5.0429 dB to 1.8520 dB.
- Decision: `RETEST`, because the best checkpoint was the final epoch and the
  representation ceiling has not yet been reached.
- Detailed result: [`../../reports/film_siren_siren_a1_p0002_baseline_v1.md`](../../reports/film_siren_siren_a1_p0002_baseline_v1.md).
