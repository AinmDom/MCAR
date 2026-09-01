# Bounded E25 input-direction sensitivity protocol

Status: **PRE-REGISTERED VALIDATION-ONLY CHARACTERIZATION**.  This protocol is
frozen before any Q14 or Q50 Bounded-E25 prediction or metric is computed.

## Model decision

The three-member formal Bounded E25 ensemble with identity
`72B7319F8334307664A02BC6369FBD9EECE1D22383C402EC9683D1F6F05F2705` is
the paper main model.  Its three cycle-25 checkpoints, equal residual-dB
weights, gate, FiLM-SIREN, and two MCAR components remain frozen.  This study
does not select another checkpoint, retrain a component, tune an ensemble
weight, or change the main-model decision.

## Direction perturbations

- Q26 is the paper model's original input grid and is rerun as the center
  reference.
- Q14 is the already frozen nested Q14 subset from the earlier SONICOM
  sparsity study and is the single sparser perturbation.
- Q50 is the single denser perturbation.  It contains all Q26 directions and
  adds twelve left/right reflection pairs.  The pairs are selected using
  geometry only, before model inference: maximize minimum great-circle
  separation first, use the regularized order-3 real-SH Gram log determinant
  second, and source index as the deterministic final tie-break.

The resulting relation is exactly `Q14 subset Q26 subset Q50`.  All three
levels use the same 44 validation listeners.  Test paths are forbidden and
`test_subject_count_read` must remain zero.

## Frozen-model use at each Q

This is an input-sensitivity experiment, not Q-specific retraining.  At every
level:

1. recompute the order-3 MCA interpolation and correction filter from only
   that level's measured directions;
2. supply only those same measured magnitudes and coordinates to the frozen
   permutation-invariant FiLM condition encoder;
3. retain the train-only per-ear/per-frequency normalization, frozen MCAR
   residual models, frozen FiLM-SIREN, frozen bounded gates, and the formal
   one-third ensemble weights;
4. predict all 793 reference directions without parameter updates.

The Q26 branch must reproduce the already frozen formal validation residuals
within `1e-5 dB` maximum absolute error before Q14/Q50 results are accepted.

## Evaluation and statistics

One common target mask is used at all levels: exclude the complete Q50 input
set, leaving 743 directions.  This prevents the denser system from being
credited for evaluated observations and keeps every within-listener contrast
paired on identical targets.  The four frozen primary endpoints remain:

- full-sphere solid-angle-weighted ERB error;
- contralateral 25-degree ERB error;
- contralateral-hemisphere 10--20 kHz magnitude error;
- horizontal-plane strict HRIR ILD MAE.

Report each Q mean and sample standard deviation.  The two planned contrasts
are `Q14 - Q26` and `Q50 - Q26`; positive values mean higher error than the
original Q26 model.  Use 10,000 paired-listener percentile-bootstrap
replicates with seed `20260901`.  Results characterize robustness only and
cannot trigger model, grid, mask, or hyperparameter changes.
