# Stage D formal E190 FiLM-SIREN spectral-CNN freeze

Status: **FROZEN BEFORE FORMAL TRAINING**  
Date: 2026-08-28  
Data boundary: SONICOM train + validation only; test access is forbidden.

## Development selection

The pre-registered E200 runs selected best cycles `190/190/170` for seeds
`20260821/20260822/20260823`. None selected the cycle-200 endpoint. The frozen
rule therefore gives `E_final = median(190,190,170) = 190`.

At the common cycle 190, the three-seed means are objective `0.722358788956295`,
residual MAE `2.47966014255177 dB`, ERB MAE `0.962147936224937 dB`,
contralateral HF MAE `3.54623398275086 dB`, and strict ILD MAE
`0.648820926971508 dB`. These are development metrics, not final comparative
claims.

## Frozen formal members

- Seeds: `20260821/20260822/20260823`.
- Each member starts from scratch with a zero-initialized spectral CNN and its
  same-seed corrected E130 FiLM-SIREN parent; all FiLM-SIREN parameters remain
  frozen.
- Architecture, data, normalization, sampling, AdamW `1e-4/1e-4`, objective,
  two-cycle warmup, precision, and clipping are identical to D2.
- Stop at exactly 190 cycles while retaining the D2 warmup-cosine horizon 200.
- `formal_fixed_cycle=true`; `checkpoint_policy=fixed_stop_cycle_last`.
- The only authoritative checkpoint is cycle-190 `last.pt`; validation-best
  checkpoints and cycles are diagnostic and may not replace it.

## Frozen ensemble and comparison

After all three formal members pass integrity checks, combine their cycle-190
residual-dB predictions with fixed equal weights `1/3,1/3,1/3`. On the same 44
validation subjects, compare the hybrid ensemble with the corrected E130
FiLM-SIREN ensemble and MCAR v3.5.1 using the already frozen Full ERB,
contralateral-25 ERB, contralateral HF, and horizontal ILD metrics. Paired
differences are hybrid minus baseline; report 10,000 subject-level bootstrap
replicates and win/tie/loss counts. No ensemble weighting or model selection is
allowed after this comparison.

## Hard boundaries

- Do not construct or read SONICOM test paths.
- Do not initialize formal members from E40/E200 trained CNN checkpoints.
- Do not alter architecture, losses, cycle count, scheduler horizon, or weights.
- This validation comparison does not authorize a new engineering test.
