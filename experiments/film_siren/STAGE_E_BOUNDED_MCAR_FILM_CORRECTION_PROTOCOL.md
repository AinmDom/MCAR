# Stage E: frozen MCAR v3.5.1 + bounded FiLM-SIREN correction

Status: **PRE-REGISTERED D1 STRUCTURE SCREEN**.  This protocol is frozen before
the first optimization step and uses train/validation subjects only.

## Hypothesis and fixed model

The base prediction is the frozen MCAR v3.5.1 residual ensemble:

`r_mcar = 0.3 * r_previous_joint + 0.7 * r_v351b`.

The alternative field and its final sine features come from the frozen,
same-seed corrected FiLM-SIREN E130 checkpoint.  A new two-output linear head
maps those features to a query-dependent gate.  The prediction is

`r = r_mcar + 0.5 * tanh(g_theta) * (r_film - r_mcar)`.

The gate head weight and bias are exactly zero initialized.  Thus cycle-zero
output must equal MCAR v3.5.1 point-for-point (absolute tolerance 0), while the
signed correction fraction is bounded to `[-0.5, 0.5]`.  All MCAR and
FiLM-SIREN parameters remain frozen; only the 514 gate parameters train.

## D1 budget and objective

- Seed: `20260821`; 40 cycles; 262 train steps per cycle.
- AdamW, learning rate `1e-3`, weight decay `0`, two-cycle warmup followed by
  cosine decay; gradient clipping 5.
- Sampling and validation: 16 global plus 16 independent horizontal
  interpolation directions per train subject; all 767 interpolation directions
  and independent 72-direction horizontal pass every five cycles.
- Objective is frozen to the corrected D1/D2/notch Stage-C objective used by the
  current FiLM and Hybrid studies: ERB `0.75`, HF `0.25`, strict HRIR ILD
  `0.75`, band ILD `0.05`, first/second HF differences `0.25/0.15`, notch depth
  `0.30`.
- If the best validation point is cycle 40, label `RETEST`; do not promote it
  or read test. Otherwise label the budget complete and run strict 44-subject
  validation reconstruction.

## Result-before-seeing comparison

The strict validation table will contain bounded correction, unchanged MCAR
v3.5.1, formal Hybrid E190, and corrected FiLM E130 ensemble.  Metrics are the
already frozen Full ERB, Contra25 ERB, Contra HF, and horizontal HRIR ILD with
10,000 subject-paired bootstrap resamples.  D1 advances only as a structure
candidate if it improves MCAR in at least two metrics and has no mean regression
larger than `0.02 dB`; final paper promotion still requires matched seeds and a
fixed-cycle ensemble.  Dominance over both MCAR and Hybrid is reported only if
all four corresponding mean differences are negative.

## Data boundary

Only the 262 train and 44 validation rows may be resolved.  Test paths must not
be constructed or opened.  Every configuration, checkpoint, report, and
evaluation must record `test_subjects_read=0`.  Any test access requires a new
pre-registration and explicit user authorization.
