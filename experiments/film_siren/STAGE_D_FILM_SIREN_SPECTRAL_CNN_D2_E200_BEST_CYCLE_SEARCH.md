# Stage D D2: matched-seed E200 best-cycle search

Status: **PRE-REGISTERED BEFORE D2 TRAINING**  
Date: 2026-08-28  
Data boundary: SONICOM train + validation only; test access is forbidden.

## Motivation and fixed evidence

The three matched D1 E40 runs all completed, and all three selected cycle 40.
Their cycle-35 to cycle-40 objective improvements were only
`0.00006146/0.00002247/0.00005811`, but the existing endpoint rule labels all
three runs `RETEST`. At the user's direction, D2 uses a single sufficiently wide
E200 search instead of an incremental E80 extension. D2 resolves only training
budget and best-cycle location; it is not a new architecture or objective search.

## Frozen D2 runs

- Seeds: `20260821/20260822/20260823`.
- Each run starts from scratch with the zero-initialized spectral CNN and the
  same-seed corrected E130 FiLM-SIREN parent, which remains frozen.
- Architecture, seven input channels, normalization, train/validation subjects,
  direction sampling, optimizer, loss weights, precision, gradient clipping,
  and validation interval are identical to D1.
- The only optimization changes are `cycles=200` and warmup-cosine
  `horizon_cycles=200`; warmup remains two cycles.
- Checkpoint selection remains the minimum shared D1/D2+notch objective over the
  44 validation subjects, evaluated every five cycles.

## Bounded decision rule

If none of the three best cycles is 200, freeze
`E_final = median(best_cycle_20260821, best_cycle_20260822, best_cycle_20260823)`;
all values are multiples of five. Then train three fixed-cycle hybrid members
and compare their equal-weight residual ensemble with the corrected FiLM
ensemble and MCAR v3.5.1 on validation.

If any best cycle is 200, label D2 `ENDPOINT_LIMITED` and do not recursively
extend or promote the hybrid for the current paper deadline. The fallback paper
route remains the already completed FiLM-SIREN/MCAR comparison. No architecture,
loss, seed, ensemble weight, or threshold may be changed after D2 results.

## Hard boundaries

- Do not construct or read SONICOM test paths.
- Do not initialize from the trained E40 CNN; D2 is a clean matched-budget run.
- Do not unfreeze FiLM-SIREN or alter the spectral CNN.
- D2 is development evidence, not manuscript-level independent confirmation.
