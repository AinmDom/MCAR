# Stage D: FiLM-SIREN + zero-initialized MCAR spectral CNN

Status: **PRE-REGISTERED BEFORE D1 TRAINING**  
Date: 2026-08-28  
Data boundary: SONICOM train + validation only; test access is forbidden.

## Question

Can the local binaural spectral CNN from MCAR repair the remaining high-frequency
and horizontal-ILD weaknesses of the corrected FiLM-SIREN without discarding its
full-sphere and contralateral ERB gains?

## Frozen D1 architecture

The base is the authoritative cycle-130 `last.pt` from corrected FiLM-SIREN seed
`20260821`. The full FiLM-SIREN, including its Q26 encoder and all modulators, is
loaded by SHA-256 and frozen. A newly initialized `BinauralSpectralCNN` is appended.

For each direction, its seven input channels are fixed to:

1. normalized left MCA log magnitude;
2. normalized left correction log magnitude;
3. normalized left FiLM-SIREN residual;
4. normalized right MCA log magnitude;
5. normalized right correction log magnitude;
6. normalized right FiLM-SIREN residual;
7. normalized log-frequency.

The CNN is the existing MCAR local architecture: width 48, kernel 7, dilations
`[1,2,4,8]`, direction-condition width 64. Its two-channel output convolution is
zero initialized. Therefore, before optimization,

`hybrid_residual == frozen_film_siren_residual`

exactly (subject only to exact floating-point identity; tests use zero tolerance).
The final residual is `base + cnn_delta`. D1 trains only the CNN.

## Frozen D1 optimization

- Seed: `20260821`.
- Budget: 40 cycles, 262 train-subject steps per cycle.
- Optimizer: AdamW, learning rate `1e-4`, weight decay `1e-4`.
- Schedule: warmup-cosine, two warmup cycles, horizon 40.
- Validation every five cycles; checkpoint selection is the minimum shared
  D1/D2+notch Stage-C objective on the 44 validation subjects.
- Global and independent horizontal batches remain 16 directions each.
- Objective and all auditory weights are inherited unchanged from corrected C4:
  ERB `0.75`, HF `0.25`, strict ILD `0.75`, band ILD `0.05`, D1/D2
  `0.25/0.15`, notch `0.30`.
- Precision is FP32 and gradient clipping is 5.

This is a one-seed development screen, not a formal paper result and not a basis
for test access.

## Frozen D1 strict-validation analysis

Before running the strict reconstruction comparison, the four method IDs are
fixed as `HYBRID` (D1 best), `PARENT` (its seed-20260821 cycle-130 parent),
`FILMENS` (the corrected E130 three-member FiLM ensemble), and `MCAR` (v3.5.1).
The evaluator and all four metric definitions are inherited from the corrected
C4 four-method validation evaluation. For every metric it reports the paired
subject difference `HYBRID - baseline`, so negative values favor D1, together
with the mean difference, a 95% percentile interval from 10,000 paired bootstrap
replicates (seed `20260828`), and hybrid win/tie/loss counts over 44 subjects.

## Decision sequence

After D1 completes, first verify completion, finite history/ledger, checkpoint
hashes, and `test_subjects_read=0`. Then reconstruct the same 44 validation
subjects from the best D1 checkpoint and compare it against (a) its frozen
seed-20260821 FiLM parent, (b) the corrected E130 FiLM ensemble, and (c) MCAR
v3.5.1 using the already frozen four-metric evaluator.

Advance only if the paired validation evidence shows a credible complementarity
pattern: retain the FiLM ERB advantage while materially reducing at least one of
Contra-HF or horizontal ILD. No single scalar is allowed to hide a large regression
in another reported metric. If advanced, run matched seeds `20260822/20260823`,
freeze a common cycle from development best cycles, and train three fixed-cycle
hybrid members before any new manuscript-level comparison.

## Hard boundaries

- Do not construct or read SONICOM test paths.
- Do not use the historically consumed test results to tune CNN weights, cycles,
  seeds, ensemble weights, or promotion thresholds.
- Do not fine-tune the FiLM-SIREN backbone in D1.
- Do not change the seven-channel interface or CNN architecture after seeing D1
  validation results; such a change requires a new named, pre-registered stage.
