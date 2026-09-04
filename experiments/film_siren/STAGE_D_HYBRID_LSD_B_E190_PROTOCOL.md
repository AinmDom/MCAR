# Hybrid B: original objective plus full-spectrum LSD

Status: PRE-REGISTERED BEFORE TRAINING, 2026-09-04 (+08:00).

## Scope and question

User authorizes only B, three simultaneous seeds 20260821/20260822/20260823.
This supersedes the proposed four-group study in
`reports/HYBRID_LSD_ERB_AUDIT_20260904.md`. Reuse the existing Hybrid E190
validation comparator; do not retrain controls. This tests adding an LSD
objective, not removing the MCA internal ERB correction or the training ERB term.
This is exploratory validation development, not independent test confirmation.

## Frozen intervention

Clone each same-seed `sonicom_film_siren_spectral_cnn_final_seed*_e190.json`.
Only experiment identity, protocol, run directory and the two new objective
parameters change: `lsd_weight=1.0`, `lsd_epsilon_db=1e-6`.
Each run starts with its hash-verified frozen corrected FiLM E130 `last.pt` and
a newly initialized CNN with zero output convolution. Do not load trained Hybrid
weights. All MCA/Q26 preprocessing, model parameters, normalization, sample order,
16 global + 16 independent horizontal directions, losses, optimizer and schedule
are inherited. AdamW lr=0.0001, wd=0.0001; warmup=2, cosine horizon=200; FP32;
gradient clip=5; 262 steps/cycle; fixed 190 cycles (49,780 optimizer steps).

Let e[e,d,f] be predicted residual dB minus target residual dB. Because the same
MCA magnitude is added to both, this is also the corrected-HRTF dB error.

    direction_lsd[e,d] = sqrt(mean_f(e[e,d,f]^2) + (1e-6 dB)^2)
    LSD_loss = mean_ear(sum_d(normalized_solid_angle[d] * direction_lsd[e,d]))
    B_total = original_total + 1.0 * LSD_loss / target_std

Train uses the sampled global interpolation directions and all 463 bins,
86.1328125--19982.8125 Hz. Validation every 5 cycles uses all 767 interpolation
directions and 44 subjects, giving subjects equal weight. Reported
`full_sphere_lsd_db` uses epsilon=0 on detached errors, with the exact reduction
order of `mcar.evaluation.secondary_metrics.full_sphere_lsd`; training uses a
positive epsilon solely for finite derivatives at exact zero error. A training
batch LSD is a sampled objective, not a complete-sphere validation measurement.
Existing global weighted RMSE is different and must retain its own name.

## Checkpoint and analysis rules

Authoritative checkpoint is E190 `last.pt`, irrespective of the validation curve.
`best.pt` remains a diagnostic minimum of the total objective. There is no
best-LSD checkpoint selection, early stop, budget extension, coefficient sweep,
or seed removal. Three completed members form a residual-dB mean with weights
1/3 each. Report incomplete/nonfinite runs as failures; do not silently exclude.

After completion, verify 190 cycles, 49,780 steps, finite history/ledger/weights,
clean training Git provenance, all hashes, and test_subjects_read=0. Reconstruct
only the new ensemble on the same 44 validation subjects, in a new directory.
Reuse existing control predictions and per-subject results without overwriting.
Primary comparison is new-minus-old FullSphereLSD per subject; report means,
paired mean difference and 95% percentile bootstrap interval (10,000 resamples,
seed 20260904), win/tie/loss (absolute tie tolerance 1e-12 dB), subject median,
P90 and worst-subject error. Also report the unchanged full/contralateral ERB,
contralateral HF, horizontal strict ILD, and spectral diagnostics to expose
tradeoffs. Report all three new seed LSD curves and E190 values. Historical
ensemble results alone do not establish paired per-seed control effects.
No improvement claim is allowed from training loss alone. This historical
comparison is not a concurrent randomized factorial test of ERB causality.

## Existing comparator and evidence

- Frozen identity: `A3CFAC9C206E0A53FD2FA130817673AAFE07B66855322B5D34824E9173BDCCFE`
  in `configs/experiments/sonicom_film_siren_spectral_cnn_final_e190_ensemble_manifest.json`.
- Old ensemble predictions:
  `artifacts/reconstruction/sonicom_film_siren_spectral_cnn_final_e190_ensemble_validation/`.
- Secondary values: `results/sonicom_film_secondary_metrics_v1_validation/`.
  Existing Hybrid FullSphereLSD mean=3.5489256190074472 dB across 44 val subjects.
- Primary values: `results/sonicom_film_siren_spectral_cnn_final_e190_validation/`.
- New configs, verified parents/controls and source-file hashes are frozen by
  `configs/experiments/sonicom_hybrid_lsd_b_e190_preparation_manifest.json`.

## Data and execution boundary

Only 262 train and 44 validation subjects may be resolved/read. Do not construct
test subject paths, read test HDF5/SOFA, reevaluate test, or tune from historical
test values. No new dataset export or MCA computation is required. Require CUDA
and a clean committed Git tree before launch. Three processes share the existing
GPU, have distinct run/log directories, and refuse to overwrite earlier runs.
Use `scripts/run_hybrid_lsd_b.py` for a single launch; its controller waits without
polling and records exit codes. Codex stops active monitoring after one startup
verification; a later handoff checks completion and performs the frozen analysis.

Launch bookkeeping correction, before any optimizer step: the first attempt
(2026-09-04 23:36:35 +08:00) exited at the clean-Git guard because its own log
directory was untracked. Preserve that receipt and all stderr logs. Ignore only
the dedicated operational log directory, then use `--attempt 2` to write a new
receipt under `outputs/hybrid_lsd_b_e190/attempt2/`. No model, objective, data or
budget change; all three failed processes exited before subject paths were read.
