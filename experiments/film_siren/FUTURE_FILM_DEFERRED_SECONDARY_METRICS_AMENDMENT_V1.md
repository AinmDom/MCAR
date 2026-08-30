# Deferred FiLM Secondary Metrics — Result-Blind Implementation Amendment v1

Status: **IMPLEMENTATION READY; NO DEFERRED ENDPOINT RESULT INSPECTED**  
Date: 2026-08-30  
Split: **SONICOM validation only (44 subjects); test is permanently excluded**

This amendment completes the locally runnable endpoints deferred by
`FUTURE_FILM_CANDIDATE_SECONDARY_METRICS_PREREGISTRATION.md`. The existing four
metrics remain primary. Gate/correction, ITD, dominant-notch, and efficiency outcomes
are secondary or descriptive and cannot alter the frozen candidate. Model-based
localization remains unrun because no immutable AMT/SAM implementation is locally
available.

## Frozen models, data, and outputs

- Methods: BOUNDED, HYBRID, FILMENS, and MCAR, using exactly the prediction inventories
  in the already frozen v1 secondary manifest.
- Candidate mechanism: the three cycle-25 BOUNDED members in identity
  `72B7319F8334307664A02BC6369FBD9EECE1D22383C402EC9683D1F6F05F2705`.
- Aggregate mask: 767 interpolation directions. Full 793-direction member gate and
  applied-correction tensors are saved only for audit.
- Results: `results/sonicom_film_deferred_secondary_metrics_v1_validation/`.
- Full diagnostics: `artifacts/reconstruction/
  sonicom_bounded_mcar_film_gate_diagnostics_v1_validation/`.
- Efficiency: `results/sonicom_bounded_mcar_film_efficiency_v1_validation/`.
- All implementation, dependency, input-inventory, checkpoint, runtime, hardware and
  output identities are frozen in
  `configs/experiments/sonicom_film_deferred_secondary_metrics_validation_manifest.json`.

## Gate and correction

For each member the model-native `forward_grid` values are saved as gate `g` and
physical applied correction `C = normalized_correction * target_std` in dB. The base
includes `target_mean`; the correction does not. The implementation asserts
`final_residual_db = base_residual_db + applied_correction_db`.

For each subject, member, ear and Q26-distance bin, absolute gate/correction values are
summarized by mean, median, linear P95/P99 and maximum. Gate summaries also include
fractions `|g| < 0.01` and `|g| >= 0.45`. Subject summaries, not spectral points, are
the aggregate replicates. Deployed correction is the frozen weighted mean of member
corrections; no ensemble gate is defined. One subject-wise Spearman correlation uses
all ear × interpolation-direction × frequency points between `|C|` and
`|R-M_base|-|R-M_candidate|`; the report gives the median and IQR of 44 correlations.

## ITD sanity endpoint

The exact estimator is `mcar.fsp_ae_signal.estimate_itd_seconds`, the repository's
public compatibility port of the official FSP-AE procedure. It uses a 1.6 kHz
low-pass biquad, resampling from 44.1 kHz to 384 kHz, raw interaural cross-correlation,
an inclusive ±1 ms search, and argmax on the upsampled grid. There is no further
sub-sample interpolation; resolution is 1/384000 s. Exact torch/torchaudio version and
binary hashes are in the manifest. The existing official-compatibility test and a new
known-delay test qualify the implementation.

Reference ITD is estimated from measured subject SOFA `Data.IR`. Candidate ITD is
estimated from the strict reconstruction that changes selected magnitudes while
inheriting MCA phase and outside-band complex bins. Report subject solid-angle-weighted
MAE and maximum absolute error over 767 directions in microseconds, plus BOUNDED-minus-
MCAR paired statistics. This is a sanity check, not learned-phase evidence or a ranking
criterion.

## Dominant-notch location — exploratory endpoint-specific preregistration

No discrete notch-location result has been inspected, but related multi-scale notch
depth results were already available when this definition was written. Therefore this
endpoint is explicitly **exploratory**, even though its algorithm is frozen before its
own output is generated.

For each ear, form a DTF by subtracting the solid-angle-weighted directional mean over
the 767 interpolation directions. Smooth frequency with Savitzky–Golay window 11 bins,
degree 3, `mode="interp"`. Within 4–18 kHz, detect minima with prominence at least
1 dB and separation at least 500 Hz. Retain one dominant notch: greatest prominence,
with lowest frequency breaking an exact prominence tie.

Match predicted and reference dominant notches for the same ear/direction when their
distance is at most 1500 Hz. Missing or farther predicted notches receive a 1500 Hz
penalty. Report solid-angle/ear-weighted penalized MAE, matched-only MAE, miss rate,
spurious rate, and reference-notch fraction per subject. This explicitly handles
missing/extra notches and does not retrospectively replace the registered multi-scale
notch-depth endpoint.

## Efficiency

Benchmark one complete subject (P0001), batch size one, 793 directions × 463 bins,
float32, direction block 32, one CPU thread and TF32 disabled. Report a single member
and the deployed sequential three-member ensemble separately. Each compute-only row
uses five warm-ups and 30 synchronized timed repetitions, reporting median/P95 latency
and peak CUDA allocated/reserved memory. Prepared CPU arrays exclude checkpoint loading
and disk I/O. One additional ensemble measurement includes input HDF5 loading and
output HDF5 serialization.

`torch.utils.flop_counter.FlopCounterMode` from torch 2.8.0 reports native covered
FLOPs; MACs are reported as FLOPs/2. Counts also include trainable, instantiated total,
and unique deployed parameters, where the two frozen MCAR components are counted once
across the ensemble, but the three FiLM backbones and gates remain distinct. Checkpoint
MiB and provenance-qualified training elapsed seconds/GPU-hours come from frozen files.

## Localization status

The local repository contains SUpDEq but no immutable AMT or Spatial Audio Metrics
localization package/model. Installing or silently reimplementing a published listener
model would create an unqualified dependency and coordinate risk. Consequently the
three localization endpoints remain `not_run_dependency_unavailable`; no proxy metric
may be substituted. A later version requires a separately committed official package,
commit/archive hash, Monte Carlo settings, seed, coordinate tests, and reference-self
threshold before any output.

## Statistical and abort rules

Subject is the only inferential unit. Mean intervals and paired BOUNDED-minus-MCAR
statistics use 10,000 PCG64 bootstrap replicates with seed 20260829, linear quantiles,
tie tolerance `1e-12`, and the existing five-subject worst-decile rule. Abort without
aggregation on any hash, split, shape, coordinate, finite, count, or regenerated-
prediction check failure. No test path may be constructed; every report must state
`test_subject_count_read=0`.
