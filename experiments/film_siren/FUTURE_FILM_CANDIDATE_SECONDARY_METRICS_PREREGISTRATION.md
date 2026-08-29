# Future FiLM Candidate Secondary Metrics Preregistration v1

**Status:** metric/data/statistical boundary frozen before any evaluation under this
protocol; evaluator implementation and external dependency identities remain to be
frozen in a result-blind implementation amendment.

**Protocol date:** 2026-08-29 (Asia/Hong_Kong)

## 1. Scope and non-retroactivity

This protocol defines supplementary paper endpoints for future FiLM-family candidates
and for validation-only characterization of already frozen models. It does not change
the four primary endpoints already used by Stage E:

1. Full-sphere ERB error;
2. contralateral 25% ERB error;
3. contralateral high-frequency error; and
4. horizontal-plane ILD MAE.

The ongoing Bounded E25 frozen-test workflow had already generated all 44 test
predictions before this document was created. Therefore:

- no endpoint introduced here may be described as preregistered for that test run;
- this protocol does not authorize a new test read, test reconstruction, test metric,
  candidate prediction, checkpoint choice, or test-driven decision;
- any application to the current frozen Bounded E25 model is restricted to the 44
  validation subjects and must be labelled a post-freeze supplementary validation
  analysis; and
- use on a future locked test requires a separate prospective amendment, explicit
  authorization, frozen code/dependency hashes, and an unused test-access registry.

No training, candidate selection, test read, or metric computation is authorized by
the act of committing this document.

## 2. Intended claims and hierarchy

The four existing endpoints remain the only primary endpoints for the current paper
candidate. The endpoints below are secondary or descriptive. They can explain spectral,
spatial, subject-level, behavioral, or computational effects, but they cannot reverse
the primary decision and will not be combined into a composite score.

For a future candidate, a separate model-specific protocol must state before training
whether any endpoint below becomes a selection or confirmation endpoint, including its
direction, comparator, margin, and multiplicity treatment. In the absence of such a
statement, every endpoint below remains secondary.

## 3. Frozen data and reconstruction boundary

### 3.1 Current validation application

- Dataset definition: `data/processed/sonicom_residual_q26_v1/dataset_definition.json`.
- Split: validation only, 44 unique subjects.
- HRTF variant: `FreeFieldCompMinPhase_44kHz` at 44.1 kHz.
- Directions: 793 total; exclude the 26 observed Q26 directions; evaluate the 767
  interpolation directions.
- Frequency grid: the frozen 463 selected bins, 86.1328125--19982.8125 Hz with
  43.06640625 Hz spacing.
- Horizontal subset: the frozen 72 interpolation directions already used by the strict
  evaluator.
- Direction features: columns are azimuth, elevation, x, y, z, and normalized
  solid-angle weight.
- Any subset or distance bin must renormalize its included solid-angle weights to sum
  to one. Subject is the independent statistical unit.

### 3.2 Model reconstruction

For magnitude-domain endpoints, use the final reconstructed log magnitude in dB after
adding the predicted residual to the frozen MCA reconstruction. Use the dataset floor
of -200 dB before logarithmic comparisons.

For HRIR-derived endpoints, reproduce the existing strict evaluator: replace selected
frequency-bin magnitudes, retain the subject-specific original MCA selected-bin phase
and all outside-bin complex values, enforce the evaluator's conjugate-symmetric
spectrum, perform the inverse FFT, and apply the same crop/alignment rule. This
reconstruction means that the current model does not independently predict phase or
delay; ITD is consequently a sanity endpoint only.

### 3.3 Comparators

The required validation comparators for the current Bounded E25 characterization are
MCAR v3.5.1, Hybrid E190, and the frozen FiLM E130 ensemble. RANF and FSP-AE may be
included only when their already frozen, provenance-qualified predictions can be
evaluated on exactly the same subjects and masks. Missing or inapplicable endpoint
cells must be reported as `N/A`, never silently dropped.

For future candidates, the model-specific protocol must freeze the comparator set
before predictions are generated.

## 4. Endpoint definitions

Let subject, direction, ear, and selected frequency-bin indices be $s,d,e,f$. Let
$R_{sdef}$ and $P_{sdef}$ be reference and predicted log magnitudes in dB, and let
$w_{sd}$ be the solid-angle weight renormalized over the evaluated direction mask.

### 4.1 Full-sphere log-spectral distortion (LSD)

Compute an ear-specific per-direction value

\[
\operatorname{LSD}_{sde}=
\sqrt{\frac{1}{463}\sum_f(P_{sdef}-R_{sdef})^2}.
\]

The subject score is

\[
\operatorname{LSD}_s=\frac{1}{2}\sum_e\sum_{d\in I}w_{sd}
\operatorname{LSD}_{sde},
\]

where $I$ is the 767-direction interpolation mask. Units are dB; lower is better.
Also save the 793-by-2 direction/ear map before masking so its provenance can be
audited, but never include Q26 observations in aggregate performance.

### 4.2 Auditory-model localization

Report the following outcomes in degrees or percentage points:

- lateral-angle RMS error;
- local polar-angle RMS error; and
- quadrant/confusion error rate.

Use a published sagittal-plane localization model compatible with the available SOFA
HRIRs. For each subject, evaluate both the reference HRIR against itself and each
candidate HRIR against the same subject-specific reference template. The reported
model-induced degradation is candidate outcome minus reference-self outcome, so the
floor imposed by the localization simulator is explicit.

Two regions are frozen for reporting:

- horizontal-frontal: elevation within +/-30 degrees and absolute azimuth <=60 degrees;
- vertical-median: absolute lateral angle <=30 degrees.

This endpoint is not runnable until a result-blind implementation amendment freezes:
the exact software repository and commit, auditory model, SOFA convention mapping,
stimulus/noise settings, Monte Carlo repetition count, random seed, coordinate
conversion tests, and a reference-self sanity threshold. No localization output
generated before that amendment may enter the paper table.

### 4.3 Subject-level paired benefit and tail risk

For every scalar error endpoint $m$, define the paired subject difference against a
frozen comparator $b$ as

\[
\Delta_{s,m}=m_s(\text{candidate})-m_s(b),
\]

so negative is better. Report:

- win, tie, and loss counts and win rate, using a tie tolerance of (10^{-12}) in the
  endpoint's native unit;
- median paired difference;
- P90 and P95 of the 44 paired differences (linear interpolation between adjacent
  order statistics);
- worst-decile mean, defined for 44 subjects as the arithmetic mean of the five
  largest paired differences; and
- maximum paired degradation and the corresponding subject ID.

These statistics are required for the four primary endpoints, LSD, and each scalar
localization endpoint. They may also be supplied for other scalar endpoints. A positive
P95 or maximum is evidence of a tail failure, not automatically a rejection threshold;
no perceptual non-inferiority margin is claimed by this protocol.

### 4.4 ERB-band horizontal ILD profile

Reuse the repository implementation
`spectral_band_ild_smooth_l1_and_mae` in `src/mcar/losses.py`, in reporting/MAE mode:

- ERB center frequencies from 200 Hz through 18 kHz;
- power integration using the frozen ERB weights;
- ILD in dB from left/right band powers;
- the frozen 72 horizontal interpolation directions and renormalized direction weights;
- absolute candidate-reference ILD error, averaged over direction within subject.

Save one value for every subject and ERB band. Report the mean curve with a subject-level
95% confidence band and the all-band arithmetic mean as a secondary scalar. Do not tune
or merge frequency bands after inspecting results.

### 4.5 High-frequency spectral shape and notch depth

Reuse the existing repository definitions without retuning:

- first spectral-difference MAE above 4 kHz, in dB/bin;
- second spectral-difference MAE above 4 kHz, in dB/bin-squared; and
- multi-scale notch-depth MAE from 4--18 kHz with radii `(4, 8, 16)` bins, minimum
  reference depth 1.0 dB, and softplus temperature 0.5 dB.

All three are computed per subject and ear, direction-weighted over the 767 interpolation
directions, then averaged equally over ears. The exact formulas are those in the frozen
functions `high_frequency_spectral_difference_mae` and
`multi_scale_notch_depth_mae` in `src/mcar/losses.py`; the implementation amendment must
hash the file and add independent numerical tests before evaluation.

A discrete notch-location error is intentionally not defined in v1: smoothing, notch
detection, and notch matching can materially change it. It may only be added by a
result-blind amendment that freezes those three choices and handles missing/extra
notches explicitly. Until then, the multi-scale notch-depth endpoint is the registered
notch measure.

### 4.6 Error versus angular distance from Q26

For every interpolation direction, compute its minimum great-circle distance to any of
the 26 observed directions using unit Cartesian vectors:

\[
\theta_d=\frac{180}{\pi}\min_{q\in Q26}
\arccos\left(\operatorname{clip}(\mathbf{x}_d^T\mathbf{x}_q,-1,1)\right).
\]

Freeze the bins to `[0,10)`, `[10,20)`, `[20,30)`, and `[30,180]` degrees. The response
is per-direction LSD averaged equally over ears. For each subject and nonempty bin,
compute the solid-angle-weighted mean after renormalizing weights within that bin.
Report subject mean and bootstrap CI per bin, plus the paired candidate-comparator
difference. No bin boundary may be changed after results are inspected.

Also save a direction-level map of mean per-direction LSD across subjects. Plot it in
the dataset's azimuth/elevation convention with Q26 locations overlaid. The heat map is
descriptive and must not be treated as 767 independent observations.

### 4.7 Gate and correction behavior

For every FiLM correction member, save the bounded gate $g$ and applied correction
$C=g\,\Delta$ at each subject, direction, ear, and frequency bin. Report per member and
for the deployed ensemble:

- mean, median, P95, P99, and maximum of `abs(g)`;
- fractions with `abs(g) < 0.01` and `abs(g) >= 0.45`;
- mean, median, P95, P99, and maximum of `abs(C)` in dB; and
- the same gate/correction summaries by ear and by the four distance-to-Q26 bins.

For an ensemble, average the applied member corrections to obtain the deployed
correction. Do not invent an "ensemble gate" by averaging gates whose deltas differ.

As a descriptive mechanism check, define pointwise benefit

\[
B=|R-M_{base}|-|R-M_{candidate}|,
\]

where all quantities are final log magnitudes in dB. Compute Spearman correlation
between `abs(C)` and $B$ separately for every subject, then report the median and IQR
of the 44 correlations. Pooled-bin correlations are prohibited because they would
pseudoreplicate subjects and frequency bins.

### 4.8 ITD sanity check

Estimate ITD from the strictly reconstructed HRIR using the maximum interaural
cross-correlation method. Report, over the 767 interpolation directions:

- subject-level solid-angle-weighted MAE from the reference ITD in microseconds;
- subject-level maximum absolute ITD error; and
- paired differences versus MCAR.

ITD remains a sanity endpoint because selected-bin phase and outside-bin complex values
are inherited from MCA. It must not be used to claim learned timing reconstruction or
to rank magnitude-only candidates. The exact estimator implementation, lag limits,
filtering, interpolation, software version/commit, and unit tests must be frozen in the
same implementation amendment as the localization endpoint.

### 4.9 Efficiency and deployability

Report the following for every evaluated method when technically applicable:

- trainable parameter count;
- total parameter count;
- unique deployed parameter count (shared/frozen components counted once);
- serialized checkpoint size in MiB;
- MACs and FLOPs for one complete subject reconstruction at 793 directions and 463
  bins, with the profiler/tool and version stated;
- inference latency for one complete subject;
- peak allocated and peak reserved accelerator memory; and
- training wall-clock time and accelerator-hours from provenance-qualified logs.

Latency measurement excludes checkpoint loading and disk I/O, uses batch size one,
five warm-up runs, 30 timed runs, device synchronization immediately before and after
each timed region, and reports median and P95. Also report one end-to-end latency that
includes input loading and output serialization. Freeze hardware model, driver,
runtime/library versions, precision, thread count, power mode, and exact command before
the first benchmark. Single-member and deployed-ensemble costs are separate rows.

## 5. Statistical analysis

- Compute every aggregate from subject-level values; directions, ears, bands, and
  frequency bins are not independent replicates.
- Use 10,000 subject-level paired bootstrap resamples with replacement and seed
  `20260829` for candidate-comparator mean differences and 95% percentile intervals.
- For unpaired descriptive means, use the same 10,000 subject bootstrap resamples and
  seed.
- Retain all 44 subjects. Any non-finite or structurally incomplete value invalidates
  the affected method/end point table; do not silently remove a subject.
- Report exact estimates, interval endpoints, subject counts, wins/ties/losses, and
  missingness. Do not convert secondary CIs into confirmatory significance claims.
- No multiplicity-adjusted confirmatory family, equivalence margin, non-inferiority
  margin, composite rank, or weighted summary score is registered here.

## 6. Required artifacts

The implementation amendment must freeze an output directory before first evaluation.
At minimum it must contain:

- `protocol_snapshot.json`: protocol commit, data definition/hash, split, candidate and
  comparator identities, code/dependency hashes, software and hardware manifest;
- `quality_checks.json`: subject/direction/frequency counts, masks, weight sums,
  reconstruction checks, finite checks, and test-access count;
- `per_subject_metrics.csv`;
- `aggregate_metrics.csv`;
- `paired_tail_risk.csv`;
- `band_ild_profile.csv`;
- `spatial_distance_bins.csv` and `spatial_direction_map.csv`;
- `gate_correction_statistics.csv` when applicable;
- `efficiency.json`; and
- machine-readable plotting data plus paper figures.

Raw per-subject rows must include subject ID, split, method identity, endpoint name,
unit, value, and applicability. Aggregates without auditable subject rows are invalid.

## 7. Integrity and abort conditions

Before accepting results, verify:

- exactly 44 unique validation subjects for the current application;
- exactly 793 directions, 26 excluded Q26 directions, 767 interpolation directions,
  72 horizontal interpolation directions, and 463 selected frequency bins;
- direction masks are disjoint where required and subset weights renormalize to one;
- all values and reconstructed arrays are finite;
- prediction/model/comparator identities and file hashes match the frozen amendment;
- the four existing metrics reproduce their committed source values to the frozen
  tolerance when the new evaluator shares reconstruction code; and
- this supplementary workflow introduces `test_subject_count_read=0` and contains no
  test paths or test subject records.

Abort before aggregation if any check fails. Preserve partial artifacts for diagnosis,
record the failure, and do not substitute a new definition after seeing results.

## 8. Required result-blind implementation amendment

Before any endpoint is computed, commit a short amendment that freezes:

1. evaluator entry point, command, output directory, and code SHA-256 identities;
2. candidate/comparator manifests and prediction hashes;
3. numerical test fixtures for LSD, distance bins, tail statistics, band ILD, spectral
   differences, notch depth, and gate/correction summaries;
4. the exact official localization/ITD packages and immutable commits or release
   archives, all model parameters, coordinate mappings, and seeds;
5. profiler, FLOP convention, hardware/software manifest, and benchmark command; and
6. expected counts, tolerances, failure behavior, and explicit confirmation that no
   output metrics have yet been inspected.

The amendment may resolve implementation details left explicitly open above, but it
may not alter endpoint formulas, masks, bins, hierarchy, comparators, or statistical
rules after any result has been observed. A scientifically necessary change after
observation must be labelled exploratory and versioned separately.

## 9. Interpretation boundary

Evidence from this protocol may support statements about average spectral fidelity,
localization-model predictions, subject-tail behavior, spatial generalization,
correction mechanism, and computational cost on the frozen validation cohort. It does
not establish perceptual transparency, listener preference, real-time feasibility on
unmeasured hardware, population-wide worst-case safety, or independent learned phase/
ITD reconstruction. Human listening evaluation requires its own prospective protocol,
ethics/consent handling where applicable, power analysis, stimuli, listeners, trials,
randomization, exclusions, and analysis plan.

## 10. Method references for implementation qualification

- Spatial Audio Metrics (official repository):
  <https://github.com/Katarina-Poole/Spatial-Audio-Metrics>
- Spatial Audio Metrics HRTF tutorial:
  <https://spatial-audio-metrics.readthedocs.io/en/latest/hrtf_tutorial.html>
- Auditory Modeling Toolbox, `baumgartner2014` documentation:
  <https://www.amtoolbox.org/amt-0.10.0/doc/models/baumgartner2014.php>
- LAP24 localization assessment paper:
  <https://doi.org/10.1109/OJSP.2025.3588776>

These links identify candidate official sources, not yet-frozen dependency identities.
The implementation amendment must record immutable versions and local hashes.
