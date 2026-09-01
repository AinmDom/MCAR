# Bounded FiLM-SIREN Correction for Sparse HRTF Reconstruction

Status: working paper draft. The formal three-member Bounded E25 ensemble is
the paper main model as of 2026-09-01; its frozen identity is
`72B7319F8334307664A02BC6369FBD9EECE1D22383C402EC9683D1F6F05F2705`.
The model has completed its single authorized frozen engineering-test
evaluation and a validation-only Q14/Q26/Q50 input-direction sensitivity
experiment. The paper-main operating point is Bounded E25 with Q26 input.
Hybrid E190 and MCAR v3.5.1 remain key ablation/engineering comparators rather
than alternative paper-main candidates.

## Provisional titles

1. **Safe Neural Correction of a Frozen HRTF Reconstructor with Bounded
   FiLM-SIREN Gating**
2. **Bounded FiLM-SIREN Adaptation of a Frozen Baseline for Sparse HRTF
   Reconstruction**
3. **From Strong Baselines to Safe Personalized Correction: A Bounded
   Implicit Model for Sparse HRTF Reconstruction**

The first title is the recommended working title. It makes the central method
clear without claiming universal superiority.

## Central thesis

A strong frozen MCAR model can be improved without destabilizing it by learning
only a zero-initialized, query-dependent, signed gate toward a frozen
FiLM-SIREN prediction. Bounding the gate limits every correction, while the
zero initialization makes the model exactly equal to MCAR before training. The
formal three-seed ensemble improves full-sphere ERB and horizontal ILD over the
previous spectral-CNN hybrid while retaining statistically equivalent
contralateral ERB and high-frequency performance.

## Draft abstract

Sparse measurements can reduce the acquisition burden of individualized
head-related transfer functions (HRTFs), but learned reconstruction systems
often improve spectral accuracy at the expense of binaural cues. We propose a
bounded correction model that preserves a strong frozen convolutional
reconstructor and uses a frozen FiLM-conditioned sinusoidal representation to
define a personalized correction direction. A zero-initialized 514-parameter
gate predicts a signed, query-dependent mixing fraction constrained to
[-0.5, 0.5], making the initial model exactly identical to the frozen baseline
and limiting subsequent deviations. Three independently trained gates are
combined by a pre-registered equal-weight ensemble. On 44 held-out SONICOM
validation subjects reconstructed from a 26-direction sparse grid, the proposed
ensemble achieves 0.797 dB full-sphere ERB error and 0.565 dB horizontal ILD
error. Relative to a FiLM-SIREN plus spectral-CNN hybrid, it significantly
reduces full-sphere ERB error by 0.0078 dB and ILD error by 0.0648 dB, while
contralateral ERB and high-frequency errors remain statistically equivalent.
Relative to MCAR v3.5.1, it significantly improves full-sphere ERB,
contralateral ERB, and contralateral high-frequency error, with no significant
ILD degradation. These results show that a tightly constrained learned
correction can combine complementary reconstructors while avoiding the binaural
trade-off of a higher-capacity fusion network.

## Contributions

1. A safe residual-fusion formulation whose zero initialization exactly
   reproduces a frozen baseline and whose signed correction is explicitly
   bounded at every ear-direction-frequency query.
2. A parameter-efficient personalization mechanism: only a 256-to-2 linear
   gate (514 parameters) is trained, while MCAR and FiLM-SIREN remain frozen.
3. A three-seed, fixed-budget evaluation showing a better spectral/binaural
   Pareto point than MCAR, the FiLM-SIREN ensemble, and the spectral-CNN hybrid
   on the locked validation cohort.
4. A frozen-model input-direction sensitivity study showing that Q14 causes a
   large, consistent degradation and that supplying Q50 to the Q26-trained
   condition pathway does not yield monotonic improvement.

## Method draft

### Task and inputs

The task is binaural residual-log-magnitude reconstruction from a Q26 sparse
direction grid. The model receives the same train-only-normalized subject and
query inputs used by the frozen components. Evaluation covers 767 interpolation
directions, including an independent set of 72 horizontal directions used for
ILD analysis. The development split contains 262 training and 44 validation
subjects.

### Frozen predictors

Let the MCAR v3.5.1 residual prediction be

\[
r_{\mathrm{M}} = 0.3 r_{\mathrm{prev}} + 0.7 r_{\mathrm{v351b}},
\]

where both convolutional members are frozen. Let
\(r_{\mathrm{F}}\) be the residual prediction of a frozen, subject-conditioned
FiLM-SIREN. Its query representation combines direction, dual frequency
coordinates, and local binaural MCA magnitude, while a Q26 encoder provides a
128-dimensional global subject latent.

### Bounded correction

For the final FiLM-SIREN hidden feature \(h(q)\in\mathbb{R}^{256}\), a trainable
linear layer predicts a two-ear raw gate. The final residual is

\[
g(q)=0.5\tanh(W h(q)+b), \qquad
\hat r(q)=r_{\mathrm{M}}(q)+g(q)\odot
\left[r_{\mathrm{F}}(q)-r_{\mathrm{M}}(q)\right].
\]

Both \(W\) and \(b\) are initialized to zero. Consequently, \(g(q)=0\) and
\(\hat r(q)=r_{\mathrm{M}}(q)\) before the first optimization step. The bound
\(|g(q)|\leq0.5\) prevents unrestricted extrapolation between the frozen
predictors. Only 514 gate parameters are trainable.

### Training and formal ensemble

The gate is optimized with AdamW at a learning rate of \(10^{-3}\), zero weight
decay, two warm-up cycles, and a cosine schedule whose horizon is 40 cycles.
The objective combines residual, ERB, contralateral high-frequency, strict ILD,
spectral-difference, spectral-band ILD, and notch-depth terms inherited from the
frozen development protocol. A matched-seed E40 screen produced best cycles
25, 25, and 20; the pre-registered median rule therefore fixed the formal budget
at 25 cycles. Three from-scratch gates use cycle-25 checkpoints and equal
weights, regardless of each run's validation-best checkpoint.

### Evaluation and statistics

The four reported errors are full-sphere ERB, contralateral-25-degree ERB,
contralateral 10--20 kHz magnitude error, and horizontal-plane ILD error. All
are measured in dB and lower is better. Uncertainty is assessed with 10,000
subject-paired percentile bootstrap replicates. A difference is described as
significant only when its two-sided 95% interval excludes zero.

For input-direction sensitivity, Q14, Q26, and Q50 form an exactly nested grid.
The model is not retrained: MCA and the condition input are recomputed from the
current grid while every learned weight remains frozen. All three settings are
evaluated on the same 743 directions after excluding the complete Q50 input
set. The pre-registered comparisons are Q14 minus Q26 and Q50 minus Q26, using
10,000 listener-paired bootstrap replicates with seed 20260901.

## Main validation result

| Method | Full ERB | Contra25 ERB | Contra HF | Horizontal ILD |
|---|---:|---:|---:|---:|
| Bounded correction E25 ensemble | **0.7966** | 1.2166 | 3.5098 | **0.5650** |
| Spectral-CNN hybrid E190 | 0.8044 | **1.2146** | **3.5020** | 0.6298 |
| FiLM-SIREN E130 ensemble | 0.8275 | 1.2308 | 3.7028 | 0.6306 |
| MCAR v3.5.1 | 0.8308 | 1.2890 | 3.5381 | 0.5811 |

Values are validation-subject means in dB. Standard deviations and
publication-ready formatting are provided in Table 1 of the paper artifact
pack.

## Input-direction sensitivity

| Observed directions | Full ERB | Contra25 ERB | Contra HF | Horizontal ILD |
|---:|---:|---:|---:|---:|
| Q14 | 1.1115 | 1.7267 | 4.0524 | 0.7262 |
| Q26 | **0.7928** | **1.2139** | **3.5000** | **0.5591** |
| Q50 | 0.8105 | 1.2720 | 3.5339 | 0.6141 |

These validation means use the fixed 743-direction mask and therefore should
not be mixed with the 767-direction means in the main comparison table. Q14
minus Q26 was positive for all four endpoints: +0.3186 dB full ERB (95% CI
[+0.3013,+0.3363]), +0.5128 dB Contra25 ERB ([+0.4761,+0.5509]), +0.5524 dB
Contra HF ([+0.4847,+0.6226]), and +0.1671 dB horizontal ILD
([+0.0996,+0.2351]). Q50 also did not improve on Q26: the corresponding
differences were +0.0177 [+0.0108,+0.0240], +0.0582 [+0.0438,+0.0724],
+0.0339 [+0.0170,+0.0519], and +0.0549 dB [+0.0204,+0.0900]. Thus, the formal
operating point is Bounded E25 with Q26 input. The Q50 result diagnoses
distribution sensitivity of a Q26-trained frozen condition pathway; it is not
evidence that additional measurements are generally harmful or that a
Q50-trained model would behave the same way.

## Results draft

The bounded ensemble obtained the lowest full-sphere ERB and horizontal ILD
errors. Compared with the spectral-CNN hybrid, it reduced full-sphere ERB by
0.0078 dB (95% CI [-0.0142, -0.0020]) and horizontal ILD by 0.0648 dB
([-0.0882, -0.0434]). Its small mean increases in contralateral-25 ERB
(+0.0020 dB) and contralateral high-frequency error (+0.0078 dB) were not
significant. Thus, the bounded model improves the hybrid's full-sphere spectral
accuracy and removes its ILD penalty without a detectable loss in the two
contralateral metrics.

Compared with MCAR v3.5.1, the proposed model significantly reduced
full-sphere ERB by 0.0342 dB, contralateral-25 ERB by 0.0723 dB, and
contralateral high-frequency error by 0.0283 dB. Horizontal ILD was lower by
0.0161 dB, although its confidence interval slightly crossed zero. Against the
FiLM-SIREN ensemble, all four metrics improved significantly. These comparisons
support complementarity rather than replacement: MCAR supplies a robust
initial solution, FiLM-SIREN supplies a personalized correction direction, and
the bounded gate selects a conservative local adjustment.

## Discussion points

- **Why zero initialization matters.** Optimization starts from a verified
  baseline instead of a random fusion, so early training cannot discard the
  established MCAR solution.
- **Why bounding matters.** The correction cannot move more than half the
  FiLM--MCAR discrepancy at any query, which regularizes a fusion problem with
  only 262 training subjects.
- **Why the small gate can beat the larger hybrid.** The spectral-CNN hybrid
  trains 74,402 parameters and improves spectral metrics but inherits an ILD
  trade-off. The 514-parameter gate has a narrower hypothesis class aligned
  with the desired operation: deciding where and how much to move away from
  MCAR.
- **Why the ensemble helps.** The formal ensemble improves all four mean
  metrics over the one-seed bounded model, while using fixed equal weights and
  fixed cycle-25 checkpoints.
- **Why Q26 remains the operating point.** Reducing the input to Q14 degrades
  all endpoints substantially. Increasing it to Q50 without retraining also
  causes a smaller but consistent degradation, indicating that the learned
  condition pathway is calibrated to its Q26 training distribution.
- **What cannot be claimed.** The candidate does not have the lowest mean on
  every metric: Hybrid E190 remains slightly lower on Contra25 and HF. Those
  differences are statistically indistinguishable, so the defensible claim is
  a better Pareto balance, not strict four-metric domination.

## Recommended paper structure

1. Introduction: sparse individualized HRTFs and spectral/binaural trade-offs.
2. Related work: MCA/SUpDEq, learned sparse reconstruction, implicit neural
   representations, FiLM conditioning, and safe residual adaptation.
3. Method: frozen MCAR and FiLM predictors, bounded gate, zero initialization,
   objective, and formal ensemble.
4. Experimental protocol: SONICOM split, Q26 directions, metrics, baselines,
   bootstrap, and data-access safeguards.
5. Results: main comparison, paired statistics, development ablation, and
   subject-level examples.
6. Discussion: capacity versus constraint, ILD trade-off, limitations, and
   generalization.
7. Conclusion.

## Evidence and claim boundaries

- The formal model has completed one authorized frozen engineering-test
  evaluation. The Q14/Q26/Q50 sensitivity experiment is validation-only:
  132/132 predictions are finite, Q26 reproduces the existing formal residual
  exactly, and its report records `test_subject_count_read=0`.
- The three pre-existing comparison methods reproduce their previous
  subject-level values exactly (maximum absolute difference 0 dB).
- RANF and FSP-AE direct comparisons are frozen supplementary evidence and must
  not drive tuning. Validation and engineering-test results must remain
  explicitly separated.
- No further test access is authorized. Any new test evaluation requires a new
  result-blind pre-registration and explicit user authorization.
- Literature citations and venue-specific formatting remain placeholders; no
  source should be invented from memory.

## Paper artifact pack

- `results/sonicom_bounded_mcar_film_correction_e25_paper/table_1_main_validation_results.{csv,tex}`
- `results/sonicom_bounded_mcar_film_correction_e25_paper/table_2_paired_statistics.{csv,tex}`
- `results/sonicom_bounded_mcar_film_correction_e25_paper/table_3_development_ablation.{csv,tex}`
- `results/sonicom_bounded_mcar_film_correction_e25_paper/figure_1_main_validation.{png,pdf}`
- `results/sonicom_bounded_mcar_film_correction_e25_paper/figure_2_paired_effects.{png,pdf}`
- `results/sonicom_bounded_mcar_film_correction_e25_paper/manifest.json`
