# Bounded FiLM-SIREN Correction for Sparse HRTF Reconstruction

Status: working paper draft based on the frozen 44-subject SONICOM validation
evaluation. The new model has not been evaluated on the locked test split.

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

- Current new-model evidence is validation-only: 44/44 predictions are finite,
  and all reports record `test_subject_count_read=0`.
- The three pre-existing comparison methods reproduce their previous
  subject-level values exactly (maximum absolute difference 0 dB).
- RANF and FSP-AE values exist under the same validation protocol, but the new
  bounded model has not yet been placed in a pre-registered direct table with
  them. Any such table must be labeled as a frozen-model supplementary report
  and must not drive tuning.
- A final locked-test evaluation of the new candidate requires a separate
  pre-registration and explicit user authorization. Until that happens, the
  abstract and conclusions must say "held-out validation subjects," not
  "held-out test subjects."
- Literature citations and venue-specific formatting remain placeholders; no
  source should be invented from memory.

## Paper artifact pack

- `results/sonicom_bounded_mcar_film_correction_e25_paper/table_1_main_validation_results.{csv,tex}`
- `results/sonicom_bounded_mcar_film_correction_e25_paper/table_2_paired_statistics.{csv,tex}`
- `results/sonicom_bounded_mcar_film_correction_e25_paper/table_3_development_ablation.{csv,tex}`
- `results/sonicom_bounded_mcar_film_correction_e25_paper/figure_1_main_validation.{png,pdf}`
- `results/sonicom_bounded_mcar_film_correction_e25_paper/figure_2_paired_effects.{png,pdf}`
- `results/sonicom_bounded_mcar_film_correction_e25_paper/manifest.json`

