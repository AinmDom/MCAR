# CSMT English figure contract

## Figure 1: sparse-to-interpolated HRTF reconstruction workflow

- Core conclusion: sparse binaural observations first produce a dense MCA baseline; FSC corrects only selected-bin magnitudes, and the MCA phase plus unpredicted complex bins complete the interpolated HRIR/HRTF reconstruction.
- Results-level question: How does the full inference pipeline transform sparse measured HRTFs into interpolated binaural responses at target directions?
- Archetype: schematic-led method workflow.
- Final size: 143.5 mm wide at `0.98\linewidth`.
- Evidence hierarchy: sparse measurements -> MCA baseline -> FSC residual correction -> complex-spectrum assembly -> IFFT and dense output.
- Reviewer risk: confusing the complete inference pipeline with the model architecture, implying that FSC predicts phase, or omitting reuse of the 50 unpredicted complex MCA bins.

## Figure 2: FSC architecture

- Core conclusion: FSC combines a subject-conditioned continuous residual field with binaural spectral refinement to predict the normalized residual.
- Results-level question: What information enters the model, how are the FiLM-SIREN and spectral CNN implemented, and how are their residuals combined?
- Archetype: schematic-led method figure.
- Final size: 143.5 mm wide at `0.98\linewidth`.
- Evidence hierarchy: inputs -> frozen FiLM-SIREN base -> seven-channel feature builder -> trainable spectral CNN -> residual output.
- Reviewer risk: obsolete Hybrid/ensemble terminology, unreadable implementation detail, or including a signal-reconstruction stage that is outside the model architecture.

## Figure 3: nested sparse grids

- Core conclusion: Q14 is nested inside Q26, which is nested inside Q50, while newly added directions progressively densify the measured domain.
- Results-level question: Which directions are retained or added at each observation density?
- Archetype: quantitative geometric triptych.
- Final size: 143.5 mm wide at `0.98\linewidth`.
- Panel map: a, Q14; b, Q26; c, Q50.
- Evidence hierarchy: Q14 core directions, Q26 additions, Q50 additions.
- Reviewer risk: color-only encoding, unlabeled orientation cues, interactive-viewer residue, and inconsistent panel geometry.

## Figure 4: sparsity trend

- Core conclusion: mean full-sphere ERB error decreases as the number of observed directions increases from 14 to 50, with subject uncertainty shown explicitly.
- Results-level question: How does FSC full-sphere ERB error vary with observation density?
- Archetype: single-panel quantitative trend.
- Final size: 120 mm wide at `0.82\linewidth`.
- Statistics: mean and 95% subject-bootstrap confidence interval, `n=44` subjects; no significance claim.
- Reviewer risk: hiding between-subject uncertainty, calling an error metric simply ERB, and visually exaggerating the decrease with a narrow axis.

## Shared export contract

- Backend: Python / Matplotlib only.
- Font: embedded sans serif; every rendered glyph at least 5 pt, target at least 6 pt.
- Formats: editable SVG and PDF plus 600-dpi PNG and LZW-compressed TIFF.
- QA: source validation, panel-alignment audit where applicable, PDF text-size audit, collision audit, and visual inspection at final physical size.
- Data integrity: ground the workflow in the implemented 463-selected-bin/50-unpredicted-bin reconstruction path; use all frozen Q14/Q26/Q50 grid rows and the authoritative FullSphereERB summary without exclusions or invented values.
