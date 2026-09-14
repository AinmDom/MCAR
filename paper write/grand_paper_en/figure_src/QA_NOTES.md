# CSMT English figure QA notes

## Data and scope

- The reconstruction-workflow figure is a data-free schematic grounded in the implemented inference path: 463 selected magnitude bins are corrected, 50 remaining single-sided bins preserve their original MCA complex values, and the resulting 513-bin spectrum is conjugate-extended before a 1024-point IFFT and 256-sample crop.
- The nested-grid figure uses all 90 rows in the frozen geometry CSV: 14 Q14 rows, 26 Q26 rows, and 50 Q50 rows. Unique source-direction counts are 14, 26, and 50, and the script asserts `Q14 < Q26 < Q50` as strict set inclusion.
- The ERB trend uses exactly the three authoritative FullSphereERB aggregate rows. Each row represents 44 test subjects and supplies the plotted mean and asymmetric 95% subject-bootstrap interval.
- No subject response, checkpoint, prediction, training artifact, or test-set waveform is read. `test_subjects_read=0`.

## Rendered QA

| Figure/panel | Unique claim | Center and interval | Replicate unit | Final PDF size | Minimum glyph | Alignment | Collision | Visual result |
|---|---|---|---|---:|---:|---|---|---|
| Reconstruction workflow | Sparse observations produce a dense MCA baseline, FSC corrects only selected-bin magnitudes, and MCA phase plus unpredicted bins complete HRIR/HRTF reconstruction | N/A | N/A | Author-approved PNG, 1380 x 620 px | Not re-audited | Not applicable | Not applicable to raster replacement | Author-approved five-stage workflow; the PNG is included without resampling |
| FSC architecture | A FiLM-SIREN base and direction-conditioned spectral CNN form one detailed residual predictor | N/A | N/A | Author-approved PNG, 1434 x 690 px | Not re-audited | Not applicable | Not applicable to raster replacement | Author-approved five-region architecture; dilation 1/2/4/8 and both residual paths are explicit |
| Nested grids a--c | Q14 is nested in Q26 and Q50, with additions shown explicitly | N/A | Direction | 143.5 x 58.0 mm | 6.3 pt | PASS, 1.5 pt tolerance | PASS, 0 fail / 0 warn | Equal sphere sizes and gutters; circle/square/triangle encoding remains interpretable without color; fixed-view and far-side explanations appear only in the manuscript caption |
| ERB trend | Mean full-sphere ERB error decreases from Q14 to Q50 | Mean and 95% subject-bootstrap CI | Subject, n=44 | 120.0 x 68.0 mm | 6.2 pt | Not applicable | PASS, 0 fail / 0 warn | All uncertainty intervals and mean labels are visible; the in-axis statistics note was removed and its full meaning appears in the manuscript caption |

## Source and export checks

- Source validation: 20 PASS, 0 FAIL, 1 reviewed warning. The warning comes from the generic validator interpreting the template-specific width expression as a nonstandard journal width; the exported PDF media boxes independently confirm the intended 143.5, 143.5, 143.5, and 120.0 mm widths.
- Earlier code-generated PDF/SVG exports retain editable text and passed their recorded glyph checks. The currently included workflow and architecture assets are author-approved raster PNGs and should not inherit those vector-text claims.
- Earlier reproducible exports remain available as PDF, SVG, PNG, and TIFF. The manuscript now intentionally selects the author-approved PNGs for the workflow and architecture; the quantitative grid and trend figures remain PDF.
- The English manuscript references the two author-approved PNG schematics and the two PDF quantitative figures; all targets exist.
- The author visually reviewed and approved the replacement workflow and architecture PNGs. Their pixel dimensions, hashes, transparent canvas edges, and manuscript targets were checked without resampling. The earlier PDF collision and glyph audits remain provenance for the code-generated exports only, not for the replacement PNGs.

## Remaining boundary

Per the author's standing instruction, the LaTeX manuscript was not compiled and the figures were not inspected inside the final CSMT page layout. The author should compile locally and confirm float placement, caption wrapping, transparent-background compositing, and final raster text size.
