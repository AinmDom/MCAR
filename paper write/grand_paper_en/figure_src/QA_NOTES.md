# CSMT English figure QA notes

## Data and scope

- The reconstruction-workflow figure is a data-free schematic grounded in the implemented inference path: 463 selected magnitude bins are corrected, 50 remaining single-sided bins preserve their original MCA complex values, and the resulting 513-bin spectrum is conjugate-extended before a 1024-point IFFT and 256-sample crop.
- The nested-grid figure uses all 90 rows in the frozen geometry CSV: 14 Q14 rows, 26 Q26 rows, and 50 Q50 rows. Unique source-direction counts are 14, 26, and 50, and the script asserts `Q14 < Q26 < Q50` as strict set inclusion.
- The ERB trend uses exactly the three authoritative FullSphereERB aggregate rows. Each row represents 44 test subjects and supplies the plotted mean and asymmetric 95% subject-bootstrap interval.
- No subject response, checkpoint, prediction, training artifact, or test-set waveform is read. `test_subjects_read=0`.

## Rendered QA

| Figure/panel | Unique claim | Center and interval | Replicate unit | Final PDF size | Minimum glyph | Alignment | Collision | Visual result |
|---|---|---|---|---:|---:|---|---|---|
| Reconstruction workflow | Sparse observations produce a dense MCA baseline, FSC corrects only selected-bin magnitudes, and MCA phase plus unpredicted bins complete HRIR/HRTF reconstruction | N/A | N/A | 143.5 x 82.0 mm | 5.35 pt | Not applicable | PASS, 0 fail / 0 warn | Five stages read left to right; magnitude correction is visually separated from the phase/outside-bin bypass; 463 corrected and 50 original bins merge before conjugate extension and IFFT; no footer note is embedded in the figure |
| FSC architecture | A frozen FiLM-SIREN base and direction-conditioned spectral CNN form one detailed residual predictor | N/A | N/A | 143.5 x 104.0 mm | 5.3 pt | Not applicable | PASS, 0 fail / 0 warn | Five regions read left to right; all seven feature channels, four dilated blocks, and both residual paths are explicit; the diagram ends at the predicted residual and contains no phase/IFFT/HRIR reconstruction |
| Nested grids a--c | Q14 is nested in Q26 and Q50, with additions shown explicitly | N/A | Direction | 143.5 x 58.0 mm | 6.3 pt | PASS, 1.5 pt tolerance | PASS, 0 fail / 0 warn | Equal sphere sizes and gutters; circle/square/triangle encoding remains interpretable without color; fixed-view and far-side explanations appear only in the manuscript caption |
| ERB trend | Mean full-sphere ERB error decreases from Q14 to Q50 | Mean and 95% subject-bootstrap CI | Subject, n=44 | 120.0 x 68.0 mm | 6.2 pt | Not applicable | PASS, 0 fail / 0 warn | All uncertainty intervals and mean labels are visible; the in-axis statistics note was removed and its full meaning appears in the manuscript caption |

## Source and export checks

- Source validation: 20 PASS, 0 FAIL, 1 reviewed warning. The warning comes from the generic validator interpreting the template-specific width expression as a nonstandard journal width; the exported PDF media boxes independently confirm the intended 143.5, 143.5, 143.5, and 120.0 mm widths.
- Editable text: all four SVG files contain text nodes; all PDFs pass the 5 pt rendered-glyph floor. The workflow SVG contains 43 editable text nodes.
- Export formats per figure: PDF, SVG, 600-dpi PNG, and 600-dpi LZW TIFF.
- The English manuscript references all four PDF files and all targets exist.
- Visual inspection was performed on downscaled previews after the final collision-clean render. The workflow's five stages, three colored information paths, formula wrapping, and arrow endpoints are legible at final width, with no embedded footer or caption-like small print. The architecture was checked region by region for wrapped labels, residual-path routing, and model-only scope. The nested-grid note and the ERB in-axis statistics note were confirmed absent, while the symbol legend, mean labels, and all error-bar endpoints remain visible. No clipping, line-through-text, or ambiguous legend mapping was observed.

## Remaining boundary

Per the author's standing instruction, the LaTeX manuscript was not compiled and the figures were not inspected inside the final CSMT page layout. The author should compile locally and confirm float placement and caption wrapping; figure-internal physical sizes have already been fixed to the intended `includegraphics` widths.
