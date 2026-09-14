# Rebuilding the English manuscript figures

Run from the repository root:

```powershell
python "paper write/grand_paper_en/figure_src/make_submission_figures.py" `
  --skill-scripts "C:/Users/ZhuanZ/.codex/skills/nature-figure/scripts"
```

The script writes editable PDF/SVG and 600-dpi PNG/TIFF exports to `../figure/`,
with rendered alignment and other QA reports kept under `qa/`.
It reads only the frozen Q14/Q26/Q50 geometry CSV and the authoritative aggregate
FullSphereERB summary. The workflow is a data-free schematic grounded in the
implemented reconstruction path. The script does not read subject responses or
run a model.

For a workflow-only revision that preserves the accepted architecture, grid, and
trend exports, add `--only workflow`. For an architecture-only revision, add
`--only architecture`.

Editable structural counterparts of the current manuscript Figures 1 and 2 are
stored under `mermaid/`. These `.mmd` files preserve the figure semantics and
color families while relying on Mermaid's automatic layout.

The manuscript currently includes the author-approved PNG versions of the FSC
workflow and architecture figures. The corresponding PDF/SVG/TIFF files and the
Python generator remain as earlier reproducible exports; rerunning
`--only workflow` or `--only architecture` will overwrite the approved PNGs and
should therefore be done only when intentionally replacing the manual layout.
