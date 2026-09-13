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
