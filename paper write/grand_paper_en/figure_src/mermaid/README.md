# Editable Mermaid sources for Figures 1 and 2

- `fig1_sparse_to_interpolated_hrtf_workflow.mmd` is the end-to-end workflow from sparse binaural observations through MCA, FSC magnitude correction, spectrum assembly, and HRIR/HRTF output.
- `fig2_fsc_single_model_architecture.mmd` is the detailed single-model FSC architecture and intentionally ends at the normalized residual output.

Open an `.mmd` file in a Mermaid-compatible editor, or paste its contents into a Mermaid import panel in a diagram editor. Mermaid performs automatic layout, so these files preserve the current nodes, connections, grouping, terminology, and color families rather than the exact millimetre-level positions of the submission PDFs.

The Matplotlib source and exported PDF/SVG/PNG/TIFF files remain the authoritative submission figures. After editing a Mermaid diagram, mirror any accepted scientific-content change back into `../make_submission_figures.py` and the matching manuscript caption before final export.
