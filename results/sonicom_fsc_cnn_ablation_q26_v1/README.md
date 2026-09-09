# Formal FSC spectral-CNN ablation (Q26)

Validation-only paired comparison of fixed `last.pt` checkpoints: FiLM-SIREN E130 versus the same-seed FSC E190 spectral-CNN refinement. All eight endpoints are lower-is-better; paired delta is `FSC E190 - FiLM-SIREN E130`, so negative values favor the CNN stage.

The three seeds are analyzed separately over the same 44 validation subjects. Confidence intervals use the repository `paired_tail_statistics` implementation with 10,000 bootstrap replicates and the manifest-fixed seed. No test subject was read.
