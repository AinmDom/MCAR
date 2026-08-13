# SONICOM eight-method sparsity protocol

This experiment extends the completed SONICOM learned-method sparsity study
with the five remaining methods from the frozen horizontal comparison:
SH only, SUpDEq + SH, SUpDEq + Natural Neighbor, SUpDEq + Barycentric, and
MCA. The resulting comparison contains eight methods after adding the existing
MCAR v3.2, FSP-AE, and RANF rows.

## Frozen design

- Subjects: the same 44 listeners in the frozen SONICOM test split.
- Observations: the same nested Q6, Q14, and Q26 grids.
- Evaluation: the same 767 targets at every Q, obtained by excluding the
  complete Q26 input set.
- Baseline definitions: identical to the existing SONICOM Q26 horizontal
  evaluator. All SH methods use `Nmax=3`; SH-only and SUpDEq+SH use Tikhonov
  epsilon `1e-2`. Natural-neighbor and barycentric variants execute the
  upstream native SUpDEq interpolation directly. Exact preservation of the
  observed HRTFs is checked for every listener and Q.
- MCA: the current-Q reconstruction already frozen in the completed sparsity
  caches; it is not substituted by the Q26-only historical result.
- Learned methods: the 1,584 listener-level rows committed in `4879ca2` are
  read unchanged. MCAR, FSP-AE, and RANF are not rerun for this extension.
- Metrics and inference: the same four strict metrics, Q6-minus-Q26
  degradation, slope per direction-count halving, and 10,000 paired-listener
  bootstrap replicates with seed `20260812`.

The SONICOM test set was previously accessed. This is a frozen extension of an
existing test analysis, not a new untouched confirmation.

## Outputs

The five-baseline audit is written to
`results/sonicom_five_baseline_sparsity_v1/`. The combined tables, all-pair
comparisons, learned-versus-baseline comparisons, rankings, plots, and
machine-readable summary are written to
`results/sonicom_eight_method_sparsity_v1/`.
