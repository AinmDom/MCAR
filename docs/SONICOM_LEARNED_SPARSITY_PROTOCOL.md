# SONICOM learned-method sparsity protocol

This experiment compares MCAR v3.2, FSP-AE, and RANF on the same 44-subject
SONICOM test cohort at Q6, Q14, and Q26 observed directions. The protocol was
frozen before the new Q6/Q14 results were evaluated.

## Fixed design

- Sparse grids: nested antipodal/maximin subsets of the existing SONICOM-Q26
  grid, stored in `configs/data/sonicom_nested_sparse_grid_q6_q14_q26_v1.csv`.
- Evaluation mask: the same 767 targets for every Q and every method, formed
  by excluding all 26 directions in the complete Q26 grid.
- MCAR: epoch-39 v3.2 checkpoint, frozen; the MCA prior is recomputed from the
  current Q observations.
- FSP-AE: epoch-40 checkpoint, frozen; only the current Q observations are
  encoded. The official CPU HRIR/ITD reconstruction is used at every Q.
- RANF: one frozen Q26-pretrained checkpoint. At each Q, retrieval distances
  are recomputed from current-Q observations using training listeners only,
  then native subject-specific adaptation restarts for 1000 epochs with batch
  size 3.
- Metrics: measured-domain ERB error, contralateral-25-degree ERB error,
  contralateral-hemisphere 10--20 kHz error, and horizontal ILD MAE.
- Primary robustness summaries: paired Q6-minus-Q26 degradation and fitted
  error increase per halving of observed directions.
- Inference: all three method pairs are compared with 10,000 paired-listener
  bootstrap replicates, seed 20260812.

The SONICOM test split had already been accessed by the earlier frozen Q26
study. This is therefore a preregistered new sparsity analysis, not an
untouched external confirmation.

## Reproduction

From the project root:

```powershell
matlab -batch "addpath('matlab'); mcar.prepare_sonicom_learned_sparsity([], [], 4)"
$env:PYTHONPATH = 'src'
D:\miniconda3\envs\ml\python.exe -m mcar.evaluation.predict_sonicom_learned_sparsity --methods mcar
$env:PYTHONPATH = 'src;D:\miniconda3\envs\ml\Lib\site-packages'
D:\miniconda3\envs\asd\python.exe -m mcar.evaluation.predict_sonicom_learned_sparsity --methods fspae
```

RANF preparation and adaptation run in the existing WSL environment using
`ranf.prepare_mcar_learned_sparsity` and
`launch_mcar_learned_sparsity_all.sh` at RANF adapter commit
`dfe929b432d67e7f3506515558763859e17a1cb3`. Export the Q6/Q14 SOFA files
with `scripts/export_ranf_q26_predictions.py`, then run:

```powershell
matlab -batch "addpath('matlab'); mcar.evaluate_sonicom_learned_sparsity"
```

Formal tables, paired statistics, plots, and the machine-readable summary are
written under `results/sonicom_learned_methods_sparsity_v1/`.
