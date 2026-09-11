# ARI external FSC replication bundle

This directory is the curated result bundle for the single-seed ARI external-database replication. It compares MCA with FSC retrained on ARI using the frozen FSC architecture, losses, frequency representation, E130 FiLM-SIREN budget, E190 spectral-CNN budget, seed `20260911`, and the E190 endpoint checkpoint.

The strict preregistered decision rule was met: all five subject-paired `FSC-MCA` mean differences and bootstrap 95% upper bounds are below zero. Every metric improved for all 22 held-out subjects. See [REPORT.md](REPORT.md) for the concise numerical result.

## Authoritative result files

- `mca_fsc_subject_level_metrics.csv`: 220 rows (`22 subjects × 2 methods × 5 metrics`).
- `five_metric_summary.csv`: method means and sample standard deviations.
- `paired_bootstrap.csv`: five subject-paired differences with 10,000-replicate percentile intervals.
- `summary.json`: machine-readable status, checkpoint identity, test-access timing, and integrity flags.
- `erb_subject_level.csv` and `secondary_subject_level.csv`: the two frozen evaluator outputs merged into the authoritative subject-level matrix.

## Frozen protocol and provenance

- Source inventory: `configs/data/ari_hrtf_b_nh_source_inventory_v1.json`.
- Eligible cohort and deterministic split: `configs/data/ari_hrtf_b_nh_adapted_q26_subject_split_v2.csv`.
- Symmetric ARI-adapted Q26: `configs/data/ari_hrtf_b_nh_adapted_q26_v2.csv`.
- Pre-training protocol: `configs/experiments/ari_fsc_adapted_q26_external_replication_v2.json`.
- Training execution: `configs/experiments/ari_fsc_adapted_q26_execution_v2.json` and `configs/experiments/ari_fsc_adapted_q26_stage_d_seed20260911_e190.json`.
- One-shot test protocol: `configs/experiments/ari_fsc_adapted_q26_locked_test_v1.json`.
- Lightweight run evidence: `artifacts/ari_fsc_adapted_q26_v2/` plus the selected `configuration.json`, `history.csv`, `best_validation_per_subject.csv`, and `training_report.json` files under the two ARI training directories.

The authoritative E130 and E190 checkpoint SHA256 values are recorded in the training reports. Checkpoint binaries are intentionally not versioned.

## Execution chain

The reproducible implementation is split into independent ARI-only adapters:

1. `scripts/acquire_ari_hrtf_b_nh.py` acquires the official public SOFA files and records hashes and metadata.
2. `scripts/freeze_ari_adapted_q26_protocol.py` freezes the homogeneous cohort, v2 split, and symmetric Q26.
3. `scripts/prepare_ari_fsc_waveforms.py` and `matlab/+mcar/prepare_ari_fsc_{split,subject}.m` prepare train/validation waveforms and MCA residual features.
4. `scripts/compute_ari_fsc_training_statistics.py`, `scripts/train_ari_fsc_stage_b.py`, and `scripts/train_ari_fsc_stage_d.py` implement the frozen E130/E190 training path.
5. `scripts/run_ari_fsc_locked_test.ps1` binds the guarded test preparation, MCA/FSC prediction, five metrics, and paired-bootstrap finalization.

The committed `test_access_state.json` records that the authorized test pass has already occurred. Do not delete or bypass it to rerun the formal test. Any future repetition must be separately authorized and frozen under a new experiment identity.

## Intentionally excluded

Raw ARI SOFA files, processed HDF5 tensors, reconstruction tensors, checkpoints, validation ledgers, and runtime logs remain local and ignored. Superseded v1 Q26/execution candidates are also excluded; their rejection and supersession remain documented in `docs/EXPERIMENT_LOG.md`.

ARI and SONICOM use different measured direction grids, solid-angle weights, and Q26 definitions. Their absolute metric values must not be pooled as a single experiment.
