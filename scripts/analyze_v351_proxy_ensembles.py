"""Compare v3.5.1 continuation checkpoints on the locked validation proxy.

All candidates are scored with the original v3.5 scratch objective, regardless
of the additional loss terms used to train branches B and C.  The three best
standalone candidates are then swept as convex output ensembles with the fixed
previous-joint component.  The test split is never listed or read.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from mcar.data import Normalization, list_hdf5_files
from mcar.evaluation.evaluate_mlp_cnn_v3 import (
    infer_global_context_arguments,
    infer_model_architecture,
)
from mcar.models.residual_mlp_cnn import ResidualMLPCNN
from mcar.paths import project_root
from mcar.training.train_mlp_cnn_v3 import evaluate
from mcar.training.train_mlp_v2 import (
    make_erb_center_frequencies_hz,
    make_erb_weights,
)

from fuse_v32_previous_scratch_validation import ResidualOutputEnsemble


ROOT = project_root()
DATASET_ROOT = ROOT / "data" / "processed" / "sonicom_residual_q26_v1"
PREVIOUS_CHECKPOINT = (
    ROOT
    / "artifacts"
    / "training"
    / "sonicom_mlp_cnn_q26_v321_cnn_reinit_joint_unfreeze_e10"
    / "best.pt"
)
SOURCE_CHECKPOINT = (
    ROOT
    / "artifacts"
    / "training"
    / "sonicom_mlp_cnn_q26_v32_scratch_joint_seed20260809_e60"
    / "best.pt"
)
RESULT_ROOT = ROOT / "results" / "sonicom_mlp_cnn_q26_v351_proxy_ensembles"

BRANCH_ROOTS = {
    "a": ROOT
    / "artifacts"
    / "training"
    / "sonicom_mlp_cnn_q26_v351a_continuation_e40",
    "b": ROOT
    / "artifacts"
    / "training"
    / "sonicom_mlp_cnn_q26_v351b_band_ild005_e40",
    "c": ROOT
    / "artifacts"
    / "training"
    / "sonicom_mlp_cnn_q26_v351c_strict09_band_ild005_e40",
}

SWEEP_WEIGHTS = (0.0, 0.25, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_model(path: Path, device: torch.device) -> tuple[ResidualMLPCNN, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state"]
    width, blocks, channels = infer_model_architecture(state_dict)
    model = ResidualMLPCNN(
        mlp_width=width,
        mlp_block_count=blocks,
        cnn_channels=channels,
        **infer_global_context_arguments(checkpoint),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def resources(device: torch.device):
    files = list_hdf5_files(DATASET_ROOT, split="val")
    if len(files) != 44:
        raise RuntimeError(f"Expected 44 validation files, found {len(files)}")
    normalization = Normalization.from_json(DATASET_ROOT / "training_statistics.json")
    with h5py.File(files[0], "r") as handle:
        frequency_hz = np.squeeze(handle["frequency_hz"][:])
    weights = torch.from_numpy(make_erb_weights(frequency_hz)).to(device)
    log_weights = torch.log(torch.clamp(weights, min=1e-12)).view(
        1, 1, weights.shape[0], weights.shape[1]
    )
    centers = torch.from_numpy(make_erb_center_frequencies_hz()).to(device)
    return files, normalization, log_weights, centers


def score(
    previous: ResidualMLPCNN,
    candidate: ResidualMLPCNN,
    candidate_weight: float,
    files,
    normalization,
    log_weights,
    centers,
    arguments,
    device: torch.device,
) -> dict[str, float]:
    metrics = evaluate(
        ResidualOutputEnsemble(previous, candidate, candidate_weight),
        files,
        normalization,
        96,
        int(arguments.directions_per_batch),
        int(arguments.validation_seed),
        device,
        True,
        log_weights,
        centers,
        arguments,
    )
    return asdict(metrics)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    candidates = {
        f"v351{branch}_{kind}": root / filename
        for branch, root in BRANCH_ROOTS.items()
        for kind, filename in (("best_total", "best.pt"), ("best_strict", "best_strict_ild.pt"))
    }
    for path in (DATASET_ROOT, PREVIOUS_CHECKPOINT, SOURCE_CHECKPOINT, *candidates.values()):
        if not path.exists():
            raise FileNotFoundError(path)

    started = time.perf_counter()
    device = torch.device("cuda")
    previous, _ = load_model(PREVIOUS_CHECKPOINT, device)
    source, source_checkpoint = load_model(SOURCE_CHECKPOINT, device)
    arguments = type("Arguments", (), source_checkpoint["arguments"])()
    files, normalization, log_weights, centers = resources(device)

    reference_metrics = score(
        previous, source, 0.7, files, normalization, log_weights, centers, arguments, device
    )
    print(json.dumps({"reference_v35": reference_metrics}), flush=True)

    standalone_rows: list[dict[str, object]] = []
    loaded: dict[str, ResidualMLPCNN] = {}
    for label, path in candidates.items():
        model, checkpoint = load_model(path, device)
        loaded[label] = model
        metrics = score(
            previous, model, 1.0, files, normalization, log_weights, centers, arguments, device
        )
        row = {
            "candidate": label,
            "checkpoint": str(path.relative_to(ROOT)),
            "checkpoint_sha256": sha256(path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            **metrics,
        }
        standalone_rows.append(row)
        print(json.dumps({"standalone": row}), flush=True)

    finalists = sorted(standalone_rows, key=lambda row: float(row["total_loss"]))[:3]
    sweep_rows: list[dict[str, object]] = []
    for finalist in finalists:
        label = str(finalist["candidate"])
        for weight in SWEEP_WEIGHTS:
            metrics = score(
                previous,
                loaded[label],
                weight,
                files,
                normalization,
                log_weights,
                centers,
                arguments,
                device,
            )
            row = {"candidate": label, "candidate_weight": weight, **metrics}
            sweep_rows.append(row)
            print(json.dumps({"ensemble": row}), flush=True)

    best = min(sweep_rows, key=lambda row: float(row["total_loss"]))
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "selection_split": "val",
        "validation_subject_count": 44,
        "validation_steps": 96,
        "fixed_validation_sampler_seed": int(arguments.validation_seed),
        "objective_source": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
        "previous_checkpoint": str(PREVIOUS_CHECKPOINT.relative_to(ROOT)),
        "reference_v35": {"candidate_weight": 0.7, **reference_metrics},
        "finalists": [row["candidate"] for row in finalists],
        "best": best,
        "relative_total_improvement_vs_v35_percent": (
            100.0
            * (float(reference_metrics["total_loss"]) - float(best["total_loss"]))
            / float(reference_metrics["total_loss"])
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "test_subject_count_read": 0,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(RESULT_ROOT / "standalone_candidates.csv", standalone_rows)
    write_csv(RESULT_ROOT / "ensemble_sweep.csv", sweep_rows)
    (RESULT_ROOT / "proxy_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
