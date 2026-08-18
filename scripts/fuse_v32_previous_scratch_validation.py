"""Select and export a validation-only output ensemble of two MLP+CNNs.

The models were trained from different initializations, so the script blends
their predicted normalized residuals instead of averaging incompatible model
weights.  The scratch weight is selected on the same fixed v3.2 validation
sampler used during training.  The locked test split is never listed or read.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from torch import nn

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


ROOT = project_root()
DATASET_ROOT = ROOT / "data" / "processed" / "sonicom_residual_q26_v1"
PREVIOUS_CHECKPOINT = (
    ROOT
    / "artifacts"
    / "training"
    / "sonicom_mlp_cnn_q26_v321_cnn_reinit_joint_unfreeze_e10"
    / "best.pt"
)
SCRATCH_CHECKPOINT = (
    ROOT
    / "artifacts"
    / "training"
    / "sonicom_mlp_cnn_q26_v32_scratch_joint_seed20260809_e60"
    / "best.pt"
)
PREVIOUS_PREDICTIONS = (
    ROOT
    / "artifacts"
    / "reconstruction"
    / "sonicom_q26_validation_v321_cnn_reinit_joint_unfreeze_e10"
)
SCRATCH_PREDICTIONS = (
    ROOT
    / "artifacts"
    / "reconstruction"
    / "sonicom_q26_validation_scratch_joint_e60"
)
OUTPUT_ROOT = (
    ROOT
    / "artifacts"
    / "reconstruction"
    / "sonicom_q26_validation_fusion_previous_scratch"
)
RESULT_ROOT = (
    ROOT
    / "results"
    / "sonicom_mlp_cnn_q26_v32_previous_scratch_fusion"
)


class ResidualOutputEnsemble(nn.Module):
    """Convex output ensemble; ``scratch_weight`` is in [0, 1]."""

    def __init__(
        self,
        previous: ResidualMLPCNN,
        scratch: ResidualMLPCNN,
        scratch_weight: float,
    ) -> None:
        super().__init__()
        self.previous = previous
        self.scratch = scratch
        self.scratch_weight = float(scratch_weight)

    def forward(
        self, point_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        previous_prediction, previous_base, previous_delta = self.previous(
            point_features
        )
        scratch_prediction, scratch_base, scratch_delta = self.scratch(
            point_features
        )
        weight = self.scratch_weight
        return (
            torch.lerp(previous_prediction, scratch_prediction, weight),
            torch.lerp(previous_base, scratch_base, weight),
            torch.lerp(previous_delta, scratch_delta, weight),
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-steps", type=int, default=96)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[ResidualMLPCNN, dict[str, object]]:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
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


def validation_resources(
    device: torch.device,
) -> tuple[list[Path], Normalization, torch.Tensor, torch.Tensor]:
    files = list_hdf5_files(DATASET_ROOT, split="val")
    if len(files) != 44:
        raise RuntimeError(f"Expected 44 validation files, found {len(files)}")
    normalization = Normalization.from_json(
        DATASET_ROOT / "training_statistics.json"
    )
    with h5py.File(files[0], "r") as handle:
        frequency_hz = np.squeeze(handle["frequency_hz"][:])
    weights = torch.from_numpy(make_erb_weights(frequency_hz)).to(device)
    log_weights = torch.log(torch.clamp(weights, min=1e-12)).view(
        1, 1, weights.shape[0], weights.shape[1]
    )
    centers = torch.from_numpy(make_erb_center_frequencies_hz()).to(device)
    return files, normalization, log_weights, centers


def evaluate_weight(
    previous: ResidualMLPCNN,
    scratch: ResidualMLPCNN,
    scratch_weight: float,
    files: list[Path],
    normalization: Normalization,
    log_erb_weights: torch.Tensor,
    band_centers: torch.Tensor,
    training_arguments: argparse.Namespace,
    validation_steps: int,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    ensemble = ResidualOutputEnsemble(previous, scratch, scratch_weight)
    metrics = evaluate(
        ensemble,
        files,
        normalization,
        validation_steps,
        int(training_arguments.directions_per_batch),
        int(training_arguments.validation_seed),
        device,
        use_amp,
        log_erb_weights,
        band_centers,
        training_arguments,
    )
    return {"scratch_weight": scratch_weight, **asdict(metrics)}


def candidate_weights() -> tuple[list[float], float]:
    return [0.0, 0.25, 0.5, 0.75, 1.0], 0.05


def copy_ensemble_predictions(scratch_weight: float) -> int:
    previous_files = sorted(
        PREVIOUS_PREDICTIONS.glob("subjects/*/prediction.h5")
    )
    scratch_files = sorted(
        SCRATCH_PREDICTIONS.glob("subjects/*/prediction.h5")
    )
    if len(previous_files) != 44 or len(scratch_files) != 44:
        raise RuntimeError("Both validation prediction roots must contain 44 subjects")
    if [p.parent.name for p in previous_files] != [
        p.parent.name for p in scratch_files
    ]:
        raise RuntimeError("Prediction subject lists do not match")

    for previous_path, scratch_path in zip(previous_files, scratch_files):
        subject_label = previous_path.parent.name
        output_path = OUTPUT_ROOT / "subjects" / subject_label / "prediction.h5"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(".h5.partial")
        temporary_path.unlink(missing_ok=True)
        with h5py.File(previous_path, "r") as previous_handle, h5py.File(
            scratch_path, "r"
        ) as scratch_handle:
            previous = np.asarray(
                previous_handle["predicted_residual_db"][:], dtype=np.float32
            )
            scratch = np.asarray(
                scratch_handle["predicted_residual_db"][:], dtype=np.float32
            )
            if previous.shape != scratch.shape:
                raise RuntimeError(f"Prediction shape mismatch for {subject_label}")
            fused = previous + np.float32(scratch_weight) * (scratch - previous)
            with h5py.File(temporary_path, "w") as destination:
                destination.create_dataset(
                    "predicted_residual_db",
                    data=fused,
                    chunks=(1, min(64, fused.shape[1]), fused.shape[2]),
                    compression="gzip",
                    compression_opts=4,
                )
                for name, value in previous_handle.attrs.items():
                    destination.attrs[name] = value
                destination.attrs["model_family"] = "ResidualOutputEnsemble"
                destination.attrs["model_version"] = "v3.5"
                destination.attrs["model_name"] = "SONICOM Q26 MCAR v3.5"
                destination.attrs["checkpoint"] = "output-level ensemble"
                destination.attrs["previous_prediction"] = str(
                    previous_path.resolve()
                )
                destination.attrs["scratch_prediction"] = str(
                    scratch_path.resolve()
                )
                destination.attrs["previous_weight"] = 1.0 - scratch_weight
                destination.attrs["scratch_weight"] = scratch_weight
                destination.attrs["selection_split"] = "val"
                destination.attrs["test_subject_count_read"] = 0
        temporary_path.replace(output_path)
    return len(previous_files)


def write_results(
    sweep: list[dict[str, float]],
    best: dict[str, float],
    subject_count: int,
    validation_steps: int,
    elapsed_seconds: float,
) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with (RESULT_ROOT / "validation_proxy_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=sweep[0].keys())
        writer.writeheader()
        writer.writerows(sweep)
    report = {
        "schema_version": "1.0",
        "model_version": "v3.5",
        "model_name": "SONICOM Q26 MCAR v3.5",
        "status": "completed",
        "fusion_type": "convex output-level residual ensemble",
        "weight_formula": (
            "fused=(1-scratch_weight)*previous+scratch_weight*scratch"
        ),
        "selection_split": "val",
        "test_subject_count_read": 0,
        "validation_subject_count": 44,
        "validation_steps": validation_steps,
        "previous_checkpoint": str(PREVIOUS_CHECKPOINT.relative_to(ROOT)),
        "scratch_checkpoint": str(SCRATCH_CHECKPOINT.relative_to(ROOT)),
        "best": best,
        "prediction_subject_count": subject_count,
        "output_root": str(OUTPUT_ROOT.relative_to(ROOT)),
        "elapsed_seconds": elapsed_seconds,
        "sweep": sweep,
    }
    (RESULT_ROOT / "fusion_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_ROOT / "inference_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    cli = parse_arguments()
    if cli.validation_steps < 1:
        raise ValueError("validation-steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for path in (
        DATASET_ROOT,
        PREVIOUS_CHECKPOINT,
        SCRATCH_CHECKPOINT,
        PREVIOUS_PREDICTIONS,
        SCRATCH_PREDICTIONS,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    started = time.perf_counter()
    device = torch.device("cuda")
    previous, _ = load_model(PREVIOUS_CHECKPOINT, device)
    scratch, scratch_checkpoint = load_model(SCRATCH_CHECKPOINT, device)
    training_arguments = argparse.Namespace(**scratch_checkpoint["arguments"])
    files, normalization, log_weights, centers = validation_resources(device)
    use_amp = not cli.no_amp

    coarse, refinement_step = candidate_weights()
    results: dict[float, dict[str, float]] = {}
    for weight in coarse:
        result = evaluate_weight(
            previous,
            scratch,
            weight,
            files,
            normalization,
            log_weights,
            centers,
            training_arguments,
            cli.validation_steps,
            device,
            use_amp,
        )
        results[weight] = result
        print(json.dumps(result), flush=True)
    coarse_best = min(results.values(), key=lambda row: row["total_loss"])
    center = coarse_best["scratch_weight"]
    refinement = np.arange(
        max(0.0, center - 0.20),
        min(1.0, center + 0.20) + refinement_step / 2,
        refinement_step,
    )
    for raw_weight in refinement:
        weight = round(float(raw_weight), 2)
        if weight in results:
            continue
        result = evaluate_weight(
            previous,
            scratch,
            weight,
            files,
            normalization,
            log_weights,
            centers,
            training_arguments,
            cli.validation_steps,
            device,
            use_amp,
        )
        results[weight] = result
        print(json.dumps(result), flush=True)

    sweep = sorted(results.values(), key=lambda row: row["scratch_weight"])
    best = min(sweep, key=lambda row: row["total_loss"])
    subject_count = copy_ensemble_predictions(best["scratch_weight"])
    write_results(
        sweep,
        best,
        subject_count,
        cli.validation_steps,
        time.perf_counter() - started,
    )
    print(json.dumps({"status": "completed", "best": best}, indent=2))


if __name__ == "__main__":
    main()
