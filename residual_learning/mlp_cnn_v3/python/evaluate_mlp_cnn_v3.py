"""Exhaustively evaluate MCA, the frozen v2 MLP, and the v3 MLP + CNN."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch


SCRIPT_ROOT = Path(__file__).resolve().parent
V3_ROOT = SCRIPT_ROOT.parent
RESIDUAL_LEARNING_ROOT = V3_ROOT.parent
SHARED_PYTHON_ROOT = RESIDUAL_LEARNING_ROOT / "python"
for import_root in (SCRIPT_ROOT, SHARED_PYTHON_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from mlp_cnn_model import (  # noqa: E402
    ResidualMLPCNN,
    total_parameter_count,
)
from residual_data import (  # noqa: E402
    Normalization,
    build_input_features,
    list_hdf5_files,
)


@dataclass
class ErrorAccumulator:
    sample_count: int = 0
    mca_absolute_error: float = 0.0
    mca_squared_error: float = 0.0
    v2_absolute_error: float = 0.0
    v2_squared_error: float = 0.0
    v3_absolute_error: float = 0.0
    v3_squared_error: float = 0.0
    cnn_delta_absolute_sum: float = 0.0

    def add(
        self,
        target_db: torch.Tensor,
        v2_prediction_db: torch.Tensor,
        v3_prediction_db: torch.Tensor,
        cnn_delta_db: torch.Tensor,
    ) -> None:
        mca_error = target_db
        v2_error = target_db - v2_prediction_db
        v3_error = target_db - v3_prediction_db
        self.sample_count += target_db.numel()
        self.mca_absolute_error += float(
            torch.sum(torch.abs(mca_error), dtype=torch.float64).item()
        )
        self.mca_squared_error += float(
            torch.sum(mca_error.square(), dtype=torch.float64).item()
        )
        self.v2_absolute_error += float(
            torch.sum(torch.abs(v2_error), dtype=torch.float64).item()
        )
        self.v2_squared_error += float(
            torch.sum(v2_error.square(), dtype=torch.float64).item()
        )
        self.v3_absolute_error += float(
            torch.sum(torch.abs(v3_error), dtype=torch.float64).item()
        )
        self.v3_squared_error += float(
            torch.sum(v3_error.square(), dtype=torch.float64).item()
        )
        self.cnn_delta_absolute_sum += float(
            torch.sum(torch.abs(cnn_delta_db), dtype=torch.float64).item()
        )

    def merge(self, other: "ErrorAccumulator") -> None:
        for field_name in asdict(self):
            setattr(
                self,
                field_name,
                getattr(self, field_name) + getattr(other, field_name),
            )

    def metrics(self) -> dict[str, float | int]:
        if self.sample_count == 0:
            raise ValueError("Cannot calculate metrics without samples")
        mca_mae = self.mca_absolute_error / self.sample_count
        v2_mae = self.v2_absolute_error / self.sample_count
        v3_mae = self.v3_absolute_error / self.sample_count
        return {
            "sample_count": self.sample_count,
            "mca_zero_residual_mae_db": mca_mae,
            "mca_zero_residual_rmse_db": math.sqrt(
                self.mca_squared_error / self.sample_count
            ),
            "v2_mlp_mae_db": v2_mae,
            "v2_mlp_rmse_db": math.sqrt(
                self.v2_squared_error / self.sample_count
            ),
            "v3_mlp_cnn_mae_db": v3_mae,
            "v3_mlp_cnn_rmse_db": math.sqrt(
                self.v3_squared_error / self.sample_count
            ),
            "v2_vs_mca_mae_reduction_percent": (
                100.0 * (mca_mae - v2_mae) / mca_mae
            ),
            "v3_vs_mca_mae_reduction_percent": (
                100.0 * (mca_mae - v3_mae) / mca_mae
            ),
            "v3_vs_v2_mae_reduction_percent": (
                100.0 * (v2_mae - v3_mae) / v2_mae
            ),
            "cnn_delta_mean_absolute_db": (
                self.cnn_delta_absolute_sum / self.sample_count
            ),
        }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="val",
    )
    parser.add_argument("--directions-per-block", type=int, default=32)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required together with --split test to avoid accidental test access.",
    )
    return parser.parse_args()


def decode_attribute(value: object) -> str:
    scalar = np.asarray(value).item()
    return scalar.decode() if isinstance(scalar, bytes) else str(scalar)


def infer_model_architecture(
    state_dict: dict[str, torch.Tensor],
) -> tuple[int, int, int]:
    mlp_width = int(state_dict["mlp.input.0.weight"].shape[0])
    block_indices = {
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("mlp.blocks.")
    }
    if not block_indices:
        raise ValueError("Checkpoint has no MLP residual blocks")
    mlp_block_count = max(block_indices) + 1
    cnn_channels = int(state_dict["cnn.stem.1.weight"].shape[0])
    return mlp_width, mlp_block_count, cnn_channels


def build_point_features(
    mca_db: np.ndarray,
    correction_db: np.ndarray,
    direction_features: np.ndarray,
    frequency_hz: np.ndarray,
    normalization: Normalization,
) -> np.ndarray:
    ear_features: list[np.ndarray] = []
    direction_count = mca_db.shape[1]
    frequency_count = mca_db.shape[2]
    for ear_index in range(2):
        ear_features.append(
            build_input_features(
                mca_db[ear_index],
                correction_db[ear_index],
                direction_features,
                frequency_hz,
                ear_index,
                normalization,
            ).reshape(direction_count, frequency_count, 7)
        )
    return np.transpose(
        np.stack(ear_features, axis=0),
        (1, 0, 2, 3),
    ).astype(np.float32)


def write_per_subject_csv(
    path: Path,
    rows: list[dict[str, float | int | str]],
) -> None:
    if not rows:
        raise ValueError("No per-subject rows were produced")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main() -> None:
    arguments = parse_arguments()
    if arguments.split == "test" and not arguments.allow_test:
        raise ValueError("--split test requires explicit --allow-test")
    if arguments.directions_per_block < 1:
        raise ValueError("directions_per_block must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v3 exhaustive evaluation")
    device = torch.device("cuda")
    use_amp = not arguments.no_amp
    normalization = Normalization.from_json(
        arguments.dataset_root / "training_statistics.json"
    )
    files = list_hdf5_files(
        arguments.dataset_root,
        split=arguments.split,
    )
    checkpoint = torch.load(
        arguments.checkpoint,
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint["model_state"]
    mlp_width, mlp_block_count, cnn_channels = infer_model_architecture(
        state_dict
    )
    model = ResidualMLPCNN(
        mlp_width=mlp_width,
        mlp_block_count=mlp_block_count,
        cnn_channels=cnn_channels,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    output_dir = arguments.output_dir or arguments.checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    total = ErrorAccumulator()
    per_subject_rows: list[dict[str, float | int | str]] = []
    evaluation_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for file_index, file_path in enumerate(files, start=1):
        subject = ErrorAccumulator()
        with h5py.File(file_path, "r") as handle:
            subject_id = int(np.asarray(handle.attrs["subject_id"]).item())
            file_split = decode_attribute(handle.attrs["split"])
            if file_split != arguments.split:
                raise ValueError(
                    f"{file_path} has split={file_split}, expected {arguments.split}"
                )
            shape = handle["target_residual_db"].shape
            if len(shape) != 3 or shape[0] != 2:
                raise ValueError(f"Unexpected target shape {shape} in {file_path}")
            direction_count = int(shape[1])
            frequency_count = int(shape[2])
            frequency_hz = np.squeeze(handle["frequency_hz"][:]).astype(
                np.float32
            )
            if frequency_hz.size != frequency_count:
                raise ValueError(f"Frequency mismatch in {file_path}")

            for direction_start in range(
                0,
                direction_count,
                arguments.directions_per_block,
            ):
                direction_stop = min(
                    direction_count,
                    direction_start + arguments.directions_per_block,
                )
                direction_slice = slice(direction_start, direction_stop)
                mca_db = np.asarray(
                    handle["mca_logmag_db"][:, direction_slice, :],
                    dtype=np.float32,
                )
                correction_db = np.asarray(
                    handle["correction_logmag_db"][:, direction_slice, :],
                    dtype=np.float32,
                )
                target_db = np.asarray(
                    handle["target_residual_db"][:, direction_slice, :],
                    dtype=np.float32,
                )
                direction_features = np.asarray(
                    handle["direction_features"][direction_slice, :],
                    dtype=np.float32,
                )
                point_features = build_point_features(
                    mca_db,
                    correction_db,
                    direction_features,
                    frequency_hz,
                    normalization,
                )
                point_tensor = torch.from_numpy(point_features).to(
                    device,
                    non_blocking=True,
                )
                target_tensor = torch.from_numpy(target_db).to(
                    device,
                    non_blocking=True,
                )
                with torch.amp.autocast("cuda", enabled=use_amp):
                    v3_normalized, v2_normalized, delta_normalized = model(
                        point_tensor
                    )
                v2_prediction_db = (
                    v2_normalized.permute(1, 0, 2).float()
                    * normalization.target_std
                    + normalization.target_mean
                )
                v3_prediction_db = (
                    v3_normalized.permute(1, 0, 2).float()
                    * normalization.target_std
                    + normalization.target_mean
                )
                cnn_delta_db = (
                    delta_normalized.permute(1, 0, 2).float()
                    * normalization.target_std
                )
                subject.add(
                    target_tensor.float(),
                    v2_prediction_db,
                    v3_prediction_db,
                    cnn_delta_db,
                )

        expected_subject_samples = 2 * direction_count * frequency_count
        if subject.sample_count != expected_subject_samples:
            raise AssertionError(
                f"pp{subject_id} sample count {subject.sample_count} "
                f"!= {expected_subject_samples}"
            )
        total.merge(subject)
        subject_metrics = subject.metrics()
        per_subject_rows.append(
            {
                "subject_id": subject_id,
                **subject_metrics,
            }
        )
        print(
            f"[{file_index:02d}/{len(files):02d}] pp{subject_id} "
            f"v2_mae={subject_metrics['v2_mlp_mae_db']:.4f} dB "
            f"v3_mae={subject_metrics['v3_mlp_cnn_mae_db']:.4f} dB "
            f"v3_vs_v2={subject_metrics['v3_vs_v2_mae_reduction_percent']:.2f}%"
        )

    aggregate = total.metrics()
    expected_total_samples = sum(
        int(row["sample_count"]) for row in per_subject_rows
    )
    if total.sample_count != expected_total_samples:
        raise AssertionError("Aggregate and per-subject sample counts differ")
    v2_reference_metrics_path = (
        RESIDUAL_LEARNING_ROOT / "runs" / "mlp_n03_v2" / f"{arguments.split}_metrics.json"
    )
    v2_reference_comparison: Optional[dict[str, object]] = None
    if v2_reference_metrics_path.exists():
        reference = json.loads(
            v2_reference_metrics_path.read_text(encoding="utf-8")
        )
        mae_difference = (
            float(aggregate["v2_mlp_mae_db"])
            - float(reference["model_mae_db"])
        )
        rmse_difference = (
            float(aggregate["v2_mlp_rmse_db"])
            - float(reference["model_rmse_db"])
        )
        v2_reference_comparison = {
            "reference_path": str(v2_reference_metrics_path),
            "mae_difference_db": mae_difference,
            "rmse_difference_db": rmse_difference,
            "matches_within_1e_4_db": (
                abs(mae_difference) < 1e-4
                and abs(rmse_difference) < 1e-4
            ),
        }
        if not bool(
            v2_reference_comparison["matches_within_1e_4_db"]
        ):
            raise AssertionError(
                "Embedded v2 metrics do not match the saved v2 evaluation"
            )

    metrics = {
        "status": "completed",
        "split": arguments.split,
        "subject_file_count": len(files),
        **aggregate,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "training_stage": checkpoint.get("training_stage", "unknown"),
        "model_parameter_count": total_parameter_count(model),
        "device": torch.cuda.get_device_name(0),
        "amp": use_amp,
        "directions_per_block": arguments.directions_per_block,
        "elapsed_seconds": time.perf_counter() - evaluation_started,
        "peak_cuda_allocated_mib": (
            torch.cuda.max_memory_allocated() / (1024.0**2)
        ),
        "improved_subject_count_v3_vs_v2": sum(
            float(row["v3_mlp_cnn_mae_db"])
            < float(row["v2_mlp_mae_db"])
            for row in per_subject_rows
        ),
        "v2_reference_comparison": v2_reference_comparison,
    }
    metrics_path = output_dir / f"{arguments.split}_full_metrics.json"
    per_subject_path = (
        output_dir / f"{arguments.split}_per_subject_metrics.csv"
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    write_per_subject_csv(per_subject_path, per_subject_rows)
    print(json.dumps(metrics, indent=2))
    print(f"metrics_output={metrics_path}")
    print(f"per_subject_output={per_subject_path}")


if __name__ == "__main__":
    main()
