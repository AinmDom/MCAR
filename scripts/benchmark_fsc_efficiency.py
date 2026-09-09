"""Benchmark complete FSC parameter counts and validation inference latency."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import torch

from mcar.paths import project_root
from mcar.predictors import (
    FilmSirenSpectralCNNPredictor,
    _read_query_grid,
    conditioned_coordinate_block,
)
from mcar.q26_condition import build_q26_condition, build_sparse_condition
from mcar.q26_condition import sparse_source_indices
from mcar.training.train_film_siren import file_sha256, split_subject_paths
from mcar.training.train_siren import frequency_coordinates


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("configuration", type=Path)
    return parser.parse_args()


def manifest_identity(payload: dict[str, Any]) -> str:
    values = dict(payload)
    claimed = str(values.pop("identity_sha256")).upper()
    actual = hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest().upper()
    if actual != claimed:
        raise ValueError("Manifest identity mismatch")
    return actual


def synchronize() -> None:
    torch.cuda.synchronize()


def percentile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability, method="linear"))


def timed(
    function: Callable[[], torch.Tensor], warmups: int, repeats: int
) -> dict[str, Any]:
    for _ in range(warmups):
        function()
    synchronize()
    torch.cuda.reset_peak_memory_stats()
    values: list[float] = []
    output: torch.Tensor | None = None
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter_ns()
        output = function()
        synchronize()
        values.append((time.perf_counter_ns() - started) / 1e6)
    if output is None or output.shape != (2, 793, 463) or not bool(torch.isfinite(output).all()):
        raise FloatingPointError("FSC benchmark output is invalid")
    return {
        "warmup_count": warmups,
        "timed_repeat_count": repeats,
        "median_latency_ms": float(np.median(values)),
        "p95_latency_ms": percentile(values, 0.95),
        "minimum_latency_ms": float(np.min(values)),
        "maximum_latency_ms": float(np.max(values)),
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated() / 2**20),
        "peak_reserved_mib": float(torch.cuda.max_memory_reserved() / 2**20),
        "latencies_ms": values,
    }


def parameter_counts(predictors: list[FilmSirenSpectralCNNPredictor]) -> dict[str, int]:
    if not predictors:
        raise ValueError("At least one predictor is required")
    totals = [sum(value.numel() for value in item.model.parameters()) for item in predictors]
    trainables = [
        sum(value.numel() for value in item.model.parameters() if value.requires_grad)
        for item in predictors
    ]
    if len(set(totals)) != 1 or len(set(trainables)) != 1:
        raise AssertionError("FSC ensemble members do not have equal parameter counts")
    member_total = totals[0]
    member_trainable = trainables[0]
    frozen = member_total - member_trainable
    member_count = len(predictors)
    return {
        "member_trainable": member_trainable,
        "member_frozen": frozen,
        "member_total": member_total,
        "ensemble_trainable": member_trainable * member_count,
        "ensemble_frozen": frozen * member_count,
        "ensemble_instantiated_total": member_total * member_count,
        "ensemble_unique_deployed": member_total * member_count,
        "member_count": member_count,
        "parameter_sharing": 0,
    }


def prepare_blocks(
    predictor: FilmSirenSpectralCNNPredictor,
    source_path: Path,
    dataset_root: Path,
    split_csv: Path,
    q26_csv: Path,
    observation_count: int | None,
    directions_per_block: int,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    with h5py.File(source_path, "r") as handle:
        subject_id = int(np.asarray(handle.attrs["subject_id"]).item())
        local_mca_db = np.asarray(handle["mca_logmag_db"][:], dtype=np.float32)
        correction_db = np.asarray(handle["correction_logmag_db"][:], dtype=np.float32)
    if observation_count is None:
        condition = build_q26_condition(
            dataset_root, split_csv, subject_id, q26_csv, allow_test=False
        )
    else:
        indices = sparse_source_indices(q26_csv, observation_count)
        condition = build_sparse_condition(
            dataset_root,
            split_csv,
            subject_id,
            indices,
            allow_test=False,
            filename=f"q{observation_count}.h5",
        )
    normalized_condition = predictor.condition_normalization.normalize(
        condition.binaural_magnitude_db
    )
    with torch.no_grad():
        latent = predictor.model.encode_condition(
            torch.from_numpy(normalized_condition).unsqueeze(0).to(predictor.device),
            torch.from_numpy(condition.xyz).unsqueeze(0).to(predictor.device),
            torch.from_numpy(condition.mask).unsqueeze(0).to(predictor.device),
        )
    directions, frequency = _read_query_grid(source_path)
    expected = (2, directions.shape[0], frequency.size)
    if local_mca_db.shape != expected or correction_db.shape != expected:
        raise ValueError(f"Unexpected FSC input shape: {local_mca_db.shape}, {correction_db.shape}")
    frequency_coordinate = torch.from_numpy(
        frequency_coordinates(
            frequency,
            predictor.frequency_mode,
            predictor.frequency_minimum_hz,
            predictor.frequency_maximum_hz,
        )
    ).to(predictor.device)
    normalized_log_frequency = torch.from_numpy(
        (
            (np.log10(frequency) - predictor.normalization.log_frequency_mean)
            / predictor.normalization.log_frequency_std
        ).astype(np.float32)
    ).to(predictor.device)
    blocks = []
    for start in range(0, directions.shape[0], directions_per_block):
        stop = min(directions.shape[0], start + directions_per_block)
        xyz = torch.from_numpy(directions[start:stop, 2:5]).to(predictor.device)
        local_mca = local_mca_db[:, start:stop, :]
        query = conditioned_coordinate_block(
            xyz,
            frequency_coordinate,
            predictor.conditioning_scope,
            local_mca,
            predictor.normalization,
        )
        normalized_mca = torch.from_numpy(
            (local_mca - predictor.normalization.mca_mean)
            / predictor.normalization.mca_std
        ).to(predictor.device)
        normalized_correction = torch.from_numpy(
            (
                correction_db[:, start:stop, :] - predictor.normalization.correction_mean
            )
            / predictor.normalization.correction_std
        ).to(predictor.device)
        blocks.append((query, latent, normalized_mca, normalized_correction, xyz))
    return blocks


def make_forward_with_frequency(
    predictor: FilmSirenSpectralCNNPredictor,
    blocks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    normalized_log_frequency: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    @torch.no_grad()
    def forward() -> torch.Tensor:
        values = []
        for query, latent, normalized_mca, normalized_correction, xyz in blocks:
            final, _, _ = predictor.model.forward_grid(
                query,
                latent,
                normalized_mca,
                normalized_correction,
                normalized_log_frequency,
                xyz,
            )
            values.append(final)
        return torch.cat(values, dim=1)

    return forward


def end_to_end(
    predictors: list[FilmSirenSpectralCNNPredictor],
    source_path: Path,
    output_path: Path,
    weights: np.ndarray,
    repeats: int,
) -> dict[str, Any]:
    values: list[float] = []
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter_ns()
        predictions = [predictor.predict_residual_db(source_path) for predictor in predictors]
        prediction = np.average(np.stack(predictions), axis=0, weights=weights).astype(np.float32)
        with h5py.File(output_path, "w") as handle:
            handle.create_dataset("predicted_residual_db", data=prediction)
        synchronize()
        values.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "repeat_count": repeats,
        "median_latency_ms": float(np.median(values)),
        "p95_latency_ms": percentile(values, 0.95),
        "minimum_latency_ms": float(np.min(values)),
        "maximum_latency_ms": float(np.max(values)),
        "latencies_ms": values,
    }


def gpu_driver() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else None


def main() -> None:
    arguments = parse_arguments()
    root = project_root()
    configuration_path = arguments.configuration.resolve()
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    spec = configuration["benchmark"]
    if configuration["status"] != "frozen" or spec["split"] != "val":
        raise PermissionError("FSC efficiency benchmark is validation-only")
    if not torch.cuda.is_available():
        raise RuntimeError("FSC efficiency benchmark requires CUDA")
    torch.set_num_threads(int(spec["cpu_threads"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.get_device_name(0) != configuration["hardware"]["gpu_name"]:
        raise RuntimeError("GPU does not match frozen benchmark configuration")
    if platform.python_version() != configuration["software"]["python"]:
        raise RuntimeError("Python version does not match frozen benchmark configuration")
    if str(torch.__version__) != configuration["software"]["torch"]:
        raise RuntimeError("Torch version does not match frozen benchmark configuration")

    output_root = root / configuration["output_root"]
    partial = output_root.with_name(output_root.name + ".partial")
    if output_root.exists() or partial.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    partial.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "status": "completed",
        "split": "val",
        "test_subject_count_read": 0,
        "configuration": str(configuration_path),
        "benchmark": spec,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "driver_version": gpu_driver(),
        },
        "models": [],
    }
    for model_spec in configuration["model_manifests"]:
        manifest_path = root / model_spec["manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = manifest_identity(manifest)
        if identity != model_spec["identity_sha256"]:
            raise ValueError(f"Frozen identity mismatch for {model_spec['label']}")
        model_dataset = manifest["dataset"]
        dataset_root = (root / model_dataset["root"]).resolve()
        split_csv = (root / model_dataset["split_csv"]).resolve()
        q26_csv = (root / model_dataset["q26_csv"]).resolve()
        observation_count = model_dataset.get("observation_count")
        if observation_count is not None:
            observation_count = int(observation_count)
        subject_rows = split_subject_paths(dataset_root, split_csv, "val", observation_count or 26)
        subject_row = next(row for row in subject_rows if row[1] == spec["subject_label"])
        source_path = subject_row[2]
        predictors: list[FilmSirenSpectralCNNPredictor] = []
        checkpoint_sizes: list[int] = []
        member_provenance = []
        for member in manifest["members"]:
            checkpoint = root / member["checkpoint"]
            if file_sha256(checkpoint).upper() != member["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash mismatch for {model_spec['label']}")
            checkpoint_sizes.append(checkpoint.stat().st_size)
            predictors.append(
                FilmSirenSpectralCNNPredictor(
                    checkpoint,
                    split_csv,
                    q26_csv,
                    root / model_dataset["q26_normalization"],
                    device=torch.device("cuda"),
                    directions_per_block=int(spec["directions_per_block"]),
                    allow_test=False,
                    condition_source_indices=(
                        sparse_source_indices(q26_csv, observation_count)
                        if observation_count is not None
                        else None
                    ),
                    condition_dataset_root=(dataset_root if observation_count is not None else None),
                    condition_filename=(f"q{observation_count}.h5" if observation_count is not None else "q26.h5"),
                )
            )
            member_provenance.append({
                "seed": member["seed"],
                "checkpoint": member["checkpoint"],
                "checkpoint_sha256": member["checkpoint_sha256"],
                "checkpoint_bytes": checkpoint.stat().st_size,
                "training_report": member["training_report"],
            })
        prepared = []
        for predictor in predictors:
            prepared.append(
                prepare_blocks(
                    predictor,
                    source_path,
                    dataset_root,
                    split_csv,
                    q26_csv,
                    observation_count,
                    int(spec["directions_per_block"]),
                )
            )
        frequency = _read_query_grid(source_path)[1]
        normalized_log_frequency = torch.from_numpy(
            (
                (np.log10(frequency) - predictors[0].normalization.log_frequency_mean)
                / predictors[0].normalization.log_frequency_std
            ).astype(np.float32)
        ).to(torch.device("cuda"))
        single_forward = make_forward_with_frequency(
            predictors[0], prepared[0], normalized_log_frequency
        )

        @torch.no_grad()
        def ensemble_forward() -> torch.Tensor:
            values = [
                make_forward_with_frequency(predictor, blocks, normalized_log_frequency)()
                for predictor, blocks in zip(predictors, prepared)
            ]
            return torch.stack(values, dim=0).mean(dim=0)

        single_latency = timed(single_forward, int(spec["warmup_runs"]), int(spec["timed_runs"]))
        ensemble_latency = timed(ensemble_forward, int(spec["warmup_runs"]), int(spec["timed_runs"]))
        weights = np.asarray(manifest["ensemble"]["weights"], dtype=np.float64)
        with tempfile.TemporaryDirectory(prefix="mcar_fsc_efficiency_") as temporary:
            e2e = end_to_end(
                predictors,
                source_path,
                Path(temporary) / "prediction.h5",
                weights,
                int(spec["end_to_end_runs"]),
            )
        counts = parameter_counts(predictors)
        model_detail = {
            "label": model_spec["label"],
            "manifest": model_spec["manifest"],
            "manifest_identity_sha256": identity,
            "source_subject_label": spec["subject_label"],
            "source_path": str(source_path),
            "member_provenance": member_provenance,
            "parameter_counts": counts,
            "single_member_latency": single_latency,
            "ensemble_latency": ensemble_latency,
            "end_to_end_ensemble_latency": e2e,
        }
        details["models"].append(model_detail)
        rows.extend(
            [
                {
                    "Model": model_spec["label"],
                    "Deployment": "single_member_1",
                    "TrainableParameters": counts["member_trainable"],
                    "FrozenParameters": counts["member_frozen"],
                    "TotalParameters": counts["member_total"],
                    "UniqueDeployedParameters": counts["member_total"],
                    "CheckpointMiB": checkpoint_sizes[0] / 2**20,
                    "MedianLatency_ms": single_latency["median_latency_ms"],
                    "P95Latency_ms": single_latency["p95_latency_ms"],
                    "PeakAllocated_MiB": single_latency["peak_allocated_mib"],
                    "PeakReserved_MiB": single_latency["peak_reserved_mib"],
                    "EndToEndMedianLatency_ms": "",
                },
                {
                    "Model": model_spec["label"],
                    "Deployment": "three_member_ensemble",
                    "TrainableParameters": counts["ensemble_trainable"],
                    "FrozenParameters": counts["ensemble_frozen"],
                    "TotalParameters": counts["ensemble_instantiated_total"],
                    "UniqueDeployedParameters": counts["ensemble_unique_deployed"],
                    "CheckpointMiB": sum(checkpoint_sizes) / 2**20,
                    "MedianLatency_ms": ensemble_latency["median_latency_ms"],
                    "P95Latency_ms": ensemble_latency["p95_latency_ms"],
                    "PeakAllocated_MiB": ensemble_latency["peak_allocated_mib"],
                    "PeakReserved_MiB": ensemble_latency["peak_reserved_mib"],
                    "EndToEndMedianLatency_ms": e2e["median_latency_ms"],
                },
            ]
        )
    with (partial / "efficiency_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    details["script_sha256"] = file_sha256(Path(__file__).resolve()).upper()
    (partial / "efficiency_details.json").write_text(
        json.dumps(details, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (partial / "protocol_snapshot.json").write_text(
        json.dumps(configuration, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (partial / "README.md").write_text(
        "# FSC efficiency benchmark\n\n"
        "Validation-only benchmark for FSC-Q14@Q14, FSC-Q26@Q26, and FSC-Q50@Q50. "
        "Pure model latency excludes checkpoint loading, input disk I/O, and output serialization; "
        "it uses FP32, batch size one, five warm-ups, 30 timed runs, and CUDA synchronization "
        "around each timed region. End-to-end latency includes input loading and prediction.h5 "
        "serialization. `UniqueDeployedParameters` counts all three members because their "
        "checkpoints are independent; frozen does not imply shared. Test subjects were not read.\n",
        encoding="utf-8",
    )
    partial.replace(output_root)
    print(json.dumps({"status": "completed", "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
