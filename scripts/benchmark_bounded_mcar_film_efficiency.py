"""Frozen single-member and deployed-ensemble efficiency benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from mcar.paths import project_root
from mcar.predictors import BoundedMcarFilmCorrectionPredictor
from mcar.training.train_film_siren import file_sha256, split_subject_paths


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
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


def benchmark(function: Callable[[], np.ndarray], warmups: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmups):
        function()
    synchronize()
    torch.cuda.reset_peak_memory_stats()
    latencies = []
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter_ns()
        output = function()
        synchronize()
        latencies.append((time.perf_counter_ns() - started) / 1e6)
        if output.shape != (2, 793, 463) or not np.all(np.isfinite(output)):
            raise FloatingPointError("Benchmark prediction is invalid")
    return {
        "warmup_count": warmups,
        "timed_repeat_count": repeats,
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.quantile(latencies, 0.95, method="linear")),
        "minimum_latency_ms": float(np.min(latencies)),
        "maximum_latency_ms": float(np.max(latencies)),
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated() / 2**20),
        "peak_reserved_mib": float(torch.cuda.max_memory_reserved() / 2**20),
        "latencies_ms": latencies,
    }


def parameter_counts(predictors: list[BoundedMcarFilmCorrectionPredictor]) -> dict[str, int]:
    first = predictors[0].model
    film = sum(value.numel() for value in first.film_siren.parameters())
    previous = sum(value.numel() for value in first.previous_mcar.parameters())
    candidate = sum(value.numel() for value in first.candidate_mcar.parameters())
    gate = sum(value.numel() for value in first.gate_output.parameters())
    member_total = film + previous + candidate + gate
    if member_total != sum(value.numel() for value in first.parameters()):
        raise AssertionError("Parameter component accounting mismatch")
    return {
        "member_trainable": gate,
        "member_total": member_total,
        "member_unique_deployed": member_total,
        "ensemble_trainable": gate * 3,
        "ensemble_instantiated_total": member_total * 3,
        "ensemble_unique_deployed": film * 3 + previous + candidate + gate * 3,
        "film_per_member": film,
        "previous_mcar_shared": previous,
        "candidate_mcar_shared": candidate,
        "gate_per_member": gate,
    }


def profile_flops(function: Callable[[], np.ndarray]) -> int:
    synchronize()
    with FlopCounterMode(display=False) as counter:
        output = function()
    synchronize()
    if output.shape != (2, 793, 463):
        raise ValueError("FLOP profile output shape mismatch")
    return int(counter.get_total_flops())


def end_to_end(
    predictors: list[BoundedMcarFilmCorrectionPredictor],
    source_path: Path,
    output_path: Path,
    weights: np.ndarray,
) -> float:
    synchronize()
    started = time.perf_counter_ns()
    predictions = [predictor.predict_residual_db(source_path) for predictor in predictors]
    prediction = np.average(np.stack(predictions), axis=0, weights=weights).astype(np.float32)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("predicted_residual_db", data=prediction)
    synchronize()
    return (time.perf_counter_ns() - started) / 1e6


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_arguments()
    root = project_root()
    payload = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    identity = manifest_identity(payload)
    benchmark_spec = payload["efficiency"]
    if payload["status"] != "frozen" or payload["dataset"]["split"] != "val":
        raise PermissionError("Efficiency benchmark is validation-only")
    torch.set_num_threads(int(benchmark_spec["cpu_threads"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen benchmark requires CUDA")
    for resource in payload["implementation"]["resources"]:
        if file_sha256(root / resource["path"]) != resource["sha256"]:
            raise ValueError(f"Implementation hash mismatch: {resource['path']}")
    actual_gpu = torch.cuda.get_device_name(0)
    if actual_gpu != benchmark_spec["hardware"]["gpu_name"]:
        raise RuntimeError(f"GPU mismatch: {actual_gpu}")
    if platform.python_version() != benchmark_spec["software"]["python"]:
        raise RuntimeError("Python version mismatch")
    if str(torch.__version__) != benchmark_spec["software"]["torch"]:
        raise RuntimeError("Torch version mismatch")

    model_manifest_path = root / payload["bounded_manifest"]["path"]
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    split_csv = root / model_manifest["dataset"]["split_csv"]
    subject_rows = split_subject_paths(root / model_manifest["dataset"]["root"], split_csv, "val")
    source_path = subject_rows[0][2]
    if subject_rows[0][1] != benchmark_spec["subject_label"]:
        raise ValueError("Frozen benchmark subject changed")
    predictors = []
    checkpoint_sizes = []
    for member in model_manifest["members"]:
        checkpoint = root / member["checkpoint"]
        if file_sha256(checkpoint) != member["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        checkpoint_sizes.append(checkpoint.stat().st_size)
        predictors.append(
            BoundedMcarFilmCorrectionPredictor(
                checkpoint,
                split_csv,
                root / model_manifest["dataset"]["q26_csv"],
                root / model_manifest["dataset"]["q26_normalization"],
                device=torch.device("cuda"),
                directions_per_block=int(benchmark_spec["directions_per_block"]),
                allow_test=False,
            )
        )
    prepared = predictors[0].prepare_subject(source_path)
    weights = np.asarray(model_manifest["ensemble"]["weights"], dtype=np.float64)

    def single() -> np.ndarray:
        return predictors[0].predict_prepared_diagnostics(prepared).final_residual_db

    def ensemble() -> np.ndarray:
        values = [item.predict_prepared_diagnostics(prepared).final_residual_db for item in predictors]
        return np.average(np.stack(values), axis=0, weights=weights).astype(np.float32)

    single_latency = benchmark(single, int(benchmark_spec["warmups"]), int(benchmark_spec["repeats"]))
    ensemble_latency = benchmark(
        ensemble, int(benchmark_spec["warmups"]), int(benchmark_spec["repeats"])
    )
    single_flops = profile_flops(single)
    ensemble_flops = profile_flops(ensemble)
    counts = parameter_counts(predictors)

    output_root = root / payload["outputs"]["efficiency_root"]
    partial = output_root.with_name(output_root.name + ".partial")
    if output_root.exists() or partial.exists():
        raise FileExistsError("Refusing to overwrite efficiency output")
    partial.mkdir(parents=True)
    e2e_ms = end_to_end(predictors, source_path, partial / "end_to_end_prediction.h5", weights)
    training_reports = []
    for member in model_manifest["members"]:
        report = json.loads((root / member["training_report"]).read_text(encoding="utf-8"))
        training_reports.append(report)
    rows = [
        {
            "Deployment": "single_member_1",
            "TrainableParameters": counts["member_trainable"],
            "TotalInstantiatedParameters": counts["member_total"],
            "UniqueDeployedParameters": counts["member_unique_deployed"],
            "CheckpointMiB": checkpoint_sizes[0] / 2**20,
            "MACs": single_flops / 2.0,
            "FLOPs": single_flops,
            "MedianLatency_ms": single_latency["median_latency_ms"],
            "P95Latency_ms": single_latency["p95_latency_ms"],
            "PeakAllocated_MiB": single_latency["peak_allocated_mib"],
            "PeakReserved_MiB": single_latency["peak_reserved_mib"],
            "EndToEndLatency_ms": "",
            "TrainingWallClock_seconds": training_reports[0]["elapsed_seconds"],
            "AcceleratorHours": training_reports[0]["elapsed_seconds"] / 3600.0,
        },
        {
            "Deployment": "three_member_ensemble",
            "TrainableParameters": counts["ensemble_trainable"],
            "TotalInstantiatedParameters": counts["ensemble_instantiated_total"],
            "UniqueDeployedParameters": counts["ensemble_unique_deployed"],
            "CheckpointMiB": sum(checkpoint_sizes) / 2**20,
            "MACs": ensemble_flops / 2.0,
            "FLOPs": ensemble_flops,
            "MedianLatency_ms": ensemble_latency["median_latency_ms"],
            "P95Latency_ms": ensemble_latency["p95_latency_ms"],
            "PeakAllocated_MiB": ensemble_latency["peak_allocated_mib"],
            "PeakReserved_MiB": ensemble_latency["peak_reserved_mib"],
            "EndToEndLatency_ms": e2e_ms,
            "TrainingWallClock_seconds": max(r["elapsed_seconds"] for r in training_reports),
            "AcceleratorHours": sum(r["elapsed_seconds"] for r in training_reports) / 3600.0,
        },
    ]
    write_csv(partial / "efficiency_summary.csv", rows)
    details = {
        "status": "completed", "split": "val", "test_subject_count_read": 0,
        "manifest_identity_sha256": identity, "parameter_components": counts,
        "checkpoint_sizes_bytes": checkpoint_sizes,
        "single_member_latency": single_latency,
        "ensemble_latency": ensemble_latency,
        "flop_counter": "torch.utils.flop_counter.FlopCounterMode",
        "flop_convention": "native profiler FLOPs; MACs reported as FLOPs/2",
        "end_to_end_ensemble_latency_ms": e2e_ms,
        "training_reports": training_reports,
        "environment": benchmark_spec,
    }
    (partial / "efficiency_details.json").write_text(
        json.dumps(details, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (partial / "protocol_snapshot.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    partial.replace(output_root)
    print(json.dumps({"status": "completed", "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
