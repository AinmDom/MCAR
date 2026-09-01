"""Run frozen MCAR v3.5.1, FSP-AE, and Hybrid E190 at Q14/Q26/Q50."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from mcar.data import Normalization
from mcar.evaluation.evaluate_mlp_cnn_v3 import (
    build_point_features,
    infer_global_context_arguments,
    infer_model_architecture,
)
from mcar.fsp_ae_signal import reconstruct_hrir_with_itd
from mcar.models.fsp_ae import FreqSrcPosCondAutoEncoder
from mcar.models.residual_mlp_cnn import ResidualMLPCNN
from mcar.paths import project_root
from mcar.predictors import FilmSirenSpectralCNNPredictor
from mcar.training.train_film_siren import file_sha256, split_subject_paths


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "configs/experiments/sonicom_ten_method_direction_sensitivity_v1_manifest.json"
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("mcar", "hybrid", "fspae"),
        default=("mcar", "hybrid", "fspae"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--directions-per-block", type=int, default=32)
    parser.add_argument("--target-directions-per-chunk", type=int, default=32)
    parser.add_argument("--subject-limit", type=int)
    return parser.parse_args()


def manifest_identity(payload: dict[str, object]) -> str:
    values = dict(payload)
    claimed = str(values.pop("identity_sha256"))
    encoded = json.dumps(
        values, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    actual = hashlib.sha256(encoded).hexdigest().upper()
    if actual != claimed:
        raise ValueError("Ten-method manifest identity mismatch")
    return actual


def grid_indices(path: Path, count: int) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        values = [
            int(row["source_index_zero_based"])
            for row in csv.DictReader(handle)
            if int(row["direction_count"]) == count
        ]
    result = np.asarray(values, dtype=np.int64)
    if result.shape != (count,) or not bool(np.all(np.diff(result) > 0)):
        raise ValueError(f"Invalid frozen Q{count} grid")
    return result


def validate_hash(path: Path, expected: str) -> None:
    actual = file_sha256(path).upper()
    if actual != str(expected).upper():
        raise ValueError(f"SHA-256 mismatch for {path}: {actual}")


def atomic_hdf5(
    path: Path, datasets: dict[str, np.ndarray], attributes: dict[str, object]
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with h5py.File(partial, "w") as handle:
        for name, value in datasets.items():
            handle.create_dataset(name, data=value, compression="gzip", compression_opts=4)
        for name, value in attributes.items():
            handle.attrs[name] = value
    partial.replace(path)


@torch.no_grad()
def predict_fspae(
    cache_path: Path,
    output_path: Path,
    sparse_indices: np.ndarray,
    checkpoint_path: Path,
    model: FreqSrcPosCondAutoEncoder,
    device: torch.device,
    target_directions_per_chunk: int,
) -> None:
    with h5py.File(cache_path, "r") as source:
        magnitude = torch.from_numpy(source["hrtf_magnitude_db"][:]).to(device)
        itd = torch.from_numpy(source["itd_seconds"][:]).to(device)
        frequency = torch.from_numpy(source["frequency_hz"][:]).unsqueeze(0).to(device)
        positions = torch.from_numpy(source["source_positions_cartesian_m"][:]).to(device)
        subject_id = str(source.attrs["subject_id"])
        split = str(source.attrs["split"])
        sampling_rate_hz = float(source.attrs["sampling_rate_hz"])
    if split != "val":
        raise PermissionError(f"FSP-AE cache is not validation: {cache_path}")
    index = torch.from_numpy(sparse_indices).to(device)
    prototype = model.encode(
        magnitude[index].unsqueeze(0),
        itd[index].unsqueeze(0),
        frequency,
        positions[index].unsqueeze(0),
        dataset_name="sonicom",
    )
    magnitude_chunks: list[torch.Tensor] = []
    itd_chunks: list[torch.Tensor] = []
    for start in range(0, len(positions), target_directions_per_chunk):
        stop = min(start + target_directions_per_chunk, len(positions))
        predicted_magnitude, predicted_itd = model.decode(
            prototype,
            frequency,
            positions[start:stop].unsqueeze(0),
            dataset_name="sonicom",
        )
        magnitude_chunks.append(predicted_magnitude[0].cpu())
        itd_chunks.append(predicted_itd[0].cpu())
    predicted_magnitude = torch.cat(magnitude_chunks)
    predicted_itd = torch.cat(itd_chunks)
    oversized = reconstruct_hrir_with_itd(
        predicted_magnitude.unsqueeze(0),
        predicted_itd.unsqueeze(0),
        sampling_rate_hz=sampling_rate_hz,
    )
    predicted_hrir = oversized[0, :, :, :256].cpu()
    if not all(
        torch.all(torch.isfinite(value))
        for value in (predicted_magnitude, predicted_itd, predicted_hrir)
    ):
        raise FloatingPointError(f"Non-finite FSP-AE output for {subject_id}")
    atomic_hdf5(
        output_path,
        {
            "predicted_magnitude_db": predicted_magnitude.numpy(),
            "predicted_itd_seconds": predicted_itd.numpy(),
            "predicted_hrir": predicted_hrir.numpy(),
            "frequency_hz": frequency[0].cpu().numpy(),
            "sparse_indices_zero_based": sparse_indices.astype(np.int32),
        },
        {
            "subject_id": subject_id,
            "split": split,
            "sparse_direction_count": len(sparse_indices),
            "sampling_rate_hz": sampling_rate_hz,
            "checkpoint_path": str(checkpoint_path),
            "complete": 1,
            "test_subject_count_read": 0,
        },
    )


def method_spec(config: dict[str, object], method_id: str) -> dict[str, object]:
    matches = [method for method in config["methods"] if method["id"] == method_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one {method_id} registry entry")
    return matches[0]


def load_mcar_model(
    checkpoint_path: Path, expected_hash: str, device: torch.device
) -> ResidualMLPCNN:
    validate_hash(checkpoint_path, expected_hash)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = payload["model_state"]
    width, blocks, channels = infer_model_architecture(state)
    model = ResidualMLPCNN(
        mlp_width=width,
        mlp_block_count=blocks,
        cnn_channels=channels,
        **infer_global_context_arguments(payload),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_mcar_component(
    source_h5: Path,
    model: ResidualMLPCNN,
    normalization: Normalization,
    device: torch.device,
    directions_per_block: int,
    use_amp: bool,
) -> np.ndarray:
    with h5py.File(source_h5, "r") as source:
        split = source.attrs["split"]
        if isinstance(split, bytes):
            split = split.decode()
        if str(split) != "val":
            raise PermissionError(f"MCAR input is not validation: {source_h5}")
        mca = source["mca_logmag_db"]
        correction = source["correction_logmag_db"]
        directions = np.asarray(source["direction_features"][:], dtype=np.float32)
        frequency_hz = np.squeeze(source["frequency_hz"][:]).astype(np.float32)
        prediction = np.empty(mca.shape, dtype=np.float32)
        for start in range(0, mca.shape[1], directions_per_block):
            stop = min(start + directions_per_block, mca.shape[1])
            point_features = build_point_features(
                np.asarray(mca[:, start:stop, :], dtype=np.float32),
                np.asarray(correction[:, start:stop, :], dtype=np.float32),
                directions[start:stop],
                frequency_hz,
                normalization,
            )
            point_tensor = torch.from_numpy(point_features).to(device)
            with torch.amp.autocast(
                "cuda", enabled=use_amp and device.type == "cuda"
            ):
                normalized, _, _ = model(point_tensor)
            predicted_db = (
                normalized.permute(1, 0, 2).float() * normalization.target_std
                + normalization.target_mean
            )
            prediction[:, start:stop, :] = predicted_db.cpu().numpy()
    return prediction


def predict_mcar_v351(
    source_h5: Path,
    models: tuple[ResidualMLPCNN, ResidualMLPCNN],
    normalization: Normalization,
    device: torch.device,
    directions_per_block: int,
    use_amp: bool,
    candidate_weight: float,
) -> np.ndarray:
    previous = predict_mcar_component(
        source_h5, models[0], normalization, device, directions_per_block, use_amp
    )
    candidate = predict_mcar_component(
        source_h5, models[1], normalization, device, directions_per_block, use_amp
    )
    return previous + np.float32(candidate_weight) * (candidate - previous)


def write_residual(
    path: Path,
    prediction: np.ndarray,
    *,
    method: str,
    subject_label: str,
    count: int,
    identity: str,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    atomic_hdf5(
        path,
        {"predicted_residual_db": prediction.astype(np.float32)},
        {
            "schema_version": "1.0",
            "complete": 1,
            "method": method,
            "subject_id": int(subject_label[1:]),
            "subject_label": subject_label,
            "split": "val",
            "sparse_direction_count": count,
            "tensor_layout": "ear,direction,frequency",
            "manifest_identity_sha256": identity,
            "test_subject_count_read": 0,
        },
    )


def q26_residual_error(prediction: np.ndarray, path: Path) -> float:
    with h5py.File(path, "r") as handle:
        reference = np.asarray(handle["predicted_residual_db"][:], dtype=np.float32)
    return float(np.max(np.abs(prediction - reference)))


def q26_fspae_error(candidate: Path, reference: Path) -> float:
    maximum = 0.0
    with h5py.File(candidate, "r") as actual, h5py.File(reference, "r") as expected:
        for name in (
            "predicted_magnitude_db",
            "predicted_itd_seconds",
            "predicted_hrir",
            "frequency_hz",
        ):
            value = np.asarray(actual[name][:])
            target = np.asarray(expected[name][:])
            if value.shape != target.shape:
                raise ValueError(f"FSP-AE Q26 shape mismatch for {name}")
            maximum = max(maximum, float(np.max(np.abs(value - target))))
    return maximum


def copy_reference_hdf5(
    source: Path, destination: Path, *, identity: str, count: int
) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    shutil.copy2(source, partial)
    with h5py.File(partial, "r+") as handle:
        handle.attrs["sparse_direction_count"] = count
        handle.attrs["manifest_identity_sha256"] = identity
        handle.attrs["q26_reused_formal_artifact"] = 1
        handle.attrs["test_subject_count_read"] = 0
    partial.replace(destination)


def main() -> None:
    args = parse_arguments()
    root = project_root()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest_identity(manifest)
    if manifest["status"] != "frozen_before_inference":
        raise ValueError("Manifest is not frozen before inference")
    config_path = root / manifest["experiment_config"]
    if file_sha256(config_path).upper() != manifest["experiment_config_sha256"]:
        raise ValueError("Experiment config hash mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["dataset"]["split"] != "val" or config["inference"]["test_access_allowed"]:
        raise PermissionError("Ten-method sensitivity inference is validation-only")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    split_csv = root / config["dataset"]["split_file"]
    dataset_root = root / config["dataset"]["root"]
    subjects = split_subject_paths(dataset_root, split_csv, "val")
    if len(subjects) != 44:
        raise ValueError("Expected exactly 44 validation listeners")
    if args.subject_limit is not None:
        subjects = subjects[: args.subject_limit]
    grid_csv = root / config["dataset"]["sparse_grid_file"]
    input_root = root / config["dataset"]["prepared_input_root"]
    output_root = root / "artifacts/sparsity" / config["output_name"]
    tolerance = float(config["inference"]["q26_reproduction_tolerance"])
    requested = set(args.methods)
    started = time.perf_counter()
    counts = (26, 14, 50)
    maxima = {method: 0.0 for method in requested}
    predictions = {method: 0 for method in requested}

    normalization = root / config["inference"]["normalization"]

    hybrid_spec = method_spec(config, "HYBRID")
    hybrid_manifest = json.loads((root / hybrid_spec["manifest"]).read_text(encoding="utf-8"))
    if hybrid_manifest["identity_sha256"] != hybrid_spec["manifest_identity_sha256"]:
        raise ValueError("Hybrid E190 identity mismatch")
    q26_csv = root / hybrid_manifest["dataset"]["q26_csv"]

    mcar_spec = method_spec(config, "MCARv351")
    mcar_models: tuple[ResidualMLPCNN, ResidualMLPCNN] | None = None
    mcar_normalization: Normalization | None = None
    if "mcar" in requested:
        components = mcar_spec["components"]
        if [component["role"] for component in components] != [
            "previous_joint",
            "v351b_continuation",
        ]:
            raise ValueError("Frozen MCAR component order changed")
        mcar_models = tuple(
            load_mcar_model(
                root / component["checkpoint"], component["checkpoint_sha256"], device
            )
            for component in components
        )
        mcar_normalization = Normalization.from_json(
            root / config["inference"]["mcar_normalization"]
        )

    fsp_spec = method_spec(config, "FSPAE")
    fsp_checkpoint_path = root / fsp_spec["checkpoint"]
    validate_hash(fsp_checkpoint_path, fsp_spec["checkpoint_sha256"])
    fsp_model: FreqSrcPosCondAutoEncoder | None = None
    if "fspae" in requested:
        payload = torch.load(fsp_checkpoint_path, map_location="cpu", weights_only=False)
        fsp_model = FreqSrcPosCondAutoEncoder()
        fsp_model.load_state_dict(payload["model"])
        fsp_model.stats = payload["stats"]
        fsp_model.to(device).eval()

    for count in counts:
        indices = grid_indices(grid_csv, count)
        hybrid_predictors: list[FilmSirenSpectralCNNPredictor] = []
        if "hybrid" in requested and count != 26:
            for member in hybrid_manifest["members"]:
                checkpoint = root / member["checkpoint"]
                validate_hash(checkpoint, member["checkpoint_sha256"])
                hybrid_predictors.append(
                    FilmSirenSpectralCNNPredictor(
                        checkpoint,
                        split_csv,
                        q26_csv,
                        normalization,
                        device=device,
                        directions_per_block=args.directions_per_block,
                        allow_test=False,
                        condition_source_indices=indices,
                        condition_dataset_root=dataset_root,
                    )
                )
        for index, (_, subject_label, _) in enumerate(subjects, 1):
            source_h5 = input_root / "subjects" / subject_label / f"q{count}" / "model_input.h5"
            level_root = output_root / "subjects" / subject_label / f"q{count}"
            if not source_h5.is_file():
                raise FileNotFoundError(source_h5)
            if mcar_models is not None and mcar_normalization is not None:
                reference = root / mcar_spec["q26_reference_root"] / "subjects" / subject_label / "prediction.h5"
                if count == 26:
                    with h5py.File(reference, "r") as handle:
                        prediction = np.asarray(
                            handle["predicted_residual_db"][:], dtype=np.float32
                        )
                else:
                    prediction = predict_mcar_v351(
                        source_h5,
                        mcar_models,
                        mcar_normalization,
                        device,
                        int(mcar_spec["directions_per_block"]),
                        bool(mcar_spec["cuda_amp"]),
                        float(mcar_spec["candidate_weight"]),
                    )
                if prediction.shape != (2, 793, 463) or not np.all(np.isfinite(prediction)):
                    raise FloatingPointError(f"Invalid MCAR Q{count} {subject_label}")
                if count == 26:
                    maxima["mcar"] = max(maxima["mcar"], q26_residual_error(prediction, reference))
                write_residual(level_root / "mcar_v351_prediction.h5", prediction, method="MCARv351", subject_label=subject_label, count=count, identity=identity)
                predictions["mcar"] += 1
            if "hybrid" in requested:
                reference = root / hybrid_spec["q26_reference_root"] / "subjects" / subject_label / "prediction.h5"
                if count == 26:
                    with h5py.File(reference, "r") as handle:
                        prediction = np.asarray(
                            handle["predicted_residual_db"][:], dtype=np.float32
                        )
                else:
                    member_predictions = [
                        predictor.predict_residual_db(source_h5)
                        for predictor in hybrid_predictors
                    ]
                    prediction = np.mean(
                        np.stack(member_predictions), axis=0, dtype=np.float64
                    ).astype(np.float32)
                if prediction.shape != (2, 793, 463) or not np.all(np.isfinite(prediction)):
                    raise FloatingPointError(f"Invalid Hybrid Q{count} {subject_label}")
                if count == 26:
                    maxima["hybrid"] = max(maxima["hybrid"], q26_residual_error(prediction, reference))
                write_residual(level_root / "hybrid_e190_prediction.h5", prediction, method="HYBRID", subject_label=subject_label, count=count, identity=identity)
                predictions["hybrid"] += 1
            if fsp_model is not None:
                output = level_root / "fspae_prediction.h5"
                reference = root / fsp_spec["q26_reference_root"] / "subjects" / subject_label / "prediction.h5"
                if count == 26:
                    copy_reference_hdf5(
                        reference, output, identity=identity, count=count
                    )
                else:
                    predict_fspae(
                        root / config["dataset"]["fspae_cache_root"] / f"{subject_label}.h5",
                        output,
                        indices,
                        fsp_checkpoint_path,
                        fsp_model,
                        device,
                        args.target_directions_per_chunk,
                    )
                if count == 26:
                    maxima["fspae"] = max(maxima["fspae"], q26_fspae_error(output, reference))
                predictions["fspae"] += 1
            print(f"Q{count} [{index}/{len(subjects)}] {subject_label}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    failures = {method: value for method, value in maxima.items() if value > tolerance}
    if failures:
        raise ValueError(f"Q26 reproduction failed: {failures}")
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "split": "val",
        "subject_count": len(subjects),
        "direction_counts": [14, 26, 50],
        "methods": sorted(requested),
        "prediction_counts": predictions,
        "q26_reproduction_max_abs_error": maxima,
        "q26_reproduction_tolerance": tolerance,
        "manifest_identity_sha256": identity,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "test_subject_count_read": 0,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(sorted(requested))
    (output_root / f"inference_{suffix}_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
