"""Frozen MCAR and FSP-AE predictions for the SONICOM learned-method study."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from mcar.data import Normalization
from mcar.evaluation.predict_external_sparsity import (
    atomic_hdf5,
    validate_hash,
)
from mcar.evaluation.predict_mlp_cnn_reconstruction import (
    infer_global_context_arguments,
    infer_model_architecture,
)
from mcar.evaluation.predict_sonicom_mlp_cnn_residuals import (
    predict_subject as predict_mcar_subject,
)
from mcar.fsp_ae_signal import reconstruct_hrir_with_itd
from mcar.models.fsp_ae import FreqSrcPosCondAutoEncoder
from mcar.models.residual_mlp_cnn import ResidualMLPCNN


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/sonicom_learned_methods_sparsity_v1.json"),
    )
    parser.add_argument("--subject-limit", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--directions-per-block", type=int, default=64)
    parser.add_argument("--target-directions-per-chunk", type=int, default=32)
    parser.add_argument(
        "--methods", nargs="+", choices=("mcar", "fspae"), default=("mcar", "fspae")
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_subjects(config: dict[str, object], limit: int | None) -> list[str]:
    split_file = Path(str(config["dataset"]["split_file"]))
    rows = np.genfromtxt(split_file, delimiter=",", dtype=str, skip_header=1)
    subjects = [row[0] for row in rows if row[1] == config["dataset"]["split"]]
    if len(subjects) != int(config["dataset"]["subject_count"]):
        raise ValueError("Frozen SONICOM subject count mismatch")
    return subjects[:limit] if limit is not None else subjects


def read_sparse_indices(config: dict[str, object], count: int) -> np.ndarray:
    key = f"q{count}_source_indices_zero_based"
    indices = np.asarray(config["sparse_grid_selection"][key], dtype=np.int64)
    if indices.shape != (count,) or len(np.unique(indices)) != count:
        raise ValueError(f"Invalid frozen Q{count} indices")
    return indices


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
    oversized_hrir = reconstruct_hrir_with_itd(
        predicted_magnitude.unsqueeze(0),
        predicted_itd.unsqueeze(0),
        sampling_rate_hz=sampling_rate_hz,
    )
    predicted_hrir = oversized_hrir[0, :, :, :256].cpu()
    if not all(
        torch.all(torch.isfinite(value))
        for value in (predicted_magnitude, predicted_itd, predicted_hrir)
    ):
        raise FloatingPointError(f"Non-finite FSP-AE prediction for {subject_id}")
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
            "hrir_reconstruction": "mcar.fsp_ae_signal.reconstruct_hrir_with_itd",
            "complete": 1,
        },
    )


def main() -> None:
    args = parse_arguments()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    methods = {method["id"]: method for method in config["methods"]}
    mcar_spec = methods["MCARv32"]
    fsp_spec = methods["FSPAE"]
    mcar_checkpoint_path = Path(mcar_spec["checkpoint"])
    fsp_checkpoint_path = Path(fsp_spec["checkpoint"])
    normalization_path = Path(
        "data/processed/sonicom_residual_q26_v1/training_statistics.json"
    )
    validate_hash(mcar_checkpoint_path, mcar_spec["checkpoint_sha256"])
    validate_hash(fsp_checkpoint_path, fsp_spec["checkpoint_sha256"])
    device = torch.device(args.device)

    mcar_checkpoint = torch.load(
        mcar_checkpoint_path, map_location=device, weights_only=False
    )
    state = mcar_checkpoint["model_state"]
    width, blocks, channels = infer_model_architecture(state)
    mcar_model = ResidualMLPCNN(
        mlp_width=width,
        mlp_block_count=blocks,
        cnn_channels=channels,
        **infer_global_context_arguments(mcar_checkpoint),
    ).to(device)
    mcar_model.load_state_dict(state)
    mcar_model.eval()
    normalization = Normalization.from_json(normalization_path)

    fsp_checkpoint = torch.load(
        fsp_checkpoint_path, map_location="cpu", weights_only=False
    )
    fsp_model = FreqSrcPosCondAutoEncoder()
    fsp_model.load_state_dict(fsp_checkpoint["model"])
    fsp_model.stats = fsp_checkpoint["stats"]
    fsp_model.to(device).eval()

    subjects = read_subjects(config, args.subject_limit)
    counts = [int(value) for value in config["dataset"]["direction_counts"]]
    artifact_root = Path("artifacts/sparsity") / config["output_name"] / "subjects"
    fsp_cache_root = Path("data/processed/sonicom_fsp_ae_q26_v1/subjects/test")
    started = time.perf_counter()
    processed = 0
    for subject in subjects:
        for count in counts:
            level_root = artifact_root / subject / f"q{count}"
            input_path = level_root / "model_input.h5"
            if not input_path.is_file():
                raise FileNotFoundError(input_path)
            mcar_output = level_root / "mcar_prediction.h5"
            fsp_output = level_root / "fspae_prediction.h5"
            if "mcar" in args.methods and (args.overwrite or not mcar_output.is_file()):
                predict_mcar_subject(
                    input_path,
                    mcar_output,
                    mcar_checkpoint_path,
                    mcar_model,
                    normalization,
                    device,
                    args.directions_per_block,
                    device.type == "cuda",
                    "test",
                )
            if "fspae" in args.methods and (args.overwrite or not fsp_output.is_file()):
                predict_fspae(
                    fsp_cache_root / f"{subject}.h5",
                    fsp_output,
                    read_sparse_indices(config, count),
                    fsp_checkpoint_path,
                    fsp_model,
                    device,
                    args.target_directions_per_chunk,
                )
            processed += 1
            print(f"[{processed}/{len(subjects) * len(counts)}] {subject} Q{count}")
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "subject_count": len(subjects),
        "processed_subject_levels": processed,
        "mcar_checkpoint_epoch": int(mcar_checkpoint["epoch"]),
        "fspae_checkpoint_epoch": int(fsp_checkpoint["epoch"]),
        "device": str(device),
        "methods": list(args.methods),
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_name = (
        "learned_inference_report.json"
        if set(args.methods) == {"mcar", "fspae"}
        else "learned_inference_" + "_".join(args.methods) + "_report.json"
    )
    report_path = artifact_root.parent / report_name
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
