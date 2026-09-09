"""One-subject real-data CUDA gradient smoke for global+local Stage C."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import torch

from mcar.data import Normalization
from mcar.models.film_siren import ConditionEncoderConfig, FilmSiren, FilmSirenConfig
from mcar.paths import project_root
from mcar.q26_condition import Q26MagnitudeNormalization
from mcar.training.train_film_siren import (
    condition_tensors,
    load_condition_cache,
    read_common_grid,
    set_seed,
    split_subject_paths,
)
from mcar.training.train_film_siren_stage_c import (
    horizontal_interpolation_indices,
    loss_configuration,
    one_loss,
)
from mcar.training.train_mlp_v2 import (
    make_erb_center_frequencies_hz,
    make_erb_weights,
)
from mcar.training.train_siren import frequency_coordinates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Stage C smoke test")
    root = project_root()
    config_path = (root / args.config).resolve()
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(configuration["seed"])
    set_seed(seed)
    device = torch.device("cuda")
    dataset_root = root / configuration["dataset_root"]
    split_csv = root / configuration["subject_split_csv"]
    observation_count = int(configuration.get("observation_count", 26))
    q26_csv = root / configuration.get("q_csv", configuration["q26_csv"])
    q26_normalization = Q26MagnitudeNormalization.from_json(
        root / configuration["q26_normalization"]
    )
    train_path = split_subject_paths(dataset_root, split_csv, "train", observation_count)[0]
    subjects = load_condition_cache(
        [train_path],
        dataset_root,
        split_csv,
        q26_csv,
        q26_normalization,
        "train",
        observation_count,
    )
    subject = subjects[0]
    normalization = Normalization.from_json(dataset_root / "training_statistics.json")
    directions, frequency, interpolation_mask, _ = read_common_grid(subject.path)
    interpolation_indices = np.flatnonzero(interpolation_mask)
    horizontal_indices = horizontal_interpolation_indices(directions, interpolation_mask)
    rng = np.random.default_rng(seed)
    global_batch = np.sort(rng.choice(interpolation_indices, size=4, replace=False))
    horizontal_batch = np.sort(rng.choice(horizontal_indices, size=4, replace=False))
    mapping = configuration["frequency_mapping"]
    frequency_coordinate = torch.from_numpy(
        frequency_coordinates(
            frequency,
            mapping["mode"],
            mapping["frequency_minimum_hz"],
            mapping["frequency_maximum_hz"],
        )
    ).to(device)
    frequency_hz = torch.from_numpy(frequency).to(device)
    erb_weights = torch.from_numpy(make_erb_weights(frequency)).to(device)
    log_erb_weights = torch.log(torch.clamp(erb_weights, min=1e-12)).view(
        1, 1, erb_weights.shape[0], erb_weights.shape[1]
    )
    erb_centers = torch.from_numpy(make_erb_center_frequencies_hz()).to(device)
    model = FilmSiren(
        FilmSirenConfig(**configuration["model"]),
        ConditionEncoderConfig(**configuration["condition_encoder"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    objective = loss_configuration(configuration["objective"])
    losses: list[float] = []
    for _ in range(2):
        magnitude, xyz, mask = condition_tensors(subject, device)
        optimizer.zero_grad(set_to_none=True)
        latent = model.encode_condition(magnitude, xyz, mask)
        loss, _ = one_loss(
            model,
            latent,
            subject,
            directions,
            frequency_coordinate,
            frequency_hz,
            global_batch,
            horizontal_batch,
            normalization,
            log_erb_weights,
            erb_centers,
            objective,
            device,
            configuration["conditioning_scope"],
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
            raise FloatingPointError("Stage C smoke produced a non-finite value")
        optimizer.step()
        losses.append(float(loss.detach().item()))
    print(
        json.dumps(
            {
                "status": "passed",
                "subject_id": subject.subject_id,
                "optimizer_steps": 2,
                "conditioning_scope": configuration["conditioning_scope"],
                "losses": losses,
                "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()
                / (1024.0**2),
                "validation_subjects_read": 0,
                "test_subjects_read": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
