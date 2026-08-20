"""Shared residual-prediction protocol and SIREN adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import h5py
import numpy as np
import torch

from mcar.models.film_siren import (
    ConditionEncoderConfig,
    FilmSiren,
    FilmSirenConfig,
)
from mcar.models.siren import Siren, SirenConfig
from mcar.q26_condition import Q26MagnitudeNormalization, build_q26_condition
from mcar.training.train_siren import frequency_coordinates


@runtime_checkable
class ResidualPredictor(Protocol):
    """Predict a full-grid binaural residual field in dB."""

    def predict_residual_db(self, source_h5: Path) -> np.ndarray:
        """Return residual dB with shape ``[2,D,F]``."""
        ...


def _read_query_grid(source_h5: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(source_h5, "r") as handle:
        directions = np.asarray(handle["direction_features"][:], dtype=np.float32)
        frequency = np.asarray(handle["frequency_hz"][:], dtype=np.float32).reshape(-1)
    if directions.ndim != 2 or directions.shape[1] != 6:
        raise ValueError("direction_features must have shape [D,6]")
    if frequency.ndim != 1 or frequency.size < 1:
        raise ValueError("frequency_hz must be a non-empty vector")
    return directions, frequency


def _coordinate_block(
    xyz: torch.Tensor,
    frequency_coordinate: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            xyz[:, None, :].expand(-1, frequency_coordinate.shape[0], -1),
            frequency_coordinate[None, :, :].expand(xyz.shape[0], -1, -1),
        ),
        dim=-1,
    ).reshape(-1, xyz.shape[1] + frequency_coordinate.shape[1])


class SirenPredictor:
    """Adapter for plain-SIREN checkpoints produced by Stage A."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: torch.device | None = None,
        directions_per_block: int = 64,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        configuration = payload["siren_configuration"]
        self.model = Siren(SirenConfig(**configuration)).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        conversion = payload["residual_db_conversion"]
        self.target_mean = float(conversion["target_mean"])
        self.target_std = float(conversion["target_std"])
        mapping = payload.get("experiment_configuration", {}).get(
            "frequency_mapping",
            {},
        )
        self.frequency_mode = str(mapping.get("mode", "linear"))
        self.frequency_minimum_hz = float(
            mapping.get("frequency_minimum_hz", 86.1328125)
        )
        self.frequency_maximum_hz = float(
            mapping.get("frequency_maximum_hz", 19982.8125)
        )
        self.directions_per_block = int(directions_per_block)
        if self.directions_per_block < 1:
            raise ValueError("directions_per_block must be positive")

    @torch.no_grad()
    def predict_residual_db(self, source_h5: Path) -> np.ndarray:
        directions, frequency = _read_query_grid(source_h5)
        coordinate = frequency_coordinates(
            frequency,
            self.frequency_mode,
            self.frequency_minimum_hz,
            self.frequency_maximum_hz,
        )
        frequency_tensor = torch.from_numpy(coordinate).to(self.device)
        prediction = np.empty(
            (2, directions.shape[0], frequency.size),
            dtype=np.float32,
        )
        for start in range(0, directions.shape[0], self.directions_per_block):
            stop = min(directions.shape[0], start + self.directions_per_block)
            xyz = torch.from_numpy(directions[start:stop, 2:5]).to(self.device)
            query = _coordinate_block(xyz, frequency_tensor)
            normalized = self.model(query)
            residual = normalized.float() * self.target_std + self.target_mean
            prediction[:, start:stop, :] = (
                residual.reshape(stop - start, frequency.size, 2)
                .permute(2, 0, 1)
                .cpu()
                .numpy()
            )
        return prediction


class FilmSirenPredictor:
    """Adapter for a frozen FiLM-SIREN checkpoint."""

    def __init__(
        self,
        checkpoint: Path,
        split_csv: Path,
        q26_csv: Path,
        q26_normalization: Path,
        *,
        device: torch.device | None = None,
        directions_per_block: int = 64,
        allow_test: bool = False,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model = FilmSiren(
            FilmSirenConfig(**payload["film_siren_configuration"]),
            ConditionEncoderConfig(**payload["condition_encoder_configuration"]),
        ).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        conversion = payload["residual_db_conversion"]
        self.target_mean = float(conversion["target_mean"])
        self.target_std = float(conversion["target_std"])
        mapping = payload["experiment_configuration"]["frequency_mapping"]
        self.frequency_mode = str(mapping["mode"])
        self.frequency_minimum_hz = float(mapping["frequency_minimum_hz"])
        self.frequency_maximum_hz = float(mapping["frequency_maximum_hz"])
        self.split_csv = split_csv
        self.q26_csv = q26_csv
        self.condition_normalization = Q26MagnitudeNormalization.from_json(
            q26_normalization
        )
        self.directions_per_block = int(directions_per_block)
        self.allow_test = bool(allow_test)

    @torch.no_grad()
    def predict_residual_db(self, source_h5: Path) -> np.ndarray:
        with h5py.File(source_h5, "r") as handle:
            subject_id = int(np.asarray(handle.attrs["subject_id"]).item())
        dataset_root = source_h5.resolve().parents[2]
        condition = build_q26_condition(
            dataset_root,
            self.split_csv,
            subject_id,
            self.q26_csv,
            allow_test=self.allow_test,
        )
        normalized = self.condition_normalization.normalize(
            condition.binaural_magnitude_db
        )
        magnitude_tensor = torch.from_numpy(normalized).unsqueeze(0).to(self.device)
        xyz_tensor = torch.from_numpy(condition.xyz).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(condition.mask).unsqueeze(0).to(self.device)
        latent = self.model.encode_condition(
            magnitude_tensor,
            xyz_tensor,
            mask_tensor,
        )

        directions, frequency = _read_query_grid(source_h5)
        coordinate = frequency_coordinates(
            frequency,
            self.frequency_mode,
            self.frequency_minimum_hz,
            self.frequency_maximum_hz,
        )
        frequency_tensor = torch.from_numpy(coordinate).to(self.device)
        prediction = np.empty(
            (2, directions.shape[0], frequency.size),
            dtype=np.float32,
        )
        for start in range(0, directions.shape[0], self.directions_per_block):
            stop = min(directions.shape[0], start + self.directions_per_block)
            xyz = torch.from_numpy(directions[start:stop, 2:5]).to(self.device)
            query = _coordinate_block(xyz, frequency_tensor)
            normalized_prediction = self.model(query, latent)
            residual = (
                normalized_prediction.float() * self.target_std + self.target_mean
            )
            prediction[:, start:stop, :] = (
                residual.reshape(stop - start, frequency.size, 2)
                .permute(2, 0, 1)
                .cpu()
                .numpy()
            )
        return prediction
