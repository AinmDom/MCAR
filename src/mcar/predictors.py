"""Shared residual-prediction protocol and SIREN adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import h5py
import numpy as np
import torch

from mcar.data import Normalization
from mcar.models.film_siren import (
    ConditionEncoderConfig,
    FilmSiren,
    FilmSirenConfig,
)
from mcar.models.film_siren_spectral_cnn import (
    FilmSirenSpectralCNN,
    SpectralRefinerConfig,
)
from mcar.models.bounded_mcar_film_correction import (
    BoundedCorrectionConfig,
    BoundedMcarFilmCorrection,
)
from mcar.models.residual_mlp_cnn import ResidualMLPCNN
from mcar.models.siren import Siren, SirenConfig
from mcar.evaluation.evaluate_mlp_cnn_v3 import (
    build_point_features,
    infer_global_context_arguments,
    infer_model_architecture,
)
from mcar.paths import project_root
from mcar.q26_condition import Q26MagnitudeNormalization, build_q26_condition
from mcar.training.train_siren import frequency_coordinates
from mcar.training.train_film_siren import conditioned_coordinate_block


@runtime_checkable
class ResidualPredictor(Protocol):
    """Predict a full-grid binaural residual field in dB."""

    def predict_residual_db(self, source_h5: Path) -> np.ndarray:
        """Return residual dB with shape ``[2,D,F]``."""
        ...


@dataclass(frozen=True)
class BoundedCorrectionSubjectInputs:
    """CPU-resident inputs for one complete Stage-E subject reconstruction."""

    subject_id: int
    local_mca_db: np.ndarray
    correction_db: np.ndarray
    directions: np.ndarray
    frequency_hz: np.ndarray
    q26_magnitude_db: np.ndarray
    q26_xyz: np.ndarray
    q26_mask: np.ndarray


@dataclass(frozen=True)
class BoundedCorrectionDiagnostics:
    """Physical-unit outputs from one bounded-correction ensemble member."""

    final_residual_db: np.ndarray
    base_residual_db: np.ndarray
    applied_correction_db: np.ndarray
    gate: np.ndarray


def bounded_diagnostics_to_db(
    final_normalized: torch.Tensor,
    base_normalized: torch.Tensor,
    correction_normalized: torch.Tensor,
    gate: torch.Tensor,
    *,
    target_mean: float,
    target_std: float,
) -> BoundedCorrectionDiagnostics:
    """Convert normalized model internals without adding the mean to a delta."""
    final = final_normalized.float() * target_std + target_mean
    base = base_normalized.float() * target_std + target_mean
    correction = correction_normalized.float() * target_std
    gate_value = gate.float()
    if not torch.allclose(final, base + correction, rtol=2e-6, atol=2e-6):
        raise AssertionError("Bounded correction no longer satisfies final=base+correction")
    return BoundedCorrectionDiagnostics(
        final.cpu().numpy(),
        base.cpu().numpy(),
        correction.cpu().numpy(),
        gate_value.cpu().numpy(),
    )


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
        experiment = payload["experiment_configuration"]
        mapping = experiment["frequency_mapping"]
        self.frequency_mode = str(mapping["mode"])
        self.frequency_minimum_hz = float(mapping["frequency_minimum_hz"])
        self.frequency_maximum_hz = float(mapping["frequency_maximum_hz"])
        self.conditioning_scope = str(experiment.get("conditioning_scope", "global"))
        if self.conditioning_scope not in {"global", "global_plus_local_mca"}:
            raise ValueError(
                "Frozen FilmSirenPredictor supports global or global_plus_local_mca"
            )
        normalization_path = Path(conversion["normalization_path"])
        self.normalization = Normalization.from_json(normalization_path)
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
        local_mca_db: np.ndarray | None = None
        if self.conditioning_scope == "global_plus_local_mca":
            with h5py.File(source_h5, "r") as handle:
                local_mca_db = np.asarray(handle["mca_logmag_db"][:], dtype=np.float32)
            expected = (2, directions.shape[0], frequency.size)
            if local_mca_db.shape != expected:
                raise ValueError(
                    f"Expected local MCA shape {expected}, found {local_mca_db.shape}"
                )
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
            if self.conditioning_scope == "global":
                query = _coordinate_block(xyz, frequency_tensor)
            else:
                assert local_mca_db is not None
                query = conditioned_coordinate_block(
                    xyz,
                    frequency_tensor,
                    self.conditioning_scope,
                    local_mca_db[:, start:stop, :],
                    self.normalization,
                )
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


class FilmSirenSpectralCNNPredictor:
    """Adapter for a Stage-D FiLM-SIREN + spectral-CNN checkpoint."""

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
        if payload.get("training_stage") != "film_siren_spectral_cnn_stage_d":
            raise ValueError("Checkpoint is not a Stage-D spectral-refiner model")
        backbone = FilmSiren(
            FilmSirenConfig(**payload["film_siren_configuration"]),
            ConditionEncoderConfig(**payload["condition_encoder_configuration"]),
        )
        refiner_values = dict(payload["spectral_refiner_configuration"])
        refiner_values["dilation_schedule"] = tuple(
            refiner_values["dilation_schedule"]
        )
        self.model = FilmSirenSpectralCNN(
            backbone,
            SpectralRefinerConfig(**refiner_values),
            freeze_film_siren=bool(payload.get("film_siren_frozen", True)),
        ).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        conversion = payload["residual_db_conversion"]
        self.target_mean = float(conversion["target_mean"])
        self.target_std = float(conversion["target_std"])
        self.normalization = Normalization.from_json(Path(conversion["normalization_path"]))
        experiment = payload["experiment_configuration"]
        mapping = experiment["frequency_mapping"]
        self.frequency_mode = str(mapping["mode"])
        self.frequency_minimum_hz = float(mapping["frequency_minimum_hz"])
        self.frequency_maximum_hz = float(mapping["frequency_maximum_hz"])
        self.conditioning_scope = str(experiment["conditioning_scope"])
        if self.conditioning_scope != "global_plus_local_mca":
            raise ValueError("Stage-D predictor requires global_plus_local_mca")
        self.split_csv = split_csv
        self.q26_csv = q26_csv
        self.condition_normalization = Q26MagnitudeNormalization.from_json(
            q26_normalization
        )
        self.directions_per_block = int(directions_per_block)
        if self.directions_per_block < 1:
            raise ValueError("directions_per_block must be positive")
        self.allow_test = bool(allow_test)

    def prepare_subject(self, source_h5: Path) -> BoundedCorrectionSubjectInputs:
        """Load all subject inputs before a compute-only benchmark region."""
        with h5py.File(source_h5, "r") as handle:
            subject_id = int(np.asarray(handle.attrs["subject_id"]).item())
            local_mca_db = np.asarray(handle["mca_logmag_db"][:], dtype=np.float32)
            correction_db = np.asarray(
                handle["correction_logmag_db"][:], dtype=np.float32
            )
        dataset_root = source_h5.resolve().parents[2]
        condition = build_q26_condition(
            dataset_root,
            self.split_csv,
            subject_id,
            self.q26_csv,
            allow_test=self.allow_test,
        )
        normalized_condition = self.condition_normalization.normalize(
            condition.binaural_magnitude_db
        )
        latent = self.model.encode_condition(
            torch.from_numpy(normalized_condition).unsqueeze(0).to(self.device),
            torch.from_numpy(condition.xyz).unsqueeze(0).to(self.device),
            torch.from_numpy(condition.mask).unsqueeze(0).to(self.device),
        )

        directions, frequency = _read_query_grid(source_h5)
        expected = (2, directions.shape[0], frequency.size)
        if local_mca_db.shape != expected or correction_db.shape != expected:
            raise ValueError("Stage-D local spectral inputs have incompatible shape")
        frequency_coordinate = torch.from_numpy(
            frequency_coordinates(
                frequency,
                self.frequency_mode,
                self.frequency_minimum_hz,
                self.frequency_maximum_hz,
            )
        ).to(self.device)
        normalized_log_frequency = torch.from_numpy(
            (
                (np.log10(frequency) - self.normalization.log_frequency_mean)
                / self.normalization.log_frequency_std
            ).astype(np.float32)
        ).to(self.device)
        prediction = np.empty(expected, dtype=np.float32)
        for start in range(0, directions.shape[0], self.directions_per_block):
            stop = min(directions.shape[0], start + self.directions_per_block)
            xyz = torch.from_numpy(directions[start:stop, 2:5]).to(self.device)
            local_mca = local_mca_db[:, start:stop, :]
            query = conditioned_coordinate_block(
                xyz,
                frequency_coordinate,
                self.conditioning_scope,
                local_mca,
                self.normalization,
            )
            normalized_mca = torch.from_numpy(
                (local_mca - self.normalization.mca_mean)
                / self.normalization.mca_std
            ).to(self.device)
            normalized_correction = torch.from_numpy(
                (
                    correction_db[:, start:stop, :]
                    - self.normalization.correction_mean
                )
                / self.normalization.correction_std
            ).to(self.device)
            normalized, _, _ = self.model.forward_grid(
                query,
                latent,
                normalized_mca,
                normalized_correction,
                normalized_log_frequency,
                xyz,
            )
            prediction[:, start:stop, :] = (
                normalized.float() * self.target_std + self.target_mean
            ).cpu().numpy()
        return prediction


class BoundedMcarFilmCorrectionPredictor:
    """Adapter for the frozen-MCAR bounded FiLM correction checkpoint."""

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
        if payload.get("training_stage") != "bounded_mcar_film_correction_stage_e":
            raise ValueError("Checkpoint is not a Stage-E bounded correction model")
        experiment = payload["experiment_configuration"]
        film = FilmSiren(
            FilmSirenConfig(**payload["film_siren_configuration"]),
            ConditionEncoderConfig(**payload["condition_encoder_configuration"]),
        )
        correction_values = dict(experiment["bounded_mcar_film_correction"])
        components = correction_values.pop("components")
        mcar_models: list[ResidualMLPCNN] = []
        root = project_root()
        for component in components:
            component_payload = torch.load(
                root / component["checkpoint"],
                map_location=self.device,
                weights_only=False,
            )
            state = component_payload["model_state"]
            width, blocks, channels = infer_model_architecture(state)
            mcar = ResidualMLPCNN(
                mlp_width=width,
                mlp_block_count=blocks,
                cnn_channels=channels,
                **infer_global_context_arguments(component_payload),
            )
            mcar.load_state_dict(state)
            mcar_models.append(mcar)
        self.model = BoundedMcarFilmCorrection(
            film,
            mcar_models[0],
            mcar_models[1],
            BoundedCorrectionConfig(**correction_values),
        ).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        conversion = payload["residual_db_conversion"]
        self.target_mean = float(conversion["target_mean"])
        self.target_std = float(conversion["target_std"])
        self.normalization = Normalization.from_json(Path(conversion["normalization_path"]))
        mapping = experiment["frequency_mapping"]
        self.frequency_mode = str(mapping["mode"])
        self.frequency_minimum_hz = float(mapping["frequency_minimum_hz"])
        self.frequency_maximum_hz = float(mapping["frequency_maximum_hz"])
        self.conditioning_scope = str(experiment["conditioning_scope"])
        if self.conditioning_scope != "global_plus_local_mca":
            raise ValueError("Stage-E predictor requires global_plus_local_mca")
        self.split_csv = split_csv
        self.q26_csv = q26_csv
        self.condition_normalization = Q26MagnitudeNormalization.from_json(
            q26_normalization
        )
        self.directions_per_block = int(directions_per_block)
        if self.directions_per_block < 1:
            raise ValueError("directions_per_block must be positive")
        self.allow_test = bool(allow_test)

    @torch.no_grad()
    def predict_residual_db(self, source_h5: Path) -> np.ndarray:
        with h5py.File(source_h5, "r") as handle:
            subject_id = int(np.asarray(handle.attrs["subject_id"]).item())
            local_mca_db = np.asarray(handle["mca_logmag_db"][:], dtype=np.float32)
            correction_db = np.asarray(
                handle["correction_logmag_db"][:], dtype=np.float32
            )
        dataset_root = source_h5.resolve().parents[2]
        condition = build_q26_condition(
            dataset_root,
            self.split_csv,
            subject_id,
            self.q26_csv,
            allow_test=self.allow_test,
        )
        directions, frequency = _read_query_grid(source_h5)
        expected = (2, directions.shape[0], frequency.size)
        if local_mca_db.shape != expected or correction_db.shape != expected:
            raise ValueError("Stage-E local spectral inputs have incompatible shape")
        return BoundedCorrectionSubjectInputs(
            subject_id=subject_id,
            local_mca_db=local_mca_db,
            correction_db=correction_db,
            directions=directions,
            frequency_hz=frequency,
            q26_magnitude_db=condition.binaural_magnitude_db,
            q26_xyz=condition.xyz,
            q26_mask=condition.mask,
        )

    @torch.no_grad()
    def predict_prepared_diagnostics(
        self, inputs: BoundedCorrectionSubjectInputs
    ) -> BoundedCorrectionDiagnostics:
        """Run one prepared subject and retain gate/correction diagnostics."""
        normalized_condition = self.condition_normalization.normalize(
            inputs.q26_magnitude_db
        )
        latent = self.model.encode_condition(
            torch.from_numpy(normalized_condition).unsqueeze(0).to(self.device),
            torch.from_numpy(inputs.q26_xyz).unsqueeze(0).to(self.device),
            torch.from_numpy(inputs.q26_mask).unsqueeze(0).to(self.device),
        )
        directions = inputs.directions
        frequency = inputs.frequency_hz
        expected = (2, directions.shape[0], frequency.size)
        frequency_coordinate = torch.from_numpy(
            frequency_coordinates(
                frequency,
                self.frequency_mode,
                self.frequency_minimum_hz,
                self.frequency_maximum_hz,
            )
        ).to(self.device)
        final = np.empty(expected, dtype=np.float32)
        base = np.empty(expected, dtype=np.float32)
        correction = np.empty(expected, dtype=np.float32)
        gate = np.empty(expected, dtype=np.float32)
        for start in range(0, directions.shape[0], self.directions_per_block):
            stop = min(directions.shape[0], start + self.directions_per_block)
            xyz = torch.from_numpy(directions[start:stop, 2:5]).to(self.device)
            query = conditioned_coordinate_block(
                xyz,
                frequency_coordinate,
                self.conditioning_scope,
                inputs.local_mca_db[:, start:stop, :],
                self.normalization,
            )
            point_features = torch.from_numpy(
                build_point_features(
                    inputs.local_mca_db[:, start:stop, :],
                    inputs.correction_db[:, start:stop, :],
                    directions[start:stop, :],
                    frequency,
                    self.normalization,
                )
            ).to(self.device)
            normalized, normalized_base, normalized_correction, block_gate = (
                self.model.forward_grid(query, latent, point_features)
            )
            converted = bounded_diagnostics_to_db(
                normalized,
                normalized_base,
                normalized_correction,
                block_gate,
                target_mean=self.target_mean,
                target_std=self.target_std,
            )
            final[:, start:stop, :] = converted.final_residual_db
            base[:, start:stop, :] = converted.base_residual_db
            correction[:, start:stop, :] = converted.applied_correction_db
            gate[:, start:stop, :] = converted.gate
        return BoundedCorrectionDiagnostics(final, base, correction, gate)

    @torch.no_grad()
    def predict_diagnostics_db(
        self, source_h5: Path
    ) -> BoundedCorrectionDiagnostics:
        return self.predict_prepared_diagnostics(self.prepare_subject(source_h5))

    @torch.no_grad()
    def predict_residual_db(self, source_h5: Path) -> np.ndarray:
        return self.predict_diagnostics_db(source_h5).final_residual_db
