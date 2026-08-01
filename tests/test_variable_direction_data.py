"""Offline regression test for variable-direction HDF5 sampling."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from mcar.data import (
    BinauralSpectrumSampler,
    Normalization,
    ResidualBlockSampler,
    list_hdf5_files,
)


def main() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        subject_dir = root / "subjects" / "P0001"
        subject_dir.mkdir(parents=True)
        path = subject_dir / "q26.h5"
        rng = np.random.default_rng(20260731)
        with h5py.File(path, "w") as handle:
            shape = (2, 7, 5)
            handle.create_dataset(
                "mca_logmag_db",
                data=rng.normal(size=shape).astype(np.float32),
            )
            handle.create_dataset(
                "correction_logmag_db",
                data=rng.normal(size=shape).astype(np.float32),
            )
            handle.create_dataset(
                "target_residual_db",
                data=rng.normal(size=shape).astype(np.float32),
            )
            directions = np.zeros((7, 6), dtype=np.float32)
            directions[:, 2] = np.arange(7, dtype=np.float32)
            directions[:, 1] = np.array(
                [10.0, 0.0, 0.0, 15.0, 0.0, -10.0, 0.0],
                dtype=np.float32,
            )
            directions[:, 5] = np.arange(1, 8, dtype=np.float32)
            directions[:, 5] /= np.sum(directions[:, 5])
            handle.create_dataset(
                "direction_features", data=directions
            )
            handle.create_dataset(
                "interpolation_evaluation_mask",
                data=np.array(
                    [False, False, True, True, True, True, True],
                    dtype=np.uint8,
                ),
            )
            handle.create_dataset(
                "frequency_hz",
                data=np.linspace(100.0, 10_000.0, 5, dtype=np.float32),
            )
            handle.attrs["subject_id"] = 1
            handle.attrs["sparse_order"] = 3
            handle.attrs["split"] = "train"

        files = list_hdf5_files(root, split="train")
        assert files == [path]
        normalization = Normalization(
            mca_mean=0.0,
            mca_std=1.0,
            correction_mean=0.0,
            correction_std=1.0,
            target_mean=0.0,
            target_std=1.0,
            log_frequency_mean=3.0,
            log_frequency_std=1.0,
        )
        point_sampler = ResidualBlockSampler(
            files,
            normalization,
            directions_per_batch=3,
            frequencies_per_batch=4,
            seed=1,
        )
        features, target, _ = point_sampler.sample_batch()
        assert features.shape == (12, 7)
        assert target.shape == (12, 1)

        weighted_sampler = ResidualBlockSampler(
            files,
            normalization,
            directions_per_batch=3,
            frequencies_per_batch=4,
            seed=1,
            direction_sampling="solid_angle",
            interpolation_only=True,
        )
        weighted_features, weighted_target, _ = (
            weighted_sampler.sample_batch()
        )
        assert weighted_features.shape == (12, 7)
        assert weighted_target.shape == (12, 1)
        assert np.all(weighted_features[:, 2] >= 2.0)
        assert weighted_sampler.eligible_direction_indices.size == 5

        binaural_sampler = BinauralSpectrumSampler(
            files,
            normalization,
            directions_per_batch=4,
            seed=1,
        )
        (
            binaural_features,
            binaural_target,
            _,
            _,
            sampled_directions,
            frequency,
            _,
        ) = binaural_sampler.sample_batch()
        assert binaural_features.shape == (2, 4, 5, 7)
        assert binaural_target.shape == (2, 4, 5)
        assert sampled_directions.shape == (4, 6)
        assert frequency.shape == (5,)

        interpolation_binaural_sampler = BinauralSpectrumSampler(
            files,
            normalization,
            directions_per_batch=4,
            seed=1,
            interpolation_only=True,
        )
        interpolation_sample = (
            interpolation_binaural_sampler.sample_batch()
        )
        assert np.all(interpolation_sample[4][:, 2] >= 2.0)
        assert (
            interpolation_binaural_sampler.eligible_direction_indices.size
            == 5
        )

        horizontal_binaural_sampler = BinauralSpectrumSampler(
            files,
            normalization,
            directions_per_batch=3,
            seed=1,
            interpolation_only=True,
            horizontal_only=True,
        )
        horizontal_sample = horizontal_binaural_sampler.sample_batch()
        assert np.all(np.abs(horizontal_sample[4][:, 1]) <= 1e-6)
        assert horizontal_binaural_sampler.eligible_direction_indices.size == 3
        print(
            {
                "status": "passed",
                "direction_count": 7,
                "point_batch": features.shape,
                "binaural_batch": binaural_features.shape,
            }
        )


if __name__ == "__main__":
    main()
