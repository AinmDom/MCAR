from __future__ import annotations

import csv

import h5py
import numpy as np
import pytest

from scripts.evaluate_complete_ten_method_test_metrics import (
    DEFERRED_UNITS,
    METHODS,
    SECONDARY_UNITS,
    STANDARDIZED_METHOD_PATHS,
    derived_seed,
    load_standardized_prediction,
    verify_primary_reproduction,
)


def test_registered_shape_is_ten_methods_and_sixteen_new_endpoints() -> None:
    assert len(METHODS) == 10
    assert len(set(METHODS)) == 10
    assert len(SECONDARY_UNITS) == 9
    assert len(DEFERRED_UNITS) == 7
    assert len(STANDARDIZED_METHOD_PATHS) == 8
    assert METHODS[-2:] == ("HYBRID", "BOUNDED")


def test_derived_bootstrap_seed_is_stable_and_group_specific() -> None:
    first = derived_seed(20260902, "FullSphereLSD", "BOUNDED", "mean")
    assert first == derived_seed(20260902, "FullSphereLSD", "BOUNDED", "mean")
    assert first != derived_seed(20260902, "FullSphereLSD", "MCARv351", "mean")
    assert 0 <= first < 2 ** 32


def test_standardized_loader_checks_split_shape_and_finite(tmp_path) -> None:
    path = tmp_path / "prediction.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("predicted_magnitude_db", data=np.zeros((2, 793, 463), np.float32))
        handle.create_dataset("predicted_hrir", data=np.zeros((793, 2, 256), np.float32))
        handle.attrs["split"] = "test"
    magnitude, hrir = load_standardized_prediction(path, "test")
    assert magnitude.shape == (2, 793, 463)
    assert hrir.shape == (793, 2, 256)
    with pytest.raises(PermissionError):
        load_standardized_prediction(path, "val")


def test_primary_reproduction_requires_all_shared_rows(tmp_path) -> None:
    primary = tmp_path / "primary.csv"
    observed = tmp_path / "observed.csv"
    subjects = ["P{:04d}".format(index) for index in range(1, 45)]
    metrics = (
        "FullSphereERB", "Contralateral25ERB",
        "ContralateralHighFrequency", "HorizontalILDMAE",
    )
    with primary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("SubjectLabel", "Method", "Metric", "Value_dB"),
        )
        writer.writeheader()
        for subject_index, subject in enumerate(subjects):
            for method_index, method in enumerate(STANDARDIZED_METHOD_PATHS):
                for metric_index, metric in enumerate(metrics):
                    writer.writerow({
                        "SubjectLabel": subject, "Method": method, "Metric": metric,
                        "Value_dB": subject_index + method_index / 10 + metric_index / 100,
                    })
    with observed.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("SubjectLabel", "Method", "Metric", "Value_dB"),
        )
        writer.writeheader()
        for subject_index, subject in enumerate(subjects):
            for method_index, method in enumerate(STANDARDIZED_METHOD_PATHS):
                source_method = "MCARv32" if method == "MCARv351" else method
                for metric_index, metric in enumerate(metrics):
                    writer.writerow({
                        "SubjectLabel": subject, "Method": source_method, "Metric": metric,
                        "Value_dB": subject_index + method_index / 10 + metric_index / 100,
                    })
    assert verify_primary_reproduction(primary, observed, subjects, 1e-12) == 0.0
