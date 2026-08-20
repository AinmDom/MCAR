from __future__ import annotations

import numpy as np

from mcar.q26_condition import Q26MagnitudeNormalization


def test_q26_normalization_is_per_ear_and_frequency() -> None:
    normalization = Q26MagnitudeNormalization(
        mean_db=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        std_db=np.asarray([[2.0, 4.0], [5.0, 10.0]], dtype=np.float32),
        frequency_hz=np.asarray([100.0, 200.0], dtype=np.float32),
        training_subject_count=262,
    )
    values = np.asarray(
        [
            [[3.0, 6.0], [1.0, 2.0]],
            [[8.0, 14.0], [3.0, 4.0]],
        ],
        dtype=np.float32,
    )
    expected = np.asarray(
        [
            [[1.0, 1.0], [0.0, 0.0]],
            [[1.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(normalization.normalize(values), expected)
