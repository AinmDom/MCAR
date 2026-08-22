from __future__ import annotations

import pytest

from scripts.evaluate_siren_b_condition_ablation import cyclic_donor_indices


def test_cyclic_donor_indices_have_no_self_match() -> None:
    subjects = [101, 102, 103, 104]
    donors = cyclic_donor_indices(subjects)
    assert donors == [1, 2, 3, 0]
    assert all(index != donor for index, donor in enumerate(donors))
    assert sorted(donors) == list(range(len(subjects)))


@pytest.mark.parametrize("subjects", [[], [101], [102, 101]])
def test_cyclic_donor_indices_reject_invalid_subject_order(subjects: list[int]) -> None:
    with pytest.raises(ValueError):
        cyclic_donor_indices(subjects)
