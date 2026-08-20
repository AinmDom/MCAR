from __future__ import annotations

import pytest

from scripts.analyze_siren_b_latent_e150 import (
    EARLY_WINDOW,
    LATE_WINDOW,
    plateau_budget_decision,
)


def validation_window(early: float, late: float) -> dict[int, float]:
    return {
        **{cycle: early for cycle in EARLY_WINDOW},
        **{cycle: late for cycle in LATE_WINDOW},
    }


def test_nonterminal_best_keeps_budget_without_window_audit() -> None:
    result = plateau_budget_decision(145, {})
    assert result["budget_status"] == "KEEP"


def test_terminal_best_on_flat_window_is_kept_as_plateau() -> None:
    result = plateau_budget_decision(150, validation_window(3.1, 3.1))
    assert result["budget_status"] == "KEEP_PLATEAU"
    assert result["window_improvement_percent"] == pytest.approx(0.0)


def test_terminal_best_with_point_one_percent_window_gain_retests() -> None:
    result = plateau_budget_decision(150, validation_window(3.1, 3.0969))
    assert result["budget_status"] == "RETEST"
    assert result["window_improvement_percent"] == pytest.approx(0.1)


def test_terminal_best_requires_all_frozen_window_cycles() -> None:
    with pytest.raises(ValueError, match="Missing validation cycles"):
        plateau_budget_decision(150, {150: 3.1})
