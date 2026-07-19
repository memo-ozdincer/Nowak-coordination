import math

import pytest

from nowak_coordination.game import Action, RoundResult
from nowak_coordination.rewards import (
    forecast_calibration_reward,
    hkb_bounds,
    hkb_phase,
    hkb_raw,
    hkb_reward,
    model_reward,
    payoff_reward,
)


def result(index: int, agent: Action, partner: Action, payoff: float = 0) -> RoundResult:
    return RoundResult(index, agent, partner, payoff, 0)


@pytest.mark.parametrize(
    ("payoff", "expected"),
    [(-1, 0), (0, 0.2), (3, 0.8), (4, 1)],
)
def test_payoff_reward_covers_full_matrix(payoff: float, expected: float):
    assert payoff_reward(payoff, b=4, c=1) == pytest.approx(expected)


def test_hkb_phase_alignment_and_mismatch():
    aligned = [
        result(1, Action.COOPERATE, Action.COOPERATE),
        result(2, Action.DEFECT, Action.DEFECT),
    ]
    mismatched = [
        result(1, Action.COOPERATE, Action.DEFECT),
        result(2, Action.DEFECT, Action.COOPERATE),
    ]
    assert hkb_phase(aligned) == pytest.approx(0)
    assert hkb_phase(mismatched) == pytest.approx(math.pi)
    assert hkb_phase(aligned + mismatched) == pytest.approx(math.pi / 2)


@pytest.mark.parametrize(("q", "ratio"), [(0.1, 0.25), (0.25, 0.25), (0.8, 0.25)])
def test_hkb_analytic_bounds_match_dense_search(q: float, ratio: float):
    lower, upper = hkb_bounds(q, ratio)
    sampled = [hkb_raw(i * math.pi / 10_000, q, ratio) for i in range(10_001)]
    assert lower == pytest.approx(min(sampled), abs=1e-7)
    assert upper == pytest.approx(max(sampled), abs=1e-7)


def test_hkb_normalization_is_bounded_and_valence_blind():
    cc = [result(1, Action.COOPERATE, Action.COOPERATE)]
    dd = [result(1, Action.DEFECT, Action.DEFECT)]
    mismatch = [result(1, Action.COOPERATE, Action.DEFECT)]
    assert hkb_reward(cc, q=0.8, b=4, c=1) == pytest.approx(1)
    assert hkb_reward(dd, q=0.8, b=4, c=1) == pytest.approx(1)
    assert 0 <= hkb_reward(mismatch, q=0.8, b=4, c=1) <= 1


def test_calibration_reward():
    assert forecast_calibration_reward(0.75, 0.75) == 0
    assert forecast_calibration_reward(0, 1) == -1
    with pytest.raises(ValueError):
        forecast_calibration_reward(1.1, 1)


def test_models_a_to_d_ablation_composition():
    history = [result(1, Action.COOPERATE, Action.COOPERATE, payoff=3)]
    a = model_reward("A", history, b=4, c=1, q=0.8)
    b = model_reward("B", history, b=4, c=1, q=0.8)
    c = model_reward("C", history, b=4, c=1, q=0.8, forecast=0.5, realized_group_cooperation=1)
    d = model_reward("D", history, b=4, c=1, q=0.8, forecast=0.5, realized_group_cooperation=1)
    assert a.total == pytest.approx(0.8)
    assert b.total == pytest.approx(a.total + 0.15)
    assert c.total == pytest.approx(a.total - 0.05 * 0.25)
    assert d.total == pytest.approx(b.total - 0.05 * 0.25)


def test_forecast_models_require_forecast_inputs():
    history = [result(1, Action.DEFECT, Action.DEFECT)]
    with pytest.raises(ValueError, match="requires forecast"):
        model_reward("D", history, b=4, c=1, q=0.8)
