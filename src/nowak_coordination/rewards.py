"""Reward components for Models A--D in the implementation plan."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .game import RoundResult


def payoff_reward(agent_payoff: float, b: float, c: float) -> float:
    """Model A reward, normalized to [0, 1] for valid Donor's Game payoffs."""

    if b <= c or c <= 0:
        raise ValueError("reward requires b > c > 0")
    return (agent_payoff + c) / (b + c)


def hkb_phase(history: Sequence[RoundResult], window: int = 4) -> float:
    if window < 1:
        raise ValueError("window must be positive")
    if not history:
        raise ValueError("HKB phase requires at least one round")
    recent = history[-window:]
    mean_alignment = sum(
        int(result.agent_action) * int(result.partner_action) for result in recent
    ) / len(recent)
    return math.pi * (1 - mean_alignment) / 2


def hkb_raw(phase: float, q: float, cost_benefit_ratio: float) -> float:
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    if not 0 < cost_benefit_ratio < 1:
        raise ValueError("cost_benefit_ratio must be in (0, 1)")
    return 4 * q * math.cos(phase) + cost_benefit_ratio * math.cos(2 * phase)


def hkb_bounds(q: float, cost_benefit_ratio: float) -> tuple[float, float]:
    """Exact extrema of the HKB term over phase in [0, pi]."""

    ratio = cost_benefit_ratio
    maximum = 4 * q + ratio
    minimum = -2 * q * q / ratio - ratio if q <= ratio else ratio - 4 * q
    return minimum, maximum


def hkb_reward(
    history: Sequence[RoundResult], q: float, b: float, c: float, window: int = 4
) -> float:
    """Min-max normalize the HKB term to [0, 1] over its attainable phase range."""

    ratio = c / b
    raw = hkb_raw(hkb_phase(history, window), q, ratio)
    lower, upper = hkb_bounds(q, ratio)
    return (raw - lower) / (upper - lower)


def forecast_calibration_reward(forecast: float, realized_group_cooperation: float) -> float:
    """Model C collective-forecast reward (negative Brier score)."""

    if not 0 <= forecast <= 1:
        raise ValueError("forecast must be in [0, 1]")
    if not 0 <= realized_group_cooperation <= 1:
        raise ValueError("realized_group_cooperation must be in [0, 1]")
    return -((forecast - realized_group_cooperation) ** 2)


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    payoff: float
    hkb: float | None
    calibration: float | None
    total: float


@dataclass(frozen=True, slots=True)
class ShuffledHKBReference:
    """HKB history from an explicitly different episode and partner."""

    episode_id: str
    partner_id: str
    history: tuple[RoundResult, ...]


def model_reward(
    model: str,
    history: Sequence[RoundResult],
    *,
    b: float,
    c: float,
    q: float,
    forecast: float | None = None,
    realized_group_cooperation: float | None = None,
    calibration_applicable: bool = False,
    hkb_histories: Sequence[Sequence[RoundResult]] | None = None,
    shuffled_reference: ShuffledHKBReference | None = None,
    focal_episode_id: str | None = None,
    focal_partner_id: str | None = None,
    hkb_weight: float = 0.15,
    calibration_weight: float = 0.05,
) -> RewardBreakdown:
    """Compute a preregistered ablation reward for one completed round."""

    model = model.upper()
    if model not in {"A", "B", "C", "D", "E"}:
        raise ValueError("model must be one of A, B, C, D, E")
    if not history:
        raise ValueError("reward requires at least one completed round")

    payoff = payoff_reward(history[-1].agent_payoff, b, c)
    hkb = None
    if model in {"B", "D"}:
        histories = list(hkb_histories) if hkb_histories is not None else [history]
        if not histories or any(not item for item in histories):
            raise ValueError("HKB histories must be non-empty")
        hkb = sum(hkb_reward(item, q, b, c) for item in histories) / len(histories)
    elif model == "E":
        if shuffled_reference is None:
            raise ValueError("Model E requires a shuffled HKB reference")
        if not shuffled_reference.history:
            raise ValueError("shuffled HKB history must be non-empty")
        if shuffled_reference.episode_id == focal_episode_id:
            raise ValueError("shuffled HKB cannot use the focal episode")
        if shuffled_reference.partner_id == focal_partner_id:
            raise ValueError("shuffled HKB cannot use the focal partner")
        hkb = hkb_reward(shuffled_reference.history, q, b, c)
    calibration = None
    if model in {"C", "D"} and calibration_applicable:
        if forecast is None or realized_group_cooperation is None:
            raise ValueError(f"Model {model} requires a group forecast target")
        calibration = forecast_calibration_reward(forecast, realized_group_cooperation)

    total = payoff
    if hkb is not None:
        total += hkb_weight * hkb
    if calibration is not None:
        total += calibration_weight * calibration
    return RewardBreakdown(payoff=payoff, hkb=hkb, calibration=calibration, total=total)
