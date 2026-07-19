"""Core iterated Donor's Game types and transition logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import random
from typing import Iterable


class Action(IntEnum):
    """Binary Donor's Game action, encoded for the HKB reward."""

    DEFECT = -1
    COOPERATE = 1

    @classmethod
    def parse(cls, value: str) -> Action:
        normalized = value.strip().upper()
        if normalized.startswith("ACTION:"):
            normalized = normalized.split(":", 1)[1].strip()
        try:
            return cls[normalized]
        except KeyError as exc:
            raise ValueError(f"invalid action: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    game_id: str
    b: float
    c: float
    w: float
    q: float
    horizon_min: int = 6
    horizon_max: int = 12
    partner_policy: str = "tit_for_tat"
    noise_rate: float = 0.0
    reputation_visible: bool = True
    partner_switch_round: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.b <= 0 or self.c <= 0:
            raise ValueError("b and c must be positive")
        if self.b <= self.c:
            raise ValueError("Donor's Game requires b > c")
        for name, value in (("w", self.w), ("q", self.q), ("noise_rate", self.noise_rate)):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.horizon_min < 1 or self.horizon_max < self.horizon_min:
            raise ValueError("invalid horizon bounds")
        if self.partner_switch_round is not None and self.partner_switch_round < 1:
            raise ValueError("partner_switch_round must be positive")


@dataclass(frozen=True, slots=True)
class RoundResult:
    round_index: int
    agent_action: Action
    partner_action: Action
    agent_payoff: float
    partner_payoff: float

    @property
    def joint_action(self) -> str:
        return ("C" if self.agent_action is Action.COOPERATE else "D") + (
            "C" if self.partner_action is Action.COOPERATE else "D"
        )


def donor_payoffs(
    agent_action: Action, partner_action: Action, b: float, c: float
) -> tuple[float, float]:
    """Return simultaneous Donor's Game payoffs for agent and partner."""

    agent = b * (partner_action is Action.COOPERATE) - c * (agent_action is Action.COOPERATE)
    partner = b * (agent_action is Action.COOPERATE) - c * (partner_action is Action.COOPERATE)
    return float(agent), float(partner)


@dataclass(slots=True)
class DonorGame:
    """Seeded episode state. Partner action selection is intentionally external."""

    config: EpisodeConfig
    history: list[RoundResult] = field(default_factory=list)
    rng: random.Random = field(init=False, repr=False)
    horizon: int = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.config.seed)
        self.horizon = self.rng.randint(self.config.horizon_min, self.config.horizon_max)

    @property
    def done(self) -> bool:
        return len(self.history) >= self.horizon

    def step(self, agent_action: Action, partner_action: Action) -> RoundResult:
        if self.done:
            raise RuntimeError("episode is already complete")
        agent_payoff, partner_payoff = donor_payoffs(
            agent_action, partner_action, self.config.b, self.config.c
        )
        result = RoundResult(
            round_index=len(self.history) + 1,
            agent_action=agent_action,
            partner_action=partner_action,
            agent_payoff=agent_payoff,
            partner_payoff=partner_payoff,
        )
        self.history.append(result)
        return result

    def recent(self, window: int) -> tuple[RoundResult, ...]:
        if window < 1:
            raise ValueError("window must be positive")
        return tuple(self.history[-window:])

    def outcome_counts(self) -> dict[str, int]:
        counts = {"CC": 0, "CD": 0, "DC": 0, "DD": 0}
        for result in self.history:
            counts[result.joint_action] += 1
        return counts


def cooperation_rate(actions: Iterable[Action]) -> float:
    values = list(actions)
    if not values:
        return 0.0
    return sum(action is Action.COOPERATE for action in values) / len(values)
