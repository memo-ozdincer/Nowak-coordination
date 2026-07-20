"""Deterministic and stochastic partner policies for training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Protocol, Sequence

from .game import Action, RoundResult


class PartnerPolicy(Protocol):
    def act(self, history: Sequence[RoundResult], rng: random.Random) -> Action: ...


@dataclass(frozen=True, slots=True)
class AlwaysCooperate:
    def act(self, history: Sequence[RoundResult], rng: random.Random) -> Action:
        return Action.COOPERATE


@dataclass(frozen=True, slots=True)
class AlwaysDefect:
    def act(self, history: Sequence[RoundResult], rng: random.Random) -> Action:
        return Action.DEFECT


@dataclass(frozen=True, slots=True)
class RandomPolicy:
    cooperation_probability: float = 0.5

    def __post_init__(self) -> None:
        if not 0 <= self.cooperation_probability <= 1:
            raise ValueError("cooperation_probability must be in [0, 1]")

    def act(self, history: Sequence[RoundResult], rng: random.Random) -> Action:
        return Action.COOPERATE if rng.random() < self.cooperation_probability else Action.DEFECT


@dataclass(frozen=True, slots=True)
class TitForTat:
    initial_action: Action = Action.COOPERATE

    def act(self, history: Sequence[RoundResult], rng: random.Random) -> Action:
        return history[-1].agent_action if history else self.initial_action


@dataclass(frozen=True, slots=True)
class DelayedTitForTat:
    delay: int = 2

    def __post_init__(self) -> None:
        if self.delay < 1:
            raise ValueError("delay must be positive")

    def act(self, history: Sequence[RoundResult], rng: random.Random) -> Action:
        return history[-self.delay].agent_action if len(history) >= self.delay else Action.COOPERATE


@dataclass(frozen=True, slots=True)
class GenerousTitForTat:
    forgiveness_probability: float = 0.3

    def __post_init__(self) -> None:
        if not 0 <= self.forgiveness_probability <= 1:
            raise ValueError("forgiveness_probability must be in [0, 1]")

    def act(self, history: Sequence[RoundResult], rng: random.Random) -> Action:
        if not history or history[-1].agent_action is Action.COOPERATE:
            return Action.COOPERATE
        return Action.COOPERATE if rng.random() < self.forgiveness_probability else Action.DEFECT


@dataclass(frozen=True, slots=True)
class GrimTrigger:
    def act(self, history: Sequence[RoundResult], rng: random.Random) -> Action:
        defected = any(result.agent_action is Action.DEFECT for result in history)
        return Action.DEFECT if defected else Action.COOPERATE


@dataclass(frozen=True, slots=True)
class ForgivingGrudger:
    punishment_rounds: int = 2

    def __post_init__(self) -> None:
        if self.punishment_rounds < 1:
            raise ValueError("punishment_rounds must be positive")

    def act(self, history: Sequence[RoundResult], rng: random.Random) -> Action:
        recent = history[-self.punishment_rounds :]
        return (
            Action.DEFECT
            if any(result.agent_action is Action.DEFECT for result in recent)
            else Action.COOPERATE
        )


@dataclass(frozen=True, slots=True)
class WinStayLoseShift:
    initial_action: Action = Action.COOPERATE

    def act(self, history: Sequence[RoundResult], rng: random.Random) -> Action:
        if not history:
            return self.initial_action
        previous = history[-1]
        matched = previous.agent_action is previous.partner_action
        return previous.partner_action if matched else Action(-previous.partner_action)


@dataclass(frozen=True, slots=True)
class NoisyTitForTat:
    noise_rate: float = 0.05

    def __post_init__(self) -> None:
        if not 0 <= self.noise_rate <= 1:
            raise ValueError("noise_rate must be in [0, 1]")

    def act(self, history: Sequence[RoundResult], rng: random.Random) -> Action:
        intended = history[-1].agent_action if history else Action.COOPERATE
        return Action(-intended) if rng.random() < self.noise_rate else intended


@dataclass(frozen=True, slots=True)
class Opportunist:
    """Cooperate initially, then exploit persistent cooperation and punish defection."""

    exploit_after: int = 2

    def __post_init__(self) -> None:
        if self.exploit_after < 1:
            raise ValueError("exploit_after must be positive")

    def act(self, history: Sequence[RoundResult], rng: random.Random) -> Action:
        if len(history) < self.exploit_after:
            return Action.COOPERATE
        recent = history[-self.exploit_after :]
        if all(result.agent_action is Action.COOPERATE for result in recent):
            return Action.DEFECT
        return history[-1].agent_action


TRAINING_POLICIES: dict[str, type[PartnerPolicy]] = {
    "always_cooperate": AlwaysCooperate,
    "always_defect": AlwaysDefect,
    "tit_for_tat": TitForTat,
    "generous_tit_for_tat": GenerousTitForTat,
    "grudger": GrimTrigger,
    "grim_trigger": GrimTrigger,
    "win_stay_lose_shift": WinStayLoseShift,
    "random_p": RandomPolicy,
    "noisy_tit_for_tat": NoisyTitForTat,
    "opportunist": Opportunist,
}

EVAL_POLICIES: dict[str, type[PartnerPolicy]] = {
    "forgiving_grudger": ForgivingGrudger,
    "delayed_tit_for_tat": DelayedTitForTat,
    "probabilistic_defector": RandomPolicy,
    "copy_with_noise_10%": NoisyTitForTat,
}

DIAGNOSTIC_POLICIES: dict[str, type[PartnerPolicy]] = {
    "always_cooperate": AlwaysCooperate,
    "always_defect": AlwaysDefect,
    "tit_for_tat": TitForTat,
    "generous_tit_for_tat": GenerousTitForTat,
    "opportunist": Opportunist,
}

POLICY_REGISTRIES = {
    "training": TRAINING_POLICIES,
    "heldout": EVAL_POLICIES,
    "diagnostic": DIAGNOSTIC_POLICIES,
}


def validate_policy_assignment(name: str, split: str) -> None:
    """Reject accidental training/evaluation partner leakage at config load."""

    try:
        registry = POLICY_REGISTRIES[split]
    except KeyError as exc:
        raise ValueError(f"unknown policy split {split!r}") from exc
    if name not in registry:
        raise ValueError(f"policy {name!r} is not registered in the {split!r} split")


def is_adaptive_policy(name: str) -> bool:
    return name not in {"always_cooperate", "always_defect", "random_p", "probabilistic_defector"}


def partner_decision(
    policy: PartnerPolicy,
    history: Sequence[RoundResult],
    rng: random.Random,
) -> tuple[Action, Action]:
    """Return policy intention and post-policy-noise execution separately."""

    if isinstance(policy, NoisyTitForTat):
        intended = history[-1].agent_action if history else Action.COOPERATE
        executed = Action(-int(intended)) if rng.random() < policy.noise_rate else intended
        return intended, executed
    action = policy.act(history, rng)
    return action, action


def make_partner(name: str, *, split: str | None = None, **kwargs: object) -> PartnerPolicy:
    if split is not None:
        validate_policy_assignment(name, split)
    try:
        policy_type = (
            TRAINING_POLICIES.get(name) or EVAL_POLICIES.get(name) or DIAGNOSTIC_POLICIES[name]
        )
    except KeyError as exc:
        known = sorted(TRAINING_POLICIES.keys() | EVAL_POLICIES.keys())
        raise ValueError(f"unknown partner policy {name!r}; choose from {known}") from exc
    if name == "probabilistic_defector" and "cooperation_probability" not in kwargs:
        kwargs["cooperation_probability"] = 0.2
    if name == "copy_with_noise_10%" and "noise_rate" not in kwargs:
        kwargs["noise_rate"] = 0.1
    return policy_type(**kwargs)
