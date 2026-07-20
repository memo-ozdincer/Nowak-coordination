"""Pure seeded mechanics for dyadic, naturalistic, and group episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import random
import re
from typing import Any

from .game import Action, EpisodeConfig, RoundResult, donor_payoffs
from .partners import (
    is_adaptive_policy,
    make_partner,
    partner_decision,
    validate_policy_assignment,
)
from .rewards import ShuffledHKBReference, model_reward


_FORECAST = r"(0(?:\.\d+)?|1(?:\.0+)?)"
_NEUTRAL_LABEL_PAIRS = (
    ("CHOICE LANTERN", "CHOICE HARBOR"),
    ("CHOICE MAPLE", "CHOICE ORBIT"),
    ("CHOICE RIVER", "CHOICE SUMMIT"),
    ("CHOICE EMBER", "CHOICE MEADOW"),
)


def named_seed(root_seed: int, stream_name: str) -> int:
    """Derive a stable independent RNG seed without Python's salted hash."""

    payload = f"{root_seed}\0{stream_name}".encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"nowak-v1").digest()
    return int.from_bytes(digest, "big")


def action_code(action: Action) -> str:
    return "C" if action is Action.COOPERATE else "D"


def flip(action: Action) -> Action:
    return Action(-int(action))


@dataclass(frozen=True, slots=True)
class ReputationRecord:
    actions: tuple[Action, ...]

    @property
    def cooperation_fraction(self) -> float:
        return sum(action is Action.COOPERATE for action in self.actions) / len(self.actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [action_code(action) for action in self.actions],
            "cooperation_fraction": self.cooperation_fraction,
        }


@dataclass(frozen=True, slots=True)
class Observation:
    round_index: int
    partner_ids: tuple[str, ...]
    transition: str
    previous_partner_id: str | None
    reputations: tuple[ReputationRecord, ...]
    reputation_visible: tuple[bool, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "partner_ids": list(self.partner_ids),
            "transition": self.transition,
            "previous_partner_id": self.previous_partner_id,
            "reputations": [
                {
                    "latent": reputation.to_dict(),
                    "visible": visible,
                    "rendered": reputation.to_dict() if visible else None,
                }
                for reputation, visible in zip(
                    self.reputations, self.reputation_visible, strict=True
                )
            ],
        }


@dataclass(slots=True)
class PartnerRuntime:
    partner_id: str
    policy_name: str
    policy: Any
    reputation: ReputationRecord
    rng: random.Random
    history: list[RoundResult] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class StepResult:
    event: dict[str, Any]
    done: bool


def _policy_kwargs(name: str, config: EpisodeConfig) -> dict[str, object]:
    return {"noise_rate": config.noise_rate} if name == "noisy_tit_for_tat" else {}


def validate_config_policies(config: EpisodeConfig) -> None:
    names = set(config.replacement_policies)
    if config.interleaved_policies:
        names.update(config.interleaved_policies)
    else:
        names.add(config.partner_policy)
    if config.switch_to_policy is not None:
        names.add(config.switch_to_policy)
    for name in names:
        validate_policy_assignment(name, config.policy_split)


def _labels_for(config: EpisodeConfig) -> dict[Action, str]:
    if config.mode != "naturalistic":
        return {Action.COOPERATE: "COOPERATE", Action.DEFECT: "DEFECT"}
    rng = random.Random(named_seed(config.seed, "label_mapping"))
    pair = _NEUTRAL_LABEL_PAIRS[rng.randrange(len(_NEUTRAL_LABEL_PAIRS))]
    should_flip = (
        bool(rng.randrange(2))
        if config.naturalistic_label_flip is None
        else config.naturalistic_label_flip
    )
    if should_flip:
        pair = (pair[1], pair[0])
    return {Action.COOPERATE: pair[0], Action.DEFECT: pair[1]}


def system_prompt(config: EpisodeConfig, labels: dict[Action, str] | None = None) -> str:
    labels = labels or _labels_for(config)
    if config.mode == "naturalistic":
        return (
            "Choose exactly one available option. Reply with exactly two lines and no other "
            f"text:\nACTION: {labels[Action.COOPERATE]} or ACTION: "
            f"{labels[Action.DEFECT]}\nFORECAST_GROUP_COOP: a number from 0.00 to 1.00"
        )
    return (
        "You are playing an iterated Donor's Game.\n"
        "Reply with exactly two lines and no other text:\n"
        "ACTION: COOPERATE or ACTION: DEFECT\n"
        "FORECAST_GROUP_COOP: a number from 0.00 to 1.00"
    )


def parse_response(
    message: str, config: EpisodeConfig, labels: dict[Action, str] | None = None
) -> tuple[Action, float]:
    labels = labels or _labels_for(config)
    choices = sorted(labels.items(), key=lambda item: len(item[1]), reverse=True)
    label_pattern = "|".join(re.escape(label) for _, label in choices)
    pattern = re.compile(
        rf"\A\s*ACTION:\s*({label_pattern})\s*\nFORECAST_GROUP_COOP:\s*{_FORECAST}\s*\Z",
        re.IGNORECASE,
    )
    match = pattern.fullmatch(message)
    if match is None:
        raise ValueError("invalid output format")
    by_label = {label.upper(): action for action, label in choices}
    return by_label[match.group(1).upper()], float(match.group(2))


class _WorldBase:
    def __init__(self, config: EpisodeConfig, reward_model: str) -> None:
        validate_config_policies(config)
        self.config = config
        self.reward_model = reward_model.upper()
        if self.reward_model not in {"A", "B", "C", "D", "E"}:
            raise ValueError("reward_model must be A, B, C, D, or E")
        self.labels = _labels_for(config)
        self.horizon = random.Random(named_seed(config.seed, "horizon")).randint(
            config.horizon_min, config.horizon_max
        )
        self.visibility_rng = random.Random(named_seed(config.seed, "reputation_visibility"))
        self.selection_rng = random.Random(named_seed(config.seed, "partner_selection"))
        self.retention_rng = random.Random(named_seed(config.seed, "retention"))
        self.partner_counter = 0
        self.events: list[dict[str, Any]] = []
        self.observations: list[Observation] = []
        self.shuffled_reference = self._make_shuffled_reference()

    @property
    def done(self) -> bool:
        return len(self.events) >= self.horizon

    def _identity(self, namespace: str, index: int) -> str:
        value = named_seed(self.config.seed, f"{namespace}:{index}")
        return f"p-{value:016x}"

    def _reputation(self, policy_name: str, index: int) -> ReputationRecord:
        policy = make_partner(
            policy_name,
            split=self.config.policy_split,
            **_policy_kwargs(policy_name, self.config),
        )
        rng = random.Random(named_seed(self.config.seed, f"reputation:{index}"))
        history: list[RoundResult] = []
        partner_actions: list[Action] = []
        for round_index in range(1, self.config.reputation_length + 1):
            _, partner_action = partner_decision(policy, history, rng)
            agent_action = Action.COOPERATE if rng.random() < 0.5 else Action.DEFECT
            history.append(RoundResult(round_index, agent_action, partner_action, 0.0, 0.0))
            partner_actions.append(partner_action)
        return ReputationRecord(tuple(partner_actions))

    def _make_partner(self, policy_name: str) -> PartnerRuntime:
        index = self.partner_counter
        self.partner_counter += 1
        return PartnerRuntime(
            partner_id=self._identity("partner", index),
            policy_name=policy_name,
            policy=make_partner(
                policy_name,
                split=self.config.policy_split,
                **_policy_kwargs(policy_name, self.config),
            ),
            reputation=self._reputation(policy_name, index),
            rng=random.Random(named_seed(self.config.seed, f"partner_action:{index}")),
        )

    def _make_shuffled_reference(self) -> ShuffledHKBReference:
        candidates = (
            self.config.replacement_policies
            or self.config.interleaved_policies
            or (self.config.partner_policy,)
        )
        policy_name = candidates[
            random.Random(named_seed(self.config.seed, "shuffle_selection")).randrange(
                len(candidates)
            )
        ]
        policy = make_partner(
            policy_name,
            split=self.config.policy_split,
            **_policy_kwargs(policy_name, self.config),
        )
        rng = random.Random(named_seed(self.config.seed, "shuffle_actions"))
        history: list[RoundResult] = []
        for round_index in range(1, 5):
            focal = Action.COOPERATE if rng.random() < 0.5 else Action.DEFECT
            partner_intended, partner = partner_decision(policy, history, rng)
            focal_payoff, partner_payoff = donor_payoffs(
                focal, partner, self.config.b, self.config.c
            )
            history.append(
                RoundResult(
                    round_index,
                    focal,
                    partner,
                    focal_payoff,
                    partner_payoff,
                    focal,
                    partner_intended,
                )
            )
        return ShuffledHKBReference(
            episode_id=f"shuffle-{named_seed(self.config.seed, 'shuffle_episode'):016x}",
            partner_id=f"s-{named_seed(self.config.seed, 'shuffle_partner'):016x}",
            history=tuple(history),
        )

    def _visibility(self, count: int) -> tuple[bool, ...]:
        return tuple(self.visibility_rng.random() < self.config.q for _ in range(count))

    def _shuffled_for_round(self, round_index: int) -> ShuffledHKBReference:
        """Match Model E's history length to B while changing only its source."""

        prefix_length = min(round_index, len(self.shuffled_reference.history))
        return ShuffledHKBReference(
            episode_id=self.shuffled_reference.episode_id,
            partner_id=self.shuffled_reference.partner_id,
            history=self.shuffled_reference.history[:prefix_length],
        )

    def trace_header(self) -> dict[str, Any]:
        label_payload = "\n".join(
            f"{label}={action_code(action)}"
            for action, label in sorted(self.labels.items(), key=lambda item: item[1])
        )
        return {
            "episode_id": self.config.game_id,
            "root_seed": self.config.seed,
            "mode": self.config.mode,
            "reward_model": self.reward_model,
            "horizon": self.horizon,
            "policy_split": self.config.policy_split,
            "label_mapping_sha256": hashlib.sha256(label_payload.encode()).hexdigest(),
            "shuffled_hkb_source": {
                "episode_id": self.shuffled_reference.episode_id,
                "partner_id": self.shuffled_reference.partner_id,
            },
        }


class DyadicWorld(_WorldBase):
    """Fixed-opportunity episode with causal same-partner retention."""

    def __init__(self, config: EpisodeConfig, reward_model: str = "A") -> None:
        if config.mode == "group":
            raise ValueError("DyadicWorld cannot run group mode")
        super().__init__(config, reward_model)
        self.global_history: list[RoundResult] = []
        if config.interleaved_policies:
            self.interleaved = [self._make_partner(name) for name in config.interleaved_policies]
            self.current = self.interleaved[0]
        else:
            self.interleaved = []
            self.current = self._make_partner(config.partner_policy)
        self.current_observation = self._observe("initial", None)

    def _observe(self, transition: str, previous_partner_id: str | None) -> Observation:
        observation = Observation(
            round_index=len(self.events) + 1,
            partner_ids=(self.current.partner_id,),
            transition=transition,
            previous_partner_id=previous_partner_id,
            reputations=(self.current.reputation,),
            reputation_visible=self._visibility(1),
        )
        self.observations.append(observation)
        return observation

    def render_prompt(self) -> str:
        observation = self.current_observation
        label_c = self.labels[Action.COOPERATE]
        label_d = self.labels[Action.DEFECT]
        if self.config.mode == "naturalistic":
            lines = [
                f"Option {label_c} transfers {self.config.b:g} value units to the other "
                f"participant at a cost of {self.config.c:g} to you.",
                f"Option {label_d} transfers nothing and costs nothing.",
                f"The same participant is retained with probability w = {self.config.w:.2f}.",
                f"Background-record visibility probability: q = {self.config.q:.2f}.",
            ]
        else:
            lines = [
                f"Benefit to recipient: b = {self.config.b:g}.",
                f"Cost to donor: c = {self.config.c:g}.",
                f"Same-partner retention probability: w = {self.config.w:.2f}.",
                f"Reputation visibility probability: q = {self.config.q:.2f}.",
            ]
        lines.extend(
            (
                f"Current partner ID: {self.current.partner_id}.",
                f"Partner transition: {observation.transition}.",
            )
        )
        if observation.reputation_visible[0]:
            lines.append(
                "Observed reputation: cooperation fraction "
                f"{self.current.reputation.cooperation_fraction:.2f} over "
                f"{len(self.current.reputation.actions)} reference interactions."
            )
        else:
            lines.append("No reputation information was observed.")
        lines.extend(("", "History with this partner:"))
        if self.current.history:
            for item in self.current.history:
                lines.append(
                    f"Interaction {item.round_index}: you chose "
                    f"{self.labels[item.agent_action]}, partner chose "
                    f"{self.labels[item.partner_action]}."
                )
        else:
            lines.append("(none)")
        lines.extend(("", f"Current round: {len(self.events) + 1}.", "Your output:"))
        return "\n".join(lines)

    def _next_partner(self) -> tuple[str, bool | None, float | None, str | None]:
        previous = self.current
        next_round = len(self.events) + 1
        if self.interleaved:
            self.current = self.interleaved[(next_round - 1) % 2]
            return "interleaved", None, None, previous.partner_id
        if (
            self.config.partner_switch_round is not None
            and next_round == self.config.partner_switch_round
        ):
            self.current = self._make_partner(self.config.switch_to_policy or "")
            return "forced_switch", False, None, previous.partner_id
        draw = self.retention_rng.random()
        retained = draw < self.config.w
        if retained:
            return "retained", True, draw, previous.partner_id
        candidates = self.config.replacement_policies or (self.config.partner_policy,)
        replacement = candidates[self.selection_rng.randrange(len(candidates))]
        self.current = self._make_partner(replacement)
        return "replaced", False, draw, previous.partner_id

    def step(self, intended_action: Action, forecast: float) -> StepResult:
        if self.done:
            raise RuntimeError("episode is already complete")
        round_index = len(self.events) + 1
        rendered_observation = self.render_prompt()
        partner = self.current
        focal_executed = intended_action
        focal_perturbed = (
            self.config.perturbation_round == round_index
            and self.config.perturbation_actor == "focal"
        )
        if focal_perturbed:
            focal_executed = flip(focal_executed)
        partner_intended, partner_executed = partner_decision(
            partner.policy, partner.history, partner.rng
        )
        partner_perturbed = (
            self.config.perturbation_round == round_index
            and self.config.perturbation_actor == "partner"
        )
        if partner_perturbed:
            partner_executed = flip(partner_intended)
        focal_payoff, partner_payoff = donor_payoffs(
            focal_executed, partner_executed, self.config.b, self.config.c
        )
        result = RoundResult(
            round_index,
            focal_executed,
            partner_executed,
            focal_payoff,
            partner_payoff,
            intended_action,
            partner_intended,
            partner.partner_id,
        )
        partner.history.append(result)
        self.global_history.append(result)
        reward = model_reward(
            self.reward_model,
            partner.history,
            b=self.config.b,
            c=self.config.c,
            q=self.config.q,
            forecast=forecast,
            calibration_applicable=False,
            shuffled_reference=self._shuffled_for_round(round_index),
            focal_episode_id=self.config.game_id,
            focal_partner_id=partner.partner_id,
        )
        event: dict[str, Any] = {
            "episode_id": self.config.game_id,
            "round_index": round_index,
            "mode": self.config.mode,
            "partner_ids": [partner.partner_id],
            "partner_policy": partner.policy_name,
            "partner_adaptive": is_adaptive_policy(partner.policy_name),
            "partner_history_length_before": len(partner.history) - 1,
            "observation": self.current_observation.to_dict(),
            "rendered_observation": rendered_observation,
            "focal_intended_action": action_code(intended_action),
            "focal_executed_action": action_code(focal_executed),
            "partner_intended_actions": [action_code(partner_intended)],
            "partner_executed_actions": [action_code(partner_executed)],
            "focal_payoff": focal_payoff,
            "partner_payoffs": [partner_payoff],
            "joint_outcomes": [result.joint_action],
            "perturbation": {
                "applied": focal_perturbed or partner_perturbed,
                "actor": self.config.perturbation_actor
                if focal_perturbed or partner_perturbed
                else None,
            },
            "forecast": forecast,
            "forecast_target": None,
            "reward": asdict(reward),
            "hkb_source": (
                {
                    "episode_id": self.shuffled_reference.episode_id,
                    "partner_id": self.shuffled_reference.partner_id,
                }
                if self.reward_model == "E"
                else {"episode_id": self.config.game_id, "partner_id": partner.partner_id}
            ),
            "retention_draw": None,
            "retained_for_next": None,
            "next_partner_id": None,
            "transition_to_next": None,
        }
        self.events.append(event)
        if not self.done:
            transition, retained, retention_draw, previous_id = self._next_partner()
            event["retention_draw"] = retention_draw
            event["retained_for_next"] = retained
            event["transition_to_next"] = transition
            event["next_partner_id"] = self.current.partner_id
            self.current_observation = self._observe(transition, previous_id)
        return StepResult(event=event, done=self.done)


class GroupWorld(_WorldBase):
    """Four/five-agent simultaneous group-donor episode for genuine CFE."""

    def __init__(self, config: EpisodeConfig, reward_model: str = "D") -> None:
        if config.mode != "group":
            raise ValueError("GroupWorld requires group mode")
        super().__init__(config, reward_model)
        candidates = config.replacement_policies or (config.partner_policy,)
        self.peers = [
            self._make_partner(candidates[index % len(candidates)])
            for index in range(config.group_size - 1)
        ]
        self.focal_payoff_history: list[RoundResult] = []
        self.current_observation = self._observe()

    def _observe(self) -> Observation:
        observation = Observation(
            round_index=len(self.events) + 1,
            partner_ids=tuple(peer.partner_id for peer in self.peers),
            transition="stable_group" if self.events else "initial_group",
            previous_partner_id=None,
            reputations=tuple(peer.reputation for peer in self.peers),
            reputation_visible=self._visibility(len(self.peers)),
        )
        self.observations.append(observation)
        return observation

    def render_prompt(self) -> str:
        lines = [
            f"Group size: {self.config.group_size}.",
            f"Each cooperator pays cost c = {self.config.c:g}.",
            f"Each agent receives total benefit up to b = {self.config.b:g} from "
            "the other agents' cooperation.",
            f"Reputation visibility probability: q = {self.config.q:.2f}.",
            "Current peer IDs: " + ", ".join(peer.partner_id for peer in self.peers) + ".",
        ]
        for peer, visible in zip(
            self.peers, self.current_observation.reputation_visible, strict=True
        ):
            if visible:
                lines.append(
                    f"Observed reputation for {peer.partner_id}: cooperation fraction "
                    f"{peer.reputation.cooperation_fraction:.2f}."
                )
            else:
                lines.append(f"No reputation information was observed for {peer.partner_id}.")
        lines.extend(("", "Group history:"))
        if self.events:
            for event in self.events:
                lines.append(
                    f"Round {event['round_index']}: your action "
                    f"{event['focal_executed_action']}; group cooperation "
                    f"{event['forecast_target']:.2f}."
                )
        else:
            lines.append("(none)")
        lines.extend(("", f"Current round: {len(self.events) + 1}.", "Your output:"))
        return "\n".join(lines)

    def step(self, intended_action: Action, forecast: float) -> StepResult:
        if self.done:
            raise RuntimeError("episode is already complete")
        round_index = len(self.events) + 1
        rendered_observation = self.render_prompt()
        focal_executed = intended_action
        focal_perturbed = (
            self.config.perturbation_round == round_index
            and self.config.perturbation_actor == "focal"
        )
        if focal_perturbed:
            focal_executed = flip(focal_executed)
        decisions = [partner_decision(peer.policy, peer.history, peer.rng) for peer in self.peers]
        peer_intended = [decision[0] for decision in decisions]
        peer_executed = [decision[1] for decision in decisions]
        partner_perturbed = (
            self.config.perturbation_round == round_index
            and self.config.perturbation_actor == "partner"
        )
        if partner_perturbed:
            peer_executed[0] = flip(peer_intended[0])
        actions = [focal_executed, *peer_executed]
        cooperation_count = sum(action is Action.COOPERATE for action in actions)

        def group_payoff(index: int) -> float:
            others_cooperating = cooperation_count - (actions[index] is Action.COOPERATE)
            benefit = self.config.b * others_cooperating / (self.config.group_size - 1)
            cost = self.config.c if actions[index] is Action.COOPERATE else 0.0
            return float(benefit - cost)

        payoffs = [group_payoff(index) for index in range(self.config.group_size)]
        dyadic_results = []
        for peer, intended, executed, payoff in zip(
            self.peers, peer_intended, peer_executed, payoffs[1:], strict=True
        ):
            result = RoundResult(
                round_index,
                focal_executed,
                executed,
                payoffs[0],
                payoff,
                intended_action,
                intended,
                peer.partner_id,
            )
            peer.history.append(result)
            dyadic_results.append(result)
        target = cooperation_count / self.config.group_size
        summary_partner = (
            Action.COOPERATE
            if sum(action is Action.COOPERATE for action in peer_executed) * 2 >= len(peer_executed)
            else Action.DEFECT
        )
        summary = RoundResult(
            round_index,
            focal_executed,
            summary_partner,
            payoffs[0],
            sum(payoffs[1:]) / len(payoffs[1:]),
            intended_action,
            summary_partner,
            "group",
        )
        self.focal_payoff_history.append(summary)
        reward = model_reward(
            self.reward_model,
            self.focal_payoff_history,
            b=self.config.b,
            c=self.config.c,
            q=self.config.q,
            forecast=forecast,
            realized_group_cooperation=target,
            calibration_applicable=True,
            hkb_histories=[peer.history for peer in self.peers],
            shuffled_reference=self._shuffled_for_round(round_index),
            focal_episode_id=self.config.game_id,
            focal_partner_id="group",
        )
        event: dict[str, Any] = {
            "episode_id": self.config.game_id,
            "round_index": round_index,
            "mode": self.config.mode,
            "partner_ids": [peer.partner_id for peer in self.peers],
            "partner_policy": [peer.policy_name for peer in self.peers],
            "partner_adaptive": [is_adaptive_policy(peer.policy_name) for peer in self.peers],
            "partner_history_length_before": [len(peer.history) - 1 for peer in self.peers],
            "observation": self.current_observation.to_dict(),
            "rendered_observation": rendered_observation,
            "focal_intended_action": action_code(intended_action),
            "focal_executed_action": action_code(focal_executed),
            "partner_intended_actions": [action_code(action) for action in peer_intended],
            "partner_executed_actions": [action_code(action) for action in peer_executed],
            "focal_payoff": payoffs[0],
            "partner_payoffs": payoffs[1:],
            "joint_outcomes": [result.joint_action for result in dyadic_results],
            "perturbation": {
                "applied": focal_perturbed or partner_perturbed,
                "actor": self.config.perturbation_actor
                if focal_perturbed or partner_perturbed
                else None,
            },
            "forecast": forecast,
            "forecast_target": target,
            "reward": asdict(reward),
            "hkb_source": (
                {
                    "episode_id": self.shuffled_reference.episode_id,
                    "partner_id": self.shuffled_reference.partner_id,
                }
                if self.reward_model == "E"
                else {
                    "episode_id": self.config.game_id,
                    "partner_id": [peer.partner_id for peer in self.peers],
                }
            ),
            "retention_draw": None,
            "retained_for_next": None,
            "next_partner_id": None,
            "transition_to_next": "stable_group" if round_index < self.horizon else None,
        }
        self.events.append(event)
        if not self.done:
            self.current_observation = self._observe()
        return StepResult(event=event, done=self.done)


def make_world(config: EpisodeConfig, reward_model: str) -> DyadicWorld | GroupWorld:
    return (
        GroupWorld(config, reward_model)
        if config.mode == "group"
        else DyadicWorld(config, reward_model)
    )
