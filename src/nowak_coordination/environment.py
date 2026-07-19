"""Verifiers v1 taskset and user simulator for iterated Donor's Game episodes."""

from __future__ import annotations

import re
import random
from itertools import product

from pydantic import Field
import verifiers.v1 as vf

from .game import Action, EpisodeConfig, RoundResult, donor_payoffs
from .partners import make_partner
from .rewards import model_reward


SYSTEM_PROMPT = """You are playing an iterated Donor's Game.
Reply with exactly two lines and no other text:
ACTION: COOPERATE or ACTION: DEFECT
FORECAST_GROUP_COOP: a number from 0.00 to 1.00"""

_RESPONSE = re.compile(
    r"\A\s*ACTION:\s*(COOPERATE|DEFECT)\s*\n"
    r"FORECAST_GROUP_COOP:\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*\Z",
    re.IGNORECASE,
)


def round_prompt(config: EpisodeConfig, round_index: int, history: list[RoundResult]) -> str:
    lines = [
        f"Benefit to recipient: b = {config.b:g}.",
        f"Cost to cooperator: c = {config.c:g}.",
        f"Same-partner probability: w = {config.w:.2f}.",
        f"Reputation visibility probability: q = {config.q:.2f}.",
        "",
        "History:",
    ]
    if history:
        lines.extend(
            f"Round {item.round_index}: you "
            f"{'cooperated' if item.agent_action is Action.COOPERATE else 'defected'}, "
            f"partner {'cooperated' if item.partner_action is Action.COOPERATE else 'defected'}."
            for item in history
        )
    else:
        lines.append("(none)")
    lines.extend(("", f"Current round: {round_index}.", "Your output:"))
    return "\n".join(lines)


class DonorState(vf.State):
    game_over: bool = False
    invalid_output: bool = False
    agent_actions: list[int] = Field(default_factory=list)
    partner_actions: list[int] = Field(default_factory=list)
    agent_payoffs: list[float] = Field(default_factory=list)
    partner_payoffs: list[float] = Field(default_factory=list)
    forecasts: list[float] = Field(default_factory=list)


class DonorUser(vf.User[vf.UserConfig, DonorState]):
    """Host-driven heuristic partner that advances one game round per model reply."""

    async def setup_task(self, task: DonorData) -> None:
        self.episode = EpisodeConfig(**task.episode)
        self.horizon = random.Random(self.episode.seed).randint(
            self.episode.horizon_min, self.episode.horizon_max
        )
        self.rng = random.Random(self.episode.seed + 1)
        kwargs: dict[str, object] = {}
        if self.episode.partner_policy == "noisy_tit_for_tat":
            kwargs["noise_rate"] = self.episode.noise_rate
        self.partner = make_partner(self.episode.partner_policy, **kwargs)

    def _history(self) -> list[RoundResult]:
        return [
            RoundResult(
                round_index=index + 1,
                agent_action=Action(agent),
                partner_action=Action(partner),
                agent_payoff=self.state.agent_payoffs[index],
                partner_payoff=self.state.partner_payoffs[index],
            )
            for index, (agent, partner) in enumerate(
                zip(self.state.agent_actions, self.state.partner_actions, strict=True)
            )
        ]

    async def respond(self, message: str) -> vf.Messages:
        # A task with a simulator opens through ``respond(\"\")``.  Keeping the
        # initial round here (rather than in TaskData.prompt) ensures the null
        # harness always receives a user turn before it calls Qwen's template.
        if message == "":
            return [{"role": "user", "content": round_prompt(self.episode, 1, [])}]

        match = _RESPONSE.fullmatch(message)
        if match is None:
            self.state.invalid_output = True
            self.state.game_over = True
            return [{"role": "user", "content": "Episode ended: invalid output format."}]

        action = Action[match.group(1).upper()]
        forecast = float(match.group(2))
        history = self._history()
        partner_action = self.partner.act(history, self.rng)
        agent_payoff, partner_payoff = donor_payoffs(
            action, partner_action, self.episode.b, self.episode.c
        )
        self.state.agent_actions.append(int(action))
        self.state.partner_actions.append(int(partner_action))
        self.state.agent_payoffs.append(agent_payoff)
        self.state.partner_payoffs.append(partner_payoff)
        self.state.forecasts.append(forecast)

        history = self._history()
        if len(history) >= self.horizon:
            self.state.game_over = True
            return [{"role": "user", "content": "Episode complete."}]
        return [
            {
                "role": "user",
                "content": round_prompt(self.episode, len(history) + 1, history),
            }
        ]


class DonorTaskConfig(vf.TaskConfig):
    user: vf.UserConfig = vf.UserConfig(colocated=True)
    model: str = "A"


class DonorData(vf.TaskData):
    episode: dict


class DonorTask(vf.Task[DonorData, DonorState, DonorTaskConfig]):
    user = DonorUser

    def _history(self, trace: vf.Trace) -> list[RoundResult]:
        state = trace.state
        return [
            RoundResult(
                index + 1,
                Action(agent),
                Action(partner),
                state.agent_payoffs[index],
                state.partner_payoffs[index],
            )
            for index, (agent, partner) in enumerate(
                zip(state.agent_actions, state.partner_actions, strict=True)
            )
        ]

    @vf.stop
    async def game_over(self, trace: vf.Trace) -> bool:
        return trace.state.game_over

    @vf.reward
    async def episode_reward(self, trace: vf.Trace) -> float:
        if trace.state.invalid_output or not trace.state.agent_actions:
            return 0.0
        config = EpisodeConfig(**self.data.episode)
        history = self._history(trace)
        rewards = []
        for index in range(len(history)):
            prefix = history[: index + 1]
            partner_cooperation = sum(
                item.partner_action is Action.COOPERATE for item in prefix
            ) / len(prefix)
            rewards.append(
                model_reward(
                    self.config.model,
                    prefix,
                    b=config.b,
                    c=config.c,
                    q=config.q,
                    forecast=trace.state.forecasts[index],
                    realized_group_cooperation=partner_cooperation,
                ).total
            )
        return sum(rewards) / len(rewards)

    @vf.metric
    async def cooperation_rate(self, trace: vf.Trace) -> float:
        actions = trace.state.agent_actions
        return sum(action == int(Action.COOPERATE) for action in actions) / max(len(actions), 1)

    @vf.metric
    async def format_valid(self, trace: vf.Trace) -> float:
        return float(not trace.state.invalid_output)

    @vf.metric
    async def outcome_rates(self, trace: vf.Trace) -> dict[str, float]:
        history = self._history(trace)
        denominator = max(len(history), 1)
        return {
            f"p_{outcome.lower()}": sum(item.joint_action == outcome for item in history)
            / denominator
            for outcome in ("CC", "CD", "DC", "DD")
        }

    @vf.metric
    async def mean_agent_payoff(self, trace: vf.Trace) -> float:
        payoffs = trace.state.agent_payoffs
        return sum(payoffs) / max(len(payoffs), 1)


class DonorTasksetConfig(vf.TasksetConfig):
    num_tasks: int = Field(default=20, ge=1)
    seed: int = 0
    b: float = Field(default=4, gt=0)
    c: float = Field(default=1, gt=0)
    w: float = Field(default=0.7, ge=0, le=1)
    q: float = Field(default=0.8, ge=0, le=1)
    horizon_min: int = Field(default=6, ge=1)
    horizon_max: int = Field(default=12, ge=1)
    partner_policy: str = "noisy_tit_for_tat"
    noise_rate: float = Field(default=0.05, ge=0, le=1)
    b_values: list[float] = Field(default_factory=list)
    w_values: list[float] = Field(default_factory=list)
    q_values: list[float] = Field(default_factory=list)
    partner_policies: list[str] = Field(default_factory=list)
    episodes_per_cell: int = Field(default=1, ge=1)
    task: DonorTaskConfig = DonorTaskConfig()


class DonorTaskset(vf.Taskset[DonorTask, DonorTasksetConfig]):
    def load(self) -> list[DonorTask]:
        cells = list(
            product(
                self.config.b_values or [self.config.b],
                self.config.w_values or [self.config.w],
                self.config.q_values or [self.config.q],
                self.config.partner_policies or [self.config.partner_policy],
                range(self.config.episodes_per_cell),
            )
        )
        if not any(
            (
                self.config.b_values,
                self.config.w_values,
                self.config.q_values,
                self.config.partner_policies,
            )
        ):
            cells = [
                (
                    self.config.b,
                    self.config.w,
                    self.config.q,
                    self.config.partner_policy,
                    index,
                )
                for index in range(self.config.num_tasks)
            ]
        tasks = []
        for index, (b, w, q, partner_policy, replicate) in enumerate(cells):
            episode = EpisodeConfig(
                game_id=(f"donor_b{b:g}_w{w:.2f}_q{q:.2f}_{partner_policy}_r{replicate:02d}"),
                b=b,
                c=self.config.c,
                w=w,
                q=q,
                horizon_min=self.config.horizon_min,
                horizon_max=self.config.horizon_max,
                partner_policy=partner_policy,
                noise_rate=self.config.noise_rate,
                seed=self.config.seed + index,
            )
            tasks.append(
                DonorTask(
                    DonorData(
                        idx=index,
                        name=episode.game_id,
                        # Send the opening turn as a structured message.  The null harness
                        # forwards this verbatim via its initial-messages file, which avoids
                        # relying on its simulator-opening path before Qwen sees its first
                        # user query.  Subsequent turns still come from ``DonorUser``.
                        prompt=[
                            {
                                "role": "user",
                                "content": round_prompt(episode, 1, []),
                            }
                        ],
                        system_prompt=SYSTEM_PROMPT,
                        episode={
                            field: getattr(episode, field) for field in episode.__dataclass_fields__
                        },
                    ),
                    self.config.task,
                )
            )
        return tasks


if __name__ == "__main__":
    DonorUser.run()
