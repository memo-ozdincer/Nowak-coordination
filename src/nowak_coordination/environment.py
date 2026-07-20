"""Verifiers v1 adapter over the pure seeded coordination mechanics."""

from __future__ import annotations

from itertools import product
from typing import Any, Literal

from pydantic import Field, model_validator
import verifiers.v1 as vf

from .game import Action, EpisodeConfig
from .mechanics import make_world, parse_response, system_prompt


SYSTEM_PROMPT = """You are playing an iterated Donor's Game.
Reply with exactly two lines and no other text:
ACTION: COOPERATE or ACTION: DEFECT
FORECAST_GROUP_COOP: a number from 0.00 to 1.00"""
TRAINING_SEEDS_BY_MODEL = {
    "A": set(range(1101, 1106)),
    "B": set(range(1201, 1206)),
    "C": set(range(1301, 1306)),
    "D": set(range(1401, 1404)),
    "E": set(range(1501, 1506)),
}
VALIDATION_SEEDS = {2101, 2102, 2103, 2104, 2105}
TEST_SEEDS = {3101, 3102, 3103, 3104, 3105}


def initial_prompt(config: EpisodeConfig, reward_model: str = "A") -> str:
    """Render the deterministic opening observation for task construction."""

    return make_world(config, reward_model).render_prompt()


class DonorState(vf.State):
    """JSON-serializable scientific trace state."""

    game_over: bool = False
    invalid_output: bool = False
    terminal_reason: str | None = None
    trace_header: dict[str, Any] = Field(default_factory=dict)
    rounds: list[dict[str, Any]] = Field(default_factory=list)
    observations: list[dict[str, Any]] = Field(default_factory=list)
    terminal_event: dict[str, Any] | None = None
    # Compatibility columns retained for PRIME metrics and old engineering tools.
    agent_actions: list[int] = Field(default_factory=list)
    partner_actions: list[int] = Field(default_factory=list)
    agent_payoffs: list[float] = Field(default_factory=list)
    partner_payoffs: list[float] = Field(default_factory=list)
    forecasts: list[float] = Field(default_factory=list)
    forecast_targets: list[float | None] = Field(default_factory=list)


class DonorUser(vf.User[vf.UserConfig, DonorState]):
    """Host-driven simulator: one parsed model response advances one round."""

    async def setup_task(self, task: DonorData) -> None:
        self.episode = EpisodeConfig(**task.episode)
        self.reward_model = task.reward_model
        self.world = make_world(self.episode, self.reward_model)
        self.state.trace_header = self.world.trace_header()
        self.state.trace_header["policy_arm"] = task.policy_arm
        self.state.trace_header["seed_metadata"] = task.seed_metadata
        self.state.trace_header["sampling_metadata"] = task.sampling_metadata
        self.state.observations = [observation.to_dict() for observation in self.world.observations]

    def _finish(self, reason: str) -> None:
        self.state.game_over = True
        self.state.terminal_reason = reason
        self.state.terminal_event = {
            "episode_id": self.episode.game_id,
            "terminal_reason": reason,
            "rounds_completed": len(self.state.rounds),
            "expected_horizon": self.world.horizon,
            "complete": True,
            "label_mapping": {
                label: ("C" if action is Action.COOPERATE else "D")
                for action, label in self.world.labels.items()
            },
        }

    async def respond(self, message: str) -> vf.Messages:
        if message == "":
            return [{"role": "user", "content": self.world.render_prompt()}]

        try:
            action, forecast = parse_response(message, self.episode, self.world.labels)
        except ValueError:
            self.state.invalid_output = True
            self._finish("invalid_format")
            return [{"role": "user", "content": "Episode ended: invalid output format."}]

        result = self.world.step(action, forecast)
        event = result.event
        self.state.rounds.append(event)
        self.state.agent_actions.append(
            int(Action.COOPERATE if event["focal_executed_action"] == "C" else Action.DEFECT)
        )
        first_partner_action = event["partner_executed_actions"][0]
        self.state.partner_actions.append(
            int(Action.COOPERATE if first_partner_action == "C" else Action.DEFECT)
        )
        self.state.agent_payoffs.append(float(event["focal_payoff"]))
        self.state.partner_payoffs.append(float(event["partner_payoffs"][0]))
        self.state.forecasts.append(float(event["forecast"]))
        self.state.forecast_targets.append(event["forecast_target"])
        self.state.observations = [observation.to_dict() for observation in self.world.observations]

        if result.done:
            self._finish("horizon")
            return [{"role": "user", "content": "Episode complete."}]
        return [{"role": "user", "content": self.world.render_prompt()}]


class DonorTaskConfig(vf.TaskConfig):
    user: vf.UserConfig = vf.UserConfig(colocated=True)
    model: str = "A"


class DonorData(vf.TaskData):
    episode: dict[str, Any]
    reward_model: str = "A"
    policy_arm: str = "A"
    seed_metadata: dict[str, Any] = Field(default_factory=dict)
    sampling_metadata: dict[str, Any] = Field(default_factory=dict)


class DonorTask(vf.Task[DonorData, DonorState, DonorTaskConfig]):
    user = DonorUser

    @staticmethod
    def _persist_state(trace: vf.Trace) -> None:
        """Copy transient simulator state into the JSON-persistent info field."""

        trace.info["coordination_trace"] = trace.state.model_dump(mode="json")

    @vf.stop
    async def game_over(self, trace: vf.Trace) -> bool:
        return trace.state.game_over

    @vf.reward
    async def episode_reward(self, trace: vf.Trace) -> float:
        self._persist_state(trace)
        if trace.state.invalid_output or not trace.state.rounds:
            return 0.0
        return sum(float(item["reward"]["total"]) for item in trace.state.rounds) / len(
            trace.state.rounds
        )

    @vf.metric
    async def cooperation_rate(self, trace: vf.Trace) -> float:
        rounds = trace.state.rounds
        return sum(item["focal_executed_action"] == "C" for item in rounds) / max(len(rounds), 1)

    @vf.metric
    async def format_valid(self, trace: vf.Trace) -> float:
        return float(not trace.state.invalid_output)

    @vf.metric
    async def trace_complete(self, trace: vf.Trace) -> float:
        terminal = trace.state.terminal_event
        return float(
            terminal is not None
            and terminal.get("complete") is True
            and trace.state.terminal_reason in {"horizon", "invalid_format"}
        )

    @vf.metric
    async def outcome_rates(self, trace: vf.Trace) -> dict[str, float]:
        outcomes = [outcome for item in trace.state.rounds for outcome in item["joint_outcomes"]]
        denominator = max(len(outcomes), 1)
        return {
            f"p_{outcome.lower()}": sum(item == outcome for item in outcomes) / denominator
            for outcome in ("CC", "CD", "DC", "DD")
        }

    @vf.metric
    async def mean_agent_payoff(self, trace: vf.Trace) -> float:
        payoffs = [float(item["focal_payoff"]) for item in trace.state.rounds]
        return sum(payoffs) / max(len(payoffs), 1)

    @vf.metric
    async def mean_reward_components(self, trace: vf.Trace) -> dict[str, float]:
        rounds = trace.state.rounds
        if not rounds:
            return {"mean_payoff_reward": 0.0, "mean_hkb_reward": 0.0, "mean_cfe_reward": 0.0}

        def mean_component(name: str) -> float:
            values = [
                float(item["reward"][name]) for item in rounds if item["reward"][name] is not None
            ]
            return sum(values) / len(values) if values else 0.0

        return {
            "mean_payoff_reward": mean_component("payoff"),
            "mean_hkb_reward": mean_component("hkb"),
            "mean_cfe_reward": mean_component("calibration"),
        }


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
    mode: Literal["dyadic", "naturalistic", "group"] = "dyadic"
    policy_split: Literal["training", "heldout", "diagnostic"] = "training"
    replacement_policies: list[str] = Field(default_factory=list)
    partner_switch_round: int | None = None
    switch_to_policy: str | None = None
    interleaved_policies: list[str] = Field(default_factory=list)
    perturbation_round: int | None = None
    perturbation_actor: Literal["focal", "partner"] | None = None
    group_size: int = Field(default=4, ge=4, le=5)
    reputation_length: int = Field(default=4, ge=1)
    policy_arm: Literal["Base", "A", "B", "C", "D", "E"] | None = None
    sampling_temperature: float | None = Field(default=None, ge=0)
    sampling_top_p: float | None = Field(default=None, gt=0, le=1)
    sampling_enable_thinking: bool | None = None
    seed_role: Literal["engineering", "training", "validation", "test"] = "engineering"
    training_seed: int | None = None
    evaluation_seed: int | None = None
    checkpoint_training_seed: int | None = None
    b_values: list[float] = Field(default_factory=list)
    w_values: list[float] = Field(default_factory=list)
    q_values: list[float] = Field(default_factory=list)
    partner_policies: list[str] = Field(default_factory=list)
    episodes_per_cell: int = Field(default=1, ge=1)
    task: DonorTaskConfig = DonorTaskConfig()

    @model_validator(mode="after")
    def validate_seed_partition(self) -> DonorTasksetConfig:
        arm = self.policy_arm or self.task.model
        if self.seed_role == "training":
            if arm not in TRAINING_SEEDS_BY_MODEL:
                raise ValueError("Base has no training-seed registry")
            allowed = TRAINING_SEEDS_BY_MODEL[arm]
            if self.training_seed not in allowed or self.evaluation_seed is not None:
                raise ValueError(f"training role for Model {arm} requires its registered seed")
            if self.policy_split != "training":
                raise ValueError("training role requires the training policy pool")
            if self.checkpoint_training_seed not in (None, self.training_seed):
                raise ValueError("training checkpoint seed must match the run training seed")
        elif self.seed_role == "validation":
            if self.evaluation_seed not in VALIDATION_SEEDS or self.training_seed is not None:
                raise ValueError("validation role requires a registered validation seed")
            if self.policy_split == "training":
                raise ValueError("validation role requires a held-out policy pool")
            if arm == "Base" and self.checkpoint_training_seed is not None:
                raise ValueError("Base evaluation cannot name a checkpoint training seed")
            if arm != "Base" and self.checkpoint_training_seed not in TRAINING_SEEDS_BY_MODEL[arm]:
                raise ValueError("validation role requires checkpoint training-seed provenance")
        elif self.seed_role == "test":
            if self.evaluation_seed not in TEST_SEEDS or self.training_seed is not None:
                raise ValueError("test role requires a registered test seed")
            if self.policy_split == "training":
                raise ValueError("test role requires a held-out policy pool")
            if arm == "Base" and self.checkpoint_training_seed is not None:
                raise ValueError("Base evaluation cannot name a checkpoint training seed")
            if arm != "Base" and self.checkpoint_training_seed not in TRAINING_SEEDS_BY_MODEL[arm]:
                raise ValueError("test role requires checkpoint training-seed provenance")
        return self


class DonorTaskset(vf.Taskset[DonorTask, DonorTasksetConfig]):
    def _episode(
        self,
        *,
        index: int,
        b: float,
        w: float,
        q: float,
        partner_policy: str,
        replicate: int,
    ) -> EpisodeConfig:
        return EpisodeConfig(
            game_id=(
                f"{self.config.mode}_b{b:g}_w{w:.2f}_q{q:.2f}_{partner_policy}_r{replicate:02d}"
            ),
            b=b,
            c=self.config.c,
            w=w,
            q=q,
            horizon_min=self.config.horizon_min,
            horizon_max=self.config.horizon_max,
            partner_policy=partner_policy,
            noise_rate=self.config.noise_rate,
            mode=self.config.mode,
            policy_split=self.config.policy_split,
            replacement_policies=tuple(self.config.replacement_policies),
            partner_switch_round=self.config.partner_switch_round,
            switch_to_policy=self.config.switch_to_policy,
            interleaved_policies=tuple(self.config.interleaved_policies),
            perturbation_round=self.config.perturbation_round,
            perturbation_actor=self.config.perturbation_actor,
            group_size=self.config.group_size,
            reputation_length=self.config.reputation_length,
            naturalistic_label_flip=(index % 2 == 1)
            if self.config.mode == "naturalistic"
            else None,
            seed=self.config.seed + index,
        )

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
        tasks: list[DonorTask] = []
        for index, (b, w, q, partner_policy, replicate) in enumerate(cells):
            episode = self._episode(
                index=index,
                b=b,
                w=w,
                q=q,
                partner_policy=partner_policy,
                replicate=replicate,
            )
            world = make_world(episode, self.config.task.model)
            tasks.append(
                DonorTask(
                    DonorData(
                        idx=index,
                        name=episode.game_id,
                        prompt=[{"role": "user", "content": world.render_prompt()}],
                        system_prompt=system_prompt(episode, world.labels),
                        episode={
                            field: getattr(episode, field) for field in episode.__dataclass_fields__
                        },
                        reward_model=self.config.task.model,
                        policy_arm=self.config.policy_arm or self.config.task.model,
                        seed_metadata={
                            "role": self.config.seed_role,
                            "training_seed": self.config.training_seed,
                            "evaluation_seed": self.config.evaluation_seed,
                            "checkpoint_training_seed": (
                                self.config.checkpoint_training_seed
                                if self.config.checkpoint_training_seed is not None
                                else self.config.training_seed
                            ),
                            "episode_seed": episode.seed,
                        },
                        sampling_metadata={
                            "temperature": self.config.sampling_temperature,
                            "top_p": self.config.sampling_top_p,
                            "enable_thinking": self.config.sampling_enable_thinking,
                            "requested_seed": episode.seed,
                        },
                    ),
                    self.config.task,
                )
            )
        return tasks


if __name__ == "__main__":
    DonorUser.run()
