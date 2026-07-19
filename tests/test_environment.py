import asyncio

from nowak_coordination.environment import (
    DonorState,
    DonorTaskset,
    DonorTasksetConfig,
    DonorUser,
    SYSTEM_PROMPT,
    round_prompt,
)
from nowak_coordination.game import EpisodeConfig


def test_taskset_loads_seeded_typed_tasks():
    taskset = DonorTaskset(
        DonorTasksetConfig(
            id="local/donor",
            num_tasks=3,
            seed=10,
            horizon_min=2,
            horizon_max=2,
        )
    )
    tasks = taskset.load()
    assert len(tasks) == 3
    assert tasks[0].data.system_prompt == SYSTEM_PROMPT
    assert tasks[0].data.prompt is not None
    assert tasks[0].data.prompt[0].role == "user"
    assert tasks[0].data.prompt[0].content == round_prompt(
        EpisodeConfig(**tasks[0].data.episode), 1, []
    )
    assert tasks[0].data.episode["seed"] == 10
    assert tasks[2].data.episode["seed"] == 12


def test_taskset_expands_parameter_grid():
    tasks = DonorTaskset(
        DonorTasksetConfig(
            id="local/donor",
            b_values=[2, 8],
            w_values=[0.1, 0.9],
            q_values=[0.2],
            partner_policies=["always_defect", "tit_for_tat"],
            episodes_per_cell=2,
        )
    ).load()
    assert len(tasks) == 16
    assert {task.data.episode["b"] for task in tasks} == {2, 8}
    assert {task.data.episode["partner_policy"] for task in tasks} == {
        "always_defect",
        "tit_for_tat",
    }
    assert len({task.data.name for task in tasks}) == 16


def test_round_prompt_contains_parameters_and_history():
    config = EpisodeConfig("x", b=4, c=1, w=0.7, q=0.8)
    prompt = round_prompt(config, 1, [])
    assert "b = 4" in prompt
    assert "q = 0.80" in prompt
    assert "Current round: 1" in prompt


def test_user_simulator_advances_and_terminates_seeded_episode():
    task = DonorTaskset(
        DonorTasksetConfig(
            id="local/donor",
            num_tasks=1,
            horizon_min=2,
            horizon_max=2,
            partner_policy="tit_for_tat",
        )
    ).load()[0]
    user = DonorUser(task.config.user)
    user._inert_state = DonorState()
    asyncio.run(user.setup_task(task.data))

    opening = asyncio.run(user.respond(""))
    assert opening == [{"role": "user", "content": round_prompt(EpisodeConfig(**task.data.episode), 1, [])}]

    first = asyncio.run(user.respond("ACTION: COOPERATE\nFORECAST_GROUP_COOP: 0.75"))
    assert not user.state.game_over
    assert "Current round: 2" in first[0]["content"]
    assert user.state.agent_payoffs == [3.0]

    asyncio.run(user.respond("ACTION: DEFECT\nFORECAST_GROUP_COOP: 0.50"))
    assert user.state.game_over
    assert user.state.partner_actions == [1, 1]


def test_user_simulator_rejects_extra_text():
    task = DonorTaskset(DonorTasksetConfig(id="local/donor", num_tasks=1)).load()[0]
    user = DonorUser(task.config.user)
    user._inert_state = DonorState()
    asyncio.run(user.setup_task(task.data))
    asyncio.run(user.respond("I think this is best.\nACTION: COOPERATE\nFORECAST_GROUP_COOP: 0.50"))
    assert user.state.invalid_output
    assert user.state.game_over
