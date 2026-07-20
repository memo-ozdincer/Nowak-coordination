import asyncio

import pytest

from nowak_coordination.environment import (
    DonorState,
    DonorTask,
    DonorTaskConfig,
    DonorTaskset,
    DonorTasksetConfig,
    DonorUser,
)


def make_user(**config_overrides: object) -> tuple[DonorTask, DonorUser]:
    values: dict[str, object] = {
        "id": "local/donor",
        "num_tasks": 1,
        "horizon_min": 2,
        "horizon_max": 2,
        "partner_policy": "tit_for_tat",
        "w": 1.0,
        "q": 1.0,
    }
    values.update(config_overrides)
    task = DonorTaskset(DonorTasksetConfig(**values)).load()[0]
    user = DonorUser(task.config.user)
    user._inert_state = DonorState()
    asyncio.run(user.setup_task(task.data))
    return task, user


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
    assert tasks[0].data.prompt is not None
    assert tasks[0].data.prompt[0].role == "user"
    assert "Current partner ID: p-" in tasks[0].data.prompt[0].content
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


def test_user_simulator_logs_complete_scientific_trace():
    task, user = make_user()
    opening = asyncio.run(user.respond(""))
    assert opening == [{"role": "user", "content": user.world.render_prompt()}]
    assert opening[0]["content"] == task.data.prompt[0].content

    first = asyncio.run(user.respond("ACTION: COOPERATE\nFORECAST_GROUP_COOP: 0.75"))
    assert not user.state.game_over
    assert "Current round: 2" in first[0]["content"]
    assert user.state.agent_payoffs == [3.0]
    assert user.state.rounds[0]["focal_intended_action"] == "C"
    assert user.state.rounds[0]["reward"]["payoff"] == pytest.approx(0.8)

    asyncio.run(user.respond("ACTION: DEFECT\nFORECAST_GROUP_COOP: 0.50"))
    assert user.state.game_over
    assert user.state.partner_actions == [1, 1]
    assert user.state.terminal_reason == "horizon"
    assert user.state.terminal_event == {
        "episode_id": task.data.episode["game_id"],
        "terminal_reason": "horizon",
        "rounds_completed": 2,
        "expected_horizon": 2,
        "complete": True,
        "label_mapping": {"COOPERATE": "C", "DEFECT": "D"},
    }


def test_invalid_format_terminates_without_task_reward_and_keeps_terminal_trace():
    task, user = make_user()
    asyncio.run(user.respond(""))
    asyncio.run(user.respond("I think this is best.\nACTION: COOPERATE\nFORECAST_GROUP_COOP: 0.50"))
    assert user.state.invalid_output
    assert user.state.game_over
    assert user.state.rounds == []
    assert user.state.terminal_reason == "invalid_format"
    assert user.state.terminal_event is not None
    assert user.state.terminal_event["complete"] is True

    trace = type("Trace", (), {"state": user.state, "info": {}})()
    assert asyncio.run(task.episode_reward(trace)) == 0.0
    assert trace.info["coordination_trace"]["terminal_reason"] == "invalid_format"
    assert asyncio.run(task.trace_complete(trace)) == 1.0


def test_taskset_rejects_policy_split_leakage():
    with pytest.raises(ValueError, match="not registered"):
        DonorTaskset(
            DonorTasksetConfig(
                id="local/donor",
                policy_split="training",
                partner_policy="forgiving_grudger",
            )
        ).load()
    heldout = DonorTaskset(
        DonorTasksetConfig(
            id="local/donor",
            policy_split="heldout",
            partner_policy="forgiving_grudger",
        )
    ).load()
    assert heldout[0].data.episode["policy_split"] == "heldout"


def test_taskset_rejects_seed_partition_leakage():
    with pytest.raises(ValueError, match="registered validation seed"):
        DonorTasksetConfig(
            seed_role="validation",
            evaluation_seed=3101,
            policy_split="heldout",
        )
    with pytest.raises(ValueError, match="held-out policy pool"):
        DonorTasksetConfig(
            seed_role="test",
            evaluation_seed=3101,
            policy_split="training",
        )


def test_group_model_d_runs_through_verifiers_adapter_with_real_target():
    task, user = make_user(
        mode="group",
        policy_split="diagnostic",
        partner_policy="always_cooperate",
        task=DonorTaskConfig(model="D"),
    )
    assert "Group size: 4" in asyncio.run(user.respond(""))[0]["content"]
    asyncio.run(user.respond("ACTION: DEFECT\nFORECAST_GROUP_COOP: 0.75"))
    event = user.state.rounds[0]
    assert event["forecast_target"] == 0.75
    assert event["reward"]["calibration"] == pytest.approx(0.0)
    trace = type("Trace", (), {"state": user.state, "info": {}})()
    asyncio.run(task.episode_reward(trace))
    assert trace.info["coordination_trace"]["rounds"][0]["forecast_target"] == 0.75
    assert asyncio.run(task.mean_reward_components(trace))["mean_cfe_reward"] == 0.0


def test_naturalistic_adapter_accepts_only_its_seeded_neutral_labels():
    task, user = make_user(mode="naturalistic")
    opening = asyncio.run(user.respond(""))[0]["content"]
    assert "ACTION: COOPERATE" not in task.data.system_prompt
    assert "Current partner ID" in opening
    label = user.world.labels[next(action for action in user.world.labels if int(action) == 1)]
    asyncio.run(user.respond(f"ACTION: {label}\nFORECAST_GROUP_COOP: 0.50"))
    assert not user.state.invalid_output
    assert user.state.rounds[0]["focal_intended_action"] == "C"


def test_naturalistic_taskset_balances_the_verifier_mapping():
    tasks = DonorTaskset(
        DonorTasksetConfig(id="local/donor", mode="naturalistic", num_tasks=6)
    ).load()
    flips = [task.data.episode["naturalistic_label_flip"] for task in tasks]
    assert flips == [False, True, False, True, False, True]
