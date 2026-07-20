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
from nowak_coordination.seeded_eval import requested_sampler_seed


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


def test_first_state_synchronized_response_restores_trace_provenance():
    task, user = make_user(
        policy_arm="Base",
        sampling_seed=2901,
        sampling_temperature=0.7,
        sampling_top_p=1.0,
        sampling_enable_thinking=False,
    )
    # Verifiers setup_task runs before its remote state channel is attached.
    user._inert_state = DonorState()
    asyncio.run(user.respond("ACTION: COOPERATE\nFORECAST_GROUP_COOP: 0.50"))
    header = user.state.trace_header
    assert header["policy_arm"] == "Base"
    assert header["seed_metadata"]["episode_seed"] == task.data.episode["seed"]
    assert header["sampling_metadata"] == {
        "temperature": 0.7,
        "top_p": 1.0,
        "enable_thinking": False,
        "requested_seed": 2901,
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
    with pytest.raises(ValueError, match="Model B"):
        DonorTasksetConfig(
            seed_role="training",
            training_seed=1101,
            policy_split="training",
            task=DonorTaskConfig(model="B"),
        )
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


def test_gate4_registry_is_exact_balanced_and_seeded():
    all_tasks = []
    for evaluation_seed in range(2101, 2106):
        tasks = DonorTaskset(
            DonorTasksetConfig(
                id="local/donor",
                registry="gate4_base_characterization",
                num_tasks=100,
                seed=4_210_100 + (evaluation_seed - 2101) * 100,
                horizon_min=10,
                horizon_max=10,
                policy_split="heldout",
                policy_arm="Base",
                seed_role="validation",
                evaluation_seed=evaluation_seed,
                sampling_seed=evaluation_seed,
                sampling_temperature=0.7,
                sampling_top_p=1.0,
                sampling_enable_thinking=False,
                task=DonorTaskConfig(model="A"),
            )
        ).load()
        assert len(tasks) == 100
        assert {
            (
                task.data.episode["b"],
                task.data.episode["w"],
                task.data.episode["q"],
            )
            for task in tasks
        } == {
            (b, w, q)
            for b in (2.0, 3.0, 5.0, 8.0)
            for w in (0.1, 0.3, 0.5, 0.7, 0.9)
            for q in (0.1, 0.3, 0.5, 0.7, 0.9)
        }
        counts = {
            name: sum(task.data.analysis_targets["scenario"] == name for task in tasks)
            for name in {
                "heldout_forgiving_grudger",
                "heldout_delayed_tft",
                "heldout_probabilistic_defector",
                "heldout_noisy_copy",
                "diagnostic_switch",
                "diagnostic_exploitability",
                "heldout_group_forecast",
            }
        }
        assert counts == {
            "heldout_forgiving_grudger": 15,
            "heldout_delayed_tft": 15,
            "heldout_probabilistic_defector": 15,
            "heldout_noisy_copy": 15,
            "diagnostic_switch": 15,
            "diagnostic_exploitability": 15,
            "heldout_group_forecast": 10,
        }
        for axis in ("w", "q"):
            for value in (0.1, 0.3, 0.5, 0.7, 0.9):
                assert sum(
                    task.data.episode[axis] == value
                    for task in tasks
                    if task.data.analysis_targets["scenario"]
                    == "diagnostic_exploitability"
                ) == 3
        exploitability = [
            task
            for task in tasks
            if task.data.analysis_targets["scenario"] == "diagnostic_exploitability"
        ]
        assert all(
            task.data.analysis_targets["safe_defect_mean_payoff_provenance"]["episode_seed"]
            == task.data.episode["seed"]
            for task in exploitability
        )
        group = [
            task
            for task in tasks
            if task.data.analysis_targets["scenario"] == "heldout_group_forecast"
        ]
        assert {task.data.episode["partner_policy"] for task in group} == {
            "forgiving_grudger",
            "delayed_tit_for_tat",
            "probabilistic_defector",
            "copy_with_noise_10%",
        }
        assert {task.data.episode["group_size"] for task in group} == {4, 5}
        assert {requested_sampler_seed(task) for task in tasks} == {evaluation_seed}
        all_tasks.extend(tasks)
    assert len({task.data.episode["seed"] for task in all_tasks}) == 500
    assert sum(task.data.analysis_targets["suite"] == "forecast" for task in all_tasks) == 50
    assert sum(task.data.analysis_targets["suite"] == "switch" for task in all_tasks) == 75
    assert sum(task.data.analysis_targets["suite"] == "exploitability" for task in all_tasks) == 75
    switch_directions = {
        direction: sum(
            task.data.analysis_targets["switch_direction"] == direction for task in all_tasks
        )
        for direction in {"TFT_to_AD", "AD_to_TFT"}
    }
    assert sorted(switch_directions.values()) == [37, 38]


def test_gate4_registry_rejects_cosmetic_or_wrong_sampler_seed():
    with pytest.raises(ValueError, match="explicit model sampling seed"):
        DonorTasksetConfig(
            seed_role="validation",
            evaluation_seed=2101,
            policy_split="heldout",
            policy_arm="Base",
        )
    with pytest.raises(ValueError, match="must equal"):
        DonorTasksetConfig(
            registry="gate4_base_characterization",
            seed_role="validation",
            evaluation_seed=2101,
            sampling_seed=2102,
            policy_split="heldout",
            policy_arm="Base",
            horizon_min=10,
            horizon_max=10,
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
