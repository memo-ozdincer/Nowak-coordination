from dataclasses import replace

import pytest

from nowak_coordination.game import Action, EpisodeConfig
from nowak_coordination.mechanics import (
    DyadicWorld,
    GroupWorld,
    parse_response,
    system_prompt,
)


def config(**overrides: object) -> EpisodeConfig:
    values: dict[str, object] = {
        "game_id": "semantic-test",
        "b": 4,
        "c": 1,
        "w": 0.7,
        "q": 0.8,
        "horizon_min": 6,
        "horizon_max": 6,
        "partner_policy": "tit_for_tat",
        "policy_split": "training",
        "seed": 1234,
    }
    values.update(overrides)
    return EpisodeConfig(**values)


def rollout(world: DyadicWorld | GroupWorld, actions: list[Action]) -> list[dict]:
    for action in actions:
        world.step(action, 0.5)
    return world.events


def test_identical_seed_and_config_produce_identical_transitions():
    episode = config(replacement_policies=("always_cooperate", "always_defect", "tit_for_tat"))
    actions = [Action.COOPERATE, Action.DEFECT] * 3
    first = DyadicWorld(episode, "B")
    second = DyadicWorld(episode, "B")
    assert first.trace_header() == second.trace_header()
    assert first.render_prompt() == second.render_prompt()
    assert rollout(first, actions) == rollout(second, actions)
    assert [item.to_dict() for item in first.observations] == [
        item.to_dict() for item in second.observations
    ]


def test_w_is_causal_retention_probability_with_preregistered_tolerance():
    def observed_retention(w: float) -> float:
        retained: list[bool] = []
        for seed in range(300):
            world = DyadicWorld(
                config(seed=seed, w=w, horizon_min=8, horizon_max=8),
                "A",
            )
            rollout(world, [Action.COOPERATE] * 8)
            retained.extend(
                event["retained_for_next"]
                for event in world.events[:-1]
                if event["retained_for_next"] is not None
            )
        return sum(retained) / len(retained)

    assert observed_retention(0.0) == 0.0
    assert observed_retention(1.0) == 1.0
    assert observed_retention(0.2) == pytest.approx(0.2, abs=0.03)
    assert observed_retention(0.8) == pytest.approx(0.8, abs=0.03)


def test_q_changes_only_visibility_not_latent_records_or_transitions():
    hidden = DyadicWorld(config(q=0.0), "A")
    visible = DyadicWorld(config(q=1.0), "A")
    assert hidden.current.reputation == visible.current.reputation
    assert hidden.current_observation.reputation_visible == (False,)
    assert visible.current_observation.reputation_visible == (True,)
    assert "No reputation information was observed." in hidden.render_prompt()
    assert "Observed reputation:" not in hidden.render_prompt()

    actions = [Action.COOPERATE, Action.DEFECT] * 3
    rollout(hidden, actions)
    rollout(visible, actions)
    keys = (
        "partner_ids",
        "focal_executed_action",
        "partner_executed_actions",
        "focal_payoff",
        "retention_draw",
        "retained_for_next",
        "next_partner_id",
    )
    assert [{key: event[key] for key in keys} for event in hidden.events] == [
        {key: event[key] for key in keys} for event in visible.events
    ]
    assert [observation.reputations for observation in hidden.observations] == [
        observation.reputations for observation in visible.observations
    ]


def test_forced_switch_and_interleaving_keep_opaque_separate_histories():
    switched = DyadicWorld(
        config(
            policy_split="diagnostic",
            partner_policy="tit_for_tat",
            w=1.0,
            partner_switch_round=3,
            switch_to_policy="always_defect",
        ),
        "A",
    )
    rollout(switched, [Action.COOPERATE] * 6)
    assert switched.events[1]["transition_to_next"] == "forced_switch"
    assert switched.events[0]["partner_ids"] == switched.events[1]["partner_ids"]
    assert switched.events[2]["partner_ids"] != switched.events[1]["partner_ids"]
    assert "always_defect" not in switched.render_prompt()

    interleaved = DyadicWorld(
        config(
            policy_split="diagnostic",
            interleaved_policies=("tit_for_tat", "always_defect"),
        ),
        "A",
    )
    rollout(interleaved, [Action.COOPERATE] * 6)
    ids = [event["partner_ids"][0] for event in interleaved.events]
    assert ids[0] == ids[2] == ids[4]
    assert ids[1] == ids[3] == ids[5]
    assert ids[0] != ids[1]
    assert [event["partner_history_length_before"] for event in interleaved.events] == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]


@pytest.mark.parametrize(
    ("initial_policy", "next_policy"),
    [("tit_for_tat", "always_defect"), ("always_defect", "tit_for_tat")],
)
def test_both_registered_forced_switch_directions(initial_policy: str, next_policy: str):
    world = DyadicWorld(
        config(
            policy_split="diagnostic",
            partner_policy=initial_policy,
            w=1.0,
            partner_switch_round=3,
            switch_to_policy=next_policy,
        ),
        "A",
    )
    rollout(world, [Action.COOPERATE] * 6)
    assert [event["partner_policy"] for event in world.events[:3]] == [
        initial_policy,
        initial_policy,
        next_policy,
    ]
    assert next_policy not in world.events[2]["rendered_observation"]


@pytest.mark.parametrize("actor", ["focal", "partner"])
def test_forced_noise_changes_executed_but_not_intended_action(actor: str):
    world = DyadicWorld(
        config(
            policy_split="diagnostic",
            partner_policy="always_cooperate",
            perturbation_round=1,
            perturbation_actor=actor,
        ),
        "A",
    )
    event = world.step(Action.COOPERATE, 0.5).event
    assert event["perturbation"] == {"applied": True, "actor": actor}
    if actor == "focal":
        assert event["focal_intended_action"] == "C"
        assert event["focal_executed_action"] == "D"
    else:
        assert event["partner_intended_actions"] == ["C"]
        assert event["partner_executed_actions"] == ["D"]


def test_stochastic_partner_noise_is_logged_as_execution_not_intention():
    world = DyadicWorld(
        config(partner_policy="noisy_tit_for_tat", noise_rate=1.0),
        "A",
    )
    event = world.step(Action.COOPERATE, 0.5).event
    assert event["partner_intended_actions"] == ["C"]
    assert event["partner_executed_actions"] == ["D"]
    assert event["perturbation"]["applied"] is False


@pytest.mark.parametrize("group_size", [4, 5])
def test_group_forecast_target_is_current_unseen_executed_group_outcome(group_size: int):
    episode = config(
        mode="group",
        policy_split="diagnostic",
        partner_policy="always_cooperate",
        group_size=group_size,
        q=0.0,
    )
    world = GroupWorld(episode, "D")
    opening = world.render_prompt()
    target = (group_size - 1) / group_size
    assert f"group cooperation {target:.2f}" not in opening
    event = world.step(Action.DEFECT, target).event
    assert event["partner_executed_actions"] == ["C"] * (group_size - 1)
    assert event["forecast_target"] == target
    assert event["focal_payoff"] == 4.0
    assert event["reward"]["calibration"] == pytest.approx(0.0)
    assert len(event["partner_ids"]) == group_size - 1


def test_naturalistic_labels_hide_literal_action_names_and_are_verifier_mapped():
    episode = config(mode="naturalistic")
    world = DyadicWorld(episode, "A")
    prompt = world.render_prompt()
    instructions = system_prompt(episode, world.labels)
    combined = f"{prompt}\n{instructions}".lower()
    assert " cooperate " not in f" {combined} "
    assert " defect " not in f" {combined} "
    cooperate_label = world.labels[Action.COOPERATE]
    action, forecast = parse_response(
        f"ACTION: {cooperate_label}\nFORECAST_GROUP_COOP: 0.25",
        episode,
        world.labels,
    )
    assert action is Action.COOPERATE
    assert forecast == 0.25
    with pytest.raises(ValueError, match="invalid output"):
        parse_response(
            "ACTION: COOPERATE\nFORECAST_GROUP_COOP: 0.25",
            episode,
            world.labels,
        )


def test_model_e_source_is_never_focal_episode_or_partner():
    world = DyadicWorld(config(), "E")
    event = world.step(Action.COOPERATE, 0.5).event
    assert event["hkb_source"]["episode_id"] != event["episode_id"]
    assert event["hkb_source"]["partner_id"] not in event["partner_ids"]
    assert event["reward"]["hkb"] is not None
    one_round_reference = world._shuffled_for_round(1)
    four_round_reference = world._shuffled_for_round(4)
    assert len(one_round_reference.history) == 1
    assert len(four_round_reference.history) == 4


def test_hidden_reputation_render_never_contains_latent_value():
    world = DyadicWorld(config(q=0.0), "A")
    latent = f"{world.current.reputation.cooperation_fraction:.2f}"
    prompt = world.render_prompt()
    assert "Observed reputation" not in prompt
    assert f"reputation: cooperation fraction {latent}" not in prompt


def test_config_counterfactual_does_not_share_mutable_histories():
    base = config()
    first = DyadicWorld(base, "A")
    second = DyadicWorld(replace(base, game_id="other"), "A")
    first.step(Action.DEFECT, 0.5)
    assert len(first.current.history) == 1
    assert len(second.current.history) == 0
