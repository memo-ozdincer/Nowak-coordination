import pytest

from nowak_coordination.game import (
    Action,
    DonorGame,
    EpisodeConfig,
    cooperation_rate,
    donor_payoffs,
)


def config(**overrides: object) -> EpisodeConfig:
    values = {
        "game_id": "test",
        "b": 4,
        "c": 1,
        "w": 0.7,
        "q": 0.8,
        "horizon_min": 2,
        "horizon_max": 2,
        "seed": 7,
    }
    values.update(overrides)
    return EpisodeConfig(**values)


@pytest.mark.parametrize(
    ("agent", "partner", "expected"),
    [
        (Action.COOPERATE, Action.COOPERATE, (3.0, 3.0)),
        (Action.COOPERATE, Action.DEFECT, (-1.0, 4.0)),
        (Action.DEFECT, Action.COOPERATE, (4.0, -1.0)),
        (Action.DEFECT, Action.DEFECT, (0.0, 0.0)),
    ],
)
def test_donor_payoff_matrix(agent: Action, partner: Action, expected: tuple[float, float]):
    assert donor_payoffs(agent, partner, b=4, c=1) == expected


def test_seeded_horizon_and_terminal_guard():
    game = DonorGame(config())
    assert game.horizon == 2
    game.step(Action.COOPERATE, Action.DEFECT)
    game.step(Action.DEFECT, Action.COOPERATE)
    assert game.done
    assert game.outcome_counts() == {"CC": 0, "CD": 1, "DC": 1, "DD": 0}
    with pytest.raises(RuntimeError):
        game.step(Action.COOPERATE, Action.COOPERATE)


def test_action_parser_is_strict_but_accepts_protocol_prefix():
    assert Action.parse("ACTION: COOPERATE") is Action.COOPERATE
    assert Action.parse("defect") is Action.DEFECT
    with pytest.raises(ValueError):
        Action.parse("probably cooperate")


def test_episode_validation():
    with pytest.raises(ValueError, match="b > c"):
        config(b=1, c=1)
    with pytest.raises(ValueError, match="q"):
        config(q=1.1)
    with pytest.raises(ValueError, match="horizon"):
        config(horizon_min=3, horizon_max=2)
    with pytest.raises(ValueError, match="every possible horizon"):
        config(
            horizon_min=3,
            horizon_max=8,
            partner_switch_round=6,
            switch_to_policy="always_defect",
        )
    with pytest.raises(ValueError, match="every possible horizon"):
        config(
            horizon_min=3,
            horizon_max=8,
            perturbation_round=5,
            perturbation_actor="focal",
        )


def test_empty_and_nonempty_cooperation_rate():
    assert cooperation_rate([]) == 0
    assert cooperation_rate([Action.COOPERATE, Action.DEFECT, Action.COOPERATE]) == pytest.approx(
        2 / 3
    )
