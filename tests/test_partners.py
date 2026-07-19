import random

import pytest

from nowak_coordination.game import Action, RoundResult
from nowak_coordination.partners import (
    DelayedTitForTat,
    ForgivingGrudger,
    GenerousTitForTat,
    GrimTrigger,
    NoisyTitForTat,
    Opportunist,
    TitForTat,
    WinStayLoseShift,
    make_partner,
)


def result(index: int, agent: Action, partner: Action) -> RoundResult:
    return RoundResult(index, agent, partner, 0, 0)


def test_tit_for_tat_copies_agent():
    policy = TitForTat()
    assert policy.act([], random.Random(0)) is Action.COOPERATE
    history = [result(1, Action.DEFECT, Action.COOPERATE)]
    assert policy.act(history, random.Random(0)) is Action.DEFECT


def test_delayed_tit_for_tat_uses_configured_lag():
    policy = DelayedTitForTat(delay=2)
    history = [
        result(1, Action.DEFECT, Action.COOPERATE),
        result(2, Action.COOPERATE, Action.DEFECT),
    ]
    assert policy.act(history, random.Random(0)) is Action.DEFECT


def test_grim_and_forgiving_grudgers_differ():
    history = [
        result(1, Action.DEFECT, Action.COOPERATE),
        result(2, Action.COOPERATE, Action.DEFECT),
        result(3, Action.COOPERATE, Action.DEFECT),
    ]
    assert GrimTrigger().act(history, random.Random(0)) is Action.DEFECT
    assert ForgivingGrudger(punishment_rounds=2).act(history, random.Random(0)) is Action.COOPERATE


def test_generous_tft_extreme_probabilities():
    history = [result(1, Action.DEFECT, Action.COOPERATE)]
    assert (
        GenerousTitForTat(forgiveness_probability=1).act(history, random.Random(0))
        is Action.COOPERATE
    )
    assert (
        GenerousTitForTat(forgiveness_probability=0).act(history, random.Random(0)) is Action.DEFECT
    )


def test_noisy_tft_extreme_noise():
    history = [result(1, Action.COOPERATE, Action.COOPERATE)]
    assert NoisyTitForTat(noise_rate=1).act(history, random.Random(0)) is Action.DEFECT


@pytest.mark.parametrize(
    ("agent", "partner", "expected"),
    [
        (Action.COOPERATE, Action.COOPERATE, Action.COOPERATE),
        (Action.DEFECT, Action.DEFECT, Action.DEFECT),
        (Action.COOPERATE, Action.DEFECT, Action.COOPERATE),
        (Action.DEFECT, Action.COOPERATE, Action.DEFECT),
    ],
)
def test_win_stay_lose_shift_from_partner_perspective(
    agent: Action, partner: Action, expected: Action
):
    history = [result(1, agent, partner)]
    assert WinStayLoseShift().act(history, random.Random(0)) is expected


def test_opportunist_exploits_persistent_cooperation_then_reciprocates():
    policy = Opportunist(exploit_after=2)
    cooperative = [
        result(1, Action.COOPERATE, Action.COOPERATE),
        result(2, Action.COOPERATE, Action.COOPERATE),
    ]
    assert policy.act(cooperative, random.Random(0)) is Action.DEFECT
    mixed = cooperative + [result(3, Action.DEFECT, Action.DEFECT)]
    assert policy.act(mixed, random.Random(0)) is Action.DEFECT


def test_factory_defaults_and_unknown_name():
    assert isinstance(make_partner("copy_with_noise_10%"), NoisyTitForTat)
    with pytest.raises(ValueError, match="unknown partner"):
        make_partner("not_real")
