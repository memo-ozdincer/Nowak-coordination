"""Evolutionary-pressure cooperation RL experiments."""

from .game import Action, DonorGame, EpisodeConfig, RoundResult, donor_payoffs
from .mechanics import DyadicWorld, GroupWorld, make_world
from .rewards import RewardBreakdown, ShuffledHKBReference, model_reward

__version__ = "0.1.0"

__all__ = [
    "Action",
    "DonorGame",
    "DyadicWorld",
    "EpisodeConfig",
    "GroupWorld",
    "RewardBreakdown",
    "RoundResult",
    "ShuffledHKBReference",
    "donor_payoffs",
    "make_world",
    "model_reward",
]
