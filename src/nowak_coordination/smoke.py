"""CPU-only deterministic smoke evaluation for the game and reward pipeline."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from .game import Action, DonorGame, EpisodeConfig
from .partners import make_partner
from .rewards import model_reward


def run_smoke(num_episodes: int = 20, seed: int = 0) -> dict[str, object]:
    if num_episodes < 1:
        raise ValueError("num_episodes must be positive")
    partner_names = [
        "always_cooperate",
        "always_defect",
        "tit_for_tat",
        "random_p",
        "noisy_tit_for_tat",
    ]
    model_names = ["A", "B", "C", "D"]
    totals: dict[str, list[float]] = defaultdict(list)
    cooperation: dict[str, list[float]] = defaultdict(list)
    outcome_counts = {"CC": 0, "CD": 0, "DC": 0, "DD": 0}

    for episode_index in range(num_episodes):
        partner_name = partner_names[episode_index % len(partner_names)]
        model_name = model_names[episode_index % len(model_names)]
        config = EpisodeConfig(
            game_id=f"smoke_{episode_index:04d}",
            b=4,
            c=1,
            w=0.7,
            q=0.8,
            horizon_min=6,
            horizon_max=6,
            partner_policy=partner_name,
            noise_rate=0.05,
            seed=seed + episode_index,
        )
        game = DonorGame(config)
        partner_kwargs = (
            {"noise_rate": config.noise_rate} if partner_name == "noisy_tit_for_tat" else {}
        )
        partner = make_partner(partner_name, **partner_kwargs)
        episode_rewards = []

        while not game.done:
            agent_action = game.history[-1].partner_action if game.history else Action.COOPERATE
            partner_action = partner.act(game.history, game.rng)
            game.step(agent_action, partner_action)
            realized = sum(
                result.partner_action is Action.COOPERATE for result in game.history
            ) / len(game.history)
            forecast = realized
            episode_rewards.append(
                model_reward(
                    model_name,
                    game.history,
                    b=config.b,
                    c=config.c,
                    q=config.q,
                    forecast=forecast,
                    realized_group_cooperation=realized,
                ).total
            )

        totals[model_name].append(sum(episode_rewards) / len(episode_rewards))
        cooperation[partner_name].append(
            sum(result.agent_action is Action.COOPERATE for result in game.history)
            / len(game.history)
        )
        for outcome, count in game.outcome_counts().items():
            outcome_counts[outcome] += count

    return {
        "episodes": num_episodes,
        "seed": seed,
        "mean_reward_by_model": {
            name: sum(values) / len(values) for name, values in sorted(totals.items())
        },
        "agent_cooperation_by_partner": {
            name: sum(values) / len(values) for name, values in sorted(cooperation.items())
        },
        "outcome_counts": outcome_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.episodes, args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
