"""CPU-only deterministic smoke evaluation for the scientific mechanics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from .game import Action, EpisodeConfig
from .mechanics import DyadicWorld, make_world


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
    model_names = ["A", "B", "C", "D", "E"]
    totals: dict[str, list[float]] = defaultdict(list)
    cooperation: dict[str, list[float]] = defaultdict(list)
    outcome_counts = {"CC": 0, "CD": 0, "DC": 0, "DD": 0}

    for episode_index in range(num_episodes):
        partner_name = partner_names[episode_index % len(partner_names)]
        model_name = model_names[episode_index % len(model_names)]
        mode = "group" if model_name in {"C", "D"} else "dyadic"
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
            mode=mode,
            seed=seed + episode_index,
        )
        world = make_world(config, model_name)
        episode_rewards = []

        while not world.done:
            if isinstance(world, DyadicWorld) and world.current.history:
                agent_action = world.current.history[-1].partner_action
            else:
                agent_action = Action.COOPERATE
            event = world.step(agent_action, 0.5).event
            episode_rewards.append(float(event["reward"]["total"]))
            for outcome in event["joint_outcomes"]:
                outcome_counts[outcome] += 1

        totals[model_name].append(sum(episode_rewards) / len(episode_rewards))
        cooperation[partner_name].append(
            sum(event["focal_executed_action"] == "C" for event in world.events) / len(world.events)
        )

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
