#!/usr/bin/env python3
"""Generate a small, transparent known-answer trace fixture.

This is deliberately hand-constructed rather than sampled from the environment:
the analyzer must be checked against data whose outcomes are obvious without
trusting the implementation that produced the real traces.
"""

from __future__ import annotations

import json
from pathlib import Path


MODELS = {
    "A": "CCDCCDCCDC",
    "B": "CCCCDCCCCC",
    "D": "CDCDCDCDCD",
    "E": "DDCCDDCCDD",
}
PARTNERS = {
    1: "CCCCDCCCCC",
    2: "CCCCDCDDDD",
    3: "DDDDDDCCCC",
}
TRAINING_SEEDS = {
    "A": (1101, 1102, 1103),
    "B": (1201, 1202, 1203),
    "D": (1401, 1402, 1403),
    "E": (1501, 1502, 1503),
}


def _record(model: str, training_seed: int, ordinal: int) -> dict:
    focal = MODELS[model]
    seed_index = training_seed % 100
    partner = PARTNERS[seed_index]
    episode_id = f"synthetic-{model}-{training_seed}"
    direction = "tft_to_ad" if seed_index != 3 else "ad_to_tft"
    policy_before, policy_after = (
        ("tit_for_tat", "always_defect")
        if direction == "tft_to_ad"
        else ("always_defect", "tit_for_tat")
    )
    b = {1: 2.0, 2: 3.0, 3: 5.0}[seed_index]
    q = {1: 0.1, 2: 0.5, 3: 0.9}[seed_index]
    rounds = []
    for index, (focal_action, partner_action) in enumerate(zip(focal, partner, strict=True), 1):
        focal_payoff = (b if partner_action == "C" else 0.0) - (1.0 if focal_action == "C" else 0.0)
        outcome = focal_action + partner_action
        policy = policy_before if index <= 6 else policy_after
        rounds.append(
            {
                "episode_id": episode_id,
                "round_index": index,
                "mode": "pairwise",
                "partner_ids": ["p0"],
                "partner_policy": policy,
                "partner_adaptive": "tit_for_tat" in policy,
                "partner_history_length_before": index - 1,
                "observation": {"round_index": index},
                "rendered_observation": f"Round {index}",
                "focal_intended_action": focal_action,
                "focal_executed_action": focal_action,
                "partner_intended_actions": [partner_action],
                "partner_executed_actions": [partner_action],
                "focal_payoff": focal_payoff,
                "partner_payoffs": [0.0],
                "joint_outcomes": [outcome],
                "perturbation": {
                    "applied": index == 5,
                    "actor": "partner" if index == 5 else None,
                },
                "forecast": {"A": 0.5, "B": 0.8, "D": 0.4, "E": 0.2}[model],
                "forecast_target": float(partner_action == "C"),
                "reward": {
                    "payoff": focal_payoff,
                    "hkb": None,
                    "calibration": None,
                    "total": focal_payoff,
                },
                "hkb_source": None,
                "reputation_observation": None,
                "transition_to_next": "forced_switch" if index == 6 else None,
            }
        )
    mean_payoff = sum(item["focal_payoff"] for item in rounds) / len(rounds)
    coordination_success = sum(item["joint_outcomes"][0] in {"CC", "DD"} for item in rounds) / 10
    mismatch = 1 - coordination_success
    return {
        "id": f"trace-{ordinal:02d}",
        "is_completed": True,
        "errors": [],
        "task": {
            "data": {
                "episode": {
                    "b": b,
                    "c": 1.0,
                    "w": {1: 0.1, 2: 0.5, 3: 0.9}[seed_index],
                    "q": q,
                }
            }
        },
        "analysis_targets": {
            "suite": "repeated_2x2" if seed_index == 1 else "hkb_lock",
            "niceness_eligible": True,
            "switch_direction": direction,
            "safe_defect_mean_payoff": mean_payoff - 0.25,
            "safe_defect_mean_payoff_provenance": "synthetic-hand-replay-v1",
            "oracle_mean_payoff": mean_payoff + 0.5,
            "oracle_mean_payoff_provenance": "synthetic-hand-dp-v1",
            "value_defined_punishment": float(model in {"B", "D"}),
            "value_defined_punishment_provenance": "synthetic-hand-counterfactual-v1",
            "coordination_success": coordination_success,
            "mismatch": mismatch,
        },
        "info": {
            "coordination_trace": {
                "trace_header": {
                    "schema_version": 1,
                    "episode_id": episode_id,
                    "horizon": 10,
                    "mode": "pairwise",
                    "reward_model": model,
                    "policy_arm": model,
                    "policy_split": "training",
                    "sampling_metadata": {
                        "temperature": None,
                        "top_p": None,
                        "enable_thinking": None,
                        "requested_seed": 90_000 + ordinal,
                    },
                    "seed_metadata": {
                        "role": "training",
                        "training_seed": training_seed,
                        "evaluation_seed": None,
                        "episode_seed": 90_000 + ordinal,
                    },
                },
                "observations": [{"round_index": index} for index in range(1, 11)],
                "rounds": rounds,
                "invalid_output": False,
                "terminal_reason": "horizon",
                "terminal_event": {
                    "episode_id": episode_id,
                    "complete": True,
                    "rounds_completed": 10,
                },
            }
        },
    }


def main() -> None:
    destination = Path(__file__).with_name("synthetic_traces.jsonl")
    records = [
        _record(model, training_seed, ordinal)
        for ordinal, (model, training_seed) in enumerate(
            (
                (model, training_seed)
                for model in ("A", "B", "D", "E")
                for training_seed in TRAINING_SEEDS[model]
            ),
            1,
        )
    ]
    destination.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))


if __name__ == "__main__":
    main()
