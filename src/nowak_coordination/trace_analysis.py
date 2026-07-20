"""Strict trace validation and deterministic preregistered metric analysis."""

from __future__ import annotations

from collections import defaultdict
import argparse
import csv
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import spearmanr

from .gate4_registry import HELDOUT_POLICIES, PARAMETER_CELLS, gate4_assignment


TRAINING_SEEDS_BY_MODEL = {
    "A": set(range(1101, 1106)),
    "B": set(range(1201, 1206)),
    "C": set(range(1301, 1306)),
    "D": set(range(1401, 1404)),
    "E": set(range(1501, 1506)),
}
VALIDATION_SEEDS = {2101, 2102, 2103, 2104, 2105}
TEST_SEEDS = {3101, 3102, 3103, 3104, 3105}
REGISTERED_SUITES = {
    "nowak",
    "amtft",
    "hkb_lock",
    "recovery",
    "switch",
    "interleaved",
    "exploitability",
    "repeated_2x2",
    "forecast",
}


class TraceValidationError(ValueError):
    pass


@dataclass(frozen=True)
class AnalysisConfig:
    """Frozen constants that must be recorded with real evaluations."""

    ema_alpha: float = 0.2
    ema_initial: float = 0.5
    bootstrap_iterations: int = 10_000
    permutation_iterations: int = 10_000
    analysis_seed: int = 730_031
    input_trace_sha256: str | None = None
    analysis_spec_sha256: str | None = None

    def __post_init__(self) -> None:
        if not 0 < self.ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        if not 0 <= self.ema_initial <= 1:
            raise ValueError("ema_initial must be in [0, 1]")
        if self.bootstrap_iterations < 1 or self.permutation_iterations < 1:
            raise ValueError("resampling iterations must be positive")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TraceValidationError(f"{path}:{line_number}: malformed JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise TraceValidationError(f"{path}:{line_number}: record must be an object")
        records.append(value)
    if not records:
        raise TraceValidationError(f"{path}: no trace records")
    return records


def load_jsonl_many(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [record for path in paths for record in load_jsonl(path)]


def combined_trace_sha256(paths: Sequence[Path]) -> str:
    if len(paths) == 1:
        return hashlib.sha256(paths[0].read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for path in paths:
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def coordination_state(record: dict[str, Any]) -> dict[str, Any]:
    state = record.get("info", {}).get("coordination_trace")
    if not isinstance(state, dict):
        raise TraceValidationError(
            f"trace {record.get('id', '<missing>')}: missing info.coordination_trace"
        )
    return state


def analysis_targets(record: dict[str, Any]) -> dict[str, Any]:
    """Read targets from legacy fixtures or the persisted Verifiers task data."""

    candidates = (
        record.get("analysis_targets"),
        record.get("info", {}).get("analysis_targets"),
        record.get("task", {}).get("data", {}).get("analysis_targets"),
    )
    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
    return {}


def _check_finite(value: Any, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise TraceValidationError(f"{path}: non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _check_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite(item, f"{path}[{index}]")


def _validate_seed_metadata(
    metadata: dict[str, Any],
    policy_split: str,
    model: str,
    trace_id: str,
) -> None:
    role = metadata.get("role")
    training_seed = metadata.get("training_seed")
    evaluation_seed = metadata.get("evaluation_seed")
    checkpoint_seed = metadata.get("checkpoint_training_seed")
    if role == "training":
        if (
            model not in TRAINING_SEEDS_BY_MODEL
            or training_seed not in TRAINING_SEEDS_BY_MODEL[model]
            or evaluation_seed is not None
        ):
            raise TraceValidationError(f"trace {trace_id}: invalid training seed metadata")
        if policy_split != "training":
            raise TraceValidationError(f"trace {trace_id}: training trace uses non-training pool")
        if checkpoint_seed not in (None, training_seed):
            raise TraceValidationError(f"trace {trace_id}: training checkpoint seed mismatch")
    elif role == "validation":
        if evaluation_seed not in VALIDATION_SEEDS or training_seed is not None:
            raise TraceValidationError(f"trace {trace_id}: invalid validation seed")
        if policy_split == "training":
            raise TraceValidationError(f"trace {trace_id}: validation uses training pool")
        if (
            model in TRAINING_SEEDS_BY_MODEL
            and checkpoint_seed not in TRAINING_SEEDS_BY_MODEL[model]
        ):
            raise TraceValidationError(f"trace {trace_id}: invalid checkpoint training seed")
    elif role == "test":
        if evaluation_seed not in TEST_SEEDS or training_seed is not None:
            raise TraceValidationError(f"trace {trace_id}: invalid test seed")
        if policy_split == "training":
            raise TraceValidationError(f"trace {trace_id}: test uses training pool")
        if (
            model in TRAINING_SEEDS_BY_MODEL
            and checkpoint_seed not in TRAINING_SEEDS_BY_MODEL[model]
        ):
            raise TraceValidationError(f"trace {trace_id}: invalid checkpoint training seed")
    elif role != "engineering":
        raise TraceValidationError(f"trace {trace_id}: unknown seed role {role!r}")


def _validate_recovery_suite(
    record: dict[str, Any],
    state: dict[str, Any],
    trace_id: str,
) -> None:
    if analysis_targets(record).get("suite") != "recovery":
        return
    header = state["trace_header"]
    episode = record.get("task", {}).get("data", {}).get("episode", {})
    expected = {"b": 3.0, "c": 1.0, "w": 1.0, "q": 0.0, "noise_rate": 0.0}
    if any(
        not isinstance(episode.get(key), (int, float))
        or not math.isclose(float(episode[key]), value, abs_tol=1e-12)
        for key, value in expected.items()
    ):
        raise TraceValidationError(f"trace {trace_id}: recovery-suite parameters drifted")
    rounds = state["rounds"]
    if header.get("horizon") != 10 or len(rounds) != 10:
        raise TraceValidationError(f"trace {trace_id}: recovery suite requires ten rounds")
    if header.get("mode") != "dyadic":
        raise TraceValidationError(f"trace {trace_id}: recovery suite requires dyadic mode")
    sampling = header.get("sampling_metadata", {})
    if sampling != {
        "temperature": 0.7,
        "top_p": 1.0,
        "enable_thinking": False,
        "requested_seed": header["seed_metadata"].get("episode_seed"),
    }:
        raise TraceValidationError(f"trace {trace_id}: recovery sampling contract drifted")
    if header["seed_metadata"].get("role") != "test" or header.get("policy_split") == "training":
        raise TraceValidationError(f"trace {trace_id}: recovery suite is not held-out test data")
    perturbations = [event for event in rounds if event.get("perturbation", {}).get("applied")]
    if (
        len(perturbations) != 1
        or perturbations[0]["round_index"] != 5
        or perturbations[0]["perturbation"].get("actor") != "partner"
    ):
        raise TraceValidationError(f"trace {trace_id}: recovery perturbation contract drifted")
    for event in rounds:
        if event.get("partner_policy") != "generous_tit_for_tat":
            raise TraceValidationError(f"trace {trace_id}: recovery partner contract drifted")
        if event["focal_intended_action"] != event["focal_executed_action"]:
            raise TraceValidationError(f"trace {trace_id}: recovery suite has focal action noise")
        intended = event["partner_intended_actions"]
        executed = event["partner_executed_actions"]
        if event["round_index"] == 5:
            if intended != ["C"] or executed != ["D"]:
                raise TraceValidationError(
                    f"trace {trace_id}: recovery forced defection is malformed"
                )
        elif intended != executed:
            raise TraceValidationError(f"trace {trace_id}: recovery suite has endogenous noise")


def _validate_exploitability_suite(
    record: dict[str, Any],
    state: dict[str, Any],
    trace_id: str,
) -> None:
    targets = analysis_targets(record)
    if targets.get("suite") != "exploitability":
        return
    header = state["trace_header"]
    is_gate4_validation = (
        targets.get("registry") == "gate4_base_characterization_v1"
        and header["seed_metadata"].get("role") == "validation"
    )
    if (
        not is_gate4_validation
        and header["seed_metadata"].get("role") != "test"
    ) or header.get("policy_split") == "training":
        raise TraceValidationError(
            f"trace {trace_id}: exploitability suite is not held-out test data"
        )
    policies = {event.get("partner_policy") for event in state["rounds"]}
    if len(policies) != 1 or not policies <= {"always_defect", "opportunist"}:
        raise TraceValidationError(f"trace {trace_id}: exploitability partner contract drifted")
    if targets.get("safe_defect_mean_payoff") is None:
        raise TraceValidationError(f"trace {trace_id}: exploitability counterfactual is missing")


def _validate_sampler_seed(
    record: dict[str, Any],
    state: dict[str, Any],
    trace_id: str,
) -> None:
    """Require three-way agreement between task provenance, request, and saved trace."""

    header = state["trace_header"]
    if header.get("seed_metadata", {}).get("role") not in {"validation", "test"}:
        return
    requested = header.get("sampling_metadata", {}).get("requested_seed")
    agent_sampling = (record.get("agent") or {}).get("sampling") or {}
    effective = agent_sampling.get("seed")
    evidence = record.get("info", {}).get("sampler_seed_evidence", {})
    if not isinstance(requested, int) or requested < 0:
        raise TraceValidationError(f"trace {trace_id}: requested sampler seed is missing")
    if effective != requested:
        raise TraceValidationError(f"trace {trace_id}: trace agent sampler seed mismatch")
    if evidence != {
        "requested_seed": requested,
        "effective_seed": requested,
        "transport": "verifiers.v1 EvalClient -> OpenAI chat request -> vLLM",
        "trace_agent_sampling_recorded": True,
    }:
        raise TraceValidationError(f"trace {trace_id}: effective sampler-seed evidence is missing")
    sampling = header["sampling_metadata"]
    for trace_key, agent_key in (("temperature", "temperature"), ("top_p", "top_p")):
        if (
            sampling.get(trace_key) is not None
            and agent_sampling.get(agent_key) != sampling[trace_key]
        ):
            raise TraceValidationError(f"trace {trace_id}: recorded {trace_key} mismatch")
    thinking = sampling.get("enable_thinking")
    if (
        thinking is not None
        and agent_sampling.get("chat_template_kwargs", {}).get("enable_thinking") != thinking
    ):
        raise TraceValidationError(f"trace {trace_id}: recorded thinking mode mismatch")


def _validate_gate4_trace(
    record: dict[str, Any],
    state: dict[str, Any],
    trace_id: str,
) -> None:
    targets = analysis_targets(record)
    if targets.get("registry") != "gate4_base_characterization_v1":
        return
    header = state["trace_header"]
    episode = record.get("task", {}).get("data", {}).get("episode", {})
    evaluation_seed = header.get("seed_metadata", {}).get("evaluation_seed")
    sampling = header.get("sampling_metadata", {})
    if (
        evaluation_seed not in VALIDATION_SEEDS
        or header.get("policy_arm") != "Base"
        or header.get("horizon") != 10
        or len(state["rounds"]) != 10
        or sampling
        != {
            "temperature": 0.7,
            "top_p": 1.0,
            "enable_thinking": False,
            "requested_seed": evaluation_seed,
        }
    ):
        raise TraceValidationError(f"trace {trace_id}: Gate-4 base contract drifted")
    try:
        cell_index = PARAMETER_CELLS.index(
            (float(episode["b"]), float(episode["w"]), float(episode["q"]))
        )
    except (KeyError, ValueError) as exc:
        raise TraceValidationError(f"trace {trace_id}: Gate-4 parameter cell is invalid") from exc
    expected_episode_seed = 4_210_100 + (evaluation_seed - 2101) * 100 + cell_index
    if header["seed_metadata"].get("episode_seed") != expected_episode_seed:
        raise TraceValidationError(f"trace {trace_id}: Gate-4 episode seed drifted")
    assignment = gate4_assignment(evaluation_seed, cell_index)
    if targets.get("scenario") != assignment.scenario:
        raise TraceValidationError(f"trace {trace_id}: Gate-4 scenario assignment drifted")
    suite = targets.get("suite")
    mode = header.get("mode")
    split = header.get("policy_split")
    policies = {
        policy
        for event in state["rounds"]
        for policy in (
            event["partner_policy"]
            if isinstance(event["partner_policy"], list)
            else [event["partner_policy"]]
        )
    }
    valid = (
        suite == assignment.suite
        and mode == assignment.mode
        and split == assignment.policy_split
        and episode.get("group_size", 4) == assignment.group_size
    )
    if assignment.scenario == "diagnostic_switch":
        valid = (
            valid
            and policies == {assignment.partner_policy, assignment.switch_to_policy}
            and episode.get("partner_switch_round") == 6
            and episode.get("switch_to_policy") == assignment.switch_to_policy
            and targets.get("switch_direction") == assignment.switch_direction
        )
    elif assignment.scenario == "heldout_group_forecast":
        valid = (
            valid
            and policies == {assignment.partner_policy}
            and all(event.get("forecast_target") is not None for event in state["rounds"])
        )
    else:
        valid = valid and policies == {assignment.partner_policy}
    if assignment.partner_policy == "copy_with_noise_10%":
        valid = valid and math.isclose(float(episode.get("noise_rate", -1)), 0.1)
    if assignment.suite == "exploitability":
        expected_provenance = {
            "method": "same_seed_same_world_always_defect_replay",
            "registry": "gate4_base_characterization_v1",
            "episode_seed": expected_episode_seed,
            "horizon": 10,
        }
        valid = (
            valid
            and targets.get("safe_defect_mean_payoff") is not None
            and targets.get("safe_defect_mean_payoff_provenance") == expected_provenance
        )
    if not valid:
        raise TraceValidationError(f"trace {trace_id}: Gate-4 scenario mechanics drifted")


def validate_gate4_cohort(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Enforce the frozen 500-row Gate-4 registry before characterization."""

    gate4 = [
        record
        for record in records
        if analysis_targets(record).get("registry") == "gate4_base_characterization_v1"
    ]
    if len(gate4) != 500 or len(gate4) != len(records):
        raise TraceValidationError("Gate-4 cohort must contain exactly its 500 registered traces")
    expected_parameters = set(PARAMETER_CELLS)
    expected_scenarios = {
        "heldout_forgiving_grudger": 15,
        "heldout_delayed_tft": 15,
        "heldout_probabilistic_defector": 15,
        "heldout_noisy_copy": 15,
        "diagnostic_switch": 15,
        "diagnostic_exploitability": 15,
        "heldout_group_forecast": 10,
    }
    evaluation_seeds: set[int] = set()
    for seed in sorted(VALIDATION_SEEDS):
        block = [
            record
            for record in gate4
            if coordination_state(record)["trace_header"]["seed_metadata"].get("evaluation_seed")
            == seed
        ]
        if len(block) != 100:
            raise TraceValidationError(f"Gate-4 validation seed {seed} must have 100 traces")
        parameters = {
            (
                float(record["task"]["data"]["episode"]["b"]),
                float(record["task"]["data"]["episode"]["w"]),
                float(record["task"]["data"]["episode"]["q"]),
            )
            for record in block
        }
        if parameters != expected_parameters:
            raise TraceValidationError(f"Gate-4 validation seed {seed} parameter grid drifted")
        scenario_counts = {
            scenario: sum(analysis_targets(record).get("scenario") == scenario for record in block)
            for scenario in expected_scenarios
        }
        if scenario_counts != expected_scenarios:
            raise TraceValidationError(f"Gate-4 validation seed {seed} scenario allocation drifted")
        for record in block:
            episode = record["task"]["data"]["episode"]
            cell_index = PARAMETER_CELLS.index(
                (float(episode["b"]), float(episode["w"]), float(episode["q"]))
            )
            assignment = gate4_assignment(seed, cell_index)
            targets = analysis_targets(record)
            if (
                targets.get("scenario") != assignment.scenario
                or targets.get("suite") != assignment.suite
                or episode.get("partner_policy") != assignment.partner_policy
                or episode.get("mode", "dyadic") != assignment.mode
                or episode.get("policy_split") != assignment.policy_split
                or episode.get("group_size", 4) != assignment.group_size
            ):
                raise TraceValidationError(
                    f"Gate-4 validation seed {seed} registry row assignment drifted"
                )
        evaluation_seeds.add(seed)
    group_policies = [
        record["task"]["data"]["episode"]["partner_policy"]
        for record in gate4
        if analysis_targets(record).get("scenario") == "heldout_group_forecast"
    ]
    if set(group_policies) != set(HELDOUT_POLICIES):
        raise TraceValidationError("Gate-4 group policy rotation does not cover held-out policies")
    return {
        "status": "PASS",
        "traces": 500,
        "evaluation_seeds": sorted(evaluation_seeds),
        "parameter_cells_per_seed": 100,
    }


def validate_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    trace_ids: set[str] = set()
    episode_ids: set[str] = set()
    role_episode_seeds: dict[int, str] = {}
    round_count = 0
    for record in records:
        trace_id = record.get("id")
        if not isinstance(trace_id, str) or not trace_id:
            raise TraceValidationError("trace ID is missing")
        if trace_id in trace_ids:
            raise TraceValidationError(f"duplicate trace ID: {trace_id}")
        trace_ids.add(trace_id)
        if record.get("errors"):
            raise TraceValidationError(f"trace {trace_id}: captured errors")
        if record.get("is_completed") is not True:
            raise TraceValidationError(f"trace {trace_id}: incomplete framework trace")
        state = coordination_state(record)
        _check_finite(record, f"trace[{trace_id}]")
        header = state.get("trace_header", {})
        episode_id = header.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise TraceValidationError(f"trace {trace_id}: missing episode ID")
        if episode_id in episode_ids:
            raise TraceValidationError(f"duplicate episode ID: {episode_id}")
        episode_ids.add(episode_id)
        terminal = state.get("terminal_event")
        if not isinstance(terminal, dict) or terminal.get("complete") is not True:
            raise TraceValidationError(f"trace {trace_id}: missing complete terminal event")
        if terminal.get("episode_id") != episode_id:
            raise TraceValidationError(f"trace {trace_id}: terminal episode mismatch")
        rounds = state.get("rounds")
        if not isinstance(rounds, list):
            raise TraceValidationError(f"trace {trace_id}: rounds are missing")
        expected_indices = list(range(1, len(rounds) + 1))
        if [item.get("round_index") for item in rounds] != expected_indices:
            raise TraceValidationError(f"trace {trace_id}: missing or reordered turns")
        if terminal.get("rounds_completed") != len(rounds):
            raise TraceValidationError(f"trace {trace_id}: terminal round count mismatch")
        if state.get("terminal_reason") == "horizon" and len(rounds) != header.get("horizon"):
            raise TraceValidationError(f"trace {trace_id}: horizon trace has missing turns")
        if state.get("terminal_reason") == "invalid_format" and rounds:
            raise TraceValidationError(f"trace {trace_id}: invalid format earned task turns")
        observations = state.get("observations")
        if not isinstance(observations, list) or len(observations) < max(1, len(rounds)):
            raise TraceValidationError(f"trace {trace_id}: incomplete observations")
        for event in rounds:
            if event.get("episode_id") != episode_id:
                raise TraceValidationError(f"trace {trace_id}: round episode mismatch")
            if event.get("focal_intended_action") not in {"C", "D"}:
                raise TraceValidationError(f"trace {trace_id}: malformed focal intention")
            if event.get("focal_executed_action") not in {"C", "D"}:
                raise TraceValidationError(f"trace {trace_id}: malformed focal execution")
            partners = event.get("partner_executed_actions")
            if not partners or any(action not in {"C", "D"} for action in partners):
                raise TraceValidationError(f"trace {trace_id}: malformed partner actions")
            if len(event.get("joint_outcomes", [])) != len(partners):
                raise TraceValidationError(f"trace {trace_id}: malformed outcome decomposition")
            expected_outcomes = [
                event["focal_executed_action"] + partner_action for partner_action in partners
            ]
            if event["joint_outcomes"] != expected_outcomes:
                raise TraceValidationError(f"trace {trace_id}: outcome/action mismatch")
            episode = record.get("task", {}).get("data", {}).get("episode", {})
            b = episode.get("b")
            c = episode.get("c")
            if not isinstance(b, (int, float)) or not isinstance(c, (int, float)):
                raise TraceValidationError(f"trace {trace_id}: payoff parameters missing")
            expected_payoff = float(b) * (partners.count("C") / len(partners)) - float(c) * (
                event["focal_executed_action"] == "C"
            )
            actual_payoff = event.get("focal_payoff")
            if not isinstance(actual_payoff, (int, float)) or not math.isclose(
                float(actual_payoff), expected_payoff, abs_tol=1e-10
            ):
                raise TraceValidationError(f"trace {trace_id}: focal payoff mismatch")
            if not isinstance(event.get("rendered_observation"), str):
                raise TraceValidationError(f"trace {trace_id}: rendered observation missing")
            forecast = event.get("forecast")
            target = event.get("forecast_target")
            if not isinstance(forecast, (int, float)) or not 0 <= forecast <= 1:
                raise TraceValidationError(f"trace {trace_id}: malformed forecast")
            if target is not None and not 0 <= target <= 1:
                raise TraceValidationError(f"trace {trace_id}: malformed forecast target")
            reward = event.get("reward", {})
            components = [reward.get("payoff"), reward.get("hkb"), reward.get("calibration")]
            if components[0] is None or reward.get("total") is None:
                raise TraceValidationError(f"trace {trace_id}: reward component missing")
            if not all(
                value is None or isinstance(value, (int, float))
                for value in (*components, reward["total"])
            ):
                raise TraceValidationError(f"trace {trace_id}: malformed reward component")
            expected_total = float(components[0])
            if components[1] is not None:
                expected_total += 0.15 * float(components[1])
            if components[2] is not None:
                expected_total += 0.05 * float(components[2])
            if not math.isclose(float(reward["total"]), expected_total, abs_tol=1e-10):
                raise TraceValidationError(f"trace {trace_id}: reward total mismatch")
        metadata = header.get("seed_metadata", {})
        _validate_seed_metadata(
            metadata,
            header.get("policy_split"),
            header.get("policy_arm", header.get("reward_model")),
            trace_id,
        )
        targets = analysis_targets(record)
        if not isinstance(targets, dict):
            raise TraceValidationError(f"trace {trace_id}: analysis_targets must be an object")
        if (
            metadata.get("role") in {"validation", "test"}
            and targets.get("suite") not in REGISTERED_SUITES
        ):
            raise TraceValidationError(f"trace {trace_id}: unregistered scientific suite")
        for name in ("oracle_mean_payoff", "safe_defect_mean_payoff"):
            if targets.get(name) is not None and not targets.get(f"{name}_provenance"):
                raise TraceValidationError(
                    f"trace {trace_id}: {name} lacks counterfactual replay provenance"
                )
        for name in (
            "oracle_mean_payoff",
            "safe_defect_mean_payoff",
            "value_defined_punishment",
            "coordination_success",
            "mismatch",
        ):
            if targets.get(name) is not None and not isinstance(targets[name], (int, float)):
                raise TraceValidationError(f"trace {trace_id}: malformed analysis target {name}")
        if targets.get("value_defined_punishment") is not None and not targets.get(
            "value_defined_punishment_provenance"
        ):
            raise TraceValidationError(
                f"trace {trace_id}: value_defined_punishment lacks counterfactual provenance"
            )
        _validate_recovery_suite(record, state, trace_id)
        _validate_exploitability_suite(record, state, trace_id)
        _validate_sampler_seed(record, state, trace_id)
        _validate_gate4_trace(record, state, trace_id)
        episode_seed = metadata.get("episode_seed")
        role = metadata.get("role")
        if role != "engineering":
            if not isinstance(episode_seed, int):
                raise TraceValidationError(f"trace {trace_id}: episode seed must be an integer")
            previous_role = role_episode_seeds.setdefault(episode_seed, role)
            if previous_role != role:
                raise TraceValidationError(
                    f"trace {trace_id}: episode seed {episode_seed} leaked across splits"
                )
        round_count += len(rounds)
    return {
        "status": "PASS",
        "traces": len(records),
        "rounds": round_count,
        "episode_ids_unique": True,
        "trace_ids_unique": True,
    }


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _lock(rounds: Sequence[dict[str, Any]]) -> tuple[str | None, int | None]:
    outcomes = [event["joint_outcomes"][0] for event in rounds if len(event["joint_outcomes"]) == 1]
    for end in range(2, len(outcomes)):
        window = outcomes[end - 2 : end + 1]
        if window == ["CC"] * 3:
            return "CC", end + 1
        if window == ["DD"] * 3:
            return "DD", end + 1
        if window in (["CD", "DC", "CD"], ["DC", "CD", "DC"]):
            return "alternation", end + 1
    return None, None


def episode_metrics(record: dict[str, Any]) -> dict[str, Any]:
    state = coordination_state(record)
    header = state["trace_header"]
    rounds = state["rounds"]
    episode = record.get("task", {}).get("data", {}).get("episode", {})
    outcomes = [outcome for event in rounds for outcome in event["joint_outcomes"]]
    focal_actions = [event["focal_executed_action"] for event in rounds]
    lock_type, lock_time = _lock(rounds)
    perturbation_round = next(
        (event["round_index"] for event in rounds if event["perturbation"]["applied"]),
        None,
    )
    recovery_time = None
    recovered_within_3 = None
    partner_perturbation = next(
        (
            event["round_index"]
            for event in rounds
            if event["perturbation"]["applied"] and event["perturbation"]["actor"] == "partner"
        ),
        None,
    )
    if perturbation_round is not None:
        later = [event for event in rounds if event["round_index"] > perturbation_round]
        for index in range(2, len(later)):
            if all(item["joint_outcomes"] == ["CC"] for item in later[index - 2 : index + 1]):
                recovery_time = later[index]["round_index"] - perturbation_round
                break
        recovered_within_3 = recovery_time is not None and recovery_time <= 3
    switch_round = next(
        (
            event["round_index"] + 1
            for event in rounds
            if event.get("transition_to_next") == "forced_switch"
        ),
        None,
    )
    post_switch = (
        [event for event in rounds if switch_round <= event["round_index"] <= switch_round + 3]
        if switch_round
        else []
    )
    by_policy: dict[str, list[int]] = defaultdict(list)
    for event in rounds:
        policy = event.get("partner_policy")
        if isinstance(policy, str):
            by_policy[policy].append(event["focal_executed_action"] == "C")
    reciprocator = [
        value for policy, values in by_policy.items() if "tit_for_tat" in policy for value in values
    ]
    defector = by_policy.get("always_defect", [])
    interleaved_separation = (
        float(_mean(reciprocator) - _mean(defector)) if reciprocator and defector else None
    )
    provocation_changes: list[float] = []
    for index in range(3, len(rounds)):
        if (
            rounds[index - 2]["partner_executed_actions"][0] == "D"
            and rounds[index - 1]["partner_executed_actions"][0] == "D"
        ):
            provocation_changes.append(
                float(
                    (rounds[index]["focal_executed_action"] == "C")
                    - (rounds[index - 3]["focal_executed_action"] == "C")
                )
            )
    forgiveness = None
    retaliation_length = None
    if partner_perturbation is not None:
        perturbation_index = partner_perturbation - 1
        after = [event for event in rounds if event["round_index"] > partner_perturbation]
        is_one_round_accident = (
            perturbation_index > 0
            and perturbation_index + 1 < len(rounds)
            and rounds[perturbation_index - 1]["partner_executed_actions"] == ["C"]
            and rounds[perturbation_index]["partner_executed_actions"] == ["D"]
            and rounds[perturbation_index + 1]["partner_executed_actions"] == ["C"]
        )
        partner_return = partner_perturbation + 1 if is_one_round_accident else None
        if partner_return is not None:
            forgiveness = float(
                any(
                    event["joint_outcomes"] == ["CC"]
                    and partner_return <= event["round_index"] <= partner_return + 2
                    for event in after
                )
            )
        retaliation_length = 0
        for event in after:
            if event["focal_executed_action"] == "D":
                retaliation_length += 1
            else:
                break
    targets = analysis_targets(record)
    safe_defect = targets.get("safe_defect_mean_payoff")
    oracle = targets.get("oracle_mean_payoff")
    mean_payoff = _mean(float(event["focal_payoff"]) for event in rounds)
    adaptivity_values = [
        value
        for event in rounds
        for value in (
            event["partner_adaptive"]
            if isinstance(event["partner_adaptive"], list)
            else [event["partner_adaptive"]]
        )
    ]
    adaptivity = (
        "adaptive"
        if adaptivity_values and all(adaptivity_values)
        else "nonadaptive"
        if adaptivity_values and not any(adaptivity_values)
        else "mixed"
    )
    reciprocal_events = [
        event
        for event in rounds
        if isinstance(event.get("partner_policy"), str)
        and any(
            token in event["partner_policy"] for token in ("tit_for_tat", "generous_tit_for_tat")
        )
    ]
    policies = {
        policy
        for event in rounds
        for policy in (
            event.get("partner_policy")
            if isinstance(event.get("partner_policy"), list)
            else [event.get("partner_policy")]
        )
        if isinstance(policy, str)
    }
    cooperation_rate = _mean(action == "C" for action in focal_actions)
    action_entropy = None
    if cooperation_rate is not None:
        action_entropy = -sum(
            probability * math.log2(probability)
            for probability in (cooperation_rate, 1 - cooperation_rate)
            if probability > 0
        )
    forecast_values = [float(event["forecast"]) for event in rounds]
    forecast_bins = [min(int(forecast * 10), 9) for forecast in forecast_values]
    forecast_entropy = (
        -sum(
            probability * math.log2(probability)
            for bin_index in range(10)
            if (probability := forecast_bins.count(bin_index) / len(forecast_bins)) > 0
        )
        if forecast_bins
        else None
    )
    reward_values = [float(event["reward"]["total"]) for event in rounds]
    row: dict[str, Any] = {
        "trace_id": record["id"],
        "episode_id": header["episode_id"],
        "model": header.get("policy_arm", header["reward_model"]),
        "mode": header["mode"],
        "seed_role": header["seed_metadata"]["role"],
        "training_seed": (
            header["seed_metadata"].get("checkpoint_training_seed")
            or header["seed_metadata"].get("training_seed")
        ),
        "evaluation_seed": header["seed_metadata"].get("evaluation_seed"),
        "episode_seed": header["seed_metadata"].get("episode_seed"),
        "b": episode.get("b"),
        "c": episode.get("c"),
        "b_over_c": episode.get("b") / episode.get("c")
        if episode.get("b") and episode.get("c")
        else None,
        "w": episode.get("w"),
        "q": episode.get("q"),
        "threshold_band": (
            "below"
            if episode.get("q") is not None
            and episode.get("b")
            and episode.get("c") is not None
            and episode["q"] < episode["c"] / episode["b"] - 0.15
            else "above"
            if episode.get("q") is not None
            and episode.get("b")
            and episode.get("c") is not None
            and episode["q"] > episode["c"] / episode["b"] + 0.15
            else "near"
            if episode.get("q") is not None and episode.get("b") and episode.get("c") is not None
            else None
        ),
        "partner_adaptivity": adaptivity,
        "partner_policy": next(iter(policies)) if len(policies) == 1 else "mixed",
        "suite": targets.get("suite"),
        "registry": targets.get("registry"),
        "scenario": targets.get("scenario"),
        "switch_direction": targets.get("switch_direction"),
        "rounds": len(rounds),
        "format_valid": float(not state["invalid_output"]),
        "cooperation_rate": cooperation_rate,
        "action_entropy": action_entropy,
        "forecast_entropy": forecast_entropy,
        "total_reward_variance": float(np.var(reward_values)) if reward_values else None,
        "mean_payoff": mean_payoff,
        "p_cc": outcomes.count("CC") / len(outcomes) if outcomes else 0.0,
        "p_cd": outcomes.count("CD") / len(outcomes) if outcomes else 0.0,
        "p_dc": outcomes.count("DC") / len(outcomes) if outcomes else 0.0,
        "p_dd": outcomes.count("DD") / len(outcomes) if outcomes else 0.0,
        "niceness": (
            float(focal_actions[0] == "C")
            if focal_actions and targets.get("niceness_eligible") is True
            else None
        ),
        "provokability_delta_p_c": _mean(provocation_changes),
        "forgiveness_within_3": forgiveness,
        "nonexploitability_vs_safe_defect": mean_payoff - safe_defect
        if mean_payoff is not None and safe_defect is not None
        else None,
        "oracle_regret": oracle - mean_payoff
        if mean_payoff is not None and oracle is not None
        else None,
        "lock_type": lock_type,
        "lock_time": lock_time,
        "recovery_time": recovery_time,
        "recovered_within_3": float(recovered_within_3) if recovered_within_3 is not None else None,
        "post_switch_cooperation": _mean(
            event["focal_executed_action"] == "C" for event in post_switch
        ),
        "post_switch_payoff": _mean(float(event["focal_payoff"]) for event in post_switch),
        "interleaved_separation": interleaved_separation,
        "retaliation_length": retaliation_length,
        "value_defined_punishment": targets.get("value_defined_punishment"),
        "cooperation_with_cooperators": _mean(
            event["joint_outcomes"] == ["CC"] for event in reciprocal_events
        ),
        "coordination_success": targets.get("coordination_success"),
        "mismatch": targets.get("mismatch"),
    }
    return row


def round_rows(record: dict[str, Any], config: AnalysisConfig) -> list[dict[str, Any]]:
    state = coordination_state(record)
    header = state["trace_header"]
    episode = record.get("task", {}).get("data", {}).get("episode", {})
    suite = analysis_targets(record).get("suite")
    rows = []
    ema = config.ema_initial
    for event in state["rounds"]:
        row = {
            "trace_id": record["id"],
            "episode_id": header["episode_id"],
            "model": header.get("policy_arm", header["reward_model"]),
            "suite": suite,
            "training_seed": (
                header["seed_metadata"].get("checkpoint_training_seed")
                or header["seed_metadata"].get("training_seed")
            ),
            "evaluation_seed": header["seed_metadata"].get("evaluation_seed"),
            "round_index": event["round_index"],
            "partner_adaptivity": (
                "mixed"
                if isinstance(event["partner_adaptive"], list)
                else "adaptive"
                if event["partner_adaptive"]
                else "nonadaptive"
            ),
            "focal_action": event["focal_executed_action"],
            "focal_payoff": event["focal_payoff"],
            "forecast": event["forecast"],
            "forecast_target": event["forecast_target"],
            "ema_forecast": ema if event["forecast_target"] is not None else None,
            "group_size": (
                int(episode.get("group_size", len(event["partner_executed_actions"]) + 1))
                if event["forecast_target"] is not None
                else len(event["partner_executed_actions"])
            ),
            "p_cc": event["joint_outcomes"].count("CC") / len(event["joint_outcomes"]),
            "p_cd": event["joint_outcomes"].count("CD") / len(event["joint_outcomes"]),
            "p_dc": event["joint_outcomes"].count("DC") / len(event["joint_outcomes"]),
            "p_dd": event["joint_outcomes"].count("DD") / len(event["joint_outcomes"]),
        }
        rows.append(row)
        if event["forecast_target"] is not None:
            ema = config.ema_alpha * float(event["forecast_target"]) + (1 - config.ema_alpha) * ema
    return rows


def brier_decomposition(
    forecasts: Sequence[float],
    outcomes: Sequence[float],
    group_sizes: Sequence[int] | None = None,
    bins: int = 10,
) -> dict[str, float | None]:
    if not forecasts:
        return {
            "brier_score": None,
            "reliability": None,
            "resolution": None,
            "uncertainty": None,
        }
    if group_sizes is not None:
        expanded_f: list[float] = []
        expanded_y: list[float] = []
        for forecast, outcome, size in zip(forecasts, outcomes, group_sizes, strict=True):
            successes = int(round(outcome * size))
            if size < 1 or not math.isclose(successes / size, outcome, abs_tol=1e-10):
                raise ValueError("fractional target is incompatible with group size")
            expanded_f.extend([forecast] * size)
            expanded_y.extend([1.0] * successes + [0.0] * (size - successes))
        f = np.asarray(expanded_f, dtype=float)
        y = np.asarray(expanded_y, dtype=float)
    else:
        f = np.asarray(forecasts, dtype=float)
        y = np.asarray(outcomes, dtype=float)
    climatology = float(y.mean())
    indices = np.minimum((f * bins).astype(int), bins - 1)
    reliability = 0.0
    resolution = 0.0
    for index in range(bins):
        mask = indices == index
        if not mask.any():
            continue
        weight = float(mask.mean())
        reliability += weight * (float(f[mask].mean()) - float(y[mask].mean())) ** 2
        resolution += weight * (float(y[mask].mean()) - climatology) ** 2
    uncertainty = climatology * (1 - climatology)
    return {
        "brier_score": float(np.mean((f - y) ** 2)),
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
    }


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda index: pvalues[index])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def nested_bootstrap(
    values_by_seed: dict[int, Sequence[float]],
    *,
    iterations: int = 10_000,
    seed: int = 730_031,
) -> tuple[float, float, float]:
    if not values_by_seed:
        raise ValueError("nested bootstrap requires at least one seed")
    rng = np.random.default_rng(seed)
    seed_ids = np.asarray(sorted(values_by_seed))
    draws = np.empty(iterations)
    for iteration in range(iterations):
        sampled_seeds = rng.choice(seed_ids, size=len(seed_ids), replace=True)
        seed_means = []
        for seed_id in sampled_seeds:
            values = np.asarray(values_by_seed[int(seed_id)], dtype=float)
            seed_means.append(float(rng.choice(values, size=len(values), replace=True).mean()))
        draws[iteration] = np.mean(seed_means)
    point = float(np.mean([np.mean(values) for values in values_by_seed.values()]))
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _seed_values(rows: Sequence[dict[str, Any]], metric: str) -> dict[int, list[float]]:
    result: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        seed = row.get("training_seed")
        if value not in (None, "") and isinstance(seed, int):
            result[seed].append(float(value))
    return result


def _replication_seed(row: dict[str, Any]) -> tuple[str, int] | None:
    if isinstance(row.get("training_seed"), int):
        return "training_seed", int(row["training_seed"])
    if isinstance(row.get("evaluation_seed"), int):
        return "evaluation_seed", int(row["evaluation_seed"])
    return None


def _difference_interval(
    left: dict[int, Sequence[float]],
    right: dict[int, Sequence[float]],
    config: AnalysisConfig,
) -> tuple[float, float, float]:
    if not left or not right:
        raise ValueError("both arms require training-seed replications")
    rng = np.random.default_rng(config.analysis_seed)
    left_ids = np.asarray(sorted(left))
    right_ids = np.asarray(sorted(right))
    draws = np.empty(config.bootstrap_iterations)
    for iteration in range(config.bootstrap_iterations):
        left_draw = rng.choice(left_ids, len(left_ids), replace=True)
        right_draw = rng.choice(right_ids, len(right_ids), replace=True)
        left_mean = np.mean(
            [
                rng.choice(left[int(seed)], len(left[int(seed)]), replace=True).mean()
                for seed in left_draw
            ]
        )
        right_mean = np.mean(
            [
                rng.choice(right[int(seed)], len(right[int(seed)]), replace=True).mean()
                for seed in right_draw
            ]
        )
        draws[iteration] = right_mean - left_mean
    point = float(
        np.mean([np.mean(values) for values in right.values()])
        - np.mean([np.mean(values) for values in left.values()])
    )
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _seed_permutation_p(
    left: dict[int, Sequence[float]],
    right: dict[int, Sequence[float]],
    config: AnalysisConfig,
) -> float:
    """Two-sided label permutation over independent training-seed means."""

    left_means = [float(np.mean(values)) for _, values in sorted(left.items())]
    right_means = [float(np.mean(values)) for _, values in sorted(right.items())]
    pooled = np.asarray(left_means + right_means)
    n_left = len(left_means)
    observed = abs(float(np.mean(right_means) - np.mean(left_means)))
    possible = math.comb(len(pooled), n_left)
    exhaustive = possible <= 100_000
    if exhaustive:
        effects = []
        for selected in combinations(range(len(pooled)), n_left):
            mask = np.zeros(len(pooled), dtype=bool)
            mask[list(selected)] = True
            effects.append(abs(float(pooled[~mask].mean() - pooled[mask].mean())))
    else:
        rng = np.random.default_rng(config.analysis_seed)
        effects = []
        for _ in range(config.permutation_iterations):
            shuffled = rng.permutation(pooled)
            effects.append(abs(float(shuffled[n_left:].mean() - shuffled[:n_left].mean())))
    exceedances = sum(effect >= observed - 1e-15 for effect in effects)
    if exhaustive:
        return exceedances / len(effects)
    return (exceedances + 1) / (len(effects) + 1)


def _confirmatory_definitions() -> list[dict[str, Any]]:
    return [
        {
            "family": "efficacy",
            "comparison": "A_vs_B",
            "left": "A",
            "right": "B",
            "metric": "recovered_within_3",
            "suite": "recovery",
        },
        {
            "family": "efficacy",
            "comparison": "B_vs_E",
            "left": "E",
            "right": "B",
            "metric": "recovered_within_3",
            "suite": "recovery",
        },
        {
            "family": "safety",
            "comparison": "A_vs_B_NI",
            "left": "A",
            "right": "B",
            "metric": "p_dd",
            "suite": "recovery",
        },
        {
            "family": "safety",
            "comparison": "A_vs_B_NI",
            "left": "A",
            "right": "B",
            "metric": "nonexploitability_vs_safe_defect",
            "suite": "exploitability",
            "partner_policy": "always_defect",
        },
        {
            "family": "safety",
            "comparison": "A_vs_B_NI",
            "left": "A",
            "right": "B",
            "metric": "nonexploitability_vs_safe_defect",
            "suite": "exploitability",
            "partner_policy": "opportunist",
        },
        {
            "family": "safety",
            "comparison": "A_vs_B_NI",
            "left": "A",
            "right": "B",
            "metric": "format_valid",
            "suite": "recovery",
        },
    ]


def _expanded_episodes(episodes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded = []
    for row in episodes:
        item = dict(row)
        for lock_type in ("CC", "DD", "alternation"):
            item[f"lock_{lock_type}"] = float(row["lock_type"] == lock_type)
        expanded.append(item)
    return expanded


def _selected_rows(
    expanded: Sequence[dict[str, Any]], definition: dict[str, Any]
) -> list[dict[str, Any]]:
    return [
        row
        for row in expanded
        if all(
            row.get(key) == value
            for key, value in definition.items()
            if key in {"suite", "switch_direction", "threshold_band", "partner_policy"}
        )
    ]


def _confirmatory_rows(
    episodes: Sequence[dict[str, Any]], config: AnalysisConfig
) -> list[dict[str, Any]]:
    expanded = _expanded_episodes(episodes)
    results = []
    valid_indices = []
    raw_pvalues = []
    for definition in _confirmatory_definitions():
        selected = _selected_rows(expanded, definition)
        left = _seed_values(
            [row for row in selected if row["model"] == definition["left"]],
            definition["metric"],
        )
        right = _seed_values(
            [row for row in selected if row["model"] == definition["right"]],
            definition["metric"],
        )
        result = {
            **definition,
            "status": "UNAVAILABLE",
            "effect": None,
            "ci_lower": None,
            "ci_upper": None,
            "raw_p": None,
            "holm_p": None,
            "permutation_method": None,
            "decision": None,
            "left_training_seeds": len(left),
            "right_training_seeds": len(right),
        }
        if len(left) >= 5 and len(right) >= 5:
            effect, lower, upper = _difference_interval(left, right, config)
            result.update(
                status="ESTIMATED",
                effect=effect,
                ci_lower=lower,
                ci_upper=upper,
            )
            if definition["family"] == "efficacy":
                raw_p = _seed_permutation_p(left, right, config)
                result["raw_p"] = raw_p
                result["permutation_method"] = "exact_252_assignments"
                valid_indices.append(len(results))
                raw_pvalues.append(raw_p)
            elif definition["metric"] == "p_dd":
                result["decision"] = "PASS" if upper <= 0.05 else "FAIL"
            elif definition["metric"] == "nonexploitability_vs_safe_defect":
                result["decision"] = "PASS" if lower >= -0.10 else "FAIL"
            elif definition["metric"] == "format_valid":
                result["decision"] = "PASS" if lower >= -0.01 else "FAIL"
        elif left and right:
            result["status"] = "INSUFFICIENT_REPLICATION"
        results.append(result)
    for index, adjusted in zip(valid_indices, holm_adjust(raw_pvalues), strict=True):
        results[index]["holm_p"] = adjusted
    return results


def _confirmatory_cohort_issues(episodes: Sequence[dict[str, Any]]) -> list[str]:
    issues = []
    recovery = [row for row in episodes if row.get("suite") == "recovery"]
    for model in ("A", "B", "E"):
        expected_seeds = TRAINING_SEEDS_BY_MODEL[model]
        model_rows = [row for row in recovery if row["model"] == model]
        actual_seeds = {row["training_seed"] for row in model_rows}
        if actual_seeds != expected_seeds:
            issues.append(f"recovery {model}: training-seed cohort mismatch")
        bad_cells = 0
        for training_seed in expected_seeds:
            for evaluation_seed in TEST_SEEDS:
                count = sum(
                    row["training_seed"] == training_seed
                    and row["evaluation_seed"] == evaluation_seed
                    for row in model_rows
                )
                if count != 20:
                    bad_cells += 1
        if bad_cells:
            issues.append(f"recovery {model}: {bad_cells} seed cells are not exactly 20 episodes")
    exploitability = [row for row in episodes if row.get("suite") == "exploitability"]
    for model in ("A", "B"):
        expected_seeds = TRAINING_SEEDS_BY_MODEL[model]
        model_rows = [row for row in exploitability if row["model"] == model]
        actual_seeds = {row["training_seed"] for row in model_rows}
        if actual_seeds != expected_seeds:
            issues.append(f"exploitability {model}: training-seed cohort mismatch")
        bad_cells = 0
        for training_seed in expected_seeds:
            for evaluation_seed in TEST_SEEDS:
                cell = [
                    row
                    for row in model_rows
                    if row["training_seed"] == training_seed
                    and row["evaluation_seed"] == evaluation_seed
                ]
                policy_counts = {
                    policy: sum(row["partner_policy"] == policy for row in cell)
                    for policy in ("always_defect", "opportunist")
                }
                if policy_counts != {"always_defect": 10, "opportunist": 10}:
                    bad_cells += 1
        if bad_cells:
            issues.append(
                f"exploitability {model}: {bad_cells} seed cells lack 10 episodes per policy"
            )
    return issues


def _confirmatory_decision(
    rows: Sequence[dict[str, Any]],
    episodes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    cohort_issues = _confirmatory_cohort_issues(episodes)
    if cohort_issues:
        return {
            "status": "NOT_EVALUABLE",
            "reason": "confirmatory cohort contract failed",
            "issues": cohort_issues,
        }
    if any(row["status"] != "ESTIMATED" for row in rows):
        return {
            "status": "NOT_EVALUABLE",
            "reason": "five complete A, B, and E training runs are required",
        }
    efficacy = [row for row in rows if row["family"] == "efficacy"]
    safety = [row for row in rows if row["family"] == "safety"]
    efficacy_pass = all(
        float(row["effect"]) > 0 and float(row["ci_lower"]) > 0 and float(row["holm_p"]) <= 0.05
        for row in efficacy
    )
    safety_pass = all(row["decision"] == "PASS" for row in safety)
    return {
        "status": "PASS" if efficacy_pass and safety_pass else "FAIL",
        "efficacy_pass": efficacy_pass,
        "safety_pass": safety_pass,
        "rule": "both efficacy rows pass and all four safety rows pass",
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def analyze(
    records: Sequence[dict[str, Any]],
    output_dir: Path,
    config: AnalysisConfig | None = None,
) -> dict[str, Path]:
    config = config or AnalysisConfig()
    validate_records(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = sorted(
        (episode_metrics(record) for record in records), key=lambda row: row["trace_id"]
    )
    rounds = sorted(
        (row for record in records for row in round_rows(record, config)),
        key=lambda row: (row["trace_id"], row["round_index"]),
    )
    episode_columns = list(episodes[0])
    round_columns = (
        "trace_id",
        "episode_id",
        "model",
        "suite",
        "training_seed",
        "evaluation_seed",
        "round_index",
        "partner_adaptivity",
        "focal_action",
        "focal_payoff",
        "forecast",
        "forecast_target",
        "ema_forecast",
        "group_size",
        "p_cc",
        "p_cd",
        "p_dc",
        "p_dd",
    )
    _write_csv(output_dir / "episodes.csv", episodes, episode_columns)
    _write_csv(output_dir / "rounds.csv", rounds, round_columns)

    numeric_metrics = (
        "cooperation_rate",
        "action_entropy",
        "forecast_entropy",
        "total_reward_variance",
        "mean_payoff",
        "p_cc",
        "p_cd",
        "p_dc",
        "p_dd",
        "format_valid",
        "lock_time",
        "recovery_time",
        "recovered_within_3",
        "post_switch_cooperation",
        "post_switch_payoff",
        "interleaved_separation",
        "niceness",
        "provokability_delta_p_c",
        "forgiveness_within_3",
        "retaliation_length",
        "value_defined_punishment",
        "cooperation_with_cooperators",
        "oracle_regret",
        "nonexploitability_vs_safe_defect",
        "coordination_success",
        "mismatch",
    )
    aggregates: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int | None, str | None, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in episodes:
        grouped[
            (row["model"], row["evaluation_seed"], row["suite"], row["partner_adaptivity"])
        ].append(row)
    for (model, evaluation_seed, suite, adaptivity), group in sorted(
        grouped.items(), key=lambda item: str(item[0])
    ):
        result: dict[str, Any] = {
            "model": model,
            "evaluation_seed": evaluation_seed,
            "suite": suite,
            "partner_adaptivity": adaptivity,
            "episodes": len(group),
        }
        for metric in numeric_metrics:
            values = [float(row[metric]) for row in group if row[metric] not in (None, "")]
            result[f"mean_{metric}"] = _mean(values)
        aggregates.append(result)
    aggregate_columns = list(aggregates[0])
    _write_csv(output_dir / "aggregates.csv", aggregates, aggregate_columns)

    diagnostic_rows = []
    diagnostic_grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    diagnostic_keys = (
        "model",
        "training_seed",
        "evaluation_seed",
        "suite",
        "partner_adaptivity",
        "partner_policy",
        "threshold_band",
        "switch_direction",
    )
    for row in episodes:
        diagnostic_grouped[tuple(row.get(key) for key in diagnostic_keys)].append(row)
    for key, group in sorted(diagnostic_grouped.items(), key=lambda item: str(item[0])):
        result = dict(zip(diagnostic_keys, key, strict=True))
        result["episodes"] = len(group)
        for metric in numeric_metrics:
            values = [float(row[metric]) for row in group if row[metric] not in (None, "")]
            result[f"mean_{metric}"] = _mean(values)
        diagnostic_rows.append(result)
    diagnostic_columns = (
        list(diagnostic_keys) + ["episodes"] + [f"mean_{metric}" for metric in numeric_metrics]
    )
    _write_csv(
        output_dir / "diagnostic_cells.csv",
        diagnostic_rows,
        diagnostic_columns,
    )

    episode_lookup = {row["trace_id"]: row for row in episodes}
    stress_rows = []
    stress_grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rounds:
        episode = episode_lookup[row["trace_id"]]
        stress_grouped[
            (
                row["model"],
                episode["training_seed"],
                episode["evaluation_seed"],
                episode["suite"],
                row["partner_adaptivity"],
                episode["partner_policy"],
                episode["lock_type"],
                episode["threshold_band"],
            )
        ].append(row)
    for key, group in sorted(stress_grouped.items(), key=lambda item: str(item[0])):
        (
            model,
            training_seed,
            evaluation_seed,
            suite,
            adaptivity,
            partner_policy,
            lock_type,
            threshold_band,
        ) = key
        stress_rows.append(
            {
                "model": model,
                "training_seed": training_seed,
                "evaluation_seed": evaluation_seed,
                "suite": suite,
                "partner_adaptivity": adaptivity,
                "partner_policy": partner_policy,
                "lock_type": lock_type,
                "threshold_band": threshold_band,
                "episodes": len({row["episode_id"] for row in group}),
                "rounds": len(group),
                "mean_lock_time": _mean(
                    float(episode_lookup[row["trace_id"]]["lock_time"])
                    for row in group
                    if episode_lookup[row["trace_id"]]["lock_time"] is not None
                ),
                "mean_recovery_time": _mean(
                    float(episode_lookup[row["trace_id"]]["recovery_time"])
                    for row in group
                    if episode_lookup[row["trace_id"]]["recovery_time"] is not None
                ),
                "mean_p_cc": _mean(float(row["p_cc"]) for row in group),
                "mean_p_cd": _mean(float(row["p_cd"]) for row in group),
                "mean_p_dc": _mean(float(row["p_dc"]) for row in group),
                "mean_p_dd": _mean(float(row["p_dd"]) for row in group),
            }
        )
    _write_csv(
        output_dir / "hkb_stress.csv",
        stress_rows,
        (
            "model",
            "training_seed",
            "evaluation_seed",
            "suite",
            "partner_adaptivity",
            "partner_policy",
            "lock_type",
            "threshold_band",
            "episodes",
            "rounds",
            "mean_lock_time",
            "mean_recovery_time",
            "mean_p_cc",
            "mean_p_cd",
            "mean_p_dc",
            "mean_p_dd",
        ),
    )

    sensitivities = []
    for model in sorted({row["model"] for row in episodes}):
        replications = sorted(
            {
                (row["suite"], *replication)
                for row in episodes
                if row["model"] == model and (replication := _replication_seed(row)) is not None
            },
            key=str,
        )
        for suite, replication_kind, replication_seed in replications:
            model_rows = [
                row
                for row in episodes
                if row["model"] == model
                and row["suite"] == suite
                and _replication_seed(row) == (replication_kind, replication_seed)
            ]
            for axis in ("b_over_c", "w", "q"):
                cells: dict[float, list[float]] = defaultdict(list)
                for row in model_rows:
                    if row[axis] is not None and row["cooperation_rate"] is not None:
                        cells[float(row[axis])].append(float(row["cooperation_rate"]))
                if len(cells) >= 2:
                    x = sorted(cells)
                    y = [float(np.mean(cells[value])) for value in x]
                    candidate = float(spearmanr(x, y).statistic)
                    rho = candidate if math.isfinite(candidate) else None
                else:
                    rho = None
                sensitivities.append(
                    {
                        "model": model,
                        "suite": suite,
                        "replication_kind": replication_kind,
                        "replication_seed": replication_seed,
                        "axis": axis,
                        "spearman_rho": rho,
                    }
                )
    _write_csv(
        output_dir / "sensitivities.csv",
        sensitivities,
        (
            "model",
            "suite",
            "replication_kind",
            "replication_seed",
            "axis",
            "spearman_rho",
        ),
    )

    gate4_rows = [
        row
        for row in episodes
        if row.get("registry") == "gate4_base_characterization_v1"
    ]
    gate4_sensitivity_rows: list[dict[str, Any]] = []
    gate4_betas: dict[str, dict[int, float]] = defaultdict(dict)
    for evaluation_seed in sorted(VALIDATION_SEEDS):
        seed_rows = [row for row in gate4_rows if row["evaluation_seed"] == evaluation_seed]
        if not seed_rows:
            continue
        scenarios = sorted({str(row["scenario"]) for row in seed_rows})
        continuous = np.asarray(
            [[float(row[axis]) for axis in ("b_over_c", "w", "q")] for row in seed_rows]
        )
        means = continuous.mean(axis=0)
        scales = continuous.std(axis=0)
        if np.any(scales == 0):
            raise ValueError(f"Gate-4 seed {evaluation_seed} has a degenerate parameter axis")
        standardized = (continuous - means) / scales
        scenario_dummies = np.asarray(
            [
                [float(row["scenario"] == scenario) for scenario in scenarios[1:]]
                for row in seed_rows
            ]
        )
        design = np.column_stack((np.ones(len(seed_rows)), standardized, scenario_dummies))
        response = np.asarray([float(row["cooperation_rate"]) for row in seed_rows])
        coefficients, _, rank, singular_values = np.linalg.lstsq(design, response, rcond=None)
        if rank != design.shape[1]:
            raise ValueError(f"Gate-4 seed {evaluation_seed} sensitivity design is rank deficient")
        condition_number = float(singular_values[0] / singular_values[-1])
        for offset, axis in enumerate(("b_over_c", "w", "q"), start=1):
            beta = float(coefficients[offset])
            gate4_betas[axis][evaluation_seed] = beta
            gate4_sensitivity_rows.append(
                {
                    "row_type": "validation_seed",
                    "evaluation_seed": evaluation_seed,
                    "axis": axis,
                    "standardized_beta": beta,
                    "mean_beta": None,
                    "ci_lower": None,
                    "ci_upper": None,
                    "direction_consistent_seeds": None,
                    "episodes": len(seed_rows),
                    "scenario_reference": scenarios[0],
                    "design_rank": int(rank),
                    "design_columns": int(design.shape[1]),
                    "condition_number": condition_number,
                    "bootstrap_iterations": None,
                    "practical_effect_threshold": 0.05,
                    "recognizable_signal": None,
                }
            )
    rng = np.random.default_rng(config.analysis_seed)
    for axis in ("b_over_c", "w", "q"):
        by_seed = gate4_betas.get(axis, {})
        if not by_seed:
            continue
        if set(by_seed) != VALIDATION_SEEDS:
            raise ValueError(f"Gate-4 axis {axis} is missing validation-seed coefficients")
        values = np.asarray([by_seed[seed] for seed in sorted(by_seed)])
        sampled = rng.choice(values, size=(config.bootstrap_iterations, len(values)), replace=True)
        bootstrap_means = sampled.mean(axis=1)
        mean_beta = float(values.mean())
        majority_positive = mean_beta >= 0
        consistent = int(sum((value >= 0) == majority_positive for value in values))
        recognizable = (
            consistent >= 4 and mean_beta >= 0.05 if axis in {"b_over_c", "w"} else None
        )
        gate4_sensitivity_rows.append(
            {
                "row_type": "seed_bootstrap",
                "evaluation_seed": None,
                "axis": axis,
                "standardized_beta": None,
                "mean_beta": mean_beta,
                "ci_lower": float(np.quantile(bootstrap_means, 0.025)),
                "ci_upper": float(np.quantile(bootstrap_means, 0.975)),
                "direction_consistent_seeds": consistent,
                "episodes": len(gate4_rows),
                "scenario_reference": None,
                "design_rank": None,
                "design_columns": None,
                "condition_number": None,
                "bootstrap_iterations": config.bootstrap_iterations,
                "practical_effect_threshold": 0.05,
                "recognizable_signal": recognizable,
            }
        )
    _write_csv(
        output_dir / "gate4_sensitivity.csv",
        gate4_sensitivity_rows,
        (
            "row_type",
            "evaluation_seed",
            "axis",
            "standardized_beta",
            "mean_beta",
            "ci_lower",
            "ci_upper",
            "direction_consistent_seeds",
            "episodes",
            "scenario_reference",
            "design_rank",
            "design_columns",
            "condition_number",
            "bootstrap_iterations",
            "practical_effect_threshold",
            "recognizable_signal",
        ),
    )

    forecasts = []
    forecast_groups = sorted(
        {(row["model"], row["evaluation_seed"], row["suite"]) for row in rounds}, key=str
    )
    for model, evaluation_seed, suite in forecast_groups:
        model_rows = [
            row
            for row in rounds
            if row["model"] == model
            and row["evaluation_seed"] == evaluation_seed
            and row["suite"] == suite
            and row["forecast_target"] not in (None, "")
        ]
        if not model_rows:
            continue
        values = [float(row["forecast"]) for row in model_rows]
        targets = [float(row["forecast_target"]) for row in model_rows]
        sizes = [int(row["group_size"]) for row in model_rows]
        ema_values = [float(row["ema_forecast"]) for row in model_rows]
        decomposition = brier_decomposition(values, targets, sizes)
        model_bs = float(np.mean((np.asarray(values) - np.asarray(targets)) ** 2))
        ema_bs = (
            float(np.mean((np.asarray(ema_values) - np.asarray(targets)) ** 2)) if values else None
        )
        forecasts.append(
            {
                "model": model,
                "evaluation_seed": evaluation_seed,
                "suite": suite,
                **decomposition,
                "fraction_brier_score": model_bs,
                "ema_brier_score": ema_bs,
                "brier_skill_score": 1 - model_bs / ema_bs
                if model_bs is not None and ema_bs not in (None, 0)
                else None,
            }
        )
    _write_csv(
        output_dir / "forecast_skill.csv",
        forecasts,
        (
            "model",
            "evaluation_seed",
            "suite",
            "brier_score",
            "fraction_brier_score",
            "ema_brier_score",
            "brier_skill_score",
            "reliability",
            "resolution",
            "uncertainty",
        ),
    )

    bootstrap_rows = []
    bootstrap_groups = sorted(
        {(row["model"], row["suite"]) for row in episodes},
        key=str,
    )
    for model, suite in bootstrap_groups:
        for metric in ("cooperation_rate", "p_dd", "format_valid", "mean_payoff"):
            seed_values: dict[int, list[float]] = defaultdict(list)
            replication_kinds: set[str] = set()
            for row in episodes:
                if row["model"] != model or row["suite"] != suite or row[metric] in (None, ""):
                    continue
                replication = _replication_seed(row)
                if replication is None:
                    continue
                replication_kind, seed_id = replication
                replication_kinds.add(replication_kind)
                seed_values[int(seed_id)].append(float(row[metric]))
            if not seed_values:
                continue
            point, lower, upper = nested_bootstrap(
                seed_values,
                iterations=config.bootstrap_iterations,
                seed=config.analysis_seed,
            )
            bootstrap_rows.append(
                {
                    "model": model,
                    "suite": suite,
                    "metric": metric,
                    "estimate": point,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "replication_kind": (
                        next(iter(replication_kinds)) if len(replication_kinds) == 1 else "mixed"
                    ),
                    "replication_units": len(seed_values),
                    "bootstrap_iterations": config.bootstrap_iterations,
                }
            )
    _write_csv(
        output_dir / "bootstrap.csv",
        bootstrap_rows,
        (
            "model",
            "suite",
            "metric",
            "estimate",
            "ci_lower",
            "ci_upper",
            "replication_kind",
            "replication_units",
            "bootstrap_iterations",
        ),
    )
    confirmatory = _confirmatory_rows(episodes, config)
    confirmatory_columns = tuple(dict.fromkeys(key for row in confirmatory for key in row))
    _write_csv(output_dir / "confirmatory.csv", confirmatory, confirmatory_columns)
    (output_dir / "confirmatory_decision.json").write_text(
        json.dumps(
            _confirmatory_decision(confirmatory, episodes),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ema_alpha": config.ema_alpha,
                "ema_initial_training_pool_mean": config.ema_initial,
                "bootstrap_iterations": config.bootstrap_iterations,
                "permutation_iterations": config.permutation_iterations,
                "analysis_seed": config.analysis_seed,
                "input_trace_sha256": config.input_trace_sha256,
                "analysis_spec_sha256": config.analysis_spec_sha256,
                "trace_ids": [record["id"] for record in sorted(records, key=lambda x: x["id"])],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {
        name: output_dir / name
        for name in (
            "episodes.csv",
            "rounds.csv",
            "aggregates.csv",
            "diagnostic_cells.csv",
            "sensitivities.csv",
            "gate4_sensitivity.csv",
            "forecast_skill.csv",
            "bootstrap.csv",
            "hkb_stress.csv",
            "confirmatory.csv",
            "confirmatory_decision.json",
            "analysis_manifest.json",
        )
    }


def validate_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-gate4-cohort", action="store_true")
    args = parser.parse_args()
    records = load_jsonl_many(args.traces)
    report = validate_records(records)
    if args.require_gate4_cohort:
        report["gate4_cohort"] = validate_gate4_cohort(records)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text, end="")


def analyze_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ema-initial",
        type=float,
        required=True,
        help="training-pool mean group-cooperation rate; freeze before evaluation",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--permutation-iterations", type=int, default=10_000)
    parser.add_argument("--analysis-seed", type=int, default=730_031)
    parser.add_argument("--analysis-spec", type=Path, default=Path("docs/ANALYSIS_SPEC.md"))
    parser.add_argument("--require-gate4-cohort", action="store_true")
    args = parser.parse_args()
    spec_bytes = args.analysis_spec.read_bytes()
    records = load_jsonl_many(args.traces)
    if args.require_gate4_cohort:
        validate_gate4_cohort(records)
    analyze(
        records,
        args.output_dir,
        AnalysisConfig(
            ema_initial=args.ema_initial,
            bootstrap_iterations=args.bootstrap_iterations,
            permutation_iterations=args.permutation_iterations,
            analysis_seed=args.analysis_seed,
            input_trace_sha256=combined_trace_sha256(args.traces),
            analysis_spec_sha256=hashlib.sha256(spec_bytes).hexdigest(),
        ),
    )
