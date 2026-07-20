from __future__ import annotations

import asyncio
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from nowak_coordination.environment import (
    DonorState,
    DonorTaskConfig,
    DonorTaskset,
    DonorTasksetConfig,
    DonorUser,
)
from nowak_coordination.gate4_registry import PARAMETER_CELLS, gate4_assignment
from nowak_coordination.run_manifest import create_run
from nowak_coordination.trace_analysis import (
    AnalysisConfig,
    TraceValidationError,
    _confirmatory_decision,
    _confirmatory_rows,
    analyze,
    brier_decomposition,
    episode_metrics,
    holm_adjust,
    load_jsonl,
    validate_gate4_cohort,
    validate_records,
)


PROJECT = Path(__file__).parents[1]
FIXTURE = PROJECT / "analysis/fixtures/synthetic_traces.jsonl"
TABLE_SNAPSHOTS = {
    "aggregates.csv": "acd55d828f62faaa728a5052902b574cf015c349301b25a390e6f97136451a7d",
    "analysis_manifest.json": "ea2b20d8a71875c219378fff1b093a5681b53835aa92dce7ef859deced2b0c69",
    "bootstrap.csv": "f5b966f5d483451e79ab989c1e76421c136257ae7d3af513008a2abd6dfa889c",
    "confirmatory.csv": "75c6564bfdef348d890da6bd3b267e311ac766fd5ad425a36ddef0dc6653577b",
    "confirmatory_decision.json": "3a5ad2f4a5f7f9fd99cc5134d104ae81e000646eb5a060c1b1bd9b7d96bddc9b",
    "diagnostic_cells.csv": "c593d0ed32aed8024fc64179e9704847db3d671bcb91de8df0e5a2fa9884dc62",
    "episodes.csv": "0565ea60eac7c51da51b49d0f4a441799e6001b8f41341d1560a8c72e30af956",
    "forecast_skill.csv": "c355c396a5cf0468fd17145c3e2339f5906e76cfcdb7d07b2924450f9acfb7da",
    "gate4_sensitivity.csv": "081044237021c0c30803f67f3dea507115553ea24c010d8e6593a1b0e4c0f2f7",
    "hkb_stress.csv": "baa7474a251ca7e968b006671fa886dc8c58b0505479147dfc32e792c996c1f5",
    "rounds.csv": "d7f96cd15095ff16c1ee57c1c0685063a0ab1da3040d651444621c47f284f839",
    "sensitivities.csv": "fbc90ae0f4a6429988249dd85df5fb2e2e02e42cf382e2328419e541d626021a",
    "validation.json": "75d87529c646a42916ac35950ec99433a8e36b96c7978af11ccbe32357247201",
}


def add_sampler_evidence(record: dict) -> None:
    header = record["info"]["coordination_trace"]["trace_header"]
    sampling = header["sampling_metadata"]
    requested = sampling["requested_seed"]
    record["agent"] = {
        "sampling": {
            "temperature": sampling.get("temperature"),
            "top_p": sampling.get("top_p"),
            "seed": requested,
            "chat_template_kwargs": {"enable_thinking": sampling.get("enable_thinking")},
        }
    }
    record["info"]["sampler_seed_evidence"] = {
        "requested_seed": requested,
        "effective_seed": requested,
        "transport": "verifiers.v1 EvalClient -> OpenAI chat request -> vLLM",
        "trace_agent_sampling_recorded": True,
    }


def test_known_answer_fixture_and_episode_metrics() -> None:
    records = load_jsonl(FIXTURE)
    assert validate_records(records) == {
        "status": "PASS",
        "traces": 12,
        "rounds": 120,
        "episode_ids_unique": True,
        "trace_ids_unique": True,
    }
    first = episode_metrics(records[0])
    assert first["cooperation_rate"] == pytest.approx(0.7)
    assert first["mean_payoff"] == pytest.approx(1.1)
    assert (first["p_cc"], first["p_cd"], first["p_dc"], first["p_dd"]) == pytest.approx(
        (0.6, 0.1, 0.3, 0.0)
    )
    assert first["niceness"] == 1
    assert first["forgiveness_within_3"] == 1
    assert first["retaliation_length"] == 1
    assert first["oracle_regret"] == pytest.approx(0.5)
    assert first["nonexploitability_vs_safe_defect"] == pytest.approx(0.25)
    assert first["forecast_entropy"] == 0.0
    assert first["total_reward_variance"] > 0


def test_committed_table_snapshots() -> None:
    table_dir = PROJECT / "analysis/tables/synthetic"
    assert {path.name for path in table_dir.iterdir()} == set(TABLE_SNAPSHOTS)
    for name, expected in TABLE_SNAPSHOTS.items():
        assert hashlib.sha256((table_dir / name).read_bytes()).hexdigest() == expected
    manifest = json.loads((table_dir / "analysis_manifest.json").read_text())
    assert manifest["input_trace_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert (
        manifest["analysis_spec_sha256"]
        == hashlib.sha256((PROJECT / "docs/ANALYSIS_SPEC.md").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda records: records.append(deepcopy(records[0])), "duplicate trace ID"),
        (
            lambda records: records[0]["info"]["coordination_trace"]["rounds"].pop(3),
            "missing or reordered turns",
        ),
        (
            lambda records: records[0]["info"]["coordination_trace"]["rounds"][0].update(
                forecast=float("nan")
            ),
            "non-finite",
        ),
        (
            lambda records: records[0]["analysis_targets"].pop("oracle_mean_payoff_provenance"),
            "lacks counterfactual replay provenance",
        ),
        (
            lambda records: records[0]["info"]["coordination_trace"]["terminal_event"].update(
                complete=False
            ),
            "missing complete terminal event",
        ),
        (
            lambda records: records[0]["info"]["coordination_trace"]["rounds"][0].update(
                joint_outcomes=["DD"]
            ),
            "outcome/action mismatch",
        ),
        (
            lambda records: records[0]["info"]["coordination_trace"]["rounds"][0].update(
                focal_payoff=999
            ),
            "focal payoff mismatch",
        ),
    ],
)
def test_validator_rejects_corruption(mutation, message: str) -> None:
    records = deepcopy(load_jsonl(FIXTURE))
    mutation(records)
    with pytest.raises(TraceValidationError, match=message):
        validate_records(records)


def test_validator_rejects_seed_leakage() -> None:
    records = deepcopy(load_jsonl(FIXTURE))
    second = records[1]["info"]["coordination_trace"]["trace_header"]["seed_metadata"]
    second.update(
        role="validation",
        training_seed=None,
        evaluation_seed=2101,
        checkpoint_training_seed=1102,
    )
    second["episode_seed"] = records[0]["info"]["coordination_trace"]["trace_header"][
        "seed_metadata"
    ]["episode_seed"]
    records[1]["info"]["coordination_trace"]["trace_header"]["policy_split"] = "held_out"
    add_sampler_evidence(records[1])
    with pytest.raises(TraceValidationError, match="leaked across splits"):
        validate_records(records)


def test_validator_requires_effective_sampler_seed_evidence() -> None:
    record = deepcopy(load_jsonl(FIXTURE)[0])
    header = record["info"]["coordination_trace"]["trace_header"]
    header["policy_split"] = "heldout"
    header["seed_metadata"].update(
        role="validation",
        training_seed=None,
        evaluation_seed=2101,
        checkpoint_training_seed=1101,
    )
    header["sampling_metadata"].update(temperature=0.7, top_p=1.0, requested_seed=2101)
    record["agent"] = {"sampling": {"temperature": 0.7, "top_p": 1.0, "seed": 2101}}
    record["info"]["sampler_seed_evidence"] = {
        "requested_seed": 2101,
        "effective_seed": 2101,
        "transport": "verifiers.v1 EvalClient -> OpenAI chat request -> vLLM",
        "trace_agent_sampling_recorded": True,
    }
    assert validate_records([record])["status"] == "PASS"
    record["agent"]["sampling"]["seed"] = 2102
    with pytest.raises(TraceValidationError, match="sampler seed mismatch"):
        validate_records([record])


def test_gate4_cohort_validator_enforces_exact_registry() -> None:
    records = []
    ordinal = 0
    for evaluation_seed in range(2101, 2106):
        for cell_index, (b, w, q) in enumerate(PARAMETER_CELLS):
            assignment = gate4_assignment(evaluation_seed, cell_index)
            ordinal += 1
            records.append(
                {
                    "id": f"gate4-{ordinal}",
                    "task": {
                        "data": {
                            "episode": {
                                "b": b,
                                "w": w,
                                "q": q,
                                "partner_policy": assignment.partner_policy,
                                "mode": assignment.mode,
                                "policy_split": assignment.policy_split,
                                "group_size": assignment.group_size,
                            }
                        }
                    },
                    "analysis_targets": {
                        "registry": "gate4_base_characterization_v1",
                        "scenario": assignment.scenario,
                        "suite": assignment.suite,
                    },
                    "info": {
                        "coordination_trace": {
                            "trace_header": {"seed_metadata": {"evaluation_seed": evaluation_seed}}
                        }
                    },
                }
            )
    assert validate_gate4_cohort(records)["status"] == "PASS"
    with pytest.raises(TraceValidationError, match="exactly"):
        validate_gate4_cohort(records[:-1])


def test_gate4_trace_validator_enforces_live_registry_contract() -> None:
    task = DonorTaskset(
        DonorTasksetConfig(
            id="local/donor",
            registry="gate4_base_characterization",
            num_tasks=100,
            seed=4_210_100,
            horizon_min=10,
            horizon_max=10,
            policy_split="heldout",
            policy_arm="Base",
            seed_role="validation",
            evaluation_seed=2101,
            sampling_seed=2101,
            sampling_temperature=0.7,
            sampling_top_p=1.0,
            sampling_enable_thinking=False,
            task=DonorTaskConfig(model="A"),
        )
    ).load()[0]
    user = DonorUser(task.config.user)
    user._inert_state = DonorState()
    asyncio.run(user.setup_task(task.data))
    for _ in range(10):
        asyncio.run(user.respond("ACTION: COOPERATE\nFORECAST_GROUP_COOP: 0.50"))
    trace = type("Trace", (), {"state": user.state, "info": {}})()
    asyncio.run(task.episode_reward(trace))
    record = {
        "id": "gate4-live-contract",
        "is_completed": True,
        "errors": [],
        "task": {"data": task.data.model_dump(mode="json")},
        "agent": {
            "sampling": {
                "temperature": 0.7,
                "top_p": 1.0,
                "seed": 2101,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        },
        "info": {
            **trace.info,
            "sampler_seed_evidence": {
                "requested_seed": 2101,
                "effective_seed": 2101,
                "transport": "verifiers.v1 EvalClient -> OpenAI chat request -> vLLM",
                "trace_agent_sampling_recorded": True,
            },
        },
    }
    assert validate_records([record])["status"] == "PASS"
    record["task"]["data"]["episode"]["seed"] += 1
    record["info"]["coordination_trace"]["trace_header"]["seed_metadata"]["episode_seed"] += 1
    with pytest.raises(TraceValidationError, match="episode seed drifted"):
        validate_records([record])


def test_forecast_decomposition_expands_fractional_group_observations() -> None:
    result = brier_decomposition([0.5], [0.5], [4])
    assert result == pytest.approx(
        {
            "brier_score": 0.25,
            "reliability": 0.0,
            "resolution": 0.0,
            "uncertainty": 0.25,
        }
    )
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_analyzer_is_byte_deterministic_and_emits_every_table(tmp_path: Path) -> None:
    records = load_jsonl(FIXTURE)
    config = AnalysisConfig(
        ema_initial=0.5,
        bootstrap_iterations=100,
        permutation_iterations=100,
        analysis_seed=17,
    )
    first = analyze(records, tmp_path / "first", config)
    second = analyze(records, tmp_path / "second", config)
    assert set(first) == {
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
    }
    for name in first:
        assert first[name].read_bytes() == second[name].read_bytes()
    with first["confirmatory.csv"].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {row["status"] for row in rows} == {"UNAVAILABLE"}
    diagnostics = list(csv.DictReader(first["diagnostic_cells.csv"].open(newline="")))
    assert diagnostics
    assert all(
        row["partner_adaptivity"] in {"adaptive", "nonadaptive", "mixed"} for row in diagnostics
    )
    assert json.loads(first["confirmatory_decision.json"].read_text())["status"] == (
        "NOT_EVALUABLE"
    )


def test_incomplete_primary_cohort_cannot_pass(tmp_path: Path) -> None:
    source = {}
    for record in load_jsonl(FIXTURE):
        source.setdefault(episode_metrics(record)["model"], record)
    seed_sets = {
        "A": range(1101, 1106),
        "B": range(1201, 1206),
        "E": range(1501, 1506),
    }
    records = []
    ordinal = 0
    for model, seeds in seed_sets.items():
        for seed in seeds:
            ordinal += 1
            record = deepcopy(source[model])
            trace_id = f"primary-{ordinal:02d}"
            episode_id = f"primary-{model}-{seed}"
            record["id"] = trace_id
            record["analysis_targets"]["suite"] = "recovery"
            record["task"]["data"]["episode"].update(
                b=3.0,
                c=1.0,
                w=1.0,
                q=0.0,
                noise_rate=0.0,
            )
            state = record["info"]["coordination_trace"]
            header = state["trace_header"]
            header["episode_id"] = episode_id
            header["mode"] = "dyadic"
            header["policy_split"] = "heldout"
            header["sampling_metadata"] = {
                "temperature": 0.7,
                "top_p": 1.0,
                "enable_thinking": False,
                "requested_seed": 200_000 + ordinal,
            }
            header["seed_metadata"].update(
                role="test",
                training_seed=None,
                evaluation_seed=3101 + ((ordinal - 1) % 5),
                checkpoint_training_seed=seed,
                episode_seed=200_000 + ordinal,
            )
            state["terminal_event"]["episode_id"] = episode_id
            for event in state["rounds"]:
                event["episode_id"] = episode_id
                event["partner_policy"] = "generous_tit_for_tat"
                event["partner_adaptive"] = True
                if event["round_index"] == 5:
                    event["partner_intended_actions"] = ["C"]
            if model == "B":
                perturbed = state["rounds"][4]
                perturbed.update(
                    focal_intended_action="C",
                    focal_executed_action="C",
                    joint_outcomes=["CD"],
                )
            for event in state["rounds"]:
                payoff = 3.0 * (event["partner_executed_actions"] == ["C"]) - (
                    event["focal_executed_action"] == "C"
                )
                event["focal_payoff"] = payoff
                event["reward"].update(payoff=payoff, total=payoff)
            mean_payoff = sum(event["focal_payoff"] for event in state["rounds"]) / 10
            record["analysis_targets"]["safe_defect_mean_payoff"] = mean_payoff - 0.25
            record["analysis_targets"]["oracle_mean_payoff"] = mean_payoff + 0.5
            records.append(record)
    for source_record in list(records):
        model = source_record["info"]["coordination_trace"]["trace_header"]["reward_model"]
        if model not in {"A", "B"}:
            continue
        ordinal += 1
        record = deepcopy(source_record)
        state = record["info"]["coordination_trace"]
        episode_id = f"exploit-{model}-{ordinal}"
        record["id"] = f"exploit-{ordinal:02d}"
        record["analysis_targets"]["suite"] = "exploitability"
        state["trace_header"]["episode_id"] = episode_id
        state["trace_header"]["seed_metadata"]["episode_seed"] = 300_000 + ordinal
        state["terminal_event"]["episode_id"] = episode_id
        for event in state["rounds"]:
            event["episode_id"] = episode_id
            event["partner_policy"] = "opportunist"
            event["partner_adaptive"] = True
        records.append(record)
    for record in records:
        add_sampler_evidence(record)
    paths = analyze(
        records,
        tmp_path,
        AnalysisConfig(bootstrap_iterations=100, permutation_iterations=10_000),
    )
    rows = list(csv.DictReader(paths["confirmatory.csv"].open(newline="")))
    efficacy = [row for row in rows if row["family"] == "efficacy"]
    safety = [row for row in rows if row["family"] == "safety"]
    assert [float(row["raw_p"]) for row in efficacy] == pytest.approx([2 / 252, 2 / 252])
    assert [float(row["holm_p"]) for row in efficacy] == pytest.approx([4 / 252, 4 / 252])
    assert all(float(row["ci_lower"]) > 0 for row in efficacy)
    assert {row["decision"] for row in safety} == {"", "PASS"}
    decision = json.loads(paths["confirmatory_decision.json"].read_text())
    assert decision["status"] == "NOT_EVALUABLE"
    assert decision["reason"] == "confirmatory cohort contract failed"


def test_exact_complete_primary_cohort_can_reach_registered_pass() -> None:
    episodes = []
    for model, seeds, recovery in (
        ("A", range(1101, 1106), 0.0),
        ("B", range(1201, 1206), 1.0),
        ("E", range(1501, 1506), 0.0),
    ):
        for training_seed in seeds:
            for evaluation_seed in range(3101, 3106):
                for _ in range(20):
                    episodes.append(
                        {
                            "model": model,
                            "training_seed": training_seed,
                            "evaluation_seed": evaluation_seed,
                            "suite": "recovery",
                            "partner_policy": "generous_tit_for_tat",
                            "lock_type": "CC",
                            "recovered_within_3": recovery,
                            "p_dd": 0.0,
                            "format_valid": 1.0,
                        }
                    )
    for model, seeds in (("A", range(1101, 1106)), ("B", range(1201, 1206))):
        for training_seed in seeds:
            for evaluation_seed in range(3101, 3106):
                for policy in ("always_defect", "opportunist"):
                    for _ in range(10):
                        episodes.append(
                            {
                                "model": model,
                                "training_seed": training_seed,
                                "evaluation_seed": evaluation_seed,
                                "suite": "exploitability",
                                "partner_policy": policy,
                                "lock_type": "DD",
                                "nonexploitability_vs_safe_defect": 0.0,
                            }
                        )
    config = AnalysisConfig(bootstrap_iterations=100, permutation_iterations=1)
    rows = _confirmatory_rows(episodes, config)
    efficacy = [row for row in rows if row["family"] == "efficacy"]
    assert {row["permutation_method"] for row in efficacy} == {"exact_252_assignments"}
    assert [row["raw_p"] for row in efficacy] == pytest.approx([2 / 252, 2 / 252])
    assert _confirmatory_decision(rows, episodes)["status"] == "PASS"

    harmed = deepcopy(episodes)
    for row in harmed:
        if (
            row["model"] == "B"
            and row["suite"] == "exploitability"
            and row["partner_policy"] == "opportunist"
        ):
            row["nonexploitability_vs_safe_defect"] = -1.0
    harmed_rows = _confirmatory_rows(harmed, config)
    opportunist = next(row for row in harmed_rows if row.get("partner_policy") == "opportunist")
    always_defect = next(row for row in harmed_rows if row.get("partner_policy") == "always_defect")
    assert opportunist["decision"] == "FAIL"
    assert always_defect["decision"] == "PASS"
    assert _confirmatory_decision(harmed_rows, harmed)["status"] == "FAIL"


def test_analyzer_uses_evaluation_seeds_as_base_replications(tmp_path: Path) -> None:
    records = deepcopy(load_jsonl(FIXTURE)[:2])
    for index, record in enumerate(records):
        header = record["info"]["coordination_trace"]["trace_header"]
        header["policy_arm"] = "Base"
        header["policy_split"] = "heldout"
        header["seed_metadata"].update(
            role="test",
            training_seed=None,
            evaluation_seed=3101 + index,
        )
        add_sampler_evidence(record)
    paths = analyze(
        records,
        tmp_path,
        AnalysisConfig(bootstrap_iterations=10, permutation_iterations=10),
    )
    bootstrap = list(csv.DictReader(paths["bootstrap.csv"].open(newline="")))
    assert bootstrap
    assert {row["replication_kind"] for row in bootstrap} == {"evaluation_seed"}
    assert {row["replication_units"] for row in bootstrap} == {"1"}
    assert {row["suite"] for row in bootstrap} == {"repeated_2x2", "hkb_lock"}


def test_analyzer_handles_all_invalid_format_traces(tmp_path: Path) -> None:
    record = deepcopy(load_jsonl(FIXTURE)[0])
    state = record["info"]["coordination_trace"]
    state.update(rounds=[], invalid_output=True, terminal_reason="invalid_format")
    state["terminal_event"]["rounds_completed"] = 0
    paths = analyze(
        [record],
        tmp_path,
        AnalysisConfig(bootstrap_iterations=10, permutation_iterations=10),
    )
    assert paths["rounds.csv"].read_text().count("\n") == 1
    assert list(csv.DictReader(paths["forecast_skill.csv"].open(newline=""))) == []


def test_run_manifest_success_failure_and_overwrite_refusal(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[run]\nname = "fixture"\n')
    common = {
        "project": PROJECT,
        "stage": "test",
        "variant": "fixture",
        "seed": 1101,
        "config": config,
        "output_root": tmp_path / "results",
        "run_id": "fixed",
        "analysis_spec": PROJECT / "docs/ANALYSIS_SPEC.md",
        "seed_role": "training",
        "model_arm": "A",
        "training_seed": 1101,
    }
    run_dir = create_run(command=[sys.executable, "-c", "print('ok')"], **common)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert (run_dir / "STATUS").read_text() == "COMPLETED\n"
    assert manifest["status"] == "COMPLETED"
    assert manifest["exit_code"] == 0
    assert (run_dir / "logs/stdout.log").read_text() == "ok\n"
    assert (run_dir / "resolved_config.toml").read_bytes() == config.read_bytes()
    with pytest.raises(FileExistsError):
        create_run(command=[sys.executable, "-c", "pass"], **common)

    failed = create_run(
        command=[sys.executable, "-c", "raise SystemExit(7)"],
        **{**common, "run_id": "failed"},
    )
    failed_manifest = json.loads((failed / "manifest.json").read_text())
    assert (failed / "STATUS").read_text() == "FAILED\n"
    assert failed_manifest["exit_code"] == 7


def test_run_manifest_rejects_seed_role_mismatch(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[run]\nname = "fixture"\n')
    with pytest.raises(ValueError, match="registered validation seed"):
        create_run(
            project=PROJECT,
            stage="test",
            variant="fixture",
            seed=1101,
            config=config,
            output_root=tmp_path / "results",
            command=[sys.executable, "-c", "pass"],
            seed_role="validation",
            evaluation_seed=3101,
        )


def test_run_manifest_records_sigterm_cancellation(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[run]\nname = "cancel"\n')
    output = tmp_path / "results"
    command = [
        sys.executable,
        "-m",
        "nowak_coordination.run_manifest",
        "--project",
        str(PROJECT),
        "--stage",
        "test",
        "--variant",
        "cancel",
        "--seed",
        "1",
        "--config",
        str(config),
        "--output-root",
        str(output),
        "--run-id",
        "cancelled",
        "--",
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
    ]
    process = subprocess.Popen(command, cwd=PROJECT)
    manifest = output / "test/cancel/cancelled/manifest.json"
    for _ in range(100):
        if manifest.exists():
            break
        time.sleep(0.05)
    assert manifest.exists()
    time.sleep(0.1)
    process.terminate()
    assert process.wait(timeout=15) == 1
    run_dir = manifest.parent
    assert (run_dir / "STATUS").read_text() == "CANCELLED\n"
    assert json.loads(manifest.read_text())["exit_code"] == 130
