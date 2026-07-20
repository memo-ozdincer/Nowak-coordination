from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from nowak_coordination.run_manifest import create_run
from nowak_coordination.trace_analysis import (
    AnalysisConfig,
    TraceValidationError,
    analyze,
    brier_decomposition,
    episode_metrics,
    holm_adjust,
    load_jsonl,
    validate_records,
)


PROJECT = Path(__file__).parents[1]
FIXTURE = PROJECT / "analysis/fixtures/synthetic_traces.jsonl"
TABLE_SNAPSHOTS = {
    "aggregates.csv": "04309fd19251c11bb667ab5967852db279708ec93b5530af55bb61afbb478206",
    "analysis_manifest.json": "9121b192dcc424c41db0df607d7a3c6e958fba0876c55bfdc10ae73e3431f6ed",
    "bootstrap.csv": "0eb3da389363c82f6ccf7023e7b55e3695594d5739e32640fa6b61b1df64fb61",
    "confirmatory.csv": "d9049e3de13c3dc6065795d3f7f8502326ded36930cc198be43d86aac922bd51",
    "episodes.csv": "1af420abe6d4a9aee3f9e6386ea1b1991d1f451f3d3e3cae7f7f38dc2d23c496",
    "forecast_skill.csv": "9e210dcfe25687a045ab9c262a4cbb49bb79e1dad1847cba17a64bdd9a7eebe6",
    "hkb_stress.csv": "90d65851053faeb440ec8f6a9e4f57d367dfadd281fc9c39717c9af3b7dcda59",
    "rounds.csv": "d178ceaf466eae949b6a604b8da49253dc5fbaf4ac45d79a18a6786b11855ad4",
    "sensitivities.csv": "4c7fe595f846a272c9d6dc6ee0b7235a2b9f2b6635b22c36dfb4963bbd2aea96",
    "validation.json": "75d87529c646a42916ac35950ec99433a8e36b96c7978af11ccbe32357247201",
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
    second.update(role="validation", training_seed=None, evaluation_seed=2101)
    second["episode_seed"] = records[0]["info"]["coordination_trace"]["trace_header"][
        "seed_metadata"
    ]["episode_seed"]
    records[1]["info"]["coordination_trace"]["trace_header"]["policy_split"] = "held_out"
    with pytest.raises(TraceValidationError, match="leaked across splits"):
        validate_records(records)


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
        "sensitivities.csv",
        "forecast_skill.csv",
        "bootstrap.csv",
        "hkb_stress.csv",
        "confirmatory.csv",
        "analysis_manifest.json",
    }
    for name in first:
        assert first[name].read_bytes() == second[name].read_bytes()
    with first["confirmatory.csv"].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 33
    assert {row["status"] for row in rows} == {
        "ESTIMATED",
        "INSUFFICIENT_REPLICATION",
        "UNAVAILABLE",
    }
    assert all(row["holm_p"] for row in rows if row["status"] == "ESTIMATED")


def test_analyzer_uses_evaluation_seeds_as_base_replications(tmp_path: Path) -> None:
    records = deepcopy(load_jsonl(FIXTURE)[:2])
    for index, record in enumerate(records):
        header = record["info"]["coordination_trace"]["trace_header"]
        header["reward_model"] = "Base"
        header["policy_split"] = "heldout"
        header["seed_metadata"].update(
            role="test",
            training_seed=None,
            evaluation_seed=3101 + index,
        )
    paths = analyze(
        records,
        tmp_path,
        AnalysisConfig(bootstrap_iterations=10, permutation_iterations=10),
    )
    bootstrap = list(csv.DictReader(paths["bootstrap.csv"].open(newline="")))
    assert bootstrap
    assert {row["replication_kind"] for row in bootstrap} == {"evaluation_seed"}
    assert {row["replication_units"] for row in bootstrap} == {"2"}


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
