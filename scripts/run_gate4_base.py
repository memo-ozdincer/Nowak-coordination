"""Launch and fail-closed validate the five-block Gate-4 base characterization."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from nowak_coordination.run_manifest import create_run


VALIDATION_SEEDS = tuple(range(2101, 2106))


def _checked(command: list[str], *, cwd: Path, stdout: Path | None = None) -> None:
    if stdout is None:
        subprocess.run(command, cwd=cwd, check=True)
        return
    with stdout.open("w") as handle:
        subprocess.run(command, cwd=cwd, check=True, stdout=handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument(
        "--analysis-python",
        type=Path,
        help="Python with the project analysis dependencies (default: PROJECT/.venv/bin/python)",
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--prime-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        choices=VALIDATION_SEEDS,
        default=list(VALIDATION_SEEDS),
        help="Validation blocks to generate; use with --existing-trace to resume safely",
    )
    parser.add_argument(
        "--existing-trace",
        type=Path,
        action="append",
        default=[],
        help="Previously completed Gate-4 trace block to include in final cohort validation",
    )
    args = parser.parse_args()

    project = args.project.resolve()
    # Do not resolve interpreter symlinks: doing so strips virtualenv context.
    runtime_python = args.runtime_python.absolute()
    analysis_python = (
        args.analysis_python.absolute()
        if args.analysis_python
        else project / ".venv/bin/python"
    )
    output_root = args.output_root.resolve()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains a duplicate validation seed")
    trace_paths = [path.resolve() for path in args.existing_trace]
    if any(not path.is_file() for path in trace_paths):
        raise FileNotFoundError("every --existing-trace must be an existing file")
    run_dirs: list[Path] = [path.parent for path in trace_paths]
    for evaluation_seed in args.seeds:
        config = project / f"configs/eval_gate4_base_s{evaluation_seed}.toml"
        command = [
            str(runtime_python),
            "-m",
            "nowak_coordination.seeded_eval",
            str(config),
            "--output-dir",
            "{RUN_DIR}",
        ]
        # create_run must know the final output path before starting its child.
        run_id = (
            f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-"
            f"gate4-base-s{evaluation_seed}"
        )
        run_dir = (
            output_root
            / "gate4"
            / "base_characterization"
            / run_id
        )
        command[-1] = str(run_dir)
        created = create_run(
            project=project,
            stage="gate4",
            variant="base_characterization",
            seed=evaluation_seed,
            config=config,
            output_root=output_root,
            command=command,
            run_id=run_id,
            prime_dir=args.prime_dir.resolve(),
            model_path=args.model_path.resolve(),
            tokenizer_path=args.tokenizer_path.resolve(),
            analysis_spec=project / "docs/ANALYSIS_SPEC.md",
            seed_role="validation",
            model_arm="Base",
            evaluation_seed=evaluation_seed,
            runtime_python=runtime_python,
        )
        if (created / "STATUS").read_text().strip() != "COMPLETED":
            raise RuntimeError(f"Gate-4 block failed: {created}")
        trace_paths.append(created / "traces.jsonl")
        run_dirs.append(created)

    cohort = (
        output_root
        / "gate4"
        / "base_characterization_cohort"
        / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    )
    cohort.mkdir(parents=True, exist_ok=False)
    trace_args = [str(path) for path in trace_paths]
    _checked(
        [
            str(analysis_python),
            str(project / "scripts/validate_traces.py"),
            *trace_args,
            "--require-gate4-cohort",
            "--report",
            str(cohort / "validation_report.json"),
        ],
        cwd=project,
    )
    _checked(
        [
            str(analysis_python),
            str(project / "scripts/analyze.py"),
            *trace_args,
            "--output-dir",
            str(cohort / "analysis"),
            "--ema-initial",
            "0.5",
            "--require-gate4-cohort",
        ],
        cwd=project,
    )
    (cohort / "cohort_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "COMPLETED",
                "formal_gate4": True,
                "required_registry_flag": "--require-gate4-cohort",
                "ema_initial": 0.5,
                "analysis_python": str(analysis_python),
                "generation_python": str(runtime_python),
                "validation_seeds": list(VALIDATION_SEEDS),
                "generated_validation_seeds": list(args.seeds),
                "run_directories": [str(path) for path in run_dirs],
                "trace_paths": [str(path) for path in trace_paths],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(cohort)


if __name__ == "__main__":
    main()
