"""Unique run-directory creation and reproducibility manifests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import socket
import subprocess
import tomllib
from typing import Any, Sequence


TRAINING_SEEDS = {1101, 1102, 1103}
VALIDATION_SEEDS = {2101, 2102, 2103, 2104, 2105}
TEST_SEEDS = {3101, 3102, 3103, 3104, 3105}


def _run(command: Sequence[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            command, cwd=cwd, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def git_snapshot(repo: Path) -> dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"], repo)
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
    ).stdout
    untracked = _run(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--exclude=.cache/**",
            "--exclude=.tmp/**",
            "--exclude=results/**",
            "--exclude=analysis/tables/**",
            "--exclude=analysis/figures/**",
        ],
        repo,
    ).splitlines()
    payload = bytearray(diff)
    for relative in sorted(untracked):
        path = repo / relative
        if path.is_file():
            payload.extend(relative.encode() + b"\0" + path.read_bytes())
    return {
        "commit": commit or None,
        "dirty": bool(payload),
        "dirty_diff_sha256": sha256_bytes(bytes(payload)),
    }


def package_versions() -> dict[str, str | None]:
    packages = ("torch", "transformers", "vllm", "verifiers")
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def hardware_snapshot() -> dict[str, Any]:
    query = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpus": query.splitlines() if query else [],
        "cuda_home": os.environ.get("CUDA_HOME"),
        "nvcc": _run(["nvcc", "--version"]) or None,
        "loaded_modules": os.environ.get("LOADEDMODULES", "").split(":")
        if os.environ.get("LOADEDMODULES")
        else [],
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def create_run(
    *,
    project: Path,
    stage: str,
    variant: str,
    seed: int,
    config: Path,
    output_root: Path,
    command: Sequence[str],
    run_id: str | None = None,
    prime_dir: Path | None = None,
    model_path: Path | None = None,
    tokenizer_path: Path | None = None,
    analysis_spec: Path | None = None,
    seed_role: str = "engineering",
    training_seed: int | None = None,
    evaluation_seed: int | None = None,
    checkpoint_parent: str | None = None,
    wandb_run_id: str | None = None,
) -> Path:
    if not command:
        raise ValueError("run command cannot be empty")
    if seed_role == "training" and (
        training_seed not in TRAINING_SEEDS or evaluation_seed is not None
    ):
        raise ValueError("training runs require a registered training seed only")
    if seed_role == "validation" and (
        evaluation_seed not in VALIDATION_SEEDS or training_seed is not None
    ):
        raise ValueError("validation runs require a registered validation seed")
    if seed_role == "test" and (evaluation_seed not in TEST_SEEDS or training_seed is not None):
        raise ValueError("test runs require a registered test seed")
    config_bytes = config.read_bytes()
    tomllib.loads(config_bytes.decode())
    project_state = git_snapshot(project)
    short = (project_state["commit"] or "nogit")[:7]
    now = datetime.now(timezone.utc)
    run_id = run_id or f"{now:%Y%m%dT%H%M%S%fZ}-{short}-s{seed}"
    run_dir = output_root / stage / variant / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    logs = run_dir / "logs"
    logs.mkdir()
    shutil.copyfile(config, run_dir / "resolved_config.toml")

    prime_state = git_snapshot(prime_dir) if prime_dir and prime_dir.is_dir() else None
    patches = {
        path.name: sha256_file(path) for path in sorted((project / "patches").glob("*.patch"))
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "RUNNING",
        "stage": stage,
        "variant": variant,
        "run_id": run_id,
        "started_at": now.isoformat(),
        "completed_at": None,
        "project": project_state,
        "prime": prime_state,
        "patches_sha256": patches,
        "config": {
            "source": str(config.resolve()),
            "sha256": sha256_bytes(config_bytes),
        },
        "analysis_spec": {
            "path": str(analysis_spec.resolve()) if analysis_spec else None,
            "sha256": sha256_file(analysis_spec),
        },
        "model": {
            "path": str(model_path.resolve()) if model_path else None,
            "config_sha256": sha256_file(model_path / "config.json")
            if model_path and model_path.is_dir()
            else None,
        },
        "tokenizer": {
            "path": str(tokenizer_path.resolve()) if tokenizer_path else None,
            "config_sha256": sha256_file(tokenizer_path / "tokenizer_config.json")
            if tokenizer_path and tokenizer_path.is_dir()
            else None,
        },
        "versions": package_versions(),
        "hardware": hardware_snapshot(),
        "seeds": {
            "role": seed_role,
            "run_seed": seed,
            "training_seed": training_seed,
            "evaluation_seed": evaluation_seed,
        },
        "checkpoint_parent": checkpoint_parent,
        "wandb_run_id": wandb_run_id,
        "command": list(command),
        "exit_code": None,
    }
    write_json(run_dir / "manifest.json", manifest)

    status = "FAILED"
    exit_code = 1
    process: subprocess.Popen[bytes] | None = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def cancel_run(_signum: int, _frame: Any) -> None:
        if process is not None and process.poll() is None:
            process.terminate()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, cancel_run)
    try:
        with (logs / "stdout.log").open("wb") as stdout, (logs / "stderr.log").open("wb") as stderr:
            process = subprocess.Popen(command, cwd=project, stdout=stdout, stderr=stderr)
            exit_code = process.wait()
        status = "COMPLETED" if exit_code == 0 else "FAILED"
    except KeyboardInterrupt:
        status = "CANCELLED"
        exit_code = 130
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        manifest["status"] = status
        manifest["exit_code"] = exit_code
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(run_dir / "manifest.json", manifest)
        (run_dir / "STATUS").write_text(status + "\n")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--stage", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--run-id")
    parser.add_argument("--prime-dir", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--analysis-spec", type=Path, default=Path("docs/ANALYSIS_SPEC.md"))
    parser.add_argument(
        "--seed-role",
        choices=("engineering", "training", "validation", "test"),
        default="engineering",
    )
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--evaluation-seed", type=int)
    parser.add_argument("--checkpoint-parent")
    parser.add_argument("--wandb-run-id")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    run_dir = create_run(
        project=args.project.resolve(),
        stage=args.stage,
        variant=args.variant,
        seed=args.seed,
        config=args.config.resolve(),
        output_root=args.output_root.resolve(),
        command=command,
        run_id=args.run_id,
        prime_dir=args.prime_dir,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        analysis_spec=args.analysis_spec,
        seed_role=args.seed_role,
        training_seed=args.training_seed,
        evaluation_seed=args.evaluation_seed,
        checkpoint_parent=args.checkpoint_parent,
        wandb_run_id=args.wandb_run_id,
    )
    status = (run_dir / "STATUS").read_text().strip()
    print(run_dir)
    raise SystemExit(0 if status == "COMPLETED" else 1)


if __name__ == "__main__":
    main()
