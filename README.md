# Nowak-coordination

For the current project state, mandatory next gate, scientific completion
criteria, and verifiable end-to-end execution path, start with
[`docs/PROJECT_EXECUTION_PLAN.md`](docs/PROJECT_EXECUTION_PLAN.md).

For cluster paths, reusable environments, PRIME launch commands, implementation
notes, and known Qwen/FP8 fixes, use
[`docs/CLUSTER_RL_RUNBOOK.md`](docs/CLUSTER_RL_RUNBOOK.md).

## Environment

This project uses Python 3.12 and `uv`. The local model cache contains the rollout
target at `/home/memoozd/scratch/models/Qwen3.6-35B-A3B-FP8`.

```bash
uv sync
source .venv/bin/activate
```

The GPU group installs the current Transformers/vLLM/Torch stack required by the
Qwen3.6-MoE checkpoint. Run it from an allocated H100 node:

```bash
MODEL_DIR=/home/memoozd/scratch/models/Qwen3.6-35B-A3B-FP8 \
  ./scripts/setup_gpu_env.sh
```

That script also clones and installs PRIME-RL at
`/home/memoozd/scratch/rl/prime-rl`, initializes its submodules, and verifies
CUDA, Transformers, vLLM, and the local model config.

## Local validation

Run the unit suite and the deterministic 20-episode CPU smoke evaluation:

```bash
uv run pytest -q
uv run python -m nowak_coordination.smoke --episodes 20
```

The environment integration uses the composable Verifiers v1 API: a typed
taskset, task, rollout state, and colocated user simulator.
