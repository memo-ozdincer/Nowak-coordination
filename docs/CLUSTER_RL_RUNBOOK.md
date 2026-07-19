# Cluster RL runbook

## Paths and environments

`/home/memoozd/scratch` resolves to `/scratch/memoozd`; either spelling reaches
the same storage. Current paths:

| Item | Path |
|---|---|
| Project | `/scratch/memoozd/rl/Nowak-coordination` |
| PRIME-RL | `/home/memoozd/scratch/rl/prime-rl` |
| Model | `/home/memoozd/scratch/models/Qwen3.6-35B-A3B-FP8` |
| Pilot config | `configs/train_model_a_pilot.toml` |
| Pilot output | `results/model_a_pilot` |

There are two venvs with different jobs:

- `Nowak-coordination/.venv`: project development, CPU tests, and smoke
  evaluations. Create/update with `uv sync`.
- `prime-rl/.venv`: PRIME training and its exact Torch/vLLM/Transformers stack.
  Reuse this for new local Verifiers tasksets instead of rebuilding the GPU
  stack.

Install this or another taskset editable into the PRIME venv:

```bash
PROJECT=/scratch/memoozd/rl/Nowak-coordination
PRIME=/home/memoozd/scratch/rl/prime-rl
uv pip install --python "$PRIME/.venv/bin/python" --no-deps -e "$PROJECT"
source "$PRIME/.venv/bin/activate"
```

Run `rl` only after activating the PRIME venv: the launcher starts
`inference`, `orchestrator`, and `torchrun` by name from `PATH`.

Verified PRIME venv snapshot (2026-07-17): Python 3.12.13, Torch
2.11.0+cu128, Transformers 5.6.2, vLLM 0.24.0+cu129, Verifiers
0.2.1.dev47, uv 0.11.29, PRIME commit `5f7e3ffca`. Treat these as a known
working snapshot, not version requirements.

`scripts/setup_gpu_env.sh` is a bootstrap script for a fresh checkout. It
fetches/checks out PRIME `origin/main`; do not use it as a routine activation
script, and re-check the local PRIME patch below after any update. On shared
storage, prefer `UV_LINK_MODE=symlink uv sync ...` when rebuilding.

## Validate and launch

Fast project checks:

```bash
uv run pytest -q
uv run python -m nowak_coordination.smoke --episodes 20
```

Training checks and launch:

```bash
source /home/memoozd/scratch/rl/prime-rl/.venv/bin/activate
rl @ configs/train_model_a_pilot.toml --dry-run

# Put only WANDB_API_KEY=... in this gitignored file; never commit it.
set -a
source .secrets/wandb.env
set +a

CUDA_VISIBLE_DEVICES=0,1,2,3 \
XDG_CACHE_HOME="$PWD/.cache/xdg" \
TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/torchinductor" \
TMPDIR="$PWD/.tmp" \
rl @ configs/train_model_a_pilot.toml
```

Use `tmux` for a persistent run. Do not place API keys directly in the tmux
command because they then appear in process listings. The current pilot uses
two trainer GPUs and two tensor-parallel inference GPUs.

Useful diagnostics:

```bash
nvidia-smi
tail -f results/model_a_pilot/logs/orchestrator.log
tail -f results/model_a_pilot/logs/inference.log
tail -f results/model_a_pilot/logs/envs/train/donor-model-a.log
rg -n -i 'error|exception|step' results/model_a_pilot/logs
```

Stop the persistent pilot with
`tmux kill-session -t nowak_model_a_pilot`, then confirm GPU memory is
released with `nvidia-smi`.

### FlashInfer GDN JIT requires a CUDA toolkit

The latest training launch (2026-07-17) loaded the model but died on its first
generation because FlashInfer attempted to JIT-compile a Qwen GDN prefill
kernel and could not find `nvcc`:

```text
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

This is the current blocker; the later engine-dead and connection errors are
consequences. On an allocated GPU node, discover a compatible site CUDA module,
load it, and expose its toolkit root before launching:

```bash
module spider cuda
# Load a compatible available CUDA 12.x module selected from the site output.
command -v nvcc
nvcc --version
export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
test -x "$CUDA_HOME/bin/nvcc"
```

Record the chosen module, `CUDA_HOME`, and `nvcc --version` in the run manifest.
The fix is accepted only after a real TP=2 Qwen generation forces the GDN
compile and a new one-update PRIME smoke succeeds. Do not reuse
`results/model_a_pilot`: its current logs are failed attempts and its
`rollouts/step_1` directory is empty/stale. Follow Gate 0 in
`docs/PROJECT_EXECUTION_PLAN.md`.

### PRIME and Verifiers must use the pinned submodule revision

PRIME commit `3d2dbae5f` expects the Verifiers API pinned by its git
submodule, `d5604bd3dfbfe402535c7ee7034f0ea03c02b4e2`.  A newer detached
Verifiers checkout removes `TrainRunInfo`, `EvalRunInfo`, and `Trace.stamp`.
It lets generation finish but aborts the orchestrator before the trainer update
with `AttributeError: module 'verifiers.v1' has no attribute 'TrainRunInfo'`.

Before a Gate-0 retry, verify and restore the pinned revision when necessary:

```bash
PRIME=/home/memoozd/scratch/rl/prime-rl
git -C "$PRIME" ls-tree HEAD deps/verifiers
git -C "$PRIME/deps/verifiers" checkout --detach \
  d5604bd3dfbfe402535c7ee7034f0ea03c02b4e2
```

If the commit is not in the local clone, fetch it from the public HTTPS remote
first; the configured SSH remote may require interactive host-key setup.  Then
confirm `hasattr(verifiers.v1, "TrainRunInfo")` and
`hasattr(verifiers.v1.Trace, "stamp")` are both true.  This restores PRIME's
declared dependency rather than applying an API shim.

### Disable trainer compilation for dynamic Qwen MoE RL batches

On this stack, `torch.compile` can cache a stride-specialized backward graph
for one packed rollout shape and then fail on another with an Inductor
`assert_size_stride` error.  Preserve and apply
`patches/prime-rl-disable-trainer-compile.patch`, which makes model compilation
opt-in instead of its PRIME default.  The Gate-0 pilot intentionally leaves
`[trainer.model.compile]` absent.  Verify the resolved trainer config has no
`[model.compile]` block before launching; this is a feasibility safeguard, not
a final performance choice.

## Verifiers taskset pattern

The discovery module is `src/donor_coord_v1/__init__.py`; its normalized module
name matches taskset ID `donor-coord-v1` and exports exactly one `Taskset` via
`__all__`. The implementation is in
`src/nowak_coordination/environment.py`.

- `DonorTaskset.load()` expands the `(b, w, q, partner policy, replicate)` grid
  into deterministic typed tasks.
- Round 1 is an explicit structured `user` message in `TaskData.prompt`.
  This avoids depending on simulator-opening behavior. `DonorUser` supplies
  subsequent turns.
- `DonorUser` is colocated with the null harness. Per-rollout mutable data
  belongs in `DonorState`; seeded RNGs make horizons and partner behavior
  reproducible.
- The simulator parses the model's exact two-line protocol, advances the game,
  writes actions/payoffs/forecasts to state, and sets `game_over`. Invalid
  output terminates the episode and receives zero reward.
- `@vf.stop` ends the interaction from shared state. `@vf.reward` and
  `@vf.metric` score the completed trace. Do not keep rollout state on the
  shared `Task` instance.
- Unit-test task expansion, prompts, simulator transitions, termination,
  reward components, and seeded partner behavior before loading a model.

For another environment, add an importable discovery package, install the
project editable into the PRIME venv, add a `[[orchestrator.train.env]]` entry,
run `--dry-run`, and smoke one rollout before a multi-GPU launch.

## Qwen/PRIME fixes and constraints

### Explicit Qwen 3.6 renderer

PRIME renderer auto-detection uses exact canonical Hugging Face IDs. A local
model path is not recognized. The fallback `default` renderer performs
incremental prefix rendering for token attribution; its first prefix contains
only the system message, which Qwen rejects with:

```text
TemplateError: No user query found in messages.
```

The message was not dropped. Select the dedicated renderer:

```toml
[orchestrator.renderer]
name = "qwen3.6"
```

Keep `enable_thinking = false` in sampling if short protocol-only outputs are
required. The dedicated renderer was verified token-for-token against Qwen's
chat template on the actual `system -> user` prompt (138 prompt tokens).

### FP8 MoE expert conversion

The local FP8 checkpoint stores both `gate_proj.weight` and
`gate_proj.weight_scale_inv` for every routed expert. PRIME counted every key
containing `gate_proj`, inferred 512 experts instead of 256, and failed with:

```text
KeyError: model.layers.0.mlp.experts.256.gate_proj.weight
```

The local PRIME checkout counts only keys ending in `.gate_proj.weight`.
The change lives outside this repository and can be lost during an update.
Its preserved patch is
`patches/prime-rl-qwen35-fp8-expert-count.patch`.

Check/reapply it after updating PRIME:

```bash
PRIME=/home/memoozd/scratch/rl/prime-rl
PROJECT=/scratch/memoozd/rl/Nowak-coordination
git -C "$PRIME" diff -- \
  src/prime_rl/trainer/models/qwen3_5_moe/converting_qwen3_5_moe.py
git -C "$PRIME" apply \
  "$PROJECT/patches/prime-rl-qwen35-fp8-expert-count.patch"
```

Apply only when the change is absent; `git apply --check` can test first.

### Pilot-specific training choices

- LoRA targets attention and Qwen hybrid linear-attention projections:
  `q/k/v/o_proj`, `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`,
  `out_proj`.
- Routed and shared MoE experts remain frozen; adapting all 256 experts is
  intentionally avoided in the feasibility pilot.
- Filesystem weight broadcast, sequence length 4096, TP=2 inference, and a
  separate adapter checkpoint are configured in the pilot TOML.
- `mooncake-transfer-engine` is not available in the verified cluster stack;
  avoid Mooncake-dependent transport unless a compatible build is installed.
