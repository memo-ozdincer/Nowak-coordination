#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/home/memoozd/scratch/models/Qwen3.6-35B-A3B-FP8}"
PRIME_RL_DIR="${PRIME_RL_DIR:-/home/memoozd/scratch/rl/prime-rl}"

cd "${PROJECT_ROOT}"

echo "== GPU environment =="
hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "== model =="
test -f "${MODEL_DIR}/config.json"
echo "MODEL_DIR=${MODEL_DIR}"

uv python install 3.12
uv sync --group gpu

if [[ ! -d "${PRIME_RL_DIR}/.git" ]]; then
    git clone https://github.com/PrimeIntellect-ai/prime-rl.git "${PRIME_RL_DIR}"
fi
git -C "${PRIME_RL_DIR}" fetch --quiet origin
git -C "${PRIME_RL_DIR}" checkout --quiet origin/main
git -C "${PRIME_RL_DIR}" submodule update --init --recursive
(
    cd "${PRIME_RL_DIR}"
    uv sync --all-extras
)

uv run --group gpu python - <<'PY'
import os
from pathlib import Path

import torch
import transformers
import vllm

model_dir = Path(os.environ.get("MODEL_DIR", "/home/memoozd/scratch/models/Qwen3.6-35B-A3B-FP8"))
config = transformers.AutoConfig.from_pretrained(model_dir, local_files_only=True)
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
print(f"transformers={transformers.__version__} vllm={vllm.__version__}")
print(f"model_type={config.model_type} architectures={config.architectures}")
assert torch.cuda.is_available(), "This script must run inside an allocated GPU job"
PY

echo "GPU environment setup and model compatibility checks passed."
