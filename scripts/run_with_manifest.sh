#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$PROJECT_DIR/.venv/bin/python" -m nowak_coordination.run_manifest \
  --project "$PROJECT_DIR" "$@"
