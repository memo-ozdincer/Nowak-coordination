#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${project_dir}/.venv/bin/python"
fixture="${project_dir}/analysis/fixtures/synthetic_traces.jsonl"
tables="${project_dir}/analysis/tables/synthetic"
figures="${project_dir}/analysis/figures/synthetic"

"${python_bin}" "${project_dir}/analysis/fixtures/generate_synthetic.py"
rm -rf "${tables}" "${figures}"
"${python_bin}" "${project_dir}/scripts/validate_traces.py" "${fixture}" \
  --report "${tables}/validation.json"
"${python_bin}" "${project_dir}/scripts/analyze.py" "${fixture}" \
  --output-dir "${tables}" --ema-initial 0.5
"${python_bin}" "${project_dir}/scripts/build_figures.py" \
  --tables-dir "${tables}" --output-dir "${figures}"
