# Gate 3 reproducibility and analysis evidence

**Status:** complete — audited 2026-07-20 EDT; statistical blocker subsequently
resolved by `docs/GATE3A_DESIGN_REPAIR_EVIDENCE.md`.

This ledger covers the CPU-only reproducibility and analysis gate. It is not a
scientific evaluation and contains no model result. The synthetic arms are
hand-authored known-answer data, not generated behavior.

## Implemented interfaces

- `scripts/run_with_manifest.sh` creates a new run directory with
  `mkdir(exist_ok=False)`, validates the TOML snapshot, captures project and
  PRIME revisions/dirty hashes, external patch hashes, package/hardware state,
  model/tokenizer/spec hashes, seed roles, the exact argv, logs, exit code,
  timestamps, and `COMPLETED`, `FAILED`, or `CANCELLED` state. Registered seed
  partitions are enforced at both task-config and run-wrapper boundaries.
- The Verifiers adapter copies its transient `trace.state` into
  `trace.info.coordination_trace` before serialization. This repair was
  necessary because `vf.Trace.state` is excluded from raw JSONL. Without it,
  Gate 2's complete in-memory trace would have disappeared in a real run.
- `scripts/validate_traces.py` fails closed on malformed JSON, duplicate trace
  or episode IDs, non-finite values, missing/reordered rounds, inconsistent
  terminal state, invalid actions/outcomes/forecasts/rewards, captured errors,
  incomplete traces, split/policy leakage, and counterfactual values without
  provenance.
- `scripts/analyze.py` emits deterministic episode, round, aggregate,
  HKB-stress, sensitivity, forecast-skill, nested-bootstrap, and confirmatory
  tables plus an analysis manifest. Confirmatory cells with missing data or
  fewer than three training seeds are explicit, never silently dropped.
- `scripts/build_figures.py` reads analyzer tables only. It does not
  independently recompute a metric.
- `scripts/regenerate_synthetic.sh` rebuilds the committed fixture, all tables,
  and both figures from scratch.

The analyzer implements the frozen estimands: `CC/CD/DC/DD`, lock type/time,
stable-CC recovery, direction-specific switch behavior, interleaved separation,
Nowak-axis Spearman coefficients per training seed, oracle regret,
safe-defect non-exploitability, niceness eligibility, corrected two-defection
provokability, one-round-accident forgiveness, retaliation length, cooperation
with reciprocators, value-defined-punishment input, 2x2 outcomes, threshold
bands, nested bootstrap intervals, seed-level permutation tests, Holm
adjustment, and the three non-inferiority decisions.

The EMA has frozen `alpha=0.2`; its initial training-pool mean is a required CLI
argument and is written into `analysis_manifest.json`. BSS uses squared error
against the observed group-cooperation fraction. Murphy reliability,
resolution, and uncertainty expand each group fraction into its underlying
binary partner actions. This is equivalent to forecasting cooperation of a
random group member and avoids applying a binary-event decomposition
uncritically to a fractional observation.

## Known-answer and failure evidence

The committed fixture has 12 traces, 120 rounds, four synthetic arms, and three
training seeds. Exact assertions cover payoff, cooperation, every joint-outcome
cell, niceness, forgiveness, retaliation, oracle regret, non-exploitability,
fractional-group Murphy decomposition, and Holm adjustment. Snapshot hashes
cover every committed table. Mutation tests prove rejection of duplicates,
missing turns, NaN, incomplete terminal state, missing counterfactual
provenance, and cross-split seed reuse.

Run-wrapper tests cover successful and failed commands, stdout capture,
resolved-config preservation, exit codes, seed-role rejection, and overwrite
refusal. Two complete invocations of `scripts/regenerate_synthetic.sh` produced
byte-identical tables.

Acceptance commands and results:

```text
.venv/bin/ruff format --check src tests scripts analysis/fixtures/generate_synthetic.py
.venv/bin/ruff check src tests scripts analysis/fixtures/generate_synthetic.py
.venv/bin/pytest -q
```

- Ruff formatting and lint: passed.
- Full suite after Gate-3A hardening: **76 passed**.
- Targeted Gate-2 semantic suite after persistence repair: **37 passed**.
- Pinned PRIME environment (`verifiers==0.2.1.dev47`):
  `tests/test_environment.py` **9 passed**.
- Synthetic validator: **12 traces / 120 rounds, PASS**.
- Deterministic regeneration: all ten table/manifest hashes matched on the
  second run.

## Critical boundaries and newly discovered design blocker

The analyzer consumes oracle, safe-defect, and value-defined-punishment
counterfactual outputs and requires provenance; it does not invent them from
observed actions. Gate 4's evaluation harness must implement the finite-horizon
dynamic-programming oracle and paired counterfactual replays before producing
those fields. The repeated-2x2 registry is also still later work. Therefore,
Gate 3 completion does not imply that a scientific suite is launch-ready.

More importantly, Analysis Spec v1.2 combines three training seeds per arm,
seed-level two-sided permutation tests, 33 confirmatory hypotheses, and a
Holm-significance requirement. With 3+3 independent seeds there are only
`choose(6,3)=20` label assignments; the smallest attainable two-sided exact
p-value is 0.10. A positive Holm-corrected result is mathematically impossible.
Permuting episodes would be pseudoreplication because treatment was assigned
at training-run level. Even 4–6 seeds per arm cannot reach the smallest
Holm threshold `0.05/33`; seven per arm is the minimum that makes it
attainable (`2/choose(14,7) ≈ 0.000583`).

No confirmatory result was opened. Gate 3A repaired this by narrowing the
efficacy family to two forced-recovery hypotheses, using five disjoint A/B/E
training streams, and enforcing exact cohort and partner-specific safety
rules. See `docs/GATE3A_DESIGN_REPAIR_EVIDENCE.md`.

## Gate decision

Gate 3's software acceptance checks pass. Gate 3A also now passes; Gate 4 is
the first incomplete gate.
