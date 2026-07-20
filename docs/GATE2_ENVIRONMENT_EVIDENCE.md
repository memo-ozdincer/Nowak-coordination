# Gate 2 environment evidence

**Status:** complete — audited 2026-07-20 EDT.

This ledger is the durable acceptance record for Gate 2 in
`docs/PROJECT_EXECUTION_PLAN.md`. It covers CPU semantic mechanics and the
pinned PRIME/Verifiers adapter only. It does not constitute a base-model
evaluation or scientific result, and no confirmatory result was inspected.
The governing environment contract is `docs/ENVIRONMENT_SPEC.md` v1.1,
SHA-256 `645244ee7dba8c24651e7fefd132be5357b1629c0c84c6c19fe7c3a9a92d5c97`.

## Implemented causal world

- `w` is a per-transition same-partner retention probability. The transition
  draw, retained/replaced event, outgoing/current/next opaque IDs, and
  per-identity history are logged.
- `q` gates prompt visibility of an immutable latent reputation generated from
  four disjoint seeded reference interactions. Latent truth, visibility, and
  rendered content are distinct trace fields.
- Forced TFT→AD and AD→TFT switches and seeded reciprocator/defector
  interleaving preserve separate identities and histories without prompt-level
  policy names.
- Focal and partner intentions are distinct from executed actions. This
  includes both explicit round perturbations and endogenous noisy-TFT execution
  errors.
- Group mode uses four or five simultaneous agents. The non-leaking CFE target
  is the current executed group cooperation fraction, and the group-donor
  payoff remains in `[-c,b]`.
- Naturalistic mode uses a seeded neutral-label mapping, commits its hash
  before rollout, omits literal C/D action names and policy classes from the
  prompt, and records the mapping only in the terminal event.
- Models A–E are implemented. Model E requires a reference with both a
  non-focal episode ID and non-focal partner ID. CFE is scored only on real
  group rounds; no dyadic running average substitutes for a group target.
- Training, held-out, and fixed diagnostic policy registries are validated at
  task construction.
- The adapter copies `vf.Trace.state` into
  `trace.info.coordination_trace` before JSON serialization and records the
  registered seed role. This was added after the Gate-3 audit established that
  Verifiers excludes `state` from raw JSONL; in-memory completeness alone was
  not durable evidence.

Primary implementation:

- `src/nowak_coordination/mechanics.py`
- `src/nowak_coordination/environment.py`
- `src/nowak_coordination/game.py`
- `src/nowak_coordination/partners.py`
- `src/nowak_coordination/rewards.py`

## Counterfactual acceptance results

The preregistered `w` check used 300 root seeds, eight-round episodes, and
2,100 post-round transitions per condition:

| Configured `w` | Retained | Frequency | Acceptance |
|---:|---:|---:|---|
| 0.0 | 0 / 2,100 | 0.000000000 | exact pass |
| 0.2 | 413 / 2,100 | 0.196666667 | within ±0.03 |
| 0.8 | 1,649 / 2,100 | 0.785238095 | within ±0.03 |
| 1.0 | 2,100 / 2,100 | 1.000000000 | exact pass |

The `q=0` versus `q=1` counterfactual at the same root seed and focal action
sequence produced byte-equal partner IDs, partner actions, payoffs, retention
draws/events, and next IDs. All eight `q=0` visibility draws were false; all
eight `q=1` draws were true. Latent reputation records were equal.

## Tests and runtime compatibility

Commands:

```bash
./.venv/bin/ruff check src tests
./.venv/bin/ruff format --check src tests
./.venv/bin/python -m pytest -q
./.venv/bin/python -m pytest \
  tests/test_mechanics.py tests/test_environment.py tests/test_rewards.py -vv
```

Results:

- full suite: **76 passed**;
- targeted Gate-2 suite: **37 passed**;
- Ruff lint and formatting checks: passed;
- local environment: Python 3.12.13, Verifiers 0.2.0;
- pinned PRIME environment: Verifiers 0.2.1.dev47;
- pinned-environment environment suite: **9 passed** under
  Verifiers 0.2.1.dev47, including persistent normal/invalid terminal traces.

Acceptance tests explicitly cover deterministic replay, causal `w`, isolated
`q`, hidden reputation content, both switch directions, interleaved histories,
forced and endogenous noise, group forecast timing, naturalistic labels,
training/evaluation leakage, A–E hand calculations and HKB extrema,
shuffled-HKB source exclusion, invalid-format zero reward, and complete normal
and invalid terminal traces.

## Audited source hashes

| File | SHA-256 |
|---|---|
| `mechanics.py` | `9f03fcb5226a3a7b8213d933aa4c2394cf4187a395294b800b9de90ca6f6a43e` |
| `environment.py` | `ed73005303a56f11b11a27974554d679116c281169c30fab269c0f3a31a517da` |
| `game.py` | `99552dd2dc6726871b0cf4ca0a8cae120245dd7f8ee9591fd124acd2876335aa` |
| `partners.py` | `ed7a80fbe8b22eba21a7252de9c982c20252ca25dfbef6b71b490c94572fb22f` |
| `rewards.py` | `ea2165680a7a752ee26ad4a47e1d949a2b4ccc7f25fd9b4455ff73c9358edce7` |
| `test_mechanics.py` | `86b6e26697dcf684b6cd06fc9cc77d0d8ca44daf19f308965dbe071601a77436` |
| `test_environment.py` | `e57011fd8e90f51d72299fae8cc7cdc9b68b1683356d3a8762abdbb34b965fee` |
| `test_rewards.py` | `e408cc7ba2f3857adf7801dacbcc5be39b969132da96ace09d4656189ee9453e` |

These hashes identify the code exercised by the recorded targeted suite.
Changes to any hashed file require rerunning Gate-2 acceptance before a
scientific launch.

## Gate decision

Gate 2 is complete after the raw-serialization repair. Every required semantic
check passes, and the same task objects construct and execute under the pinned
PRIME environment. Gate 3A subsequently repaired the confirmatory decision
design; Gate 4 (base-model characterization) is now the first incomplete gate.
