# Gate 4 base-characterization evidence

**Status:** formal cohort passed; curriculum and evaluation-budget decisions
frozen 2026-07-20.

## Eligible cohort

The formal cohort contains 500 complete ten-round traces, exactly 100 from
each validation stream `2101–2105`. Strict validation reports 500 unique trace
IDs, 500 unique episode IDs, 5,000 ordered rounds, all 100 registered
`(b/c,w,q)` cells in every stream, and `PASS` for the exact Gate-4 registry.

Evidence root:
`results/gate4/base_characterization_cohort/20260720T150000Z-7034325`.
The validation report SHA-256 is
`3778563154fa88dcb4cabf5601a3e6209d16ca4a45719426b6370f4fed5ae6a9`.
The analysis manifest records Analysis Spec v1.4 SHA-256
`708f1b16484d567177cf3fa6a75360ec2039704c83f46bbbbb1ec356b040fbec`,
combined input SHA-256
`bed412a9591a5fac18d374eb2b03c9f57a5d02e97cb9a87770dc67790e061cc8`,
and analysis seed `730031`.

| Validation seed | Eligible run | Trace SHA-256 |
|---:|---|---|
| 2101 | `20260720T142936813037Z-gate4-base-s2101` | `44016cb23179343b404b38091ef5d350392ae9b018130a08ce28fbd9497d4043` |
| 2102 | `20260720T143210107816Z-gate4-base-s2102` | `42d31ee896c2e4ab9275ab4bab025b7f018a1826ddb67d753e74f08497b7e08f` |
| 2103 | `20260720T144442714968Z-gate4-base-s2103` | `b3393623395334220eac11b75461b9bf976c2385d21f3c1a118703c8b7d20b39` |
| 2104 | `20260720T144824964007Z-gate4-base-s2104` | `37af1bdf710cffa28f50df732e7c709bdae76850db3d1736274dc0e16217d382` |
| 2105 | `20260720T145151121237Z-gate4-base-s2105` | `fa309f7c3d4b72bb4594e33c2e82dcf52bfe5533b5860d2167cab2ea1900b3b0` |

Every eligible generation manifest is clean and records Python 3.12.13,
Torch 2.11.0+cu128, Transformers 5.6.2, Verifiers 0.2.1.dev47, and
vLLM 0.24.0+cu129. Seeds 2101–2102 used maximum request concurrency 32.
After a vLLM shared-memory broadcast hang, seeds 2103–2105 used concurrency
16. This scheduling limit changes no task, prompt, RNG seed, sampler setting,
or estimand.

Two ineligible attempts are preserved:

- `20260720T142759564280Z-gate4-base-s2101` is `FAILED`; resolving the
  virtualenv Python symlink stripped its environment, and it generated no
  trace.
- `20260720T143426625286Z-gate4-base-s2103` is `CANCELLED`; vLLM stopped
  producing tokens, repeatedly reported shared-memory broadcast starvation,
  and it generated no complete trace.

The first combined analysis attempt also failed closed after validation because
the analyzer used peer count `N-1` where the forecast target used full group
size `N`. Commit `7034325` corrected that bookkeeping error. The final
analysis was regenerated from the unchanged validated traces in a new output
directory.

## Base-model result

Format validity is `1.000`. Across the cohort, mean episode cooperation is
`0.7366` and mean binary action entropy is `0.4213` bits. The per-seed
cooperation rates are `0.734`, `0.721`, `0.747`, `0.673`, and `0.808`.

The frozen practical parameter-signal rule does not pass:

| Axis | Mean standardized coefficient | Direction agreement | Practical signal |
|---|---:|---:|---|
| `b/c` | `+0.00448` | 2/5 | no |
| `w` | `-0.03974` | 5/5 negative | no |
| `q` | `-0.04158` | 5/5 negative | descriptive only |

The model is partner-contingent, but not safe:

| Diagnostic | Episodes | Cooperation | Relevant outcome |
|---|---:|---:|---:|
| always-defect exploitability | 40 | 0.343 | payoff minus safe-defect `-0.343` |
| opportunist exploitability | 35 | 0.843 | payoff minus safe-defect `-0.834` |
| TFT→AD switch | 38 | post-switch 0.539 | post-switch payoff 1.796 |
| AD→TFT switch | 37 | post-switch 0.716 | post-switch payoff 1.378 |

The group forecast slice has mean forecast entropy `0.775` bits and mean
within-episode total-reward variance `0.011`. Brier skill versus the frozen EMA
baseline varies substantially by validation stream:
`-0.639, +0.151, +0.269, +0.001, -0.017`. This is not a stable forecast-skill
signal.

## Frozen decisions

1. Preserve the complete `b/c,w,q` curriculum. The base model did not pass the
   practical `b/c` and `w` tracking rule.
2. Emphasize partner adaptation, defector discrimination, and safe
   non-exploitability. High opportunist cooperation is a particularly clear
   failure mode.
3. Do not add seeds or optimize for a Gate-4 p-value. Gate 4 is a
   characterization, not a confirmatory efficacy test.
4. Keep the confirmatory recovery and exploitability budgets at 100 episodes
   per checkpoint; the independent training run remains the inference unit.
5. Reduce the broad diagnostic Nowak grid from 100 to 20 episodes per cell
   total, four from each of five test streams. This remains 10,000 episodes per
   checkpoint and preserves all 500 cells, but avoids spending approximately
   29 wall-clock hours per checkpoint on a non-confirmatory 50,000-episode
   grid. The observed stable-throughput blocks took roughly 2.1 wall seconds
   per ten-round episode on two H100s, making the revised grid roughly
   5.8 hours per checkpoint. Cell-level values are descriptive; registered
   inference aggregates over cells and independent training runs.

The budget change is a pre-trained-result Analysis Spec v1.5 amendment. It
does not change the recovery claim, safety margins, training-seed counts,
test-seed identities, or exact run-level permutation.
