# Gate 3A confirmatory-design repair evidence

**Status:** complete — independently audited 2026-07-20 EDT.

No confirmatory result was inspected before or during this repair. The frozen
contract is `docs/ANALYSIS_SPEC.md` v1.3, SHA-256
`412eaab56ef68533da91dec5b7a83fe845670c43943dd16397f94e236c0fa9e0`.

## Why v1.2 was invalid

Version 1.2 combined three training runs per arm, a two-sided seed-label
permutation, 33 Holm-adjusted hypotheses, and a significance requirement.
There are only `choose(6,3)=20` assignments for 3+3 run-level units, so the
smallest exact two-sided p-value is 0.10. Counting episodes as independent
would be pseudoreplication because the reward objective is assigned to the
trained checkpoint, not to an evaluation episode.

## Frozen repair

The confirmatory claim is deliberately narrow:

> In the fixed generous-TFT forced-disruption suite, does real HKB training
> improve recovery within three rounds over payoff-only A and shuffled-HKB E,
> without worsening defection, partner-specific exploitability, or format
> validity?

Only two efficacy hypotheses are in the Holm family:

1. B minus A on `recovered_within_3`;
2. B minus E on `recovered_within_3`.

A, B, and E use five disjoint training streams. The exact test enumerates all
`choose(10,5)=252` assignments regardless of the CLI resampling setting. Its
minimum two-sided raw p-value is `2/252`, and the minimum two-test
Holm-adjusted value is `4/252`, so the decision is attainable only for a strong,
consistent run-level effect.

The safety gate has four rows: recovery-suite `P(DD)`, recovery-suite format
validity, and separate safe-defect non-inferiority checks for held-out
always-defect and opportunist partners. All four must pass. The adversarial
fixture proves that harm against opportunists fails the overall decision even
when always-defect safety passes.

The exact required cohort is enforced before a decision can be evaluated:

- recovery: A/B/E × five registered checkpoint seeds × five test seeds ×
  20 episodes = 1,500 traces;
- exploitability: A/B × five checkpoint seeds × five test seeds ×
  (`10` always-defect + `10` opportunist) = 1,000 traces.

Missing, extra, imbalanced, off-seed, duplicate, or off-contract traces yield
`NOT_EVALUABLE`, not an estimate promoted to a claim. Recovery eligibility
freezes dyadic mode, generous TFT, `b=3`, `c=1`, `w=1`, `q=0`, ten rounds,
round-5 forced partner defection, no other action noise, and the requested
sampling settings. Requested sampler-seed metadata is provenance only; Gate 4
must demonstrate actual launcher/engine seed injection before a scientific
trace is accepted.

## Scope discipline

The execution plan and broader paper plan now explicitly supersede the old
broad confirmatory wording. Nowak, amTFT, switch, interleaving, threshold,
lock, and repeated-2x2 outcomes remain mandatory diagnostics but cannot be
used to rescue or generalize the recovery claim.

All scientific traces require a registered suite label. Derived tables retain
suite identity; stress tables additionally retain training seed, policy, and
adaptivity. `diagnostic_cells.csv` is stratified by suite, training run,
adaptivity, partner policy, threshold band, and switch direction and includes
action entropy plus the full outcome/behavior metric set.

Policy-arm identity is separate from the environment reward calculation.
Evaluation traces preserve checkpoint-training-seed provenance while keeping
validation/test RNG seeds in their disjoint registries.

## Independent audit

`/root/spec_audit` performed an adversarial prose/code/fixture audit. Drafts
were rejected for:

- failing to narrow the operational headline claim;
- incomplete recovery and exploitability cohort enforcement;
- configurable Monte Carlo fallback despite promising an exact test;
- pooled/under-specified diagnostics;
- overclaiming that recorded sampler metadata proved engine seed use;
- allowing always-defect performance to mask opportunist harm.

Every finding was reconciled. The final audit passed after the partner-specific
adversarial safety test and complete suite-stratification review.

## Acceptance evidence

- full local suite: **76 passed**;
- targeted Gate-2 semantic suite: **37 passed**;
- pinned PRIME/Verifiers environment suite: **9 passed** under
  Verifiers 0.2.1.dev47;
- Ruff formatting and lint: passed;
- synthetic validator: 12 traces / 120 rounds, PASS;
- two complete regenerations: byte-identical tables;
- `g9` readiness check: four idle NVIDIA H100 80GB GPUs, Torch CUDA available
  with four devices; no scientific GPU run was launched.

## Gate decision

Gate 3A is complete. Gate 4 is the first incomplete gate. Gate 4 must implement
and test effective sampler-seed injection plus the exact named-suite builders
before formal base characterization.
