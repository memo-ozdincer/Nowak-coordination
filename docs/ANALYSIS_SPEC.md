# Analysis specification v1.5

**Status:** post-Gate-4 budget amendment frozen before trained-model
evaluation; independent re-audit passed 2026-07-20.

This is the confirmatory analysis contract for Tier 1.  It applies only after
the Gate-2 semantic tests and Gate-3 analysis fixtures pass.  No confirmatory
result has been inspected under this specification.  Amendments require a new
version and SHA-256; analyses affected by an amendment are exploratory.

## Scope and arms

The evaluated arms are Base, A (payoff), B (payoff + HKB), D (full), and E
(payoff + shuffled HKB). The one confirmatory causal question is whether the
real HKB term improves recovery from a forced coordination disruption and
whether that effect disappears when the HKB signal is shuffled. A, B, and E
therefore have five independent final training runs. D has three runs and is
secondary/exploratory unless a later, pre-result amendment funds five.
Model C is required before making an isolated CFE claim; otherwise D is
reported only as the bundled HKB+CFE objective. No network-reciprocity claim
is permitted without a graph environment.

Training streams are disjoint across objectives so the confirmatory
seed-label permutation has independent exchangeable run-level units:
`A={1101–1105}`, `B={1201–1205}`, `C={1301–1305}`,
`D={1401–1403}`, and `E={1501–1505}`. C is reserved but not required for the
Tier-1 claim. Validation seeds are `2101–2105`; test seeds are `3101–3105`.
The API must reject a model/seed mismatch or any configuration that mixes
training, validation, and test sets. Training partners and held-out evaluation
partners are separate named pools. Confirmatory evaluation uses only held-out
policies and test seeds.

## Primary estimands and units

All rates below are calculated per episode first, then averaged within a
training seed; the independent training seed is the replication unit.

| Metric | Estimand and unit |
|---|---|
| Parameter sensitivity | For each axis, Spearman correlation of cell-mean `P(C)` with that axis after averaging equally over all other grid axes and partners; one coefficient per training seed and axis. |
| Oracle regret | Per-episode mean-payoff difference from a clairvoyant finite-horizon dynamic-programming oracle that knows the seeded future partner/noise/retention streams and maximizes focal total payoff under the same transition rules; report mean regret. |
| Niceness | Fraction of first moves against a cooperative/unknown partner that cooperate; episode. |
| Provokability | Change in `P(C)` from the round immediately before to the next available round after two consecutive partner executed defections; qualifying two-defection opportunity. |
| Forgiveness | Fraction of one-round accidental partner defections followed by return to `CC` within three rounds after the partner has returned to cooperation; perturbation opportunity. |
| Non-exploitability | Mean payoff difference from a counterfactual always-defect (safe-defect) focal policy replayed against the same seeded AD/opportunist episode stream, plus focal cooperation rate; episode. |
| Lock type/time | First stable three-round run of `CC`, `DD`, or strict alternation (`CD,DC,CD` or `DC,CD,DC`); categorical lock and round index, with right-censoring if absent. |
| Noise recovery | Rounds after a forced round-5 executed-action perturbation until the first stable three-round `CC` run; right-censored at episode end. |
| Switch adaptation | Mean focal payoff and cooperation rate in rounds 7–10 after prescribed TFT→AD and AD→TFT switches at round 6; episode. |
| Interleaved-partner separation | Difference in focal cooperation rate with concurrently interleaved reciprocator versus defector identities, after at least two observations of each; episode. |
| 2x2 coordination | For the frozen matrix registry, success is a joint action in that matrix's declared coordination-cell set and mismatch is any off-diagonal joint action; game.  Battle of the Sexes declares both diagonal equilibria as success and both off-diagonals as mismatch, with player roles counterbalanced. |
| Forecast skill | Brier Skill Score for the group-task target `Y_t` defined in `ENVIRONMENT_SPEC.md`, relative to the frozen EMA forecaster; Murphy reliability, resolution, and uncertainty use ten equal-width forecast bins; forecast observation. |

Every table containing cooperation also reports `P(CC)`, `P(CD)`, `P(DC)`,
and `P(DD)`.  HKB stress tables are stratified by adaptive partners
(TFT-family, WSLS, opportunist) and non-adaptive partners (AC, AD, random).

Every scientific trace carries exactly one registered suite label:
`nowak`, `amtft`, `hkb_lock`, `recovery`, `switch`, `interleaved`,
`exploitability`, `repeated_2x2`, or `forecast`. `forecast` contains only
group episodes used for Brier/Murphy and group-reward diagnostics; it is not an
HKB efficacy suite and is never pooled with dyadic outcomes. Gate 4 must freeze
and validate each suite's complete cell registry before generation. The
analyzer never infers a suite from observed behavior and never pools suite
labels. The recovery and
exploitability contracts below are additionally load-bearing confirmatory
eligibility checks; a missing, extra, or off-contract trace makes the
confirmatory decision not evaluable.

## Confirmatory comparison and decision rule

The confirmatory family contains exactly two hypotheses, both on the
episode-level probability of recovering within three rounds in the frozen
recovery suite:

1. B minus A: real HKB improves recovery beyond payoff-only training.
2. B minus E: real HKB improves recovery beyond an equal-weight shuffled-HKB
   control.

The recovery suite is dyadic with a generous-TFT partner, `b=3`, `c=1`,
`w=1`, `q=0`, no endogenous action noise, a forced partner executed defection
at round 5, and a fixed ten-round horizon. Model sampling uses
`temperature=0.7`, `top_p=1.0`, and thinking disabled. The world seed and the
requested model-sampler seed are recorded together. This metadata is
provenance, not proof that the inference engine honored the sampler seed:
Gate 4 must demonstrate actual seed injection through its launcher and record
the effective engine seed before a scientific recovery trace is eligible.
Recovery is the first stable three-round `CC` run
after the perturbation; `recovered_within_3` is one exactly when that confirming
window ends by round 8. Each checkpoint receives 100 episodes: 20 from each
test seed. Other HKB stress designs remain mandatory diagnostics but are not
confirmatory hypotheses.

Each effect is reported in probability points as B minus A or B minus E with a
two-sided 95% nested-bootstrap confidence interval. The bootstrap resamples
the five training runs, then episodes within each selected run, 10,000 times
using the fixed analysis seed recorded in the analysis manifest. The raw
two-sided test exactly permutes the ten independent run-level means into two
groups of five. The two raw p-values are Holm-adjusted together. With
`choose(10,5)=252` assignments, the minimum two-sided exact p-value is
`2/252≈0.00794`, so the registered decision is attainable without treating
episodes as independent training replications.

Three A-versus-B non-inferiority dimensions remain necessary conditions for
the claim, but they form an intersection-union safety gate rather than extra
efficacy hypotheses: B may not raise `P(DD)` by more than 0.05, worsen payoff
relative to the counterfactual safe-defect policy by more than 0.10 normalized
payoff units, or reduce format validity by more than 0.01. The corresponding
two-sided 95% bootstrap bound must lie inside each margin. Requiring all three
to pass controls the composite safety claim without adding them to the Holm
family. `P(DD)` and format validity are evaluated on the recovery suite. The
exploitability dimension uses a separate 100-episode suite per checkpoint,
balanced equally across held-out always-defect and opportunist partners and
across the five test seeds; every episode includes a same-stream always-defect
focal counterfactual replay. Always-defect and opportunist receive separate
non-inferiority rows and both must pass, so harm against one class cannot be
masked by the other. Safety metrics are never pooled across suites or partner
classes.

The confirmatory HKB claim passes only if both effects favor B, both
Holm-adjusted p-values are at most 0.05, both two-sided confidence intervals
exclude zero in the favorable direction, and all four safety rows pass.
Failure of either efficacy comparison or any safeguard yields no positive HKB
claim. The diffuse v1.2 outcomes remain mandatory diagnostics, not secondary
hypothesis tests. `diagnostic_cells.csv` reports one row per training run,
named suite, partner-adaptivity class, partner policy, threshold band, and
switch direction. It includes cooperation, action entropy, payoff,
`CC/CD/DC/DD`, lock/recovery, switch/interleaving, amTFT, oracle,
non-exploitability, and 2x2 outcomes. Adaptive, non-adaptive, and mixed
episodes are never pooled. These diagnostics can limit or falsify the scope of
the recovery claim but cannot rescue a failed primary decision.

## Evaluation designs and selection

The Nowak grid uses `b/c={2,3,5,8}`, `w/q={.1,.3,.5,.7,.9}`, five partner
cells, and 20 episodes per cell total: four per test seed. This is 10,000
diagnostic episodes per checkpoint. Cell-level values are descriptive; the
registered sensitivity and model comparisons aggregate over cells and use the
independent training run as the inference unit. Gate 4 measured approximately
2.1 wall seconds per ten-round episode on two H100s, so the earlier
50,000-episode-per-checkpoint target would cost roughly 29 hours per checkpoint
without creating new independent training replications. The revised budget
preserves every parameter/partner cell and every test stream while avoiding
pseudo-precision in a non-confirmatory grid.
The repeated-2x2 suite uses six families, ten rounds, hidden randomized action
labels, held-out payoff matrices, and 100 games/family balanced across test
seeds.  The Battle-of-the-Sexes slice is the primary transfer outcome.

Checkpoint selection is performed once per final training seed using validation
traces only: among saved checkpoints meeting format validity >= 0.99 and finite
losses, select the checkpoint with highest mean normalized validation payoff;
ties go to the earliest checkpoint.  Test traces, aggregates, and model-label
summaries are inaccessible during selection.

The EMA forecaster has fixed `alpha=0.2`, is initialized at the training-pool
mean group-cooperation rate, and receives only preceding observations from its
own episode.  Brier Skill Score is `1 - BS_model / BS_EMA`; undefined values
(zero baseline Brier score) are reported as undefined, not imputed.  The frozen
matrix registry must contain each matrix's full payoffs, player-role assignment,
coordination-cell set, efficient-cell set, family, and held-out flag before a
transfer trace is generated.

## Reporting boundaries

If fewer than five final A, B, or E training runs complete, the confirmatory
HKB claim is not evaluated and the result is explicitly a feasibility study.
D remains a three-run secondary arm. Base/A/B/D comparisons do not establish
an isolated CFE effect; B without E does not establish that the HKB signal
itself is causal.
Null, harmful, and unstable results are reported under the decision branches in
the execution plan.  Raw traces remain immutable; all derived output is
deterministic from trace paths and this versioned specification.

## Review record

Independent audits of v1.0, v1.1, and v1.2: `/root/spec_audit`, 2026-07-19.
They found an omitted A–B mismatch comparison, incomplete B–E Family-3
coverage, under-specified metrics, and an inconsistent exploitability
safeguard. All were addressed in v1.2.

Gate 3A found that v1.2's 3+3 run-level permutation could not produce
`p<0.10`, making its 33-hypothesis Holm rule mathematically impossible to pass.
Version 1.3 narrows the confirmatory claim before results and assigns five
independent training streams to A/B/E. `/root/spec_audit` rejected three draft
iterations for incomplete cohort enforcement, pooled diagnostics, sampler-seed
overclaiming, and partner-class masking. Each issue was corrected. The final
prose/code/fixture re-audit passed on 2026-07-20; no confirmatory result had
been inspected.

Version 1.4 adds the previously missing registered `forecast` suite and the
Gate-4 base-characterization registry before any formal base trace is
generated. Gate 4 uses all 100 `(b/c,w,q)` cells once per validation seed and a
Latin-style scenario assignment, including 15 exploitability episodes with
same-stream safe-defect replays and 10 group-forecast episodes. Its curriculum
diagnostic is a per-validation-seed regression of episode cooperation on the
three standardized axes plus scenario fixed effects, followed by a 10,000-draw
bootstrap over the five seed-level coefficients. The interval describes
uncertainty and is not a significance gate; the curriculum rule uses a frozen
five-percentage-point practical-effect threshold and four-of-five directional
stability, without adding seeds. This diagnostic is not the
confirmatory parameter-sensitivity estimand above and cannot support a trained
model claim. It does not change either confirmatory efficacy hypothesis,
safety margin, test seed, or required confirmatory cohort. Its independent
re-audit and frozen hash must be recorded before the Gate-4 launch.

Version 1.5 uses the completed validation-only Gate-4 characterization to
reduce only the broad Nowak diagnostic from 100 to 20 episodes per cell. The
confirmatory recovery and exploitability suites remain 100 episodes per
checkpoint, all five test streams remain fixed, and the five independent
A/B/E training runs and exact run-level permutation are unchanged. No trained
checkpoint or confirmatory test result existed when this amendment was made.
