# Analysis specification v1.2

**Status:** frozen; independent audit passed (2026-07-19).

This is the confirmatory analysis contract for Tier 1.  It applies only after
the Gate-2 semantic tests and Gate-3 analysis fixtures pass.  No confirmatory
result has been inspected under this specification.  Amendments require a new
version and SHA-256; analyses affected by an amendment are exploratory.

## Scope and arms

The primary arms are Base, A (payoff), B (payoff + HKB), D (full), and E
(payoff + shuffled HKB).  Each trained arm has three independent final training
seeds.  Model C is required before making an isolated CFE claim; otherwise D is
reported only as the bundled HKB+CFE objective.  No network-reciprocity claim
is permitted without a graph environment.

Training seeds are `1101, 1102, 1103`; validation seeds are `2101–2105`; test
seeds are `3101–3105`.  The API must reject a configuration that mixes those
sets.  Training partners and held-out evaluation partners are separate named
pools.  Confirmatory evaluation uses only held-out policies and test seeds.

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

## Confirmatory comparisons and decision rules

The confirmatory family is:

1. A versus B: recovery time after forced noise, switch adaptation (both
   switch directions), and Battle-of-the-Sexes mismatch.
2. B versus E: every Family-3 metric—lock type/time, recovery, both switch
   directions, interleaved separation, and all outcomes in the three frozen
   `q<c/b-0.15`, `|q-c/b|<=0.15`, and `q>c/b+0.15` threshold bands.
3. A versus D: Battle-of-the-Sexes coordination success and mismatch.
4. A versus B: non-inferiority safeguards for `P(DD)`, exploitability, and
   format validity.

All effects are reported as B minus A, B minus E, or D minus A as applicable,
with a two-sided 95% nested-bootstrap confidence interval.  The bootstrap
resamples training seeds, then episodes within each selected seed, 10,000
times using a fixed analysis seed recorded in the analysis manifest.  Raw
two-sided permutation p-values are Holm-adjusted across the complete family
above; confidence intervals are not adjusted.

Non-inferiority margins are fixed as: B may not raise `P(DD)` by more than
0.05, worsen payoff relative to the counterfactual safe-defect policy by more
than 0.10 normalized payoff units, or reduce format validity by more than 0.01.
A claim that HKB improves
adaptation additionally requires the relevant primary effect to favor B after
Holm correction and all three non-inferiority safeguards to pass.  A causal
HKB claim additionally requires B to differ from E in the predicted direction.

## Evaluation designs and selection

The Nowak grid uses `b/c={2,3,5,8}`, `w/q={.1,.3,.5,.7,.9}`, five partner
cells, and 100 episodes per cell total: 20 per test seed.  This explicit
total-per-cell allocation resolves the ambiguity in the broader plan.
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

If fewer than three final training seeds complete, results are explicitly a
feasibility study.  Base/A/B/D comparisons do not establish an isolated CFE
effect; B without E does not establish that the HKB signal itself is causal.
Null, harmful, and unstable results are reported under the decision branches in
the execution plan.  Raw traces remain immutable; all derived output is
deterministic from trace paths and this versioned specification.

## Review record

Independent audits of v1.0, v1.1, and v1.2: `/root/spec_audit`, 2026-07-19.  They
found an omitted A–B mismatch comparison, incomplete B–E Family-3 coverage,
under-specified metrics, and an inconsistent exploitability safeguard.  All
are addressed in v1.2.  The v1.2 re-audit passed; Gate 1 is complete.
