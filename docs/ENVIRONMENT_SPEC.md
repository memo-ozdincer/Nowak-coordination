# Environment specification v1.0

**Status:** design frozen for Gate-2 implementation (2026-07-19).

This document defines the causal world required for the claimed variables.  It
supersedes the engineering prototype's prompt-only `w` and `q`; implementation
and semantic tests remain required before any scientific run.

## Episode streams and identities

Each episode derives named, reproducible RNG streams from `(episode_seed,
stream_name)`: retention, partner selection, partner action/noise, reputation,
perturbation, and label mapping.  A trace records the root seed and every
realized transition, never only the input parameters.  Partners have opaque
stable IDs (for example `p-03a1`), policy class, adaptivity stratum, and their
own history.  Prompts expose IDs but never policy class.

Training and held-out policy registries are distinct configuration fields and
disjoint by validation.  A task cannot be constructed with a held-out policy in
the training registry or vice versa.

## Direct reciprocity: `w`

`w` is the probability of **same-partner retention** between two decision
rounds.  After each nonterminal dyadic round, sample `retain ~ Bernoulli(w)`.
On retain, the next round uses the same identity and that partner's accumulated
history.  Otherwise the current partner retires and a newly sampled opaque
identity begins with empty focal history; the episode continues until its fixed
decision horizon.  The prompt reports whether the previous identity was
retained and the current ID.  The trace records the retention draw, outgoing
and incoming IDs, and histories.  Thus `w` changes repeated-partner mechanics,
not merely wording or horizon length.

Forced switch tests override the normal draw at the configured boundary: round
6 changes TFT→AD or AD→TFT, with a new opaque ID and an explicit
`forced_switch` event.  Interleaving selects two persistent identities in a
fixed seeded schedule (reciprocator and defector) and renders only their IDs;
each retains a separate history.

## Indirect reciprocity: `q`

Every newly created partner receives a latent reputation record generated from
its policy on four seeded reference interactions that are disjoint from the
focal episode.  The record contains the four executed actions and their
cooperation fraction.  It is immutable for that identity during the episode.
Before each focal decision, independently sample `visible ~ Bernoulli(q)`.
Only when visible is true, the prompt shows the current partner ID and the
record's cooperation fraction; when false it states that no reputation was
observed and includes no reputation content.  The trace stores latent content,
visibility, and rendered observation separately.  Changing `q` may change
visibility frequency but must never change latent records or partner behavior.

## Actions, perturbations, and payoffs

The model emits an intended action and forecast.  The verifier samples any
configured accidental-action perturbation after parsing and before payoff or
partner response.  It records intended and executed actions for each player;
payoffs, histories, partner-policy input, joint outcomes, and rewards use
executed actions.  A recovery test forces exactly one accidental focal or
partner action at round 5, records its source, and performs no further forced
noise in that episode.  Invalid output terminates immediately, receives no task
reward, and retains a complete terminal trace.

## Group/CFE task

The group task has four agents: the focal model and three seeded partner agents.
At each round the focal model submits an intended binary action and a forecast
`f_t` before any peer actions for that round are generated.  The forecast target
is `Y_t`, the fraction of all four **executed** actions that cooperate on that
same round.  Peer actions and action perturbations are not available to the
model until after its output, so the target is not leaked.  The trace records
`f_t`, `Y_t`, all intended/executed actions, group composition, and the CFE
component.  Model D/C uses this task; a dyadic running partner average is not a
substitute.

## Naturalistic curriculum stage

For each stage-3 episode, two randomly generated labels replace literal action
words.  The verifier-only mapping maps each label to C or D; prompts describe
the labels' payoff consequences but do not contain the words “cooperate” or
“defect.”  Label mapping is seeded, logged after the rollout, and balanced over
episodes.  The parser and all rewards use the verifier mapping.

## Required trace schema and tests

Each round records episode/round IDs; focal and partner IDs; intended and
executed actions; payoffs; retention event; latent and visible reputation;
rendered observation; switch/interleaving/perturbation events; forecast and
target; reward components; and terminal reason.  Gate 2 passes only when the
execution plan's deterministic-transition, `w`, `q`, hidden-content,
identity/history, perturbation, hand-calculated reward (A–E), shuffled-HKB,
invalid-format, and complete-terminal-trace tests pass with durable evidence.
