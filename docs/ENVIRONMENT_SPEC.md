# Environment specification v1.1

**Status:** implemented; Gate-2 semantic acceptance passed (2026-07-20).

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

Named seeds use BLAKE2b over the integer root seed and stream name; they never
use Python's process-salted `hash()`. Visibility, retention, selection, partner
action, reputation reference play, horizon, shuffled-HKB reference, and label
mapping have independent streams. Therefore changing only `q` cannot alter a
partner action or retention draw, and changing only `w` cannot alter the
underlying random draws used by other mechanisms.

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

Semantic acceptance uses 300 root seeds and eight-round episodes (2,100
post-round transitions per condition). `w=0` and `w=1` must produce exact
retention frequencies 0 and 1; observed frequencies for `w=0.2` and `w=0.8`
must lie within absolute tolerance 0.03 of their configured probabilities.

Forced switch tests override the normal draw at the configured boundary: round
6 changes TFT→AD or AD→TFT, with a new opaque ID and an explicit
`forced_switch` event.  Interleaving selects two persistent identities in a
fixed seeded schedule (reciprocator and defector) and renders only their IDs;
each retains a separate history. Configured forced-switch and perturbation
rounds must not exceed the minimum horizon, so an intervention cannot silently
disappear in a shorter seeded episode.

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

The counterfactual acceptance test uses identical root seeds/actions at `q=0`
and `q=1`. Latent records, identities, retention draws, executed actions, and
payoffs must be byte-equal; visibility must be uniformly false versus true.

## Actions, perturbations, and payoffs

The model emits an intended action and forecast.  The verifier samples any
configured accidental-action perturbation after parsing and before payoff or
partner response.  It records intended and executed actions for each player;
payoffs, histories, partner-policy input, joint outcomes, and rewards use
executed actions.  A recovery test forces exactly one accidental focal or
partner action at round 5, records its source, and performs no further forced
noise in that episode.  Invalid output terminates immediately, receives no task
reward, and retains a complete terminal trace.

Models A–E use the same payoff component. CFE is applicable only to group
rounds: C reduces to payoff-only and D to payoff+HKB on dyadic/naturalistic
rounds. This avoids the invalid prototype behavior of scoring a dyadic running
partner average as if it were a group forecast. Model E uses an independently
seeded HKB history whose episode and partner IDs are both required to differ
from the focal source; equality is a hard error. Its history is prefix-matched
to the focal round (one observation at round 1 through the four-observation HKB
window), so source mismatch is the intended difference rather than history
length.

## Group/CFE task

The group task defaults to four agents and supports a configured size of four
or five: the focal model and `n-1` seeded partner agents.
At each round the focal model submits an intended binary action and a forecast
`f_t` before any peer actions for that round are generated.  The forecast target
is `Y_t`, the fraction of all `n` **executed** actions that cooperate on that
same round.  Peer actions and action perturbations are not available to the
model until after its output, so the target is not leaked.  The trace records
`f_t`, `Y_t`, all intended/executed actions, group composition, and the CFE
component.  Model D/C uses this task; a dyadic running partner average is not a
substitute.

The group payoff preserves the dyadic reward range. For agent `i` in a group of
size `n`,

`pi_i = b * (# cooperating other agents)/(n-1) - c * 1[i cooperates]`.

Thus `pi_i` remains in `[-c,b]` and the existing payoff normalization remains
valid. HKB is intrinsically dyadic, so on a group round B/D average the
normalized HKB value over the focal agent's separate history with each peer; it
is not computed from a thresholded group mean. The standing theoretical caveat
remains: with `delta_omega=0`, the shaping is valence-blind and rewards CC and
DD alignment equally. The environment does not silently “fix” that registered
hypothesis.

## Naturalistic curriculum stage

For each stage-3 episode, two randomly generated labels replace literal action
words.  The verifier-only mapping maps each label to C or D; prompts describe
the labels' payoff consequences but do not contain the words “cooperate” or
“defect.”  Label mapping is seeded, logged after the rollout, and balanced over
episodes.  The parser and all rewards use the verifier mapping.

Before rollout, the private trace records only a SHA-256 commitment to the
mapping. The full label-to-C/D mapping is written in the terminal event after
the rollout. Neither policy names nor the verifier mapping appear in model
prompts.

## Required trace schema and tests

Each round records episode/round IDs; focal and partner IDs; intended and
executed actions; payoffs; retention event; latent and visible reputation;
rendered observation; switch/interleaving/perturbation events; forecast and
target; reward components; and terminal reason.  Gate 2 passes only when the
execution plan's deterministic-transition, `w`, `q`, hidden-content,
identity/history, perturbation, hand-calculated reward (A–E), shuffled-HKB,
invalid-format, and complete-terminal-trace tests pass with durable evidence.

Implementation: `src/nowak_coordination/mechanics.py` (pure worlds and named
RNGs), `src/nowak_coordination/environment.py` (Verifiers v1 adapter), and
`src/nowak_coordination/rewards.py` (A–E composition).
