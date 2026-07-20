# Gate 4 base-characterization specification

**Status:** corrected registry draft after independent audit; no formal trace generated.

## Purpose and decision boundary

This is a validation-only characterization of the unmodified
Qwen3.6-35B-A3B base model. It is allowed to determine curriculum difficulty
and the later evaluation budget, so it uses validation streams `2101–2105`,
not confirmatory test streams `3101–3105`. It cannot support a trained-model
or HKB claim.

The cohort is exactly 500 ten-round episodes: 100 episodes in each validation
stream. Each stream contains every `(b/c, w, q)` combination once:

- `b/c = {2, 3, 5, 8}` with `c=1`;
- `w = {0.1, 0.3, 0.5, 0.7, 0.9}`;
- `q = {0.1, 0.3, 0.5, 0.7, 0.9}`.

This is a 500-episode stratified characterization, not the later 50,000-episode
full Nowak evaluation (500 partner-parameter cells × 100 episodes per cell).

## Prespecified scenario allocation

Within every 100-episode validation block, parameter cells are lexicographically
ordered by `(b, w, q)`. For zero-based indices of `b`, `w`, `q`, and validation
seed, the Latin category is
`(b_index + w_index + q_index + seed_index) mod 5`. Each category contains
exactly 20 cells and is balanced over every `w` and `q` level. Five cells in
each category are deterministically reserved so that the reserved subset also
contains every `w` and `q` level once and is as balanced over `b` as the
integer counts permit. The exact tie-break is implemented in
`gate4_registry.py` and tested as part of the registry contract.

| Scenario | Episodes/seed | Suite | Mechanics |
|---|---:|---|---|
| held-out forgiving grudger | 15 | `nowak` | adaptive dyad |
| held-out delayed TFT | 15 | `nowak` | adaptive dyad |
| held-out probabilistic defector | 15 | `nowak` | non-adaptive dyad |
| held-out 10%-noisy copy | 15 | `amtft` | adaptive noisy dyad |
| diagnostic forced switch | 15 | `switch` | balanced TFT→AD / AD→TFT at round 6 |
| diagnostic exploitability | 15 | `exploitability` | AD/opportunist dyads with same-stream safe-defect replay |
| held-out group forecast | 10 | `forecast` | groups of four/five, all four held-out policies rotated |

Categories 0–4 assign their 15 unreserved cells respectively to forgiving
grudger, delayed TFT, probabilistic defector, noisy copy, and switch. Reserved
cells in categories 0–2 form the exploitability slice; reserved cells in
categories 3–4 form the forecast slice. The ten forecast cells are ranked
within seed, then cycle across all four held-out policies and alternate group
sizes, guaranteeing policy coverage and five episodes at each group size.
Diagnostic bots are never represented as held out.

For each exploitability episode, the environment creates a second world from
the identical immutable episode configuration and advances it for all ten
rounds with the focal action fixed to defect and forecast fixed to zero.
The resulting mean focal payoff and provenance
(`same_seed_same_world_always_defect_replay`, registry version, episode seed,
and horizon) are saved before model generation. This is a deterministic policy
counterfactual, not a claim that it reproduces the model's endogenous history.

Episode root seeds are disjoint blocks `4210100–4210199` through
`4210500–4210599`. The model sampler seed is the validation stream seed
`2101–2105`, paired across that stream's 100 distinct tasks. Sampling is
`temperature=0.7`, `top_p=1.0`, at most 64 new tokens per turn, and thinking
disabled.

The Gate-4 EMA forecast baseline starts at the prespecified neutral value
`0.5`, because no training-pool observation exists before base
characterization. This is confined to the validation-only Gate-4 forecast
diagnostic; later trained-model analyses use the training-pool initialization
required by the Analysis Spec.

## Sampler-seed gate

Before these traces are eligible:

1. the task registry, Verifiers `Trace.agent.sampling`, and saved
   `sampler_seed_evidence` must agree on the requested seed;
2. a live vLLM probe must show identical token output for repeated identical
   requests with the same seed and at least one differing output across a
   preregistered set of different seeds;
3. the effective request seed, model revision, Verifiers revision, vLLM
   revision, and probe payload hash must be recorded as engineering evidence.

Merely writing a seed into task metadata does not pass this gate.

## Acceptance and curriculum decision

The combined raw cohort must pass strict trace validation and the exact
500-row registry check before analysis. Tables must report, at minimum:

- per-validation-seed and suite-stratified action and forecast entropy;
- format validity and complete-trace rate;
- a per-seed scenario-adjusted sensitivity coefficient for `b/c`, `w`, and `q`;
- payoff and `CC/CD/DC/DD`;
- adaptive, non-adaptive, and mixed partner strata;
- switch-direction behavior;
- group forecast Brier skill/Murphy components and variance of total reward.

Action entropy is binary Shannon entropy from the episode cooperation rate.
Forecast entropy is Shannon entropy after assigning the ten round forecasts to
the same ten equal-width bins used by the Murphy decomposition. Total-reward
variance is the population variance across round-level verifier totals within
an episode and is reported separately for the `forecast` suite.

For each validation seed, the diagnostic sensitivity model is OLS at the
episode level:

`P(C) ~ 1 + z(b/c) + z(w) + z(q) + scenario fixed effects`.

All 100 cells enter once. The three continuous axes are standardized within
seed; one prespecified lexicographically first scenario is the dummy reference.
The analyzer must reject a rank-deficient design and report its rank and
condition number. It then bootstraps the five seed-level coefficients as the
independent units, with 10,000 resamples and analysis seed `730031`, to obtain
an interval for their mean. This adjusted regression is a Gate-4 curriculum
diagnostic only; it does not replace the final Analysis Spec estimand for
trained checkpoints.

The interval is an uncertainty description, not a Gate-4 significance test.
An axis is called a recognizable base-model signal when its coefficient has
the theory-consistent direction in at least four of five validation streams
and the absolute mean standardized coefficient is at least `0.05` (a
five-percentage-point change in episode cooperation per one standard deviation
of the prompt parameter). The expected directions are positive for `b/c` and
`w`. The sign of `q` is reported but not required globally because visibility
can rationally increase or decrease cooperation depending on the displayed
partner history.

The curriculum is made harder only if both `b/c` and `w` meet that practical
signal rule and the prespecified switch/exploitability tables show
policy-contingent rather than unconditional cooperation. Otherwise the full
parameter curriculum is preserved. No p-value or confidence-interval exclusion
is required, and adding seeds to obtain one is prohibited. This rule may
increase later training difficulty, but cannot alter confirmatory metrics,
test seeds, or the frozen recovery claim.

## Mandatory invocation

Formal Gate-4 validation and analysis commands must include
`--require-gate4-cohort`. A wrapper or command transcript omitting that flag is
ineligible even if its output happens to contain 500 traces.
