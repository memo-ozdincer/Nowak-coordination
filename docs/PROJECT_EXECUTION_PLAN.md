# Project state and completion path

**Operational source of truth — last audited 2026-07-20**

This document turns `paper-and-plan/final_plan_v2.md` into a gated execution
contract. It is deliberately stricter than a task list: later work is blocked
until earlier acceptance checks have durable evidence. A null or adverse result
can complete the project; an unverified positive claim cannot.

## 1. Finish line

The Tier-1 project is complete when it provides a reproducible answer to:

> Does post-training a Qwen3.6-35B-A3B agent with verifiable Donor's-Game
> rewards produce parameter-sensitive, adaptive, non-exploitable coordination,
> and do HKB/CFE additions improve it over payoff-only RL or merely change the
> equilibrium it selects?

Completion requires all of the following:

1. The implemented environment causally instantiates every parameter named in
   a headline claim. In particular, `w` changes repeated-partner mechanics and
   `q` changes reputation information; they are not prompt-only covariates.
2. Base, Model A (payoff), Model B (payoff+HKB), and Model D (full) have
   reproducible checkpoints from frozen configs. Model E (shuffled-HKB) is also
   required for a claim that the HKB signal itself matters. Model C is required
   for any claim that isolates the CFE term; without C, D is only a bundled
   objective.
3. Base and trained models are evaluated on the preregistered Nowak, amTFT,
   HKB-stress, and repeated-2x2 transfer suites with uncertainty, outcome
   decompositions, held-out partners, and corrected confirmatory tests.
4. The result package can be regenerated from a clean checkout plus the
   documented external PRIME patch: manifests, resolved configs, checkpoints,
   raw traces, analysis tables, figures, and a final paper/report.
5. The conclusion follows the observed decision branch, including the valid
   possibilities that payoff alone is sufficient, HKB is inert, or HKB
   stabilizes mutual defection.

Tier 2 (Trust-and-Split, GovSim, MACHIAVELLI, network reciprocity, and extra
reward controls) is not part of this finish line. Do not let it delay Tier 1.
Network reciprocity must be absent from claims and figures unless a real
graph-structured evaluation is added.

## 2. Current state

### State ledger

| Area | State | Verified evidence / remaining gap |
|---|---|---|
| Research design | **Gate 1 complete** | `docs/ANALYSIS_SPEC.md` v1.2 is independently audited and frozen; environment design is `docs/ENVIRONMENT_SPEC.md` |
| Project setup | **Working** | `.venv`, PRIME venv, and commands documented in `docs/CLUSTER_RL_RUNBOOK.md` |
| Core environment | **Gate 2 complete** | Causal `w/q`, opaque identities, per-partner histories, switches, interleaving, action perturbations, group/CFE, naturalistic labels, and complete traces pass semantic tests |
| Reward code | **Gate 2 complete** | A–E composition, normalized dyadic HKB, shuffled-HKB source exclusion, and genuine group Brier target are unit-tested |
| Tests | **Passing** | `./.venv/bin/python -m pytest -q` → 72 passed on 2026-07-20; pinned PRIME/Verifiers environment tests also pass |
| Live model protocol | **Working** | `results/live_smoke/traces.jsonl`: 3/3 complete; `results/base_grid/traces.jsonl`: 48/48 complete, no trace errors |
| Base evaluation | **Engineering sample only** | 48 deterministic, three-round episodes; too small/narrow for scientific inference |
| Training runtime | **Working (Gate 0 passed)** | CUDA 12.9 `nvcc`, pinned Verifiers, selective norm-only checkpointing, and disabled default trainer compilation passed a real vLLM/FlashInfer generation and ten-update pilot; evidence: `docs/GATE0_RUNTIME_EVIDENCE.md` |
| Model A training | **Feasibility pilot completed; scientific training not started** | `results/gate0/model_a_pilot/20260717T213000Z-6b7c288-s2005` completed 10 updates with a loadable step-10 adapter and full export; it is not a frozen scientific training run |
| Models B/C/D/E | **Not started** | No frozen training configs or checkpoints |
| Formal analysis pipeline | **Gate 3 complete** | Strict manifests/validation, deterministic registered metrics, bootstrap/permutation/Holm output, known-answer snapshots, and table-driven figures pass; evidence: `docs/GATE3_ANALYSIS_EVIDENCE.md` |
| Statistical decision feasibility | **Blocked before Gate 4** | With 3+3 training seeds, the smallest exact two-sided seed-permutation p-value is 0.10, so the 33-test Holm rule can never pass; Gate 3A must repair the frozen design before results |
| Stress/transfer environments | **Internal mechanics working; external transfer absent** | Switch, interleaving, forced noise, and group mechanics pass Gate 2; Akata-style 2x2 suite remains Gate 8 |
| Paper | **Incomplete** | `paper-and-plan/incomplete_paper.pdf`; no result-complete manuscript |

### What the existing 48-episode sample establishes

The deterministic base grid used `b={2,8}`, `w={0.1,0.9}`,
`q={0.1,0.9}`, partners `{AC, AD, TFT}`, two replicates, and three rounds.
All 48 traces completed with valid output.

| Slice | Episodes | Mean cooperation | Mean payoff | P(CC) | P(DD) |
|---|---:|---:|---:|---:|---:|
| Always cooperate | 16 | 1.000 | 4.000 | 1.000 | 0.000 |
| Always defect | 16 | 0.333 | -0.333 | 0.000 | 0.667 |
| Tit-for-tat | 16 | 1.000 | 4.000 | 1.000 | 0.000 |
| Each tested `b` value | 24 | 0.778 | varies with `b` | 0.667 | 0.222 |
| Each tested `w` value | 24 | 0.778 | 2.556 | 0.667 | 0.222 |
| Each tested `q` value | 24 | 0.778 | 2.556 | 0.667 | 0.222 |

Interpretation: the base model responds to partner history in this small sample
but its actions are exactly flat across the tested `b`, `w`, and `q` values.
That is a feasibility signal, not a statistical conclusion: decoding was
temperature 0, the grid omitted most planned values/partners, and there were
only two replicates per cell.

### Resolved runtime blocker

The original FlashInfer missing-`nvcc` failure was resolved by loading CUDA 12.9
and exporting its discovered `CUDA_HOME`; FlashInfer compiled the Qwen GDN
prefill kernel during the successful Gate-0 run.  Two subsequent compatibility
issues were also corrected: the PRIME-pinned Verifiers revision was restored,
and default trainer compilation was made opt-in because dynamic Qwen-MoE shapes
violate its cached graph assumptions.  Exact evidence, commands, failed-run
history, and measured cost are in `docs/GATE0_RUNTIME_EVIDENCE.md` and
`docs/CLUSTER_RL_RUNBOOK.md`.

## 3. Execution rules

1. Work only on the first incomplete gate below, except for read-only
   investigation or documentation that directly unblocks it.
2. A checkbox means “acceptance criteria passed,” not “code was written” or
   “a job was submitted.” Record an evidence path next to it.
3. Never reuse an output directory. Use:
   `results/<stage>/<variant>/<UTC>-<git-short>-s<seed>/`.
4. Every run directory must contain:
   `resolved_config.toml`, `manifest.json`, logs, and an explicit terminal
   status (`COMPLETED`, `FAILED`, or `CANCELLED`). Only `COMPLETED` runs enter
   analysis.
5. `manifest.json` must record project commit and dirty diff hash, PRIME commit
   and patch hash, model/tokenizer path and config hash, Python/Torch/vLLM/
   Transformers/Verifiers versions, hostname/GPU/driver/CUDA toolkit, all RNG
   seeds, start/end times, checkpoint parent, and W&B run ID if enabled.
6. Freeze configs and `docs/ANALYSIS_SPEC.md` before confirmatory runs. Any
   post-freeze change gets a new version and makes the affected analysis
   exploratory.
7. Training, validation, and test seeds and partner pools must not overlap.
8. Always report `P(CC), P(CD), P(DC), P(DD)` with cooperation. HKB results
   must be split by adaptive versus non-adaptive partners.
9. Keep raw traces immutable. Analysis writes derived files elsewhere and must
   be deterministic from trace paths plus its config.
10. Do not start Tier 2 until Gate 10 passes.

## 4. Mandatory gates

### Gate 0 — Make the GPU path perform one real update

**Purpose:** separate cluster/runtime feasibility from research design.

- [x] On an allocated H100 node, expose a compatible CUDA 12.x toolkit to the
  job. Discover available modules with `module spider cuda`; do not hard-code a
  site version without checking it. Verify:

  ```bash
  command -v nvcc
  nvcc --version
  export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  test -x "$CUDA_HOME/bin/nvcc"
  ```

- [x] Record `CUDA_HOME`, `nvcc --version`, driver, GPU, and loaded modules in
  the run manifest. The toolkit version must be validated by a real FlashInfer
  compile; version strings alone are insufficient.
- [x] Run one Qwen generation through the same vLLM/TP=2 path used by PRIME.
  It must return the exact two-line protocol and produce no fatal GDN/FlashInfer
  error.
- [x] Create a unique one-update Model-A smoke config/output. It must produce
  one non-empty rollout group, a finite loss/gradient, and a recorded trainer
  update with zero infrastructure errors.
- [x] Rerun the ten-update Model-A feasibility pilot in a fresh directory. It
  must reach step 10, save a non-empty loadable adapter, keep format validity
  at least 0.99, and have no NaN/Inf or engine-dead errors.
- [x] Measure and record rollout/update throughput, peak GPU memory, wall time,
  and storage. Use this measurement—not an estimate—to budget the final matrix.

If loading a CUDA module is impossible, the allowed alternative is a
documented vLLM/FlashInfer backend or precompiled-kernel configuration that
supports this exact Qwen checkpoint. It must pass the same generation and
ten-update acceptance checks. Do not treat disabling a warning as a fix.

**Evidence:** `docs/GATE0_RUNTIME_EVIDENCE.md`; completed terminal run
`results/gate0/model_a_pilot/20260717T213000Z-6b7c288-s2005`.

### Gate 1 — Freeze the claim and analysis contract

**Purpose:** prevent implementation choices and statistical tests from moving
after results are visible.

- [x] Create `docs/ANALYSIS_SPEC.md` v1.2; future confirmatory manifests must record its hash.
- [x] Define the estimand and unit for every primary metric: parameter
  sensitivity, oracle regret, niceness, provokability, forgiveness,
  non-exploitability, lock type/time, recovery, switch adaptation,
  interleaved-partner separation, 2x2 coordination success/mismatch, and
  forecast skill.
- [x] Freeze the primary comparisons:
  A vs B on recovery, switch adaptation, and mismatch; B vs E on Family-3
  metrics; A vs D on repeated-2x2 coordination. Freeze non-inferiority margins
  for P(DD), exploitability, and format validity.
- [x] Specify five independent evaluation seeds, at least three independent
  final training seeds per trained variant, episode-level bootstrap confidence
  intervals nested by training seed, and Holm correction over the confirmatory
  family. If the budget cannot support three training seeds, label the result a
  feasibility study rather than a robust training comparison.
- [x] Define checkpoint selection from validation data only and prohibit test
  peeking.
- [x] Resolve scientific scope:
  no network-reciprocity claim without a graph environment; no isolated CFE
  claim without Model C; no causal HKB claim without E.
- [x] Have a second reader audit the spec against
  `paper-and-plan/final_plan_v2.md` before results are opened.

**Evidence:** `docs/ANALYSIS_SPEC.md` v1.2, SHA-256
`e8e1304da0d1e67d164c17f72c1c9f78a36dbc601b1642d4203cbf68aaec0719`;
independent audit by `/root/spec_audit` on 2026-07-19 passed after two
documented reconciliation rounds.

### Gate 2 — Make the task semantics match the claims

**Purpose:** ensure the verifier changes the world, not merely the wording.

Write `docs/ENVIRONMENT_SPEC.md` first, then implement it.

- [x] **Direct reciprocity (`w`):** define and implement whether `w` is
  continuation probability or same-partner retention probability. The
  transition RNG, partner identity/state, prompt, and logged event must agree.
  A fixed-horizon episode with a printed `w` does not pass.
- [x] **Indirect reciprocity (`q`):** create a latent, seeded partner-reputation
  record and reveal it with probability `q`; log latent truth and visibility
  separately. A printed scalar or its use only inside HKB reward does not pass.
- [x] **Partner identity:** log an identity and maintain per-partner history.
  Implement forced TFT→AD and AD→TFT switches plus interleaved reciprocator/
  defector identities without leaking the policy name.
- [x] **Perturbations:** implement a forced accidental action at a specified
  round, distinguishing intended from executed action, so recovery is
  measurable.
- [x] **Group/CFE:** implement the 4–5-agent group/public-goods episode needed
  by Model D. Forecasts must target a precisely defined future group outcome;
  avoid scoring a vague running mean or leaking the target.
- [x] **Naturalistic stage:** implement label-randomized, verifier-only C/D
  mappings for the stage-3 curriculum so transfer is not just memorization of
  the words “cooperate” and “defect.”
- [x] Log round-level agent/partner IDs, intended/executed actions, payoffs,
  observations, reputation visibility, perturbations, forecast target, and
  reward components.
- [x] Separate training policies from held-out policies at the API/config
  level, not by convention.

Required semantic tests:

- identical seed/config produces identical transitions;
- changing only `w` changes retention/continuation frequencies over many
  seeded episodes within a preregistered tolerance;
- changing only `q` changes observation frequency, never latent reputation;
- hidden reputation content never appears when visibility is false;
- switch/interleaving preserves separate histories and correct identity;
- forced noise changes executed but not intended action;
- hand-computed A/B/C/D/E rewards match code, including HKB extrema;
- shuffled-HKB never draws from the focal episode/partner;
- invalid format terminates and cannot earn task reward;
- all terminal paths retain a complete trace.

**Evidence:** `docs/GATE2_ENVIRONMENT_EVIDENCE.md`; 56-test full suite,
36-test targeted semantic suite, preregistered `w/q` counterfactuals, and
pinned PRIME/Verifiers adapter checks passed on 2026-07-20.

### Gate 3 — Build the reproducible evaluation/analysis pipeline

- [x] Add a run wrapper that creates unique directories and manifests, resolves
  configs, records terminal state, and refuses to overwrite.
- [x] Add a trace validator that fails on duplicate IDs, missing turns,
  malformed metrics, seed leakage, non-finite values, or incomplete traces.
- [x] Add one deterministic analyzer that emits tidy per-episode, per-round,
  aggregate, and bootstrap tables. Plots must consume these tables rather than
  independently reimplementing metrics.
- [x] Implement the EMA base-rate forecaster, Brier Skill Score, and Murphy
  reliability/resolution/uncertainty decomposition.
- [x] Implement outcome decomposition, Spearman sensitivities, oracle regret,
  all amTFT metrics, HKB stress metrics split by lock type and partner
  adaptivity, and Holm-adjusted confirmatory results.
- [x] Commit a tiny synthetic fixture with known answers and snapshot tests for
  every metric and table.
- [x] One command regenerates all synthetic tables/figures from scratch and
  produces byte-identical tables on a second run.

Expected stable interfaces:

```text
scripts/run_with_manifest.sh
scripts/validate_traces.py
scripts/analyze.py
scripts/build_figures.py
analysis/fixtures/
analysis/tables/
analysis/figures/
```

Names may change, but the capabilities and one-command regeneration may not.

**Evidence:** `docs/GATE3_ANALYSIS_EVIDENCE.md`; 72-test full suite, 12-trace /
120-round known-answer fixture, pinned Verifiers adapter tests, overwrite and
failure-path tests, committed table snapshots, and two byte-identical complete
regenerations passed on 2026-07-20.

### Gate 3A — Repair confirmatory decision feasibility

**Purpose:** prevent a mathematically unattainable positive decision rule from
driving expensive GPU runs.

Analysis Spec v1.2 assigns treatment at the training-run level, requires
two-sided permutation p-values over independent training seeds, Holm-adjusts
33 hypotheses, and budgets only three seeds per arm. With 3+3 seeds there are
only `choose(6,3)=20` label assignments, so the smallest exact two-sided
p-value is 0.10. Episode-level permutation would be pseudoreplication. At least
seven seeds per compared arm are required even to make the smallest Holm
threshold `0.05/33` attainable under the current family.

- [ ] Choose a defensible repair before any confirmatory trace is opened:
  fund at least seven final training seeds per arm, or reduce/structure the
  confirmatory family based on the theory rather than anticipated results.
- [ ] Publish `docs/ANALYSIS_SPEC.md` v1.3 with the revised replication,
  hypothesis-family, and exact/Monte-Carlo permutation rules.
- [ ] Independently audit v1.3 against the scientific plan and the implemented
  33-row hypothesis registry.
- [ ] Record the v1.3 hash in this plan and require it in every later
  validation/test manifest.

**Evidence:** blocker derivation and implementation audit in
`docs/GATE3_ANALYSIS_EVIDENCE.md`; no confirmatory results inspected.

### Gate 4 — Characterize the base model before full RL

Run this only after Gates 1–3, because the current 48 traces lack causal `w/q`
mechanics and formal metrics.

- [ ] Run a 500-episode stratified base characterization spanning all
  `b/c={2,3,5,8}`, `w/q={.1,.3,.5,.7,.9}`, adaptive and non-adaptive partners,
  held-out partners, switches/noise, and five evaluation seeds.
- [ ] Verify trace integrity and generate the preregistered table without
  manually editing results.
- [ ] Record action/forecast entropy, format validity, parameter sensitivity,
  outcome decomposition, exploitability, and group reward variance.
- [ ] Make the curriculum decision before training:
  if base behavior is unconditional, preserve the full parameter curriculum;
  if it already tracks parameters, increase partner/adaptation difficulty and
  treat A as a sufficiency test. Record the decision without changing test
  metrics.
- [ ] Use observed variance and Gate-0 throughput to freeze the confirmatory
  sample budget. The target full Nowak grid is 500 cells
  (4 ratios × 5 `w` × 5 `q` × 5 partners), 100 episodes per cell total,
  balanced across five evaluation seeds. If precision/cost forces a smaller
  design, revise and version the analysis spec before trained results exist.

**Evidence:** current engineering sample is
`results/base_grid/traces.jsonl`; formal evidence `PENDING`

### Gate 5 — Freeze training and prove learning signal quality

- [ ] Create immutable configs for Base evaluation and Models A/B/C/D/E. All
  shared hyperparameters, curriculum samples, initial checkpoint, LoRA targets,
  and training seeds must match; only the declared reward changes.
- [ ] Keep the ten-step config labeled engineering-only. Final configs use the
  frozen curriculum: reciprocity, adaptation, naturalistic, then group/CFE.
- [ ] For each variant, run a 50-update screening seed before the full run.
  Continue only if: trace/format validity ≥0.99; all losses finite; at least 25%
  of GRPO groups have non-zero within-group reward variance; no single reward
  value occupies >95% of samples; and validation behavior has not collapsed to
  one action/forecast.
- [ ] Verify every saved adapter loads on the frozen base model and reproduces
  a recorded validation response.
- [ ] Freeze any hyperparameter correction across variants. A variant-specific
  rescue is a new exploratory experiment, not part of the primary ablation.

The current pilot uses group size 4 and attention/linear-attention LoRA targets
for feasibility. Do not silently treat those as the final group size/target
choice; record the Gate-0 measurement and freeze the final choice here.

**Evidence:** `PENDING`

### Gate 6 — Train the Tier-1 ablation matrix

- [ ] Train A, B, and D for three independent final seeds with identical frozen
  curricula and selection rules.
- [ ] Train E for three seeds if making an HKB-mechanism claim.
- [ ] Train C for three seeds if making an isolated CFE claim; otherwise state
  that D evaluates only the combined objective.
- [ ] Checkpoints, manifests, validation traces, reward-component distributions,
  throughput, and terminal `COMPLETED` markers exist for every accepted seed.
- [ ] Select checkpoints using the frozen validation rule. Do not choose by
  confirmatory test performance.
- [ ] Produce a matrix audit showing exactly one intended reward difference
  between comparable variants.

A failed scientific hypothesis does not trigger ad-hoc reward redesign. Finish
the registered comparison first; any valenced-HKB rescue is separately labeled
future/exploratory work.

**Evidence:** `PENDING`

### Gate 7 — Run the internal confirmatory suites

- [ ] Run the frozen full Nowak grid for Base and every accepted Tier-1
  checkpoint with identical prompts, sampling, and seed allocation.
- [ ] Run amTFT/Axelrod tests: niceness, provokability, forgiveness,
  non-exploitability, cooperation with reciprocators, retaliation length, and
  value-defined punishment.
- [ ] Run HKB stress tests: lock-to-CC/DD/alternation, recovery after round-5
  noise, TFT→AD and AD→TFT round-6 switches, interleaved partners, and the
  `q≈c/b` threshold bands.
- [ ] Report adaptive/non-adaptive partner strata and all four joint outcomes.
- [ ] Validate traces before unblinding aggregate model labels. Run the frozen
  bootstrap/Holm analysis once and preserve its machine-readable output.

**Evidence:** `PENDING`

### Gate 8 — Implement and run external repeated-2x2 transfer

- [ ] Implement six payoff-table families: win-win, Prisoner's Dilemma, unfair,
  cyclic, biased (including Battle of the Sexes), and second-best.
- [ ] Use ten rounds, a full payoff table in the prompt, hidden randomized
  action labels, 100 games per family total balanced across five seeds, and
  held-out payoff matrices.
- [ ] Compare Base, Base+SCoT, A, B, and D under identical decoding. Include C/E
  only where their registered claims require it.
- [ ] Report coordination success, mismatch, individual/joint payoff,
  exploitability, and adaptation by family; Battle of the Sexes is the primary
  coordination transfer slice.
- [ ] Test the environment/analyzer on hand-solvable payoff tables and archive
  raw traces plus manifests.

**Evidence:** `PENDING`

### Gate 9 — Analyze without moving the goalposts

- [ ] Execute the preregistered analysis on immutable accepted runs.
- [ ] Publish effect sizes and nested-bootstrap CIs, raw and Holm-adjusted
  p-values, non-inferiority outcomes, training-seed dispersion, and exact sample
  counts. Separate confirmatory from exploratory output.
- [ ] Audit failure modes: P(DD) inflation, unconditional cooperation,
  exploitation by AD/opportunist, format collapse, forecast-to-mean behavior,
  and training/eval contamination.
- [ ] Assign exactly one evidence-based result branch:

  1. payoff-only is sufficient;
  2. HKB improves adaptation without harmful P(DD)/exploitability;
  3. HKB is inert/decorative (especially if E matches B);
  4. HKB stabilizes bad equilibria;
  5. full reward transfers beyond SCoT;
  6. transfer does not occur.

- [ ] Have a second reader reproduce headline numbers from manifests and raw
  traces, not from a manually assembled spreadsheet.

**Evidence:** `PENDING`

### Gate 10 — Release the complete Tier-1 artifact

- [ ] Regenerate all final tables and figures from a clean analysis output
  directory with one documented command.
- [ ] Complete the manuscript/report with scoped claims, implementation
  details, negative results, limitations, compute, statistical testing, and
  artifact locations.
- [ ] Include the method diagram, Nowak sensitivity, amTFT, HKB/outcome,
  transfer, and CFE figures only when their corresponding gates support them.
- [ ] Run a clean-checkout CPU test and dry-run; verify the external PRIME patch
  and cluster setup from `docs/CLUSTER_RL_RUNBOOK.md`.
- [ ] Create a final artifact index mapping every paper number/figure to its
  config, accepted run IDs, analysis table, and code revision.
- [ ] Search the repository/artifact for secrets and remove machine-specific
  transient caches. Never remove immutable scientific traces.
- [ ] Tag the release only after the artifact index and reproduction check pass.

**Evidence:** `PENDING`

## 5. Exact next actions

Do these in order:

1. Build the Gate-3 run wrapper, immutable manifest/status protocol, and
   overwrite refusal.
2. Build the trace validator and deterministic tidy analyzer.
3. Implement every registered metric plus EMA/Brier/Murphy and Holm/nested
   bootstrap logic against synthetic known-answer fixtures.
4. Prove one-command, byte-identical synthetic table/figure regeneration.
5. Only then allocate GPUs for the formal 500-episode base characterization;
   validate its traces and freeze the final sample budget/curriculum decision.
6. Freeze configs and run 50-update screens; repair only failures allowed by the
   frozen gate.
7. Run final seeds, internal suites, transfer, registered analysis, and the
   release gates without inserting Tier 2.

## 6. Known risks and stop decisions

| Risk | Detection | Required response |
|---|---|---|
| FlashInfer JIT cannot compile | First generation fails, missing `nvcc` | Stay at Gate 0; fix toolkit/backend and repeat one request |
| Prompt parameter without mechanic | Counterfactual semantic test is flat | Stay at Gate 2; implement or remove the causal claim |
| GRPO has zero advantages | Within-group reward variance below gate | Change task diversity/sampling before full training; rerun all screens consistently |
| HKB rewards DD | B raises P(DD), slows recovery, or worsens exploitability | Complete registered analysis; use adverse-result branch, do not hide it |
| HKB is decorative | E matches B | Reject specific HKB-mechanism claim |
| Forecasts regress to mean | Low Brier skill/resolution | Restrict CFE claim; report conformity failure |
| Training-seed instability | Effects change sign across seeds | Do not report a robust effect; add seeds only under a versioned analysis amendment |
| Base already solves task | Strong preregistered parameter/adaptation metrics | Treat A as sufficiency test and increase training difficulty only before trained results |
| Transfer fails | D does not beat Base+SCoT on biased games | Scope claim to in-domain coordination |
| Compute exceeds budget | Gate-0 extrapolation exceeds allocation | Reduce scope/sample plan before unblinding, or classify as feasibility work |

## 7. Audit commands

Use these to establish state at the start of a work session:

```bash
git status --short
git rev-parse HEAD
git -C /home/memoozd/scratch/rl/prime-rl rev-parse HEAD
git -C /home/memoozd/scratch/rl/prime-rl diff -- \
  src/prime_rl/trainer/models/qwen3_5_moe/converting_qwen3_5_moe.py
uv run pytest -q
source /home/memoozd/scratch/rl/prime-rl/.venv/bin/activate
rl @ configs/train_model_a_pilot.toml --dry-run
```

On a GPU node, also record:

```bash
hostname
nvidia-smi
module list
command -v nvcc
nvcc --version
python -c 'import torch,transformers,vllm,verifiers; print(torch.__version__, transformers.__version__, vllm.__version__, verifiers.__version__)'
```

The current repository snapshot was audited at project commit `6b7c288` with
uncommitted project work and PRIME commit `5f7e3ffca` plus the preserved
FP8-expert-count patch. Scientific manifests must record the actual revisions
at launch, not copy these values.
