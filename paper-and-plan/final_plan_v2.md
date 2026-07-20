# Evolutionary-Pressure Cooperation RL — Implementation Plan (v2)

**Status:** consolidated from `final_plan.txt`, `First_notes_on_evals.txt`, `literature_review.txt`, and the original NeurIPS draft, verified against current literature and current tooling docs (July 2026). Written for an AI implementation agent to execute. Every section below either confirms, corrects, or extends the source documents — corrections are marked **[FIX]**, new material is marked **[NEW]**, open decisions are marked **[DECIDE]**.

---

## 0. North star claim

> We train LLM agents with verifiable social-dilemma rewards and test whether they acquire reusable coordination skills — conditional cooperation, partner-specific adaptation, recovery after noisy defection, and resistance to exploitation — and whether these skills transfer from Donor's Game training to broader repeated games and commons-governance settings.

Not "we make the model moral." Not "HKB instills theory of mind." Both of the stronger claims are unsupported by the reward's actual fixed points (see §6 and the standing critique below) — keep the paper's claims scoped to what's tested.

**[FIX] Internal consistency:** the original draft's abstract claims three trained mechanisms (direct reciprocity, indirect reciprocity, group selection) while its Positioning section claims to be first to implement "all four" of Nowak's mechanisms, and its own Limitations section admits kin selection is only implicit and network reciprocity is "a diagnostic," not a trained pressure. This plan trains **three** mechanisms (direct, indirect, group) and evaluates a **cost-benefit axis**; it does not implement network reciprocity as a trained pressure or as an eval (see §8, Family 1). State this plainly in the paper rather than claiming four.

---

## 1. Scope and feasibility — read this before anything else

**[NEW]** This is the single most important addition to the plan. The eval suite across the three source documents is comprehensive — arguably too comprehensive for one paper cycle. Two independent scope inflations happened: the eval plan grew from "one radar chart" to seven eval families with dozens of sub-metrics, and the training plan grew from a 4B dense model (original draft) to a 35B-total MoE model with five to seven trained variants. Combined, this is a multi-month project scoped as if it were a one-week sprint.

**Recommendation:** split into a **minimum viable pass (Tier 1)** and a **stretch pass (Tier 2)**, and do not let Tier 2 block submission.

- **Tier 1 (required for any publishable claim):** Models A (Term 1), B (Term 1+HKB), D (Full); Nowak sweeps; amTFT metrics; HKB-specific stress tests (recovery, partner-switch, interleaved partners); one external transfer eval (Akata 2×2, prioritizing Battle-of-the-Sexes as the coordination test). This alone is substantial.
- **Tier 2 (only if Tier 1 finishes early and cleanly):** Model C (Term 1+CFE) as its own arm, shuffled-HKB / wrong-threshold / random-potential controls, Trust-and-Split, GovSim, MACHIAVELLI, SOTOPIA/SOTOPIA-ToM.

Do not run Tier 2 evals on a model that hasn't cleared Tier 1 sanity checks (§13, Phase 1).

**[FIX] Timeline realism:** §13's "Day 1 … Day 7" schedule is a good *sequence* of milestones but not a credible *calendar*. A single 250-step GRPO run at batch 128 × 8 rollouts/prompt is 256,000+ multi-turn generations; two such runs plus a full HKB/CFE/Nowak/amTFT implementation do not fit in three days on top of onboarding a training stack whose environment-authoring API changed days ago (§3). Treat "Day N" as "Phase N." Expect Phase 1 (env + baseline) to take 3–7 days, each training phase (Phases 3, 5) to take 3–10 days depending on cluster contention and debugging, and the full Tier 1 pass to take 3–6 weeks. Budget for at least one full "reward diversity is degenerate, restart with adjusted hyperparameters" cycle — this is normal for multi-turn RL, not a sign of failure.

---

## 2. Model and infrastructure

### 2.1 Model choice

`Qwen/Qwen3.6-35B-A3B` — **confirmed real**: released April 2026 by Alibaba, 35B total / 3B active MoE (256 experts, 8 routed + 1 shared active), Apache 2.0, native 262K context, compatible with Transformers/vLLM/SGLang. It is positioned and benchmarked primarily as an **agentic coding** model; there is no public evidence of particular strength at social role-play or game-theoretic reasoning specifically, unlike the models used in the LLM-cooperation literature this project builds on (Claude, GPT-4o, Gemini). This is not disqualifying, but:

**[NEW] Run the Day/Phase-1 baseline check before committing.** The original NeurIPS draft's own preliminary finding is that ~4B dense models cooperate near 100% regardless of game parameters (a sycophancy-like failure mode) while ~27B+ dense models track parameters naturally. Qwen3.6-35B-A3B has **3B active parameters per token** — closer in per-token compute to the "small, fails" regime than the "large, succeeds" regime, despite its 35B total weight count. Which regime it falls into is an open empirical question this plan doesn't currently answer before committing a training budget to it. Run the 500-episode base-model eval first (§13, Phase 1) and check for parameter-sensitivity before assuming this model needs the same curriculum as a dense 4B model.

**[DECIDE]** If the base-model check shows near-100% unconditional cooperation (the sycophancy failure mode) or near-100% unconditional defection, consider a fallback to a smaller dense model (e.g., Qwen3-4B, matching the original draft, for a clean small-model story) or a mid-size dense model (e.g., Qwen3.5-27B-class) rather than assuming the MoE model automatically inherits "large model" behavior.

### 2.2 Compute

- Minimum: 4× H100 80GB. Preferred: 8× H100 80GB.
- Use short training contexts (donor-game episodes are short); do not train at the model's native 262K context.
- **[NEW]** Before committing to the full training matrix, time a small number of rollout+update steps (e.g., 10 steps of Model A) and extrapolate wall-clock and cost. Do not assume a compute budget in advance — measure it empirically on Phase 1 hardware, since throughput for MoE rollout serving varies significantly by framework version and parallelism config.

### 2.3 Training stack

`prime-rl` + `verifiers`, SGLang or vLLM for rollouts, LoRA adapters, BF16 (QLoRA/4-bit fallback if memory-constrained).

**Confirmed:** both `prime-rl` and `verifiers` are real, actively maintained Prime Intellect projects. `prime-rl` explicitly supports MoE models, LoRA (including multi-tenant concurrent LoRA), and async rollouts with a dedicated orchestrator/trainer split. `verifiers` is the standard environment-authoring library, tightly integrated with `prime-rl` and the Environments Hub.

**[FIX] — important and time-sensitive:** `verifiers` shipped a **v1 rewrite within the last several days** (as of this writing) that restructures environments from a single bundled task+harness+reward object into three composable pieces: a **taskset** (data, tools, scoring), a **harness** (produces a rollout — ReAct loop, CLI agent, or custom), and a **runtime** (local or sandboxed execution). The environment-authoring code sketched in §4 below (and in the original `final_plan.txt`) reflects the pre-v1 bundled convention. **Before writing any environment code, check the current `verifiers` docs and confirm whether you're targeting v0.x (bundled) or v1 (taskset/harness/runtime) conventions** — the JSON schema and class structure differ. This is the single highest-risk "the plan is already stale" item in the whole document, precisely because it changed so recently.

**[DECIDE]** Confirm exact `pip`/`uv` install commands and package names against current docs rather than the commands below, which are illustrative:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv tool install prime          # Prime CLI
uv pip install verifiers open_spiel nashpy pandas scipy numpy matplotlib
# prime-rl: check current docs — may be a git clone + install rather than a pip package
```

Use OpenSpiel/Nashpy only for game-theory sanity checks (payoff-matrix validation, Nash-equilibrium reference points), not as the rollout engine — both are real, stable, well-established libraries and don't need re-verification.

### 2.4 MoE + LoRA — unresolved technical detail

**[NEW]** The target-module list inherited from the original (dense-model) paper — `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` — does not translate cleanly to an MoE model with 256 experts (8 routed + 1 shared active per token):

- Applying LoRA to `gate_proj,up_proj,down_proj` on **every expert** (not just the ones active for a given token) is standard practice for MoE LoRA but multiplies adapter parameter count by the expert count — a very different (and larger) adapter than the dense-model equivalent.
- Applying LoRA only to attention projections + the shared expert (leaving routed experts frozen) is cheaper and more stable but may under-adapt the model's per-domain behavior.
- Whether to adapt the **router/gating network** at all is a separate, higher-risk decision — routing instability under LoRA fine-tuning is a known failure mode in the MoE fine-tuning literature.

**[DECIDE]** Pick one of the above explicitly and document the choice; do not silently inherit the dense-model target list. Consult current `prime-rl` MoE+LoRA documentation for its default recommendation before deciding.

---

## 3. Repository structure

```
coop-rl/
  envs/
    donor_coord/
      __init__.py
      taskset.py       # or task/harness split, per verifiers v1 — confirm before writing
      harness.py
      verifier.py
      games.py
      partners.py
      rewards.py
      prompts.py
      configs/
        train_t1.toml
        train_t1_hkb.toml
        train_t1_cfe.toml
        train_full.toml
        eval_all.toml
  evals/
    donor_sweeps.py
    amtft_metrics.py            # [NEW] split out from coordination_stress for clarity
    coordination_stress.py
    forecast_calibration.py     # [NEW] Term-3-specific controls, see §8 Family 7
    akata_2x2_transfer.py
    trust_and_split.py
    govsim_wrapper.py
    analyze.py
  scripts/
    run_train.sh
    run_eval.sh
    make_tables.py
  results/
  paper/
```

---

## 4. The environment

**Confirm verifiers v1 vs pre-v1 conventions first (§2.3).** The task/episode design below is convention-agnostic and should port to either.

Each episode is one multi-turn iterated Donor's Game. Episode config (illustrative schema — adapt field names to whatever `verifiers` convention you land on):

```json
{
  "game_id": "donor_direct_001",
  "mode": "simple",
  "b": 4,
  "c": 1,
  "w": 0.7,
  "q": 0.8,
  "horizon_min": 6,
  "horizon_max": 12,
  "partner_policy": "noisy_tit_for_tat",
  "noise_rate": 0.05,
  "reputation_visible": true,
  "partner_switch_round": null,
  "seed": 123
}
```

Prompt (strict output format, short generations — 32–64 max new tokens, no long chain-of-thought during training):

```
You are playing a repeated social-dilemma game.

In each round, choose one action:
- COOPERATE: pay a cost to benefit the other agent.
- DEFECT: do not pay the cost.

Benefit to recipient: b = 4.
Cost to cooperator: c = 1.
The same partner may appear again with probability w = 0.7.
Reputation information is visible with probability q = 0.8.

You must output exactly:
ACTION: COOPERATE or ACTION: DEFECT
FORECAST_GROUP_COOP: a number from 0.00 to 1.00

History:
Round 1: you cooperated, partner cooperated.
Round 2: you cooperated, partner defected.

Current round: 3.
Your output:
```

Use a token-preserving multi-turn client (don't re-tokenize/fragment rollouts across turns — check current `prime-rl` guidance on this, and be aware Qwen-family chat templates with `<think>`-style tags are a documented common source of multi-turn parsing bugs).

---

## 5. Partner policies

Training pool: `always_cooperate, always_defect, tit_for_tat, generous_tit_for_tat, grudger/grim_trigger, win_stay_lose_shift, random_p, noisy_tit_for_tat, opportunist`.

Held-out eval-only pool (prevents memorizing the training partner set): `forgiving_grudger, delayed_tit_for_tat, probabilistic_defector, copy_with_noise_10%`.

**[NEW] — flag from the earlier reward-design critique, relevant to partner-pool design:** several training partners (`always_cooperate`, `always_defect`, `random_p`) are non-adaptive — they do not respond to the agent's own actions. The HKB coordination term (§6, Model B) is motivated by *bidirectional coupling* between two adapting agents; against a non-adaptive partner, "coordination" reduces to "conform to a fixed or noisy external signal," which is a different and weaker claim than the one the term's citations (Kelso, Tognoli, Hancock et al.) support. Report HKB-specific metrics (§8, Family 3) **broken out by partner adaptivity** (adaptive: TFT-family, WSLS, opportunist vs. non-adaptive: AC, AD, random) rather than pooled — pooling risks averaging away exactly the distinction that matters for interpreting whether Term 2 is doing anything like coupling.

---

## 6. Reward functions

Train the following variants. All rewards other than Model A are added on top of the Term-1 payoff, not in place of it.

### Model A — Term 1 only (payoff-only RL)

```
R_1 = (π_i + c) / (b + c),   π_i = b·1[a_j=C] − c·1[a_i=C]
```

Baseline: is ordinary game payoff sufficient for conditional cooperation?

### Model B — Term 1 + HKB

```
s_t = a_i^t · a_j^t                          (C=+1, D=−1)
s̄_t = (1/W) Σ_{τ=t-W+1}^{t} s_τ,   W = 4
φ_t = π(1 − s̄_t) / 2
R_2 = 4q·cos(φ_t) + (c/b)·cos(2φ_t)
R = R_1 + λ_2 · norm(R_2),   λ_2 = 0.15
```

`norm(R_2)` (rescaling the HKB term into a comparable range to `R_1` before weighting) is a genuine improvement over the original draft, which added the raw, unnormalized HKB term directly — worth keeping.

**Standing theoretical caveat (unchanged from prior review, restated for the implementer):** this reward is valence-blind — mutual cooperation and mutual defection both register as "in-phase" (φ≈0) and score identically. Nothing in Term 2 alone favors the cooperative well over the defective one; that pull comes entirely from Term 1. Additionally, the reward is maximized *deep in a well* (commitment) and *minimized at the ridge* (φ≈π/2) — but the cited HKB/metastability literature (Kelso 2012; Hancock et al. 2024) identifies the ridge, not the well, as the regime that provides flexible, low-barrier transition capacity. The reward, taken literally, rewards rigidity and calls it flexibility. **This is precisely why the stress tests in §8 Family 3 (recovery-after-noise, partner-switch adaptation) are not optional extras — they are the actual empirical test of whether this theoretical concern materializes in practice.** If recovery and adaptation numbers come back strong for Model B relative to Model A, that's real evidence against the theoretical worry regardless of the reward's fixed-point structure; if they don't, the theoretical worry is confirmed. Do not skip these evals even under Tier-1 time pressure — they are more informative about whether Term 2 "works" than the Nowak-threshold eval is.

**[FIX] — δω justification:** the original draft sets the HKB detuning term δω=0 uniformly, justified by "self-play, both sides sampled from the same weights." Stage 1/2 training (§7) uses a **fixed heuristic partner pool** (TFT, AC, AD, random, grudger, etc.), not self-play, for a large fraction of training. By the draft's own stated logic ("non-zero δω would only be appropriate when training two distinct policies against each other"), a static non-learning bot is exactly such a case. Either justify δω=0 against fixed partners on different grounds, or acknowledge this as an unresolved asymmetry in the writeup.

### Model C — Term 1 + CFE (collective forecast calibration)

```
R_3 = −(ĝ_t − c̄_t)²          (Brier loss; more numerically stable than clipped Bernoulli KL for training)
R = R_1 + λ_3 · R_3,   λ_3 = 0.05
```

Call this **"collective forecast calibration"** or **"group self-modeling pressure"** in the paper, not "cheap talk," unless the peer-prediction/proper-scoring-rule grounding is added explicitly (see §8 Family 7 — Crawford & Sobel's cheap-talk model assumes genuinely costless communication and preference misalignment as the source of coarse equilibria; this term imposes a real training cost for miscalibration, which is a costly-signaling / proper-scoring-rule mechanism, not a cheap-talk one).

### Model D — Full

```
R = R_1 + 0.15·norm(R_2) + 0.05·R_3
```

### Model E — shuffled-HKB control (Tier 1 if time allows, otherwise Tier 2)

Same as Model B, but φ is computed from a mismatched episode/partner. If this performs comparably to real HKB, the HKB signal isn't doing anything — this is the load-bearing negative control for Model B's entire claim.

### Models F, G — additional controls (Tier 2)

- **Wrong-threshold HKB:** deliberately mis-set the coupling/bistability constants so the phase transition does *not* land on q=c/b. Tests whether the specific Nowak-aligned threshold matters, or whether any bistable shaping helps equally.
- **Random-potential HKB:** replace the HKB potential with an unrelated smooth potential of similar magnitude. Tests whether the specific HKB functional form matters, or whether any shaping signal of the right scale would do.

### KL-to-reference anchor

`KL_to_reference: 0.02, increasing to 0.05 if outputs degenerate` — this is a good practical addition not present in the original paper's zero-KL DAPO setup, and it helps guard against outright policy collapse/degenerate outputs.

**[NEW] — scope of what this does and doesn't fix:** a KL anchor to a reference policy is not the same thing as Ng-Harada-Russell potential-based reward shaping, and does not on its own guarantee that Term 2/3 preserve whatever conditional-cooperation policy Term 1 alone would train (see the standing critique: shaping terms not expressed as a discounted potential difference across a transition are not proven policy-invariant). It's a useful collapse-prevention safety net, not a substitute for the Model A vs. B vs. D ablation comparison, which remains the only real evidence of whether Term 2/3 distort or refine Term 1's behavior.

---

## 7. Training hyperparameters and curriculum

```
algorithm: GRPO / DAPO-style GRPO
LoRA rank: 16, alpha: 32, dropout: 0.05     [see §2.4 — target modules undecided for MoE]
learning rate: 1e-5
KL to reference: 0.02 → 0.05 if outputs degenerate
clip range: 0.2
rollouts per prompt: 8
batch size: 128 episodes
temperature: 0.7, top_p: 0.95
max_new_tokens: 64
max context length (training): 4096
gradient checkpointing: on
eval every: 25 steps, save every: 50 steps
```

**Curriculum (800 steps total, 4 stages):**

1. **Steps 0–250 — direct/indirect reciprocity.** Simple donor game, partners AC/AD/TFT/random/grudger, vary b/c/w/q. Goal: parameter-sensitive conditional cooperation.
2. **Steps 250–500 — robustness/adaptation.** Noisy TFT, opportunist, WSLS, delayed TFT; noise 0.05–0.15; partner switches and interleaving enabled. Goal: forgiveness, adaptation, non-exploitability.
3. **Steps 500–700 — abstract naturalistic scenes.** Everyday scenarios (borrowing a ladder, unreliable collaborator, etc.) with randomized `CHOICE A/CHOICE B` labels, verifier-only mapping to C/D. Goal: transfer away from game-language memorization.
4. **Steps 700–800 — group/CFE episodes.** 4–5 agents, small public-goods/group donor game, forecast required. Goal: group forecast calibration.

Before committing to 800 steps for every model: run each variant's Stage-1 slice first and check for reward diversity (per Prime's own guidance — if reward is ~0% or ~80%+ immediately, something is wrong with task difficulty before you've spent the rest of the budget).

---

## 8. Evaluation suite

### Statistical rigor — read before running any eval **[NEW]**

None of the three source documents specify a seed count beyond "3," nor a significance-testing plan, despite the original paper's own NeurIPS checklist item 7 ("statistical significance") being marked `[TODO]`. Given how many metrics × ablations × eval families this plan collects, garden-of-forking-paths risk is real. Before running Tier 1:

- Use **at least 5 seeds** per condition where compute allows (3 is thin for LLM sampling variance); report bootstrap confidence intervals, not just point estimates.
- **Pre-register the confirmatory comparisons** before looking at results — at minimum: (Model A vs. B) on {recovery time, partner-switch adaptation, mismatch rate}; (Model A vs. D) on {Akata coordination success}; (Model B vs. E) on all Family-3 metrics. Apply a multiple-comparisons correction (Holm or Bonferroni) across this pre-registered set. Everything else collected is exploratory and should be reported as such.
- Report the outcome decomposition `P(CC), P(CD), P(DC), P(DD)` alongside every `P(C)` figure — a model can look "cooperative" while exploitable, or "coordinated" while stabilizing mutual defection.

### Family 1 — Nowak parameter sweeps

```
b/c: [2,3,5,8]   w: [0.1,0.3,0.5,0.7,0.9]   q: [0.1,0.3,0.5,0.7,0.9]
partners: TFT, AD, AC, random, grudger
episodes/cell: 100, seeds: ≥5 (see stats note above)
```

Report `Spearman ρ(P(C), w)`, `ρ(P(C), q)`, `ρ(P(C), −c/b)`, mean reward, regret vs. oracle, plus the `P(CC)/P(CD)/P(DC)/P(DD)` decomposition.

**[FIX] scope:** this plan does not implement a network-reciprocity (graph-topology, degree-k) manipulation anywhere in the training or eval design, despite the original paper's Figure 2 radar chart listing "network/group selection" as an axis. Either add an explicit network-structured multi-agent eval (who-plays-whom determined by graph degree k, per Nowak's b/c > k condition) or drop network reciprocity from any figure/claim — don't let it survive as a label with no matching experiment, repeating the original inconsistency this plan was meant to fix.

### Family 2 — amTFT / Axelrod behavioral metrics

Correctly attributed: the "nice, retaliatory, forgiving, clear" framing is Axelrod's (1984) description of why Tit-for-Tat succeeded in his tournaments; amTFT (Lerer & Peysakhovich, 2017) is the RL paper that operationalizes testing an agent against these desiderata in Markov social dilemmas, notably defining "defection" in terms of value lost rather than the literal action taken — worth citing both explicitly rather than "the amTFT tradition."

| Metric | Operationalization |
|---|---|
| Niceness | P(C) on round 1 vs. unknown partner |
| Provokability | drop in P(C) after 2 consecutive partner defections |
| Forgiveness | P(return to CC within 3 rounds) after one noisy defection |
| Non-exploitability | payoff loss vs. AD/opportunist relative to a safe-defect policy |
| Cooperation with cooperators | P(CC) vs. TFT/generous-TFT |
| Retaliation length | rounds of defection after one accidental partner defection |
| **[NEW] Value-defined punishment** | does the model punish actions that reduce *joint value*, or merely actions labeled D? (amTFT's actual innovation — worth testing directly rather than assuming action-level punishment is equivalent) |

### Family 3 — HKB-specific coordination stress tests

The load-bearing eval family for Model B's claim. Compare Base, A, B, C, D, E (§6).

- **Phase-locking time** `T_lock`: rounds until action-relation is stable for 3 consecutive rounds. Report separately for lock-to-CC, lock-to-DD, lock-to-alternation — collapsing these together hides exactly the valence-blindness concern above.
- **Recovery after noise:** cooperative partner (generous TFT), forced accidental defection at round 5, measure `T_recover` and `P(recover within 3 rounds)`.
- **Partner-switch adaptation:** TFT→AD and AD→TFT at round 6. Measure `T_adapt`, post-switch regret.
- **Interleaved partners:** alternating partner A (reciprocator) / partner B (defector). Measure `ΔP(C) = P(C|A) − P(C|B)` — direct test of the per-partner φ-window design (histories reset across games/partners).
- **q≈c/b threshold band:** three bands (`q < c/b−0.15`, `q≈c/b`, `q > c/b+0.15`), measuring P(C), action entropy, phase-locking time, recovery, P(CC) vs. P(DD). If Model B doesn't differ from Model A here, the HKB-Nowak alignment claim isn't earning its keep.
- **[NEW]** Break out every metric above by partner adaptivity (adaptive vs. non-adaptive partner pool — see §5).

### Family 4 — Akata-style repeated 2×2 transfer

```
Game families: win-win, Prisoner's Dilemma, unfair, cyclic, biased, second-best
[FIX: six families total — Battle of the Sexes is the canonical example within "biased," not a 7th family]
10 rounds, full payoff table in prompt, 100 games/family, ≥5 seeds
Baselines: base model, base+SCoT prompt ("predict what the other player is likely to do, then choose"),
           Term 1 only, Term 1+HKB, Full
```

Akata et al. found LLMs perform well in self-interested games (PD-family) but poorly in coordination games (Battle-of-the-Sexes, the flagship biased-game example) — this remains the single most direct, well-supported transfer question this plan can ask: does Donor's-Game RL improve coordination generally, or only PD-like play specifically? If the trained model can't beat the cheap SCoT-prompting baseline on coordination games, the training story is weak regardless of Donor's-Game internal results.

### Family 5 — Trust-and-Split (communication transfer)

Simplified version of the environment introduced alongside Advantage Alignment for LLMs: private HIGH/LOW valuation, one message, then allocation proposals. Efficient outcome: allocate to whoever values it more; greedy outcome: lie or demand everything.

Metrics: truthful-communication rate, efficient-allocation rate, greedy-claim rate, exploitability vs. a lying partner, joint payoff, fairness over repeated rounds. This is the cleanest test of whether Term 3 (forecast calibration / group self-modeling) does anything for real communication, as opposed to the toy in-game forecast token.

### Family 6 — GovSim transfer (Tier 2)

Fishery scenario first (5 agents, communication enabled, greedy-newcomer perturbation in half of runs). Metrics: survival time (months), collapse rate, resource overuse, inequality (Gini), greedy-newcomer robustness, negotiation quality, belief-about-others score.

**[NEW] Manage expectations explicitly.** Independent reproductions of GovSim show even frontier-scale models (GPT-4o, GPT-4-turbo) frequently fail to sustain the commons, and most tested LLMs (43 of 45 model×scenario combinations in the original paper) collapse rather than cooperate. A LoRA-tuned 3B-active-parameter model trained only on 2-player Donor's Game transferring cleanly to 5-agent, natural-language commons negotiation is a genuine long shot. Treat a null result here as informative (donor-game training doesn't automatically transfer to open-ended multi-party negotiation) rather than as a sign the training method failed — and don't let this be the eval the paper's headline claim depends on.

### Family 7 — Term-3-specific controls **[NEW]**

Not present in any of the three source documents, and needed given the earlier critique that naive consensus-scoring rewards conformity over genuinely informative reporting (the exact failure mode proper-scoring/peer-prediction mechanisms like Bayesian Truth Serum were built to counteract):

- **Base-rate-forecaster baseline:** implement a simple non-learned forecaster (exponential moving average of the recent group cooperation rate) and compute a **Brier Skill Score** for the trained model's forecasts relative to this baseline. If the trained forecaster barely beats the moving average, CFE training is teaching calibration-to-mean, not situational reasoning.
- **Murphy (1973) decomposition:** split Brier score into reliability (calibration), resolution (sharpness/discrimination), and uncertainty components. A model that improves reliability but not resolution is reporting accurate-on-average but uninformative forecasts — exactly the conformity failure mode to watch for.
- **Episode-specificity check:** compare forecasts across episodes with deliberately different group compositions/histories; a genuinely situational forecaster should show meaningfully different `ĝ` values across these, not a near-constant output.

### Optional secondary eval — MACHIAVELLI

Keep as secondary, not headline, per the original literature review's own recommendation — contamination and "ethics benchmark score" ambiguity make it a weaker central claim than the amTFT/Akata/Trust-and-Split trio. Use it as a warning light: if ethical-violation rate rises alongside reward, say so plainly rather than headlining the reward gain.

---

## 9. Ablation / control matrix (compiled)

| Model | Reward | Tier | Purpose |
|---|---|---|---|
| Base | none | 1 | existing behavior |
| A | Term 1 | 1 | payoff-only baseline |
| B | Term 1 + HKB | 1 | isolates coordination shaping |
| C | Term 1 + CFE | 2 | isolates forecast calibration |
| D | Full | 1 | final method |
| E | Term 1 + shuffled-HKB | 1 if time, else 2 | is the HKB signal real or decorative? |
| F | Term 1 + wrong-threshold HKB | 2 | does the specific q=c/b alignment matter? |
| G | Term 1 + random-potential HKB | 2 | does any shaping of similar scale work equally? |
| SCoT-prompted base | no training | 1 (transfer only) | cheap strong baseline for coordination transfer |
| Naive GRPO/MARL baseline | payoff-only, no shaping, self-play | 2 | reviewer-proofing against the Advantage-Alignment finding that naive MARL/GRPO drives greedy behavior |

Evaluate all Tier-1 models on: Nowak sweeps, amTFT metrics, HKB stress tests (Family 3), Akata 2×2 transfer. Tier-2 models additionally on: Trust-and-Split, GovSim, Term-3 controls, MACHIAVELLI.

**Primary confirmatory comparison (pre-registered, see §8):** Term 1 + HKB vs. Term 1 only must improve phase-locking time, recovery-after-noise, partner-switch adaptation, and interleaved-partner separation **without** increasing P(DD) or Battle-of-the-Sexes mismatch rate. This is the cleanest evidence Term 2 matters, and the cleanest way to falsify it.

---

## 10. Figures and tables

1. **Method diagram:** episode → action+forecast → verifier computes payoff/HKB/CFE → GRPO update → eval on internal + external games.
2. **Nowak radar:** direct reciprocity, indirect reciprocity, cost sensitivity axes only (network/group selection dropped or backed by a real eval — see §8 Family 1 fix). Rows: Base, A, B, D.
3. **amTFT bar chart:** niceness, provokability, forgiveness, non-exploitability, cooperation-with-reciprocators, value-defined punishment.
4. **HKB ablation line plots:** recovery-after-noise, partner-switch adaptation, phase-locking time, P(CC) vs. P(DD) — split by partner adaptivity.
5. **Transfer:** Akata 2×2 families, joint payoff / coordination success, with SCoT baseline shown alongside.
6. **Trust-and-Split / GovSim (if Tier 2 completes):** survival/collapse rate, truthfulness, efficient allocation, greedy behavior.
7. **[NEW] Term-3 calibration:** Brier skill score vs. base-rate baseline, reliability/resolution decomposition.

---

## 11. Success criteria and decision tree

The paper works if you can show:

1. Term 1 beats base on parameter sensitivity (tracks w, q, c/b).
2. Term 1+HKB beats Term 1 on coordination-dynamics metrics (Family 3), without merely inflating P(DD).
3. Full model transfers to at least one external setting better than the SCoT-prompted baseline.
4. Ethics/exploitability doesn't quietly degrade wherever reward improves (report as a warning if it does; don't headline reward gains that come with rising ethical-violation or exploitability rates).

**If Term 1 alone already tracks everything:** publishable as "payoff-only RL is sufficient for conditional cooperation; HKB is unnecessary" — use HKB ablation as an informative negative result, not a null publication.

**If HKB helps adaptation:** the strongest version — "coordination-shaping improves partner-specific adaptation beyond payoff-only RL."

**If HKB increases mutual defection:** report this directly rather than suppressing it — "valence-neutral coordination rewards can stabilize bad equilibria" is itself a real, publishable finding, and motivates a valenced-HKB rescue term (e.g., gating R_2 on joint welfare exceeding a safe baseline) as future work, introduced *after* reporting the original result, not instead of it.

**If Tier 2 transfer (GovSim/Trust-and-Split) succeeds:** headline claim — "Donor-game training transfers to realistic social-dilemma coordination." Given §8 Family 6's feasibility caveat, don't build the paper's central claim on this succeeding.

---

## 12. Phased plan (relabeled from "days" — see §1)

**Phase 1 (env + baseline, expect 3–7 days):** Confirm `verifiers` v0 vs. v1 conventions. Build `donor_coord` environment: simple donor game, partner policies, strict parser, R1, episode logging. Run base-model eval on 500 episodes — check for parameter sensitivity vs. sycophancy-like unconditional cooperation/defection (§2.1) before proceeding.

**Phase 2 (reward/eval infra):** Implement HKB reward, CFE/Brier reward, Nowak sweep harness, amTFT metric harness, base-rate-forecaster baseline (§8 Family 7). Run sanity plots on the base model.

**Phase 3 (Tier-1 training, part 1):** Train Model A and Model B. Check reward diversity after the Stage-1 slice before committing to the full 800 steps (§7). Do not proceed to full training if reward is near-0% or near-ceiling immediately.

**Phase 4 (Tier-1 eval, part 1):** Evaluate A/B on Nowak sweeps, HKB stress tests, outcome decomposition. Decision point: does HKB help, per the pre-registered comparison (§9)?

**Phase 5 (Tier-1 training, part 2):** Train Model D (Full); Model C and controls E/F/G if Tier-2 budget allows.

**Phase 6 (transfer evals):** Akata 2×2 transfer (Tier 1); Trust-and-Split, GovSim, MACHIAVELLI if Tier 2.

**Phase 7 (writeup):** Build tables/figures from the pre-registered comparisons first, exploratory results clearly labeled as such. Decide final framing based on which branch of §11's decision tree the results landed in — do not decide the framing before the results are in.

---

## 13. Paper framing

> LLM agents will increasingly interact with other agents in mixed-motive environments where individual incentives conflict with collective welfare. Prior work shows LLMs can behave selfishly in repeated games, struggle with coordination, and may become greedier under naive multi-agent RL. We study whether verifiable evolutionary-game rewards can post-train LLM agents for adaptive, non-exploitable coordination. We train LoRA policies in iterated Donor's Games with payoff, dyadic coordination, and group-forecast rewards, and evaluate not only cooperation rate but niceness, provokability, forgiveness, exploitability, recovery after noise, partner-switch adaptation, and transfer to repeated 2×2 games [and commons-governance environments, if Tier 2 completes]. We argue that social-dilemma post-training should be evaluated as adaptive coordination, not raw cooperation rate or generic morality — and we report where our own coordination-shaping term helps, where it doesn't, and where its own literature basis (HKB metastability) predicts a different mechanism than a naive reading of "more coordination is better" would suggest.

Candidate titles: *"Evolutionary Pressures for Adaptive Coordination in Language Model Agents"* or *"Training Language Model Agents for Non-Exploitable Cooperation."* Avoid "cooperative language models" / "morality" framing unless results specifically support ethics claims beyond the MACHIAVELLI secondary check.

---

## 14. Consolidated risk register

| Risk | Where addressed |
|---|---|
| `verifiers` v1 API change makes environment code stale | §2.3, §4 |
| Qwen3.6-35B-A3B's 3B active params may behave like "small" not "large" regime | §2.1 |
| MoE + LoRA target-module choice unresolved | §2.4 |
| 7-day schedule not credible at this scope | §1, §12 |
| GovSim/SOTOPIA-ToM transfer may fail regardless of method quality | §8 Family 6 |
| Term 2 valence-blindness / rigidity-vs-metastability | §6 Model B, §8 Family 3 |
| Term 3 reward-hacking toward base-rate conformity | §6 Model C, §8 Family 7 |
| Network reciprocity claimed but not tested | §8 Family 1, §10 |
| δω=0 justification inconsistent with fixed-partner training | §6 Model B |
| No pre-registered confirmatory comparisons / multiple-comparisons risk | §8 |
| Reward terms not proven policy-invariant (Ng-Harada-Russell) | §6, KL-anchor note |

---

## 15. Key sources to keep on hand

Nowak (2006); Kelso (2012); Hancock et al. (2024); Axelrod (1984); Lerer & Peysakhovich (2017, amTFT); Tennant et al. (2025, Moral Alignment for LLM Agents); Duque et al. (2025, Advantage Alignment) and Piché et al. (2025, Learning Robust Social Strategies with LLMs); Akata et al. (2025, Nature Human Behaviour); Piatti et al. (2024, GovSim); Vallinder & Hughes (2024, Donor Game — note: prompting-based, no loss function); Zhou et al. (2024, SOTOPIA); SOTOPIA-TOM (2026); Ng, Harada & Russell (1999, potential-based reward shaping); Murphy (1973, Brier score decomposition); Prelec (2004, Bayesian Truth Serum).
> **Operational supersession (2026-07-20):** This document preserves the broad
> scientific motivation and diagnostic suite, but its broad HKB confirmatory
> family is superseded by `docs/ANALYSIS_SPEC.md` v1.3 and
> `docs/PROJECT_EXECUTION_PLAN.md`. The only confirmatory HKB claim is rapid
> cooperative recovery in the fixed generous-TFT forced-disruption suite,
> compared B−A and B−E with the registered safety gate. Switch, interleaving,
> threshold, amTFT, and transfer outcomes remain mandatory diagnostics and
> must not be presented as confirmatory or generalized from a recovery result.
