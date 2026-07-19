# Agent operating instructions

This repository is an active research project, not a finished implementation.
Before changing code or launching a run, read:

1. `docs/PROJECT_EXECUTION_PLAN.md` — current state, mandatory gates, evidence,
   and the exact order of work.
2. `docs/CLUSTER_RL_RUNBOOK.md` — environments, cluster commands, and local
   PRIME/Qwen fixes.
3. `paper-and-plan/final_plan_v2.md` — scientific motivation and the broader
   Tier-1/Tier-2 design.

`docs/PROJECT_EXECUTION_PLAN.md` is the operational source of truth. Follow its
first incomplete gate; do not skip a gate because a later experiment is easier
to launch. A gate is complete only when its acceptance checks pass and its
evidence path is recorded in the plan. Logs or an output directory alone do not
count as a completed run.

For scientific runs:

- Never overwrite a previous result directory. Use a unique run ID and save the
  resolved config plus a manifest containing code/environment revisions and
  seeds.
- Keep training, validation, and confirmatory-evaluation seeds separate.
- Do not inspect confirmatory results before freezing `docs/ANALYSIS_SPEC.md`.
- Do not infer indirect/direct reciprocity from prompt parameters unless the
  corresponding `q`/`w` mechanics have passed the semantic tests in Gate 2.
- Do not report coordination without the `CC/CD/DC/DD` decomposition or pool
  adaptive and non-adaptive partners in HKB analyses.
- Keep Tier 2 quarantined until every Tier-1 completion requirement passes.

After completing or invalidating a gate, update the status date, checkbox, and
evidence in `docs/PROJECT_EXECUTION_PLAN.md` in the same change. Preserve
secrets in `.secrets/`; never commit or print API keys.
