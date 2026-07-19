# Gate 0 runtime evidence ledger

**Status:** complete — last updated 2026-07-17 21:57 EDT.

This ledger is the durable record for Gate 0 in
`docs/PROJECT_EXECUTION_PLAN.md`.  It distinguishes failed infrastructure
attempts from the first successful one-update smoke; none of these directories
is reused.

## Environment verified on the H100 allocation

- Allocation: Slurm job `17895062`, one 4×H100 80GB node.
- CUDA module: `cuda/12.9`.
- `nvcc`: CUDA toolkit `12.9.86` at
  `/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.9.1/bin/nvcc`.
- `CUDA_HOME`:
  `/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.9.1`.
- PRIME commit: `3d2dbae5f`.
- Qwen checkpoint:
  `/home/memoozd/scratch/models/Qwen3.6-35B-A3B-FP8`.
- Inference topology: vLLM TP=2 on two H100s; two trainer ranks on the
  remaining H100s.

FlashInfer successfully invoked that `nvcc` to compile
`gdn_prefill_sm90`.  This removes the original missing-`nvcc` blocker; do not
regress to a shell without the CUDA module and `CUDA_HOME` export.

## External PRIME compatibility fixes

1. The FP8 expert-count patch remains
   `patches/prime-rl-qwen35-fp8-expert-count.patch`.
2. PRIME `3d2dbae5f` requires its pinned Verifiers submodule revision
   `d5604bd3dfbfe402535c7ee7034f0ea03c02b4e2`.  A detached newer submodule
   (`19fdd9cae`) removed `TrainRunInfo`, `EvalRunInfo`, and `Trace.stamp`,
   causing the orchestrator to fail after rollout collection.  The local
   submodule was restored to the pinned revision.  See
   `docs/CLUSTER_RL_RUNBOOK.md` for the exact verification.
3. Dynamic packed Qwen-MoE batches are incompatible with the default trainer
   `torch.compile` graph cache in this PRIME revision.  The local PRIME change
   preserved in `patches/prime-rl-disable-trainer-compile.patch` makes
   compilation opt-in.  Gate-0 pilots intentionally have no
   `[trainer.model.compile]` block.
4. Full activation checkpointing also failed on dynamic routed-token shapes.
   The current pilot uses PRIME's selective norm-only checkpointing.  This
   avoids whole-block routing recomputation while retaining some memory relief.

These are external-stack feasibility fixes, not scientific changes.  Any
future PRIME update must reapply and revalidate them before a scientific run.

## Immutable run ledger

| Run directory | Status | Evidence |
|---|---|---|
| `results/gate0/model_a_smoke/20260717T190000Z-6b7c288-s2000` | Failed | Generation succeeded after FlashInfer compilation, but the newer Verifiers submodule raised `AttributeError: TrainRunInfo` before trainer handoff. |
| `results/gate0/model_a_smoke/20260717T192500Z-6b7c288-s2003` | Completed one-update smoke | `logs/orchestrator.log` records a clean step 1 with 16 rollouts, 4 trainable rollouts, and 0% errors.  It saved `weights/step_1/lora_adapters/adapter_model.safetensors` (65,061,752 bytes), `run_default/broadcasts/step_1/adapter_model.safetensors` (65,055,296 bytes), raw traces, and the trainer checkpoint.  Trainer evidence: loss `0.1595`, gradient norm `0.0796`, peak GPU memory 71.0 GiB. |
| `results/gate0/model_a_pilot/20260717T190000Z-6b7c288-s2001` | Failed after four updates | Full activation checkpointing raised `torch.utils.checkpoint.CheckpointError` during step 5: recomputed routed-token tensors had 10,743 rather than 10,744 rows.  Steps 1–4 had finite losses and gradient norms. |
| `results/gate0/model_a_pilot/20260717T204500Z-6b7c288-s2004` | Failed before first trainer update | Selective checkpointing removed the prior error, but default `torch.compile` failed on a cached dynamic stride assertion (`4096 != 8192`) in the compiled backward graph. |
| `results/gate0/model_a_pilot/20260717T213000Z-6b7c288-s2005` | Completed ten-update pilot | Fresh pilot with selective norm-only checkpointing and trainer compilation disabled.  All 10 trainer updates had finite loss/gradient; all rollout steps reported 0% errors; step-10 format validity was 1.00; and vLLM loaded `run_default/broadcasts/step_10/adapter_model.safetensors` (65,055,296 bytes) successfully.  The completed full export is `weights/step_10` (70,218,759,216 bytes) and the DCP checkpoint is `checkpoints/step_10/trainer`. |

## Gate 0 decision

Gate 0 is **complete**.  The `s2005` pilot reached step 10 and saved a
loadable, non-empty adapter.  Its final-step format validity was 1.00, all
rollout steps reported 0% infrastructure errors, and no NaN/Inf or engine-dead
error appears in its logs.  The terminal status and reproducibility record are
`STATUS`, `manifest.json`, and `resolved_config.toml` in the run directory.

## Measured feasibility budget

- End-to-end wall time: 12m 09s (21:45:04–21:57:13 EDT); the orchestrator
  step loop took 3m 55s.
- Trainer peak GPU memory: 71.0 GiB per trainer GPU; inference used the two
  remaining H100s with TP=2.
- Trainer update throughput: 116–1,407 tokens/s across steps (step 10: 546
  tokens/s).  Inference aggregate throughput in `final_summary.json`: 860.3
  tokens/s.
- Rollout generation: 16 rollouts per step, 10 recorded steps, 0% errors;
  aggregate final-step format validity 1.00.
- Storage after the full export: 190 GiB for the run directory; 70.2 GB is the
  consolidated step-10 model export, in addition to the DCP checkpoint,
  rollouts, logs, and W&B records.

## Session transfer

The compute-node Codex conversation history was copied to the login node at
`/home/memoozd/codex-history/Nowak-coordination/history-20260717T212300Z.jsonl`.
Source and destination SHA-256 both equal
`1ad22ff9ddbc82cb9b98ddfcfaa0673be4b1b581dc98ee642c91c04f63c44e1b`.

No scientific training, validation, or confirmatory evaluation has begun.
