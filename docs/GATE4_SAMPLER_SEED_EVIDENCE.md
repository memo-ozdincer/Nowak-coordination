# Gate 4 sampler-seed and launch-path evidence

**Status:** engineering prerequisite passed; formal 500-episode cohort not
started.

## Why metadata alone was rejected

The Gate-3A implementation initially copied each world seed into
`sampling_metadata.requested_seed`. That proved only what the task intended,
not what Verifiers sent or vLLM used. Gate 4 now requires agreement among:

1. the task's requested seed;
2. `Trace.agent.sampling.seed`, produced from the actual per-task
   `ModelContext` used by the Verifiers client;
3. an immutable `sampler_seed_evidence` record written only after the rollout;
4. a direct black-box vLLM same-seed/different-seed probe.

`nowak_coordination.seeded_eval` constructs a separate sampling object for
every task and fails the run if any trace records a different seed, contains a
framework error, stops with `error`, or fails to complete. The strict trace
validator independently requires the same three-way agreement for validation
and test traces.

## Preserved engineering attempts

| Run | Result | Finding |
|---|---|---|
| `results/gate4/seed_transport_smoke/20260720T095500Z-ad6d2d7-s2901` | Invalid despite its old wrapper status | The inherited `qwen-nowak` alias was not advertised by this server; vLLM returned 404. This exposed that Verifiers lifecycle completion is not equivalent to an error-free rollout. The evaluator now rejects captured errors and `stop_condition=error`. |
| `results/gate4/seed_transport_smoke/20260720T100300Z-ad6d2d7-s2901` | Invalid | The real checkpoint path reached vLLM and produced valid responses, but the scientific trace header was empty. `setup_task()` ran before the per-rollout state channel was attached. Provenance initialization now occurs again inside the first state-synchronized response. |
| `results/gate4/seed_transport_smoke/20260720T101200Z-ad6d2d7-s2901` | **Passed** | Three valid model turns, zero errors, complete terminal trace, strict trace validation PASS, and requested/effective seed `2901` agrees in task header, agent sampling, and evidence. |

No failed or engineering trace is eligible for the formal cohort.

## Black-box vLLM proof

The immutable probe is
`results/gate4/sampler_seed_probe/20260720T102000Z-ad6d2d7-s2901`.
Its terminal status and `seed_probe.json` both report `PASS`.

- Prompt SHA-256:
  `dfebf35cbcaaa4de8f2e799f945bce674c702495f76f3d6505cd8a8f6c5ca794`.
- Seed sequence: `2901, 2901, 2902, 2903, 2904, 2905`.
- Both seed-2901 requests returned the identical string
  `KJXNQWPLMRTBHZGDFYSAUVCE`.
- Every recorded different seed produced a different response from seed 2901.
- Each exact request payload, including its seed, has its own SHA-256 in
  `seed_probe.json`.

This is black-box evidence that the live vLLM endpoint consumes the request
seed. It does not claim cross-version bitwise reproducibility; manifests still
freeze the full serving stack.

## Live environment

The proof ran interactively on `g9.nibi.sharcnet` using two H100 80GB GPUs for
TP=2 inference. CUDA 12.9 and its `nvcc` were exposed through `CUDA_HOME`.
The checkpoint was
`/home/memoozd/scratch/models/Qwen3.6-35B-A3B-FP8`; Torch was
`2.11.0+cu128` and vLLM was `0.24.0`. Exact project/PRIME revisions, dirty
diff hashes, package versions, model/tokenizer hashes, hardware, command, and
terminal state are retained in each run manifest.

## Gate decision

The sampler-seed injection prerequisite is passed. Launch authorization still
depends on the independent audit and frozen hash of Analysis Spec v1.4 plus
the exact Gate-4 registry tests. No formal validation trace has been generated.
