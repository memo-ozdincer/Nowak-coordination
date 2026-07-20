"""Scientific Verifiers evaluation with an audited per-task sampler seed."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import tomllib


def requested_sampler_seed(task: object) -> int:
    """Return the task's registered model-sampler seed or fail closed."""

    data = getattr(task, "data")
    metadata = getattr(data, "sampling_metadata")
    seed = metadata.get("requested_seed")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("every scientific task requires a non-negative requested sampler seed")
    return seed


async def run_seeded_eval(config_path: Path, output_dir: Path) -> None:
    """Run one immutable evaluation block, varying sampling through each task's registry."""

    from verifiers.v1.clients import ModelContext, resolve_client
    from verifiers.v1.cli.output import append_trace, save_config
    from verifiers.v1.configs.eval import EvalConfig
    from verifiers.v1.env import Environment
    from verifiers.v1.trace import EvalRunInfo

    if (output_dir / "traces.jsonl").exists():
        raise FileExistsError(f"refusing to overwrite {output_dir / 'traces.jsonl'}")
    raw = tomllib.loads(config_path.read_text())
    raw["output_dir"] = str(output_dir)
    config = EvalConfig.model_validate(raw)
    if config.num_rollouts != 1:
        raise ValueError("scientific seeded evaluation requires exactly one rollout per task")
    if config.shuffle:
        raise ValueError("scientific seeded evaluation forbids task shuffling")

    env = Environment(config)
    tasks = env.taskset.select(config.num_tasks, False)
    if not tasks:
        raise ValueError("evaluation registry produced no tasks")
    client = resolve_client(config.client)
    semaphore = asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
    write_lock = asyncio.Lock()
    save_config(config, output_dir)

    async def on_complete(trace: object) -> None:
        requested = requested_sampler_seed(trace.task)
        sampling = trace.agent.sampling if trace.agent is not None else None
        effective = getattr(sampling, "seed", None)
        if effective != requested:
            raise RuntimeError(
                f"trace {trace.id}: effective sampler seed {effective!r} "
                f"does not match requested seed {requested}"
            )
        trace.info["sampler_seed_evidence"] = {
            "requested_seed": requested,
            "effective_seed": effective,
            "transport": "verifiers.v1 EvalClient -> OpenAI chat request -> vLLM",
            "trace_agent_sampling_recorded": True,
        }
        trace.stamp(EvalRunInfo(id=config.uuid))
        await append_trace(output_dir, trace, write_lock)

    try:
        async with env.serving():
            episodes = []
            for task in tasks:
                seed = requested_sampler_seed(task)
                sampling = config.sampling.model_copy(update={"seed": seed})
                context = ModelContext(client=client, model=config.model, sampling=sampling)
                episodes.append(env.episode(task, context, n=1))
            results = await asyncio.gather(
                *(episode.run(semaphore, on_complete) for episode in episodes)
            )
        traces = [trace for group in results for trace in group]
        if (
            len(traces) != len(tasks)
            or any(not trace.is_completed for trace in traces)
            or any(trace.errors for trace in traces)
            or any(trace.stop_condition == "error" for trace in traces)
        ):
            raise RuntimeError("evaluation block did not produce one error-free trace per task")
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    asyncio.run(run_seeded_eval(args.config.resolve(), args.output_dir.resolve()))


if __name__ == "__main__":
    main()
