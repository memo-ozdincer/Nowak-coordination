#!/usr/bin/env python3
"""Black-box proof that vLLM receives and uses the OpenAI request seed."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


PROMPT = (
    "Return exactly 24 uppercase letters sampled independently from A through Z. "
    "Return the letters only, with no spaces or explanation."
)
SEEDS = (2901, 2901, 2902, 2903, 2904, 2905)


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def request_json(url: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(
        url,
        data=body,
        headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    records = []
    for seed in SEEDS:
        payload = {**base_payload, "seed": seed}
        response = request_json(f"{args.base_url.rstrip('/')}/chat/completions", payload)
        records.append(
            {
                "seed": seed,
                "request_sha256": canonical_hash(payload),
                "response_id": response["id"],
                "response_model": response["model"],
                "output": response["choices"][0]["message"]["content"],
                "finish_reason": response["choices"][0]["finish_reason"],
            }
        )
    same_seed = [record["output"] for record in records if record["seed"] == 2901]
    distinct_seed_outputs = {record["output"] for record in records if record["seed"] != 2901}
    status = (
        "PASS"
        if len(set(same_seed)) == 1
        and any(output != same_seed[0] for output in distinct_seed_outputs)
        else "FAIL"
    )
    result = {
        "schema_version": 1,
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "server_models": request_json(f"{args.base_url.rstrip('/')}/models"),
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "base_payload_sha256": canonical_hash(base_payload),
        "seed_sequence": list(SEEDS),
        "acceptance": {
            "same_seed_outputs_identical": len(set(same_seed)) == 1,
            "different_seed_changes_output": any(
                output != same_seed[0] for output in distinct_seed_outputs
            ),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if status != "PASS":
        raise SystemExit("vLLM sampler-seed probe failed")


if __name__ == "__main__":
    main()
