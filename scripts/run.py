"""Run the prompt grid against one or more models via OpenRouter.

Resumable: re-running skips (cell_id, model, rollout) keys already in the
output file. Kill it any time; start it again and it continues.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    python scripts/run.py --model anthropic/claude-... --rollouts 5
    python scripts/run.py --model meta-llama/llama-3.3-70b-instruct --rollouts 5
    python scripts/run.py --model <same> --rollouts 5 --seed-tag B   # seed null

    python scripts/run.py --model x --mock    # offline smoke test, no API calls
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import hashlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Decode parameters are PINNED and asserted. Providers ship different defaults
# and a silent temperature difference will move refusal rates enough to invent
# a finding. Report these numbers in the paper.
DECODE = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 1024,
}


def load_prompts(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def done_keys(path: pathlib.Path) -> set[tuple]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # truncated final line from a kill; ignore
        keys.add((r["cell_id"], r["model"], r["rollout"], r.get("seed_tag", "A")))
    return keys


async def call_mock(client, model, prompt):
    await asyncio.sleep(0.001)
    return f"[mock response to {len(prompt)} chars]", {"provider": "mock"}


async def call_real(client, model, prompt):
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        extra_body={
            # Pin the provider so routing is reproducible. A refusal from a
            # router-side moderation layer is not a refusal from the model,
            # and you must be able to say which you measured.
            "provider": {"allow_fallbacks": False},
        },
        **DECODE,
    )
    text = resp.choices[0].message.content or ""
    extra = getattr(resp, "model_extra", None) or {}
    meta = {"provider": extra.get("provider"),
            "resolved_model": getattr(resp, "model", None),
            "finish_reason": resp.choices[0].finish_reason}
    return text, meta


async def worker(sem, client, call, model, seed_tag, job, out_f, lock, counters):
    prompt, rollout = job["prompt"], job["rollout"]
    async with sem:
        for attempt in range(5):
            try:
                text, meta = await call(client, model, prompt)
                break
            except Exception as e:  # noqa: BLE001 - log and back off
                if attempt == 4:
                    text, meta = None, {"error": repr(e)}
                    break
                await asyncio.sleep(2 ** attempt + random.random())
        rec = {
            "cell_id": job["cell_id"],
            "scenario_id": job["scenario_id"],
            "terminology": job["terminology"],
            "framing": job["framing"],
            "drb_category": job["drb_category"],
            "model": model,
            "seed_tag": seed_tag,
            "rollout": rollout,
            "prompt": prompt,
            "response": text,
            "decode": DECODE,
            "ts": time.time(),
            **{f"meta_{k}": v for k, v in meta.items()},
        }
        async with lock:
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()  # flush every record: a killed run loses nothing
            counters["n"] += 1
            if counters["n"] % 25 == 0:
                print(f"  {counters['n']}/{counters['total']}", flush=True)


async def main_async(args) -> int:
    prompts = load_prompts(ROOT / "data" / "prompts.jsonl")
    out_path = ROOT / "out" / "responses.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    already = done_keys(out_path)

    jobs = []
    for p in prompts:
        for rollout in range(args.rollouts):
            key = (p["cell_id"], args.model, rollout, args.seed_tag)
            if key in already:
                continue
            jobs.append({**p, "rollout": rollout})

    # Randomise so a rate-limit stall or a kill drops cells uniformly across
    # conditions rather than wiping out one framing entirely.
    # hash() is salted per process; use a stable digest so the shuffle order
    # is reproducible across runs and machines.
    random.seed(int(hashlib.sha256(args.seed_tag.encode()).hexdigest()[:8], 16))
    random.shuffle(jobs)

    print(f"model={args.model} seed_tag={args.seed_tag}")
    print(f"{len(jobs)} jobs to run ({len(already)} already done, skipped)")
    if not jobs:
        return 0

    if args.mock:
        client, call = None, call_mock
    else:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            print("pip install openai", file=sys.stderr)
            return 1
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            print("set OPENROUTER_API_KEY", file=sys.stderr)
            return 1
        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        call = call_real

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    counters = {"n": 0, "total": len(jobs)}
    t0 = time.time()
    with out_path.open("a") as out_f:
        await asyncio.gather(*[
            worker(sem, client, call, args.model, args.seed_tag,
                   j, out_f, lock, counters)
            for j in jobs
        ])
    print(f"done in {time.time() - t0:.0f}s -> {out_path}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rollouts", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed-tag", default="A",
                    help="use B for the same-model seed-null replicate")
    ap.add_argument("--mock", action="store_true")
    raise SystemExit(asyncio.run(main_async(ap.parse_args())))
