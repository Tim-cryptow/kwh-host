"""Verifier calibration on real cards (HOST-CLIENT.md §7, step 3 prototype).

Two phases, so each model can have the whole GPU in turn:

    generate  greedy outputs from the model under test, at the certified concurrency (32),
              for word-salad prompts from the I-1 set (flat next-token distributions:
              the worst case for an honest host) and natural chat prompts
    score     teacher-force those outputs through the reference model one at a time and
              record, per position, the rank and the gap to the reference's top choice

Run `generate` on the reference model and on a substitute (e.g. a 4-bit quantization of
the same weights) with identical prompt ids (`--prompts-from`), `score` both against the
reference, and compare: how large do honest gaps get, and how many confident
disagreements does the substitute produce?
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import httpx
from kwh_bench import reference as ref
from kwh_bench.prompts import canonical_prompts

from .jobs import VLLMExecutor
from .jobspec import RequestSpec
from .platform.verifier import VLLMScorer, judge_greedy

Log = Callable[[str], None]

CHAT_PROMPTS = [
    "Explain how photosynthesis works to a ten-year-old.",
    "Write a Python function that checks whether a string is a palindrome, with a short docstring.",
    "What are the main causes of inflation? Answer in three bullet points.",
    "Summarize the plot of Romeo and Juliet in four sentences.",
    "Give me a simple recipe for jollof rice.",
    "What is the difference between TCP and UDP?",
    "Translate 'Good morning, how did you sleep?' into French and Spanish.",
    "Write a polite email declining a meeting invitation for Friday.",
    "List five tips for improving sleep quality.",
    "Explain what a hash function is and give one real-world use.",
    "How does compound interest work? Include a short numerical example.",
    "Write a haiku about rain in Lagos.",
    "What are the pros and cons of electric cars?",
    "Describe the water cycle in order, step by step.",
    "Write a SQL query that returns the ten customers with the highest total order value.",
    "Why is the sky blue?",
    "Give a short introduction to the Yoruba language.",
    "What should I check before buying a used laptop?",
    "Explain the difference between a virus and a bacterium.",
    "Write a short motivational message for a student preparing for exams.",
    "How do vaccines train the immune system?",
    "Explain recursion with a simple example in JavaScript.",
    "What is the capital of Australia, and why is it not Sydney?",
    "Draft three interview questions for a junior data analyst role.",
    "What does a GPU do differently from a CPU?",
    "Give me a 5-day beginner workout plan with no equipment.",
    "Explain supply and demand using the market for tomatoes.",
    "What are the main parts of a plant cell and what do they do?",
    "Write a limerick about a cat who loves coffee.",
    "How can a small business start accepting online payments?",
    "Explain what an API is to someone who does not program.",
    "What is the Pythagorean theorem? Show one worked example.",
]


async def served_model(url: str) -> tuple[str, int]:
    async with httpx.AsyncClient(base_url=url.rstrip("/"), timeout=30.0) as c:
        r = await c.get("/v1/models")
        r.raise_for_status()
        card = (r.json().get("data") or [{}])[0]
        return card["id"], int(card.get("max_model_len") or ref.MAX_MODEL_LEN)


async def build_prompts(url: str, model: str, n_canonical: int, chat: bool) -> List[dict]:
    out = []
    async with httpx.AsyncClient(base_url=url.rstrip("/"), timeout=60.0) as c:
        for p in canonical_prompts()[:n_canonical]:
            r = await c.post("/tokenize", json={"model": model, "prompt": p.text, "add_special_tokens": False})
            r.raise_for_status()
            out.append({"kind": "canonical", "id": p.id, "prompt_token_ids": r.json()["tokens"][:ref.PROMPT_TOKENS]})
        if chat:
            for i, msg in enumerate(CHAT_PROMPTS):
                r = await c.post("/tokenize", json={"model": model, "messages": [{"role": "user", "content": msg}],
                                                    "add_generation_prompt": True})
                r.raise_for_status()
                out.append({"kind": "chat", "id": i, "prompt_token_ids": r.json()["tokens"]})
    return out


async def generate(url: str, out_path: Path, n_canonical: int = 32, chat: bool = True, max_tokens: int = 128,
                   prompts_from: Optional[Path] = None, label: str = "", log: Log = lambda s: print(s, file=sys.stderr)) -> dict:
    model, _ = await served_model(url)
    if prompts_from:
        prompts = [{k: p[k] for k in ("kind", "id", "prompt_token_ids")}
                   for p in json.loads(prompts_from.read_text())["items"]]
    else:
        prompts = await build_prompts(url, model, n_canonical, chat)
    log(f"generating {len(prompts)} greedy outputs on {model} at concurrency {ref.CONCURRENCY}")
    ex = VLLMExecutor(url, model)
    sem = asyncio.Semaphore(ref.CONCURRENCY)

    async def one(i: int, p: dict) -> dict:
        spec = RequestSpec(index=i, prompt_token_ids=tuple(p["prompt_token_ids"]), max_tokens=max_tokens,
                           temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0, seed=None,
                           stop=(), stop_token_ids=(), ignore_eos=False, logprobs=None)
        async with sem:
            return await ex.generate(spec)

    t0 = time.perf_counter()
    try:
        outs = await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))
    finally:
        await ex.aclose()
    items = [{**p, "token_ids": o["token_ids"], "text": o["text"], "finish_reason": o["finish_reason"]}
             for p, o in zip(prompts, outs)]
    doc = {"phase": "generate", "label": label, "model": model, "url": url, "max_tokens": max_tokens,
           "concurrency": ref.CONCURRENCY, "seconds": round(time.perf_counter() - t0, 2), "items": items}
    out_path.write_text(json.dumps(doc) + "\n")
    log(f"wrote {out_path} ({len(items)} outputs, {sum(len(i['token_ids']) for i in items)} tokens)")
    return doc


def _summary(rows: List[dict], taus: Sequence[float]) -> dict:
    gaps = [g for r in rows for g in r["gaps"]]
    out = {"requests": len(rows), "positions": len(gaps),
           "top1_rate": round(sum(1 for r in rows for k in r["ranks"] if k == 1) / len(gaps), 5) if gaps else None,
           "max_gap": round(max(gaps), 5) if gaps else None, "by_tau": {}}
    for t in taus:
        out["by_tau"][str(t)] = {
            "positions_over": sum(1 for g in gaps if g > t),
            "requests_flagged": sum(1 for r in rows if any(g > t for g in r["gaps"])),
        }
    return out


async def score(url: str, in_path: Path, out_path: Path, taus: Sequence[float] = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
                log: Log = lambda s: print(s, file=sys.stderr)) -> dict:
    gen = json.loads(in_path.read_text())
    model, _ = await served_model(url)
    scorer = VLLMScorer(url, model)
    rows = []
    try:
        for it in gen["items"]:
            pos = await scorer.positions(it["prompt_token_ids"], it["token_ids"])
            j = judge_greedy(pos, max(taus))
            rows.append({"kind": it["kind"], "id": it["id"], "n": len(pos), "max_gap": j["max_gap"],
                         "top1_rate": j["top1_rate"], "mean_logprob": j["mean_logprob"],
                         "gaps": [round(p["gap"], 5) for p in pos], "ranks": [p["rank"] for p in pos]})
    finally:
        await scorer.aclose()
    doc = {"phase": "score", "generated_by": gen["model"], "label": gen.get("label", ""), "scored_by": model,
           "taus": list(taus), "all": _summary(rows, taus),
           "by_kind": {k: _summary([r for r in rows if r["kind"] == k], taus) for k in sorted({r["kind"] for r in rows})},
           "rows": rows}
    out_path.write_text(json.dumps(doc) + "\n")
    a = doc["all"]
    log(f"{gen['model']} scored by {model}: {a['positions']} positions, top-1 {a['top1_rate']}, max gap {a['max_gap']}")
    for t in taus:
        b = a["by_tau"][str(t)]
        log(f"  tau {t}: {b['positions_over']} positions over, {b['requests_flagged']}/{a['requests']} requests flagged")
    return doc
