"""`kwh-host burst`: the reference model working for the platform, on its operator's GPU.

A host proves its engine with fresh challenge continuations (§4), and the platform judges the
greedy outputs hosts deliver by teacher forcing (§7). Both need the reference model: the certified
engine at the locked version, serving the locked checkpoint. The platform keeps no GPU of its own,
so its operator runs a burst on a rented one now and then. It is this module because a rented
machine with kwh-host installed already has exactly that engine, sandboxed, hash-checked, at the
locked version.

1. Ask the platform how far its challenge pool is short of its target and which greedy outputs are
   waiting to be judged (`GET /burst/work`). Nothing to do: stop there, the GPU untouched.
2. Start the engine. For each continuation wanted: a fresh prompt from the benchmark's prompt
   generator under a random seed (no host can have seen it), the reference's greedy 32 tokens after
   it, and its teacher-forced mean log-probability of them, exactly as the lock made its canaries.
3. Judge each queued output: per-position gaps against the reference's top choice
   (`platform/verifier.py`).
4. Post both back (`POST /burst/continuations`, `POST /burst/verdicts`).

The engine must be the reference or its numbers would not be the reference, so the burst checks
that before it makes anything.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from typing import Any, AsyncContextManager, Callable, List, Optional

import httpx
from kwh_bench import reference as ref
from kwh_bench.engines.base import Engine
from kwh_bench.lockfile import load_lock
from kwh_bench.prompts import generate_prompts

from .platform.verifier import DEFAULT_TAU, VLLMScorer, judge_greedy

BATCH = 500


def _first_error(results: List[Any]) -> str:
    for r in results:
        if isinstance(r, BaseException):
            return f"{type(r).__name__}: {r}"[:300]
    return ""


async def _gather_bounded(n: int, items: List[Any], fn) -> List[Any]:
    sem = asyncio.Semaphore(max(1, n))

    async def one(x):
        async with sem:
            return await fn(x)
    return await asyncio.gather(*(one(x) for x in items), return_exceptions=True)


async def make_continuations(engine: Engine, count: int, seed: int, concurrency: int,
                             log: Callable[[str], None] = lambda s: None) -> List[dict]:
    prompts = generate_prompts(count=count, seed=seed)
    done = 0

    async def one(p):
        nonlocal done
        ids = (await engine.tokenize(p.text))[:ref.PROMPT_TOKENS]
        cont = (await engine.complete(ids, ref.CANARY_TOKENS, want_token_ids=True)).token_ids
        if not cont or len(cont) != ref.CANARY_TOKENS:
            raise RuntimeError(f"the engine returned {len(cont or [])} tokens, not {ref.CANARY_TOKENS}")
        lps = await engine.score_continuation(ids, cont)
        done += 1
        if done % 500 == 0:
            log(f"  {done}/{count} continuations")
        return {"prompt_text": p.text, "prompt_tokens": ref.PROMPT_TOKENS, "continuation_token_ids": list(cont),
                "reference_mean_logprob": sum(lps) / len(lps)}
    out = await _gather_bounded(concurrency, prompts, one)
    made = [o for o in out if isinstance(o, dict)]
    if len(made) < len(out):
        log(f"  {len(out) - len(made)} continuation(s) failed; the first: {_first_error(out)}")
    return made


async def judge_outputs(scorer: Any, items: List[dict], tau: float, concurrency: int) -> List[dict]:
    async def one(it):
        try:
            verdict = judge_greedy(await scorer.positions(it["prompt_token_ids"], it["output_token_ids"]), tau)
        except Exception as e:  # noqa: BLE001 - the reference failing to judge is not the host's fault
            verdict = {"pass": None, "error": f"{type(e).__name__}: {e}"[:300]}
        return {"id": it["id"], "verdict": verdict}
    return [v for v in await _gather_bounded(concurrency, items, one) if isinstance(v, dict)]


def undecided_to_post(verdicts: List[dict]) -> List[dict]:
    """Outputs the reference could not judge. A few are posted as undecided: something about those
    outputs. Many mean the engine itself failed, so they stay queued for the next burst."""
    errored = [v for v in verdicts if v["verdict"].get("pass") is None]
    return errored if len(errored) <= max(1, len(verdicts) // 10) else []


async def check_reference(engine: Engine) -> Optional[str]:
    """Why this engine cannot stand in for the reference, or None."""
    info = await engine.info()
    if info.name == "mock":
        return None                                   # tests: the benchmark's mock engine is its own reference
    lock = load_lock()
    if info.model_id not in (ref.MODEL_ID, *ref.MODEL_ALIASES):
        return f"the engine serves {info.model_id}, not the reference checkpoint {ref.MODEL_ID}"
    if lock.vllm_version and info.version != lock.vllm_version:
        return f"the engine is vLLM {info.version}; the reference is the locked {lock.vllm_version}"
    return None


def scorer_for(engine: Any) -> VLLMScorer:
    """The judge talks to the same engine, however it is reached (the sandbox's Unix socket, or a port)."""
    transport = engine.make_transport() if hasattr(engine, "make_transport") else None
    return VLLMScorer(engine.base_url, transport=transport)


async def run_burst(platform_url: str, token: str, *, engine: Optional[Engine] = None, scorer: Any = None,
                    engine_url: Optional[str] = None, max_continuations: int = 3000, max_verify: int = 2000,
                    concurrency: int = 32, seed: Optional[int] = None, log: Callable[[str], None] = print,
                    gpu: Optional[Callable[[], AsyncContextManager]] = None,
                    transport: Optional[httpx.AsyncBaseTransport] = None) -> dict:
    """One burst. `engine` is the reference (default: a vLLM already serving at `engine_url`);
    `gpu`, if given, is entered once there is work, before the engine starts, and left after it
    stops: `kwh-host burst` uses it to pause the host's own service around the burst."""
    t0 = time.monotonic()
    async with httpx.AsyncClient(base_url=platform_url.rstrip("/"), headers={"authorization": f"Bearer {token}"},
                                 timeout=300, transport=transport) as api:
        r = await api.get("/burst/work", params={"verify_limit": max_verify})
        r.raise_for_status()
        work = r.json()
        wanted = max(0, min(int(work["continuations_wanted"]), max_continuations))
        queued = work["verify"][:max_verify]
        log(f"burst {work['burst_id']}: pool {work['pool_unissued']} unissued, {wanted} continuation(s) wanted, "
            f"{len(queued)} output(s) to judge")
        if not wanted and not queued:
            return {"ok": True, "burst_id": work["burst_id"], "continuations": 0, "continuations_wanted": 0,
                    "verdicts": 0, "queued": 0, "seconds": round(time.monotonic() - t0, 1), "note": "nothing to do"}
        if engine is None:
            if not engine_url:
                raise ValueError("give an engine, or engine_url: a vLLM serving the reference model at the locked version")
            from kwh_bench.engines.vllm import VLLMEngine
            engine = VLLMEngine(server_url=engine_url)
        if scorer is None:
            scorer = scorer_for(engine)
        added = judged = 0
        try:
            async with (gpu() if gpu else contextlib.AsyncExitStack()):
                async with engine:
                    why = await check_reference(engine)
                    if why:
                        raise RuntimeError(f"not the reference: {why}")
                    log(f"  the engine is up after {time.monotonic() - t0:.0f} s")
                    if wanted:
                        s = seed if seed is not None else random.SystemRandom().getrandbits(48)
                        items = await make_continuations(engine, wanted, s, concurrency, log)
                        log(f"  made {len(items)} continuation(s) after {time.monotonic() - t0:.0f} s")
                        for i in range(0, len(items), BATCH):
                            r = await api.post("/burst/continuations", json={"burst_id": work["burst_id"],
                                                                            "items": items[i:i + BATCH]})
                            r.raise_for_status()
                            added += r.json()["added"]
                    if queued:
                        verdicts = await judge_outputs(scorer, queued, float(work.get("tau") or DEFAULT_TAU), concurrency)
                        post = [v for v in verdicts if v["verdict"].get("pass") is not None] + undecided_to_post(verdicts)
                        if len(post) < len(verdicts):
                            first = next(v["verdict"].get("error") for v in verdicts if v["verdict"].get("pass") is None)
                            log(f"  {len(verdicts) - len(post)} output(s) could not be judged and stay queued; the first: {first}")
                        for i in range(0, len(post), BATCH):
                            r = await api.post("/burst/verdicts", json={"items": post[i:i + BATCH]})
                            r.raise_for_status()
                            judged += r.json()["judged"]
                        log(f"  judged {judged} output(s)")
        finally:
            if hasattr(scorer, "aclose"):
                await scorer.aclose()
        return {"ok": added >= (wanted * 9) // 10 and judged == len(queued), "burst_id": work["burst_id"],
                "continuations": added, "continuations_wanted": wanted, "verdicts": judged, "queued": len(queued),
                "seconds": round(time.monotonic() - t0, 1)}
