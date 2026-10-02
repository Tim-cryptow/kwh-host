"""Thin wrapper over kwh_bench (HOST-CLIENT.md §4, §5): the full certified run, the 1/8-job
micro-benchmark, and challenge scoring. Nothing here re-implements the spec."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

from kwh_bench import reference as ref
from kwh_bench.engines.base import Engine
from kwh_bench.load import PreparedPrompt, prepare_prompts, run_job
from kwh_bench.prompts import canonical_prompts
from kwh_bench.report import write_report
from kwh_bench.runner import run_benchmark
from kwh_bench.verify import verify_report

from .identity import Identity

Log = Callable[[str], None]

MICRO_REQUESTS = ref.REQUESTS_PER_JOB // 8          # 32: one eighth of a reference job
MICRO_FRACTION = MICRO_REQUESTS / ref.REQUESTS_PER_JOB


async def full_benchmark(engine: Engine, identity: Identity, out: Path, runs: int = ref.DEFAULT_MEASURED_JOBS,
                         log: Log = lambda s: print(s, file=sys.stderr), ignore_preflight: bool = False) -> dict:
    """`kwh-bench run` with a daemon-owned engine, then sign and verify. The engine is started
    and stopped inside (the §6 pre-flight needs an idle GPU before launch)."""
    report = await run_benchmark(engine, measured_jobs=runs, log=log, ignore_preflight=ignore_preflight)
    identity.sign_report(report)
    write_report(report, out)
    ok, problems = verify_report(out)
    if not ok:
        raise RuntimeError("report failed verification: " + "; ".join(problems))
    return report


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def context_mismatch(report: dict, max_model_len: int) -> Optional[str]:
    """Why the engine may not serve at `max_model_len` with this report, or None. A host serves
    at the context length it certified with (D8); a report from a non-vLLM engine says nothing."""
    from kwh_bench.report import launch_max_model_len
    certified = launch_max_model_len((report.get("engine") or {}).get("launch_args") or [])
    if certified is None or certified == max_model_len:
        return None
    return (f"the report was certified at --max-model-len {certified} and the config serves {max_model_len}; "
            f"run `kwh-host bench` again (or `kwh-host init --max-model-len {certified}`)")


async def prepare_micro_prompts(engine: Engine) -> List[PreparedPrompt]:
    """The first 32 canonical prompts, tokenized by the running engine. Done once per engine start."""
    return await prepare_prompts(engine, canonical_prompts()[:MICRO_REQUESTS])


async def micro_benchmark(engine: Engine, prepared: List[PreparedPrompt]) -> dict:
    """One eighth of a reference job at full concurrency; reports the equivalent units/hour."""
    res = await run_job(engine, prepared, concurrency=ref.CONCURRENCY)
    secs = round(res.job_seconds, 4)
    return {"job_seconds": secs, "generated_tokens": res.generated_tokens, "request_failures": res.failures,
            "tokens_per_second": round(res.generated_tokens / secs, 2) if secs else None,
            "units_per_hour": round(3600.0 / (secs / MICRO_FRACTION), 3) if secs else 0.0}


async def score_challenge(engine: Engine, challenge: dict) -> dict:
    """Teacher-forced mean logprob of each of the platform's continuations, exactly as the
    benchmark scores its canaries: one request at a time, so the batch is the same shape the
    reference node scored them in. The platform judges the mean of the deltas (D9)."""
    t0 = time.perf_counter()
    means: List[float] = []
    for item in challenge["items"]:
        ids = await engine.tokenize(item["prompt_text"])
        n = int(item.get("prompt_tokens") or ref.PROMPT_TOKENS)
        if len(ids) < n:
            raise ValueError(f"challenge prompt tokenizes to {len(ids)} < {n}")
        lps = await engine.score_continuation(ids[:n], list(item["continuation_token_ids"]))
        means.append(round(sum(lps) / len(lps), 5))
    return {"challenge_id": challenge["challenge_id"], "mean_logprobs": means,
            "elapsed_ms": int((time.perf_counter() - t0) * 1000)}
