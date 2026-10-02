"""Job execution on the host (HOST-CLIENT.md §7).

The daemon receives job envelopes over the WebSocket, runs every request against the
engine it owns, and returns one signed result per job. The result is bound to the job
by the job's hash and to the host by its signature: whatever the platform's verifier
later finds in those outputs is attributable to this host and nobody else.

The host returns generated token ids, not just text. Text cannot be re-tokenized
reliably, and the verifier scores the exact tokens the host produced.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Callable, Optional, Protocol

import httpx

from .identity import Identity
from .jobspec import JobInvalid, RequestSpec, job_hash, parse_job
from .mockmodel import ToyLM

MAX_ERROR_CHARS = 500


class ExecutorError(RuntimeError):
    pass


class Executor(Protocol):
    async def generate(self, spec: RequestSpec) -> dict: ...


# --- vLLM ---------------------------------------------------------------------

class VLLMExecutor:
    """Runs one request on the daemon's vLLM over its OpenAI-compatible server.

    Prompts go in as token ids; `return_token_ids` brings the generated ids back on every
    streamed chunk (vLLM >= 0.10.2), so the buyer's own logprob request is the only reason
    logprobs are ever computed. Streaming gives the host a TTFT it can report."""

    def __init__(self, base_url: str, model: str, transport: Optional[httpx.AsyncBaseTransport] = None,
                 timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._http = httpx.AsyncClient(base_url=self.base_url, transport=transport,
                                       timeout=httpx.Timeout(timeout, connect=10.0))

    async def aclose(self) -> None:
        await self._http.aclose()

    def body(self, spec: RequestSpec) -> dict:
        body = {
            "model": self.model,
            "prompt": list(spec.prompt_token_ids),
            "max_tokens": spec.max_tokens,
            # Explicit, always: unset fields would come from the model's generation_config.
            "temperature": spec.temperature,
            "top_p": spec.top_p,
            "top_k": spec.top_k,
            "min_p": spec.min_p,
            "repetition_penalty": spec.repetition_penalty,
            "seed": spec.seed,
            "stop": list(spec.stop),
            "stop_token_ids": list(spec.stop_token_ids),
            "ignore_eos": spec.ignore_eos,
            "skip_special_tokens": True,
            "stream": True,
            "stream_options": {"include_usage": True},
            "return_token_ids": True,
        }
        if spec.logprobs is not None:
            body["logprobs"] = spec.logprobs
            body["return_tokens_as_token_ids"] = True
        return body

    async def generate(self, spec: RequestSpec) -> dict:
        token_ids, text = [], []
        lp_tokens, lp_values, lp_top = [], [], []
        finish = stop_reason = usage = None
        t0 = time.perf_counter()
        first = None
        async with self._http.stream("POST", "/v1/completions", json=self.body(spec)) as resp:
            if resp.status_code >= 400:
                detail = (await resp.aread()).decode("utf-8", "replace")
                raise ExecutorError(f"engine {resp.status_code}: {detail[:MAX_ERROR_CHARS]}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                if obj.get("error"):
                    raise ExecutorError(f"engine error: {str(obj['error'])[:MAX_ERROR_CHARS]}")
                for ch in obj.get("choices") or []:
                    ids = ch.get("token_ids") or []
                    if ids and first is None:
                        first = time.perf_counter()
                    token_ids.extend(int(t) for t in ids)
                    text.append(ch.get("text") or "")
                    lp = ch.get("logprobs")
                    if lp:
                        lp_tokens.extend(lp.get("tokens") or [])
                        lp_values.extend(lp.get("token_logprobs") or [])
                        lp_top.extend(lp.get("top_logprobs") or [])
                    if ch.get("finish_reason"):
                        finish, stop_reason = ch["finish_reason"], ch.get("stop_reason")
                if obj.get("usage"):
                    usage = obj["usage"]
        t1 = time.perf_counter()
        if finish is None:
            raise ExecutorError("stream ended without a finish_reason")
        if usage is not None and usage.get("completion_tokens") != len(token_ids):
            raise ExecutorError(f"engine reported {usage.get('completion_tokens')} completion tokens, "
                                f"stream carried {len(token_ids)} token ids")
        out = {"token_ids": token_ids, "text": "".join(text), "finish_reason": finish, "stop_reason": stop_reason,
               "ttft_ms": round(((first or t1) - t0) * 1000, 1), "total_ms": round((t1 - t0) * 1000, 1)}
        if spec.logprobs is not None:
            out["logprobs"] = {"tokens": lp_tokens, "token_logprobs": lp_values, "top_logprobs": lp_top}
        return out


# --- toy model (tests, --mock-engine) ------------------------------------------

class MockExecutor:
    def __init__(self, model: Optional[ToyLM] = None, step_s: float = 0.0, fail: bool = False):
        self.model = model or ToyLM()
        self.step_s, self.fail = step_s, fail

    async def generate(self, spec: RequestSpec) -> dict:
        if self.fail:
            raise ExecutorError("mock executor configured to fail")
        t0 = time.perf_counter()
        tokens, finish = self.model.generate(spec.prompt_token_ids, spec.max_tokens, spec.temperature, spec.top_p,
                                             spec.seed, spec.stop_token_ids, spec.ignore_eos)
        await asyncio.sleep(self.step_s * len(tokens))
        t1 = time.perf_counter()
        out = {"token_ids": tokens, "text": self.model.detokenize(tokens), "finish_reason": finish,
               "stop_reason": tokens[-1] if finish == "stop" else None,
               "ttft_ms": round(min(t1 - t0, self.step_s) * 1000, 1), "total_ms": round((t1 - t0) * 1000, 1)}
        if spec.logprobs is not None:
            out["logprobs"] = {"tokens": [f"token_id:{t}" for t in tokens], "token_logprobs": [], "top_logprobs": []}
        return out


# --- one job ----------------------------------------------------------------------

async def execute_job(job: dict, executor: Executor, sem: asyncio.Semaphore, *, identity: Identity, host_id: str,
                      engine: dict, max_model_len: int, clock: Callable[[], float] = time.time) -> dict:
    """Run every request in the job and return the signed result.

    status: completed (every request finished) | failed (deadline or a request errored;
    the platform re-routes) | rejected (the envelope is invalid or does not fit this
    engine; nothing was run). Cancellation (platform cancel, daemon stop) propagates."""
    received_at = clock()
    base = {"job_id": job.get("job_id") if isinstance(job, dict) else None, "job_sha256": job_hash(job),
            "host_id": host_id, "engine": engine, "received_at": received_at}
    try:
        specs = parse_job(job, max_model_len)
    except JobInvalid as e:
        return identity.sign_result({**base, "status": "rejected", "reason": f"invalid: {e}"[:MAX_ERROR_CHARS],
                                     "outputs": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                                     "finished_at": clock(), "result_sha256": None, "signature": None})
    timeout = float(job["timeout_s"])

    async def one(spec: RequestSpec) -> dict:
        async with sem:
            return await executor.generate(spec)

    tasks = [asyncio.ensure_future(one(s)) for s in specs]
    outputs, reason = [], None
    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)
    except asyncio.TimeoutError:
        status, reason = "failed", f"deadline: not finished within {timeout:.1f}s"
    else:
        errors = 0
        for spec, r in zip(specs, results):
            if isinstance(r, BaseException):
                errors += 1
                outputs.append({"index": spec.index, "error": f"{type(r).__name__}: {r}"[:MAX_ERROR_CHARS]})
            else:
                outputs.append({"index": spec.index, **r, "error": None})
        status = "completed" if errors == 0 else "failed"
        if errors:
            reason = f"{errors} of {len(specs)} request(s) failed"
    usage = {"prompt_tokens": sum(len(s.prompt_token_ids) for s in specs),
             "completion_tokens": sum(len(o.get("token_ids") or []) for o in outputs)}
    return identity.sign_result({**base, "status": status, "reason": reason, "outputs": outputs, "usage": usage,
                                 "finished_at": clock(), "result_sha256": None, "signature": None})
