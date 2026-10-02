"""Host-side job execution (HOST-CLIENT.md §7): envelope validation, execution, signed results,
and the vLLM stream parser against a fake vLLM."""

import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from kwh_bench import reference as ref

from kwh_host.identity import Identity, verify_result_signature
from kwh_host.jobs import ExecutorError, MockExecutor, VLLMExecutor, execute_job
from kwh_host.jobspec import (REQUEST_DEFAULTS, JobInvalid, job_hash, normalize_request, parse_job, units_for)
from kwh_host.mockmodel import ToyLM


def req(prompt=(5, 6, 7), **kw):
    return normalize_request({"prompt_token_ids": list(prompt), **kw})


def job(requests, timeout_s=10.0, job_id="j_test.1"):
    return {"job_id": job_id, "timeout_s": timeout_s, "units_reserved": 0.0, "requests": requests}


# --- envelope -------------------------------------------------------------------

def test_reference_job_meters_to_one_unit():
    assert units_for(ref.PROMPT_TOKENS_PER_JOB, ref.GENERATED_TOKENS_PER_JOB) == pytest.approx(1.0)
    assert units_for(0, 0) == 0.0


def test_normalize_fills_every_field_explicitly():
    r = req()
    assert set(r) == {"prompt_token_ids", *REQUEST_DEFAULTS}
    assert r["temperature"] == 1.0 and r["top_p"] == 1.0 and r["seed"] is None   # OpenAI semantics, written out


@pytest.mark.parametrize("bad, msg", [
    ({"prompt_token_ids": []}, "prompt_token_ids"),
    ({"prompt_token_ids": [1, -2]}, "prompt_token_ids"),
    ({"prompt_token_ids": [1], "temperature": 3}, "temperature"),
    ({"prompt_token_ids": [1], "top_p": 0}, "top_p"),
    ({"prompt_token_ids": [1], "logprobs": 21}, "logprobs"),
    ({"prompt_token_ids": [1], "max_tokens": 0}, "max_tokens"),
    ({"prompt_token_ids": [1], "stop": ["x"] * 9}, "stop"),
    ({"prompt_token_ids": [1], "temprature": 0.1}, "unknown"),
    ({"prompt_token_ids": [1], "ignore_eos": "yes"}, "ignore_eos"),
])
def test_normalize_rejects(bad, msg):
    with pytest.raises(JobInvalid, match=msg):
        normalize_request(bad)


def test_parse_job_requires_explicit_fields_and_fitting_context():
    good = job([req(max_tokens=8)])
    assert parse_job(good, 1024)[0].max_tokens == 8
    implicit = job([{"prompt_token_ids": [1, 2]}])                       # vLLM would fill the gaps
    with pytest.raises(JobInvalid, match="missing field"):
        parse_job(implicit, 1024)
    long = job([req(prompt=[1] * 900, max_tokens=200)])
    with pytest.raises(JobInvalid, match="exceeds this engine's context of 1024"):
        parse_job(long, 1024)
    assert parse_job(long, 8192)[0].context_needed == 1100
    with pytest.raises(JobInvalid, match="unknown job field"):
        parse_job({**good, "sample_spec": []}, 1024)
    with pytest.raises(JobInvalid, match="requests"):
        parse_job(job([]), 1024)


# --- execution -----------------------------------------------------------------

async def test_execute_job_completes_signed_and_bound():
    ident = Identity.generate()
    model = ToyLM()
    j = job([req(prompt=(1, 2, 3), max_tokens=12, temperature=0.0), req(prompt=(9,), max_tokens=5, temperature=0.0)])
    res = await execute_job(j, MockExecutor(model), asyncio.Semaphore(32), identity=ident, host_id="h_x",
                            engine={"version": "0"}, max_model_len=1024)
    assert res["status"] == "completed" and res["reason"] is None
    assert verify_result_signature(res) == ident.public_key_hex
    assert res["job_sha256"] == job_hash(j) and res["job_id"] == "j_test.1" and res["host_id"] == "h_x"
    for o, (prompt, n) in zip(res["outputs"], [((1, 2, 3), 12), ((9,), 5)]):
        expected, finish = model.generate(prompt, n)
        assert o["token_ids"] == expected and o["finish_reason"] == finish and o["error"] is None
    assert res["usage"] == {"prompt_tokens": 4, "completion_tokens": sum(len(o["token_ids"]) for o in res["outputs"])}


async def test_execute_job_rejects_what_does_not_fit():
    ident = Identity.generate()
    res = await execute_job(job([req(prompt=[1] * 1000, max_tokens=100)]), MockExecutor(), asyncio.Semaphore(4),
                            identity=ident, host_id="h_x", engine={}, max_model_len=1024)
    assert res["status"] == "rejected" and "context" in res["reason"] and res["outputs"] == []
    assert verify_result_signature(res) == ident.public_key_hex


async def test_execute_job_deadline_and_errors():
    ident = Identity.generate()
    slow = MockExecutor(step_s=0.05)
    res = await execute_job(job([req(max_tokens=40, ignore_eos=True, temperature=0.0)], timeout_s=0.3), slow,
                            asyncio.Semaphore(4), identity=ident, host_id="h_x", engine={}, max_model_len=1024)
    assert res["status"] == "failed" and res["reason"].startswith("deadline") and res["outputs"] == []
    res = await execute_job(job([req(), req()]), MockExecutor(fail=True), asyncio.Semaphore(4), identity=ident,
                            host_id="h_x", engine={}, max_model_len=1024)
    assert res["status"] == "failed" and res["reason"] == "2 of 2 request(s) failed"
    assert all("configured to fail" in o["error"] for o in res["outputs"])


async def test_cancelling_a_job_cancels_its_requests():
    ident = Identity.generate()
    t = asyncio.create_task(execute_job(job([req(max_tokens=100, ignore_eos=True, temperature=0.0)]),
                                        MockExecutor(step_s=0.05), asyncio.Semaphore(4), identity=ident,
                                        host_id="h_x", engine={}, max_model_len=1024))
    await asyncio.sleep(0.1)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t


# --- the vLLM stream parser, against a fake vLLM -------------------------------------

def fake_vllm(record: list, chunks=None, status=200, error_chunk=False):
    app = FastAPI()

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        record.append(body)
        if status != 200:
            return JSONResponse({"error": {"message": "This model's maximum context length is 1024"}}, status_code=status)
        parts = chunks or [
            {"choices": [{"index": 0, "text": "", "token_ids": [], "prompt_token_ids": body["prompt"], "finish_reason": None}]},
            {"choices": [{"index": 0, "text": "Hello", "token_ids": [9906], "finish_reason": None,
                          **({"logprobs": {"tokens": ["token_id:9906"], "token_logprobs": [-0.1],
                                           "top_logprobs": [{"token_id:9906": -0.1}]}} if body.get("logprobs") is not None else {})}]},
            {"choices": [{"index": 0, "text": " world", "token_ids": [1917], "finish_reason": None,
                          **({"logprobs": {"tokens": ["token_id:1917"], "token_logprobs": [-0.2],
                                           "top_logprobs": [{"token_id:1917": -0.2}]}} if body.get("logprobs") is not None else {})}]},
            {"choices": [{"index": 0, "text": "", "token_ids": [128009], "finish_reason": "stop", "stop_reason": 128009}]},
            {"choices": [], "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": 3, "total_tokens": 0}},
        ]

        async def gen():
            for c in parts:
                yield f"data: {json.dumps(c)}\n\n"
            if error_chunk:
                yield 'data: {"error": {"message": "engine died"}}\n\n'
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    return httpx.ASGITransport(app=app)


async def test_vllm_executor_sends_explicit_params_and_parses_token_ids():
    seen = []
    ex = VLLMExecutor("http://vllm", "RedHatAI/x", transport=fake_vllm(seen))
    spec = parse_job(job([req(prompt=(1, 2), max_tokens=16, temperature=0.0)]), 1024)[0]
    out = await ex.generate(spec)
    await ex.aclose()
    body = seen[0]
    for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "seed", "stop", "stop_token_ids", "ignore_eos"):
        assert k in body, k                       # nothing left for generation_config to fill
    assert body["prompt"] == [1, 2] and body["return_token_ids"] is True and body["stream"] is True
    assert "logprobs" not in body                 # not asked for, not computed
    assert out["token_ids"] == [9906, 1917, 128009] and out["text"] == "Hello world"
    assert out["finish_reason"] == "stop" and out["stop_reason"] == 128009 and "logprobs" not in out


async def test_vllm_executor_logprobs_errors_and_count_check():
    seen = []
    ex = VLLMExecutor("http://vllm", "m", transport=fake_vllm(seen))
    out = await ex.generate(parse_job(job([req(logprobs=2, temperature=0.0)]), 1024)[0])
    assert seen[0]["logprobs"] == 2 and seen[0]["return_tokens_as_token_ids"] is True
    assert out["logprobs"]["tokens"] == ["token_id:9906", "token_id:1917"]
    spec = parse_job(job([req()]), 1024)[0]
    with pytest.raises(ExecutorError, match="engine 400"):
        await VLLMExecutor("http://vllm", "m", transport=fake_vllm([], status=400)).generate(spec)
    with pytest.raises(ExecutorError, match="engine died"):
        await VLLMExecutor("http://vllm", "m", transport=fake_vllm([], error_chunk=True)).generate(spec)
    short = [{"choices": [{"index": 0, "text": "a", "token_ids": [1], "finish_reason": "length"}]},
             {"choices": [], "usage": {"completion_tokens": 2}}]
    with pytest.raises(ExecutorError, match="2 completion tokens"):
        await VLLMExecutor("http://vllm", "m", transport=fake_vllm([], chunks=short)).generate(spec)
