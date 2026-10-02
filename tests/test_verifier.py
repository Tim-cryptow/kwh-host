"""Teacher-forced greedy verification (step 3 prototype): honest outputs pass, a substitute's fail."""

import httpx
import pytest
from fastapi import FastAPI, Request

from kwh_host.mockmodel import ToyLM
from kwh_host.platform.verifier import ToyScorer, VLLMScorer, judge_greedy, verify_greedy_outputs

PROMPTS = [[11, 12, 13], [40, 41], [7], [300, 301, 302, 303], [99, 98]]


async def test_honest_outputs_pass_and_a_substitute_does_not():
    reference = ToyLM(seed=0)
    substitute = ToyLM(seed=0, drift=0.3)          # agrees on ~70% of contexts
    scorer = ToyScorer(reference)
    honest = [(i, p, reference.generate(p, 48, ignore_eos=True)[0]) for i, p in enumerate(PROMPTS)]
    wrong = [(i, p, substitute.generate(p, 48, ignore_eos=True)[0]) for i, p in enumerate(PROMPTS)]
    v_ok = await verify_greedy_outputs(scorer, honest)
    v_bad = await verify_greedy_outputs(scorer, wrong)
    assert v_ok["pass"] is True and all(r["top1_rate"] == 1.0 and r["confident_disagreements"] == 0 for r in v_ok["requests"])
    assert v_bad["pass"] is False
    assert sum(r["confident_disagreements"] for r in v_bad["requests"]) > 5


def test_near_ties_are_not_disagreements():
    # A kernel-noise flip: the host took the reference's #2 token, 0.003 nats behind.
    flips = [{"logprob": -0.700, "rank": 2, "top1_logprob": -0.697, "gap": 0.003}] * 3
    clean = [{"logprob": -0.1, "rank": 1, "top1_logprob": -0.1, "gap": 0.0}] * 20
    j = judge_greedy(clean + flips, tau=0.1)
    assert j["pass"] and j["confident_disagreements"] == 0 and j["top1"] == 20 and j["max_gap"] == pytest.approx(0.003)
    assert not judge_greedy(clean + [{"logprob": -2.0, "rank": 3, "top1_logprob": -0.2, "gap": 1.8}], tau=0.1)["pass"]


def fake_reference(record: list, max_model_len=1024):
    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "ref", "max_model_len": max_model_len}]}

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        record.append(body)
        prompt = body["prompt"]
        plp = [None]
        for t in prompt[1:]:
            if t == 666:                           # the reference would have said 5 here, confidently
                plp.append({"666": {"logprob": -3.0, "rank": 9, "decoded_token": "x"},
                            "5": {"logprob": -0.05, "rank": 1, "decoded_token": "y"}})
            else:
                plp.append({str(t): {"logprob": -0.2, "rank": 1, "decoded_token": "z"}})
        return {"choices": [{"index": 0, "text": "", "prompt_logprobs": plp, "finish_reason": "length"}],
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": 1}}

    return httpx.ASGITransport(app=app)


async def test_vllm_scorer_reads_ranks_and_gaps_and_respects_context():
    seen = []
    s = VLLMScorer("http://ref", transport=fake_reference(seen, max_model_len=10))
    pos = await s.positions([1, 2, 3], [4, 666, 6])
    assert seen[0]["prompt_logprobs"] == 1 and seen[0]["max_tokens"] == 1 and seen[0]["temperature"] == 0.0
    assert [p["rank"] for p in pos] == [1, 9, 1]
    assert pos[1]["gap"] == pytest.approx(2.95) and pos[0]["gap"] == 0.0
    # 8 prompt tokens + 5 output tokens > context 10: score what fits (vLLM needs one generated token)
    pos = await s.positions(list(range(1, 9)), [4, 5, 6, 7, 8])
    assert len(pos) == 1 and len(seen[1]["prompt"]) == 9
    await s.aclose()
