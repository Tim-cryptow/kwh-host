"""Output verification, prototype for build step 3 (HOST-CLIENT.md §7).

For a greedy request the question is: was every output token the reference model's
top choice, given the tokens before it? The verifier answers it by teacher-forcing the
host's output through the reference model in a single prefill pass (`prompt_logprobs`)
and looking at each position's rank and gap:

    gap = logprob(reference top-1) - logprob(token the host produced)      (>= 0)

An honest host's gaps are 0, except where two tokens were nearly tied and kernel noise
picked the other one; those gaps are tiny. A substitute model picks its own favourites
and some of them the reference model ranks well below its top choice. Positions with
gap > tau are "confident disagreements"; an honest output has none.

Cost: one prefill over prompt + output on the reference node, no decoding. Prefill is a
small fraction of generation work (the metering weight assumes 1/16 per token), so
checking a request costs a few percent of producing it, not the 100% of re-running it.

Sampled requests (temperature > 0) need a statistical test instead; that is step 3.
"""

from __future__ import annotations

import asyncio
from typing import List, Optional, Protocol, Sequence

import httpx

from ..mockmodel import ToyLM

DEFAULT_TAU = 0.1            # nats; provisional until the honest-vs-substitute experiment on real cards


class Scorer(Protocol):
    async def positions(self, prompt_ids: Sequence[int], output_ids: Sequence[int]) -> List[dict]: ...


def _position(token: int, entry: Optional[dict]) -> dict:
    if not entry:
        raise ValueError("missing prompt_logprobs entry")
    actual = entry.get(str(token)) or entry.get(token)
    if actual is None:
        raise ValueError(f"token {token} absent from its prompt_logprobs entry")
    top = min(entry.values(), key=lambda v: (v.get("rank") or 10**9) if isinstance(v, dict) else 10**9)
    a_lp = float(actual["logprob"])
    t_lp = float(top["logprob"])
    return {"token": token, "logprob": a_lp, "rank": actual.get("rank"), "top1_logprob": t_lp, "gap": max(0.0, t_lp - a_lp)}


class VLLMScorer:
    """Teacher-forced per-position ranks and gaps from a vLLM serving the reference model."""

    def __init__(self, base_url: str, model: Optional[str] = None, transport: Optional[httpx.AsyncBaseTransport] = None,
                 timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_model_len: Optional[int] = None
        self._http = httpx.AsyncClient(base_url=self.base_url, transport=transport, timeout=timeout)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _resolve(self) -> None:
        if self.model and self.max_model_len:
            return
        r = await self._http.get("/v1/models")
        r.raise_for_status()
        card = (r.json().get("data") or [{}])[0]
        self.model = self.model or card.get("id")
        self.max_model_len = int(card.get("max_model_len") or 1024)

    async def positions(self, prompt_ids: Sequence[int], output_ids: Sequence[int]) -> List[dict]:
        await self._resolve()
        out = list(output_ids)
        # vLLM must still generate one token after the scored sequence.
        room = self.max_model_len - 1 - len(prompt_ids)
        if room < len(out):
            out = out[:max(room, 0)]
        if not out:
            return []
        body = {"model": self.model, "prompt": list(prompt_ids) + out, "max_tokens": 1, "temperature": 0.0,
                "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0,
                "prompt_logprobs": 1, "stream": False}
        r = await self._http.post("/v1/completions", json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"reference engine {r.status_code}: {r.text[:300]}")
        plp = r.json()["choices"][0].get("prompt_logprobs") or []
        tail = plp[len(prompt_ids):len(prompt_ids) + len(out)]
        return [_position(t, e) for t, e in zip(out, tail)]


class ToyScorer:
    """Exact scoring under a ToyLM (tests)."""

    def __init__(self, model: ToyLM):
        self.model = model

    async def positions(self, prompt_ids: Sequence[int], output_ids: Sequence[int]) -> List[dict]:
        seq, out = list(prompt_ids), []
        for tok in output_ids:
            lps = self.model.logprobs(seq)
            top = max(lps)
            rank = 1 + sum(1 for v in lps if v > lps[tok])
            out.append({"token": tok, "logprob": lps[tok], "rank": rank, "top1_logprob": top, "gap": top - lps[tok]})
            seq.append(tok)
        await asyncio.sleep(0)
        return out


def judge_greedy(positions: List[dict], tau: float = DEFAULT_TAU) -> dict:
    n = len(positions)
    gaps = [p["gap"] for p in positions]
    confident = sum(1 for g in gaps if g > tau)
    return {
        "positions": n,
        "top1": sum(1 for p in positions if p.get("rank") == 1),
        "top1_rate": round(sum(1 for p in positions if p.get("rank") == 1) / n, 4) if n else None,
        "max_gap": round(max(gaps), 5) if gaps else 0.0,
        "mean_logprob": round(sum(p["logprob"] for p in positions) / n, 5) if n else None,
        "confident_disagreements": confident,
        "tau": tau,
        "pass": confident == 0,
    }


async def verify_greedy_outputs(scorer: Scorer, pairs: Sequence[tuple], tau: float = DEFAULT_TAU) -> dict:
    """pairs: (index, prompt_ids, output_ids) for greedy requests. Returns per-request verdicts
    and an overall pass (every request passes)."""
    per = []
    for index, prompt_ids, output_ids in pairs:
        try:
            pos = await scorer.positions(prompt_ids, output_ids)
            per.append({"index": index, **judge_greedy(pos, tau)})
        except Exception as e:  # noqa: BLE001 - the reference node failing is not the host's fault
            per.append({"index": index, "error": f"{type(e).__name__}: {e}"[:300], "pass": None})
    verdicts = [p["pass"] for p in per if p["pass"] is not None]
    overall = None if not verdicts else all(verdicts)
    return {"checked": len(per), "pass": overall, "inconclusive": sum(1 for p in per if p["pass"] is None),
            "tau": tau, "requests": per}
