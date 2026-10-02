"""Job envelope and metering (HOST-CLIENT.md §7). Shared by the host and the platform.

A job is a list of completion requests whose prompts are already token ids: the
platform owns tokenization and chat templating, so the platform, the host and the
verifier all agree on exactly which tokens were the prompt. Every sampling field is
explicit, because vLLM fills any field the request leaves unset from the model's
generation_config (Llama 3.1 Instruct ships temperature 0.6, top_p 0.9) and the job
would silently become something the buyer did not ask for.

    job = {
      "job_id": "j_…",
      "timeout_s": 59.5,              # relative, so host clock skew does not matter
      "units_reserved": 0.0123,       # upper bound (max_tokens), informational
      "requests": [ {request}, … ]    # 1..MAX_REQUESTS_PER_JOB
    }
    request = {"prompt_token_ids": [...], "max_tokens", "temperature", "top_p", "top_k",
               "min_p", "repetition_penalty", "seed", "stop", "stop_token_ids",
               "ignore_eos", "logprobs"}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from kwh_bench import reference as ref

from .identity import canonical_bytes, sha256_hex

MAX_REQUESTS_PER_JOB = 256
MAX_LOGPROBS = 20
MAX_STOP_STRINGS = 8
MAX_STOP_STRING_CHARS = 64
VOCAB_SIZE = 128256                      # Llama 3.1 tokenizer

# OpenAI semantics for anything the buyer leaves out; the envelope always carries them all.
REQUEST_DEFAULTS = {
    "max_tokens": 256,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,                          # 0 = disabled
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "seed": None,
    "stop": [],
    "stop_token_ids": [],
    "ignore_eos": False,
    "logprobs": None,
}
REQUEST_FIELDS = {"prompt_token_ids", *REQUEST_DEFAULTS}
JOB_FIELDS = {"job_id", "timeout_s", "units_reserved", "requests"}

# --- metering (provisional; HOST-CLIENT.md §9 D7) ---------------------------
# A unit is one reference job: 65,536 generated + 131,072 prompt tokens. Prefill is far
# cheaper per token than decode; on the 24 GB cards measured so far a 512-token prefill
# costs roughly 1/16 of the decode work per token. One reference job meters to exactly
# 1.0 unit under any weight, so the weight only moves units between prompt-heavy and
# output-heavy buyers.
PREFILL_WEIGHT = 1.0 / 16.0
UNIT_WORK_TOKENS = ref.GENERATED_TOKENS_PER_JOB + PREFILL_WEIGHT * ref.PROMPT_TOKENS_PER_JOB   # 73,728


def units_for(prompt_tokens: int, completion_tokens: int) -> float:
    return (completion_tokens + PREFILL_WEIGHT * prompt_tokens) / UNIT_WORK_TOKENS


class JobInvalid(ValueError):
    """The envelope or a request in it is malformed or does not fit the engine."""


@dataclass(frozen=True)
class RequestSpec:
    index: int
    prompt_token_ids: tuple
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    repetition_penalty: float
    seed: Optional[int]
    stop: tuple
    stop_token_ids: tuple
    ignore_eos: bool
    logprobs: Optional[int]

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0

    @property
    def context_needed(self) -> int:
        return len(self.prompt_token_ids) + self.max_tokens


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x) -> bool:
    return (isinstance(x, (int, float))) and not isinstance(x, bool)


def normalize_request(raw: dict, index: int = 0) -> dict:
    """Fill defaults and validate one request (platform side, before dispatch). Context
    length is checked per host at dispatch, since it depends on the host's engine."""
    if not isinstance(raw, dict):
        raise JobInvalid(f"request {index}: not an object")
    unknown = set(raw) - REQUEST_FIELDS
    if unknown:
        raise JobInvalid(f"request {index}: unknown field(s) {sorted(unknown)}")
    r = {**REQUEST_DEFAULTS, **raw}
    ids = r.get("prompt_token_ids")
    if not isinstance(ids, list) or not ids or not all(_is_int(t) and 0 <= t < VOCAB_SIZE for t in ids):
        raise JobInvalid(f"request {index}: prompt_token_ids must be a non-empty list of token ids < {VOCAB_SIZE}")
    if not _is_int(r["max_tokens"]) or r["max_tokens"] < 1:
        raise JobInvalid(f"request {index}: max_tokens must be an integer >= 1")
    if not _is_num(r["temperature"]) or not 0.0 <= r["temperature"] <= 2.0:
        raise JobInvalid(f"request {index}: temperature must be in [0, 2]")
    if not _is_num(r["top_p"]) or not 0.0 < r["top_p"] <= 1.0:
        raise JobInvalid(f"request {index}: top_p must be in (0, 1]")
    if not _is_int(r["top_k"]) or r["top_k"] < -1:
        raise JobInvalid(f"request {index}: top_k must be an integer >= -1")
    if not _is_num(r["min_p"]) or not 0.0 <= r["min_p"] <= 1.0:
        raise JobInvalid(f"request {index}: min_p must be in [0, 1]")
    if not _is_num(r["repetition_penalty"]) or not 0.0 < r["repetition_penalty"] <= 2.0:
        raise JobInvalid(f"request {index}: repetition_penalty must be in (0, 2]")
    if r["seed"] is not None and not _is_int(r["seed"]):
        raise JobInvalid(f"request {index}: seed must be an integer or null")
    stop = r["stop"]
    if isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list) or len(stop) > MAX_STOP_STRINGS or \
            not all(isinstance(s, str) and 0 < len(s) <= MAX_STOP_STRING_CHARS for s in stop):
        raise JobInvalid(f"request {index}: stop must be up to {MAX_STOP_STRINGS} strings of 1..{MAX_STOP_STRING_CHARS} chars")
    if not isinstance(r["stop_token_ids"], list) or not all(_is_int(t) and 0 <= t < VOCAB_SIZE for t in r["stop_token_ids"]):
        raise JobInvalid(f"request {index}: stop_token_ids must be a list of token ids")
    if not isinstance(r["ignore_eos"], bool):
        raise JobInvalid(f"request {index}: ignore_eos must be a boolean")
    lp = r["logprobs"]
    if lp is not None and (not _is_int(lp) or not 0 <= lp <= MAX_LOGPROBS):
        raise JobInvalid(f"request {index}: logprobs must be null or 0..{MAX_LOGPROBS}")
    r["stop"] = stop
    r["temperature"], r["top_p"], r["min_p"] = float(r["temperature"]), float(r["top_p"]), float(r["min_p"])
    r["repetition_penalty"] = float(r["repetition_penalty"])
    return r


def parse_job(job: dict, max_model_len: int) -> List[RequestSpec]:
    """Host side: validate an envelope against this host's engine. Raises JobInvalid."""
    if not isinstance(job, dict):
        raise JobInvalid("job is not an object")
    unknown = set(job) - JOB_FIELDS
    if unknown:
        raise JobInvalid(f"unknown job field(s) {sorted(unknown)}")
    if not isinstance(job.get("job_id"), str) or not job["job_id"]:
        raise JobInvalid("job_id missing")
    if not _is_num(job.get("timeout_s")) or job["timeout_s"] <= 0:
        raise JobInvalid("timeout_s must be a positive number")
    reqs = job.get("requests")
    if not isinstance(reqs, list) or not 1 <= len(reqs) <= MAX_REQUESTS_PER_JOB:
        raise JobInvalid(f"requests must be a list of 1..{MAX_REQUESTS_PER_JOB}")
    specs = []
    for i, raw in enumerate(reqs):
        # The envelope must be explicit: a missing field would let vLLM's generation_config decide.
        missing = REQUEST_FIELDS - set(raw) if isinstance(raw, dict) else REQUEST_FIELDS
        if missing:
            raise JobInvalid(f"request {i}: missing field(s) {sorted(missing)}")
        r = normalize_request(raw, i)
        spec = RequestSpec(index=i, prompt_token_ids=tuple(r["prompt_token_ids"]), max_tokens=r["max_tokens"],
                           temperature=r["temperature"], top_p=r["top_p"], top_k=r["top_k"], min_p=r["min_p"],
                           repetition_penalty=r["repetition_penalty"], seed=r["seed"], stop=tuple(r["stop"]),
                           stop_token_ids=tuple(r["stop_token_ids"]), ignore_eos=r["ignore_eos"], logprobs=r["logprobs"])
        if spec.context_needed > max_model_len:
            raise JobInvalid(f"request {i}: prompt {len(spec.prompt_token_ids)} + max_tokens {spec.max_tokens} "
                             f"exceeds this engine's context of {max_model_len} tokens")
        specs.append(spec)
    return specs


def job_hash(job: dict) -> str:
    return sha256_hex(canonical_bytes(job))
