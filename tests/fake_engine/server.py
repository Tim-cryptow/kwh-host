"""A fake vLLM OpenAI server for testing the engine sandbox without a GPU.

It speaks the parts of vLLM 0.30.0's API the benchmark and the host client use (health,
version, models, tokenize, streamed and teacher-forced completions) on a Unix socket or a
TCP port, and takes vLLM's command line (`server.py <model> --max-model-len N --uds PATH ...`),
so the sandbox launches it exactly as it launches `vllm serve`. The "model" is arithmetic:
deterministic, cheap, and nothing like Llama, so canaries fail as they should.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import zlib

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

VOCAB = 512
EOS = 2
BOS = 128000


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--served-model-name", default=None)
    p.add_argument("--uds", default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args, _ = p.parse_known_args(argv)          # --dtype, --seed, --revision ...: accepted, ignored
    return args


ARGS = parse_args(sys.argv[1:]) if __name__ == "__main__" else parse_args(["fake/model"])
MODEL = ARGS.served_model_name or ARGS.model
app = FastAPI()


def greedy(prev2: int, prev1: int) -> int:
    return (prev1 * 31 + prev2 * 17 + 7) % VOCAB


def logprob(prev2: int, prev1: int, tok: int, pos: int) -> tuple:
    """(logprob, rank) of `tok` after (prev2, prev1)."""
    g = greedy(prev2 % VOCAB, prev1 % VOCAB)
    if tok % VOCAB == g:
        return -0.1, 1
    return -(2.0 + ((tok * 13 + pos) % 50) / 10.0), 2 + (tok % (VOCAB - 2))


def tokenize_text(text: str) -> list:
    return [3 + zlib.crc32(w.encode()) % (VOCAB - 3) for w in text.split()]


def error(status: int, message: str):
    return JSONResponse({"error": {"message": message, "type": "BadRequestError", "code": status}}, status_code=status)


@app.get("/health")
async def health():
    return Response(status_code=200)


@app.get("/version")
async def version():
    return {"version": "0.30.0"}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "max_model_len": ARGS.max_model_len}]}


@app.post("/tokenize")
async def tokenize(request: Request):
    body = await request.json()
    if "messages" in body:
        text = " ".join(f"{m.get('role')}: {m.get('content')}" for m in body["messages"])
        toks = [BOS] + tokenize_text(text) + [VOCAB - 1]
    else:
        toks = ([BOS] if body.get("add_special_tokens", True) else []) + tokenize_text(body.get("prompt", ""))
    return {"tokens": toks, "count": len(toks), "max_model_len": ARGS.max_model_len}


def generate(prompt: list, n: int, temperature: float, seed, ignore_eos: bool, min_tokens: int, stop_ids: list):
    rng = random.Random(seed if seed is not None else 0)
    seq, out, finish = list(prompt), [], "length"
    for i in range(n):
        p2, p1 = (seq[-2] if len(seq) > 1 else 0), (seq[-1] if seq else 0)
        tok = greedy(p2 % VOCAB, p1 % VOCAB) if temperature == 0 else rng.randrange(VOCAB)
        out.append(tok)
        seq.append(tok)
        if not ignore_eos and len(out) >= min_tokens and (tok == EOS or tok in stop_ids):
            finish = "stop"
            break
    return out, finish


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    prompt = body.get("prompt")
    if isinstance(prompt, str):
        prompt = tokenize_text(prompt)
    prompt = [int(t) for t in prompt]
    n = int(body.get("max_tokens") or 16)
    if len(prompt) + n > ARGS.max_model_len:
        return error(400, f"This model's maximum context length is {ARGS.max_model_len} tokens. However, you "
                          f"requested {len(prompt) + n} tokens ({len(prompt)} in the messages, {n} in the completion).")
    temperature = float(body.get("temperature", 1.0))
    out, finish = generate(prompt, n, temperature, body.get("seed"), bool(body.get("ignore_eos")),
                           int(body.get("min_tokens") or 0), list(body.get("stop_token_ids") or []))
    want_lp = body.get("logprobs") is not None
    as_ids = bool(body.get("return_tokens_as_token_ids"))
    usage = {"prompt_tokens": len(prompt), "completion_tokens": len(out), "total_tokens": len(prompt) + len(out)}

    def lp_block(i: int, tok: int) -> dict:
        seq = prompt + out[:i]
        p2, p1 = (seq[-2] if len(seq) > 1 else 0), (seq[-1] if seq else 0)
        lp, _ = logprob(p2, p1, tok, len(seq))
        name = f"token_id:{tok}" if as_ids else f" t{tok}"
        return {"tokens": [name], "token_logprobs": [lp], "top_logprobs": [{name: lp}], "text_offset": [0]}

    if not body.get("stream"):
        choice = {"index": 0, "text": "".join(f" t{t}" for t in out), "finish_reason": finish, "stop_reason": None}
        k = body.get("prompt_logprobs")
        if k is not None:
            plp = [None]
            for j in range(1, len(prompt)):
                p2, p1 = (prompt[j - 2] if j > 1 else 0), prompt[j - 1]
                lp, rank = logprob(p2, p1, prompt[j], j)
                entry = {str(prompt[j]): {"logprob": lp, "rank": rank, "decoded_token": f" t{prompt[j]}"}}
                if int(k) >= 1 and rank != 1:
                    g = greedy(p2 % VOCAB, p1 % VOCAB)
                    entry[str(g)] = {"logprob": -0.1, "rank": 1, "decoded_token": f" t{g}"}
                plp.append(entry)
            choice["prompt_logprobs"] = plp
        if body.get("return_token_ids"):
            choice["token_ids"], choice["prompt_token_ids"] = out, prompt
        return {"id": "cmpl-fake", "object": "text_completion", "model": MODEL, "choices": [choice], "usage": usage}

    async def stream():
        if body.get("return_token_ids"):
            first = {"choices": [{"index": 0, "text": "", "token_ids": [], "prompt_token_ids": prompt, "finish_reason": None}]}
            yield f"data: {json.dumps(first)}\n\n"
        for i, tok in enumerate(out):
            last = i == len(out) - 1
            ch = {"index": 0, "text": f" t{tok}", "finish_reason": finish if last else None,
                  "stop_reason": None}
            if body.get("return_token_ids"):
                ch["token_ids"] = [tok]
            if want_lp:
                ch["logprobs"] = lp_block(i, tok)
            yield f"data: {json.dumps({'id': 'cmpl-fake', 'model': MODEL, 'choices': [ch]})}\n\n"
        if (body.get("stream_options") or {}).get("include_usage"):
            yield f"data: {json.dumps({'id': 'cmpl-fake', 'choices': [], 'usage': usage})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    if ARGS.uds:
        uvicorn.run(app, uds=ARGS.uds, log_level="warning", ws="none")
    else:
        uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning", ws="none")
