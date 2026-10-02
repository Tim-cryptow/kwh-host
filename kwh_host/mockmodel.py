"""A deterministic stand-in language model for tests and GPU-free demos.

Next-token log-probabilities over a small vocabulary are a pure function of
(seed, the last `context` tokens), so two ToyLMs with the same seed are the same
"model" and teacher-forced scoring is exact. `drift` makes a different model that
agrees with the reference on most contexts and follows its own table on the rest,
which is what a quantized or otherwise substituted model looks like from outside.
"""

from __future__ import annotations

import hashlib
import math
import random
import struct
from typing import Iterable, List, Optional, Sequence, Tuple


def _u64(*parts: int) -> int:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(struct.pack("<q", int(p) & 0x7FFFFFFFFFFFFFFF))
    return int.from_bytes(h.digest(), "little")


def _log_softmax(xs: Sequence[float]) -> List[float]:
    m = max(xs)
    z = m + math.log(sum(math.exp(x - m) for x in xs))
    return [x - z for x in xs]


class ToyLM:
    def __init__(self, seed: int = 0, vocab: int = 512, context: int = 3, eos: int = 2,
                 drift: float = 0.0, drift_seed: int = 7919, sharpness: float = 3.0):
        self.seed, self.vocab, self.context, self.eos = seed, vocab, context, eos
        self.drift, self.drift_seed, self.sharpness = drift, drift_seed, sharpness

    def _table(self, ctx: Tuple[int, ...], seed: int) -> List[float]:
        rng = random.Random(_u64(seed, len(ctx), *ctx))
        return [rng.gauss(0.0, 1.0) * self.sharpness for _ in range(self.vocab)]

    def logprobs(self, prefix: Sequence[int]) -> List[float]:
        ctx = tuple(prefix[-self.context:])
        seed = self.seed
        if self.drift and (_u64(self.drift_seed, *ctx) % 10_000) / 10_000.0 < self.drift:
            seed = self.drift_seed
        return _log_softmax(self._table(ctx, seed))

    def generate(self, prompt: Sequence[int], max_tokens: int, temperature: float = 0.0, top_p: float = 1.0,
                 seed: Optional[int] = None, stop_token_ids: Iterable[int] = (), ignore_eos: bool = False
                 ) -> Tuple[List[int], str]:
        stops = set(stop_token_ids)
        if not ignore_eos:
            stops.add(self.eos)
        rng = random.Random(seed if seed is not None else random.getrandbits(63))
        seq, out = list(prompt), []
        for _ in range(max_tokens):
            lps = self.logprobs(seq)
            if temperature == 0.0:
                tok = max(range(self.vocab), key=lambda t: lps[t])
            else:
                scaled = _log_softmax([lp / temperature for lp in lps])
                order = sorted(range(self.vocab), key=lambda t: -scaled[t])
                keep, mass = [], 0.0
                for t in order:
                    keep.append(t)
                    mass += math.exp(scaled[t])
                    if mass >= top_p:
                        break
                weights = [math.exp(scaled[t]) for t in keep]
                tok = rng.choices(keep, weights=weights)[0]
            out.append(tok)
            seq.append(tok)
            if tok in stops:
                return out, "stop"
        return out, "length"

    @staticmethod
    def detokenize(tokens: Iterable[int]) -> str:
        return " ".join(f"w{t}" for t in tokens)
