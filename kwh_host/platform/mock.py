"""Mock platform: the server side of HOST-CLIENT.md §8, in memory.

It exists so the daemon can be built and tested end to end before step 4. The logic
(`MockPlatform`, `Router`) is framework-free; `create_app` wraps it in FastAPI.
What it enforces is the real thing: signed requests, verified reports, the lifecycle
state machine (§2), challenge canaries (§4), per-heartbeat accrual with integer mints
(§6), job dispatch to live hosts over their WebSocket with re-routing on failure and
signed, job-bound results (§7). What it fakes: the challenge pool (a real platform
scores fresh canaries on its reference node; this one serves the public lock
canaries, which is fine for testing and useless as a guard), the buyer (a dev
endpoint, `POST /v1/mock/jobs`) and the price (metering is provisional, D7).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import secrets
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from kwh_bench import reference as ref
from kwh_bench.lockfile import Lock, load_lock
from kwh_bench.prompts import canonical_prompts
from kwh_bench.verify import verify_report

from ..identity import verify_report_signature, verify_request, verify_result_signature
from ..jobspec import MAX_REQUESTS_PER_JOB, JobInvalid, job_hash, normalize_request, units_for
from ..rebench import (MICROBENCH_REASON, REBENCH_EVERY_SECONDS, age_reason, change_reasons, iso_ts, report_gpus,
                       report_image, span)
from .verifier import DEFAULT_TAU, Scorer, verify_greedy_outputs

STATES = ("registered", "live", "degraded", "offline")


# --- challenges (§4, D9) ---------------------------------------------------

CHALLENGE_CONTINUATIONS = 4    # per challenge; the challenge passes on their mean delta
RECOVERY_PASSES = 2            # passes in a row a host needs after a failed challenge


@dataclass
class Continuation:
    """One scored continuation in the platform's pool: a prompt, the 32 tokens the reference
    node produced after it, and the reference's mean logprob for them (never sent to a host)."""
    id: str
    prompt_text: str
    prompt_tokens: int
    continuation_token_ids: List[int]
    reference_mean_logprob: float          # server-side only

    def to_wire(self) -> dict:
        return {"prompt_text": self.prompt_text, "prompt_tokens": self.prompt_tokens,
                "continuation_token_ids": list(self.continuation_token_ids)}


@dataclass
class Challenge:
    """What a host is asked to score: several continuations, judged together on the mean of
    their deltas, so one noisy continuation does not decide (D9)."""
    id: str
    items: List[Continuation]

    def to_wire(self) -> dict:
        return {"challenge_id": self.id, "items": [c.to_wire() for c in self.items]}


def judge_challenge(ch: Challenge, mean_logprobs: Any, tolerance: float) -> dict:
    """Per-continuation deltas and the verdict. An answer of the wrong shape, or any
    continuation left unscored, fails: the mean is only defined over all of them."""
    got = mean_logprobs if isinstance(mean_logprobs, list) and len(mean_logprobs) == len(ch.items) \
        else [None] * len(ch.items)
    deltas: List[Optional[float]] = []
    for g, c in zip(got, ch.items):
        try:
            deltas.append(None if g is None else round(abs(float(g) - c.reference_mean_logprob), 5))
        except (TypeError, ValueError):
            deltas.append(None)
    mean = None if any(d is None for d in deltas) else round(sum(deltas) / len(deltas), 5)
    return {"pass": mean is not None and mean <= tolerance, "delta": mean, "deltas": deltas}


class ChallengePool:
    def __init__(self, continuations: List[Continuation], seed: int = 0, per_challenge: int = CHALLENGE_CONTINUATIONS):
        if not continuations:
            raise ValueError("empty challenge pool")
        self.continuations = list(continuations)
        self.per_challenge = per_challenge
        self._rng = random.Random(seed)
        self._n = 0

    @classmethod
    def from_lock(cls, lock: Optional[Lock] = None) -> "ChallengePool":
        lock = lock or load_lock()
        if not lock.is_locked:
            raise RuntimeError("kwh-bench lock is incomplete; cannot build a challenge pool from it")
        text = {p.id: p.text for p in canonical_prompts()}
        return cls([Continuation(f"lock-{c.prompt_id}", text[c.prompt_id], ref.PROMPT_TOKENS,
                                 c.expected_token_ids, c.reference_mean_logprob) for c in lock.canaries])

    @classmethod
    async def from_engine(cls, engine, n: int = CHALLENGE_CONTINUATIONS) -> "ChallengePool":
        """Continuations scored by `engine` itself: what the real platform's reference node does
        with the reference model. With the benchmark's mock engine this lets a GPU-free demo
        host go live; it proves nothing about a model."""
        text = {p.id: p.text for p in canonical_prompts()}
        out = []
        async with engine as e:
            for pid in range(n):
                ids = (await e.tokenize(text[pid]))[:ref.PROMPT_TOKENS]
                cont = list(range(100 + pid, 100 + pid + ref.CANARY_TOKENS))
                lps = await e.score_continuation(ids, cont)
                out.append(Continuation(f"engine-{pid}", text[pid], ref.PROMPT_TOKENS, cont, sum(lps) / len(lps)))
        return cls(out)

    def issue(self) -> Challenge:
        picks = self._rng.sample(self.continuations, min(self.per_challenge, len(self.continuations)))
        self._n += 1
        # Fresh id per issue so a host cannot replay an earlier answer.
        return Challenge(f"ch-{self._n}-{secrets.token_hex(4)}", picks)


# --- settings / records ---------------------------------------------------

@dataclass
class Settings:
    heartbeat_seconds: float = 30.0
    challenge_every_seconds: float = 300.0
    microbench_every_seconds: float = 1800.0
    offline_after_seconds: float = 90.0
    max_failures: int = 5
    tolerance: float = ref.CANARY_MAX_LOGPROB_DELTA    # on a challenge's mean delta, like the benchmark's canaries
    recovery_passes: int = RECOVERY_PASSES
    microbench_tolerance: float = 0.10
    accept_uncertified: bool = False       # tests only: mock-engine reports
    allow_bare_metal: bool = False         # pod testing only (D4)
    require_engine_version: Optional[str] = None   # defaults to the lock's vLLM version
    # router (§7)
    job_timeout_seconds: float = 120.0     # buyer-side deadline when the caller sets none
    max_attempts: int = 3                  # hosts tried per job before the buyer sees a failure
    queue_factor: float = 2.0              # in-flight requests per host <= factor x its concurrency
    unit_window_seconds: float = 300.0     # in-flight units per host <= rate x window
    verify_fraction: float = 0.0           # share of completed jobs whose greedy outputs are verified
    verify_max_requests: int = 4           # greedy requests checked per verified job
    verify_tau: float = DEFAULT_TAU
    # re-benchmark and reliability (§5)
    rebench_every_seconds: float = REBENCH_EVERY_SECONDS   # a report older than this must be replaced
    rebench_max_seconds: float = 3600.0                     # a re-benchmark that takes longer counts as failures
    bucket_seconds: float = 300.0                 # reliability counters are kept per 5 minutes ...
    history_seconds: float = 7 * 86400.0          # ... for 7 days

    def wire(self) -> dict:
        return {"heartbeat_seconds": self.heartbeat_seconds, "challenge_every_seconds": self.challenge_every_seconds,
                "microbench_every_seconds": self.microbench_every_seconds,
                "offline_after_seconds": self.offline_after_seconds, "max_failures": self.max_failures,
                "rebench_every_seconds": self.rebench_every_seconds}


EVENTS_KEPT = 5000
LATENCIES_PER_BUCKET = 1000
WINDOWS = (("1h", 3600.0), ("24h", 86400.0), ("7d", 7 * 86400.0))


def new_bucket() -> dict:
    """Raw counters for one 5-minute slice of one host's life (§5). Step 3 scores from these."""
    return {"state_s": {}, "beats_accepted": 0, "beats_rejected": 0, "minted": 0,
            "challenges_passed": 0, "challenges_failed": 0, "challenge_deltas": [],
            "microbench_uph": [], "microbench_misses": 0,
            "jobs_completed": 0, "jobs_failed": 0, "bad_results": 0, "units": 0.0, "job_latency_ms": [],
            "send_failures": 0, "engine_restarts": 0, "results_undelivered": 0}


def percentile(values: List[float], q: float) -> Optional[float]:
    """Nearest-rank percentile; None for no values."""
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, max(0, math.ceil(q / 100.0 * len(s)) - 1))]


@dataclass
class HostRecord:
    host_id: str
    public_key: str
    token: str
    rate: float
    bucket: str
    report_sha256: str
    launch_mode: str
    registered_at: float
    state: str = "registered"
    reasons: List[str] = field(default_factory=list)
    last_beat_at: Optional[float] = None
    last_accepted_beat_at: Optional[float] = None
    accrual: float = 0.0
    balance: int = 0
    failures: int = 0
    pending: Optional[Challenge] = None
    last_challenge_at: Optional[float] = None
    last_challenge: Optional[dict] = None
    passes_needed: int = 0                 # passes in a row still owed after a failed challenge (D9)
    microbench_misses: int = 0
    last_microbench: Optional[dict] = None
    rebench_required: bool = False
    engine_instance: Optional[str] = None  # changes on every engine start
    jobs_completed: int = 0
    jobs_failed: int = 0                   # failed, rejected, timed out or disconnected: re-routed
    bad_results: int = 0                   # completed but malformed or unsigned: a protocol violation
    delivered_units: float = 0.0           # what the host is paid for (step 4 settles it)
    verifications_failed: int = 0
    events: List[dict] = field(default_factory=list)
    # re-benchmark (§5): what the current report was measured on, and why a new one is owed
    state_since: float = 0.0
    accounted_to: float = 0.0              # state time is in the buckets up to here
    report_at: float = 0.0                 # the report's age counts from here
    report_accepted_at: float = 0.0
    report_gpus: Dict[str, Optional[str]] = field(default_factory=dict)   # GPU uuid -> driver version
    report_image: Optional[str] = None
    rebench_reasons: List[str] = field(default_factory=list)
    benchmarking: bool = False
    benchmarking_since: Optional[float] = None
    rejected_reasons: Optional[List[str]] = None   # of the last rejected beat, so a run of them is one event
    telemetry: dict = field(default_factory=dict)  # the host's own counters, as of its last heartbeat
    buckets: Dict[int, dict] = field(default_factory=dict)                # reliability counters, per 5 minutes

    def event(self, now: float, kind: str, **data) -> None:
        self.events.append({"t": now, "kind": kind, **data})
        if len(self.events) > EVENTS_KEPT:
            del self.events[:-EVENTS_KEPT]

    def take_report(self, report: dict, now: float) -> None:
        """Remember what this report was measured on, so a change can be told apart from it. Its
        age counts from when it was measured, or from now if it claims to be from the future."""
        measured = iso_ts(report.get("finished_at"))
        self.report_accepted_at = now
        self.report_at = min(now, measured) if measured is not None else now
        self.report_gpus = report_gpus(report)
        self.report_image = report_image(report)
        self.rebench_required, self.rebench_reasons, self.microbench_misses = False, [], 0

    def summary(self) -> dict:
        return {"host_id": self.host_id, "state": self.state, "reasons": self.reasons, "state_since": self.state_since,
                "rate_units_per_hour": self.rate, "bucket": self.bucket, "accrual": round(self.accrual, 4),
                "balance": self.balance, "failures": self.failures, "last_beat_at": self.last_beat_at,
                "last_challenge": self.last_challenge, "passes_needed": self.passes_needed,
                "last_microbench": self.last_microbench, "rebench_required": self.rebench_required,
                "rebench_reasons": self.rebench_reasons, "benchmarking": self.benchmarking,
                "report_sha256": self.report_sha256,
                "jobs": {"completed": self.jobs_completed, "failed": self.jobs_failed, "bad_results": self.bad_results,
                         "delivered_units": round(self.delivered_units, 6),
                         "verifications_failed": self.verifications_failed}}


class Rejected(Exception):
    def __init__(self, status: int, detail: str):
        self.status, self.detail = status, detail
        super().__init__(detail)


def bucket_for(rate: float) -> str:
    """Placeholder until step 3 defines score buckets."""
    return f"I-1/{int(rate // 20) * 20}"


# --- the platform ---------------------------------------------------------

class MockPlatform:
    def __init__(self, pool: ChallengePool, settings: Optional[Settings] = None, clock: Callable[[], float] = time.time,
                 lock: Optional[Lock] = None):
        self.pool = pool
        self.s = settings or Settings()
        self.clock = clock
        self.lock = lock or load_lock()
        if self.s.require_engine_version is None and not self.s.accept_uncertified:
            self.s.require_engine_version = self.lock.vllm_version
        self.hosts: Dict[str, HostRecord] = {}

    # -- helpers -------------------------------------------------------
    def host(self, host_id: str) -> HostRecord:
        rec = self.hosts.get(host_id)
        if rec is None:
            raise Rejected(404, f"unknown host {host_id}")
        return rec

    def by_public_key(self, pub: str) -> Optional[HostRecord]:
        return next((h for h in self.hosts.values() if h.public_key == pub), None)

    def _sweep(self, rec: HostRecord, now: float) -> None:
        """Lazy offline detection: no accepted heartbeat within the timeout. The host went offline
        when the timeout ran out, not when someone noticed, and its state time says so."""
        if rec.state in ("live", "degraded") and rec.last_accepted_beat_at is not None:
            lapsed = rec.last_accepted_beat_at + self.s.offline_after_seconds
            if now > lapsed:
                self._set_state(rec, now, "offline", ["no accepted heartbeat within timeout"], at=lapsed)

    def _set_state(self, rec: HostRecord, now: float, state: str, reasons: List[str], at: Optional[float] = None) -> None:
        at = now if at is None else at
        self._account(rec, at)
        if state == "offline":
            rec.accrual = 0.0          # never minted late (§6)
            rec.pending = None
        if state != rec.state:
            rec.event(at, "state", **{"from": rec.state, "to": state, "reasons": reasons})
            rec.state_since = at
        rec.state, rec.reasons = state, list(reasons)

    # -- reliability counters (§5) -------------------------------------
    def _bucket(self, rec: HostRecord, t: float) -> dict:
        k = int(t // self.s.bucket_seconds)
        b = rec.buckets.get(k)
        if b is None:
            b = rec.buckets[k] = new_bucket()
            oldest = int((t - self.s.history_seconds) // self.s.bucket_seconds)
            for old in [x for x in rec.buckets if x < oldest]:
                del rec.buckets[old]
        return b

    def _account(self, rec: HostRecord, until: float) -> None:
        """Book the time since the last booking to the current state, split at bucket edges."""
        t = max(rec.accounted_to, until - self.s.history_seconds)
        bs = self.s.bucket_seconds
        while t < until:
            step = min((int(t // bs) + 1) * bs, until) - t
            b = self._bucket(rec, t)
            b["state_s"][rec.state] = b["state_s"].get(rec.state, 0.0) + step
            t += step
        rec.accounted_to = max(rec.accounted_to, until)

    def _take_telemetry(self, rec: HostRecord, b: dict, tel: Any, now: float) -> None:
        """What the host saw since its last heartbeat reached us: beats it could not send, engine
        restarts, results it could not deliver. The host can under-report these; it cannot use them
        to look better than the platform's own counters say."""
        if not isinstance(tel, dict):
            return

        def count(k: str) -> int:
            try:
                return max(0, min(int(tel.get(k) or 0), 1_000_000))
            except (TypeError, ValueError):
                return 0
        sent, restarts, undelivered = count("send_failures"), count("engine_restarts"), count("results_undelivered")
        b["send_failures"] += sent
        b["engine_restarts"] += restarts
        b["results_undelivered"] += undelivered
        err = str(tel.get("last_error") or "")[:200] or None
        rec.telemetry = {"uptime_s": tel.get("uptime_s") if isinstance(tel.get("uptime_s"), (int, float)) else None,
                         "send_failures": sent, "engine_restarts": restarts, "results_undelivered": undelivered,
                         "last_error": err, "t": now}
        if sent or restarts or undelivered:
            rec.event(now, "host_reported", send_failures=sent, engine_restarts=restarts,
                      results_undelivered=undelivered, last_error=err)

    def job_outcome(self, rec: HostRecord, now: float, outcome: str, latency_ms: Optional[float] = None,
                    units: float = 0.0) -> None:
        b = self._bucket(rec, now)
        if outcome == "completed":
            b["jobs_completed"] += 1
            b["units"] += units
            if latency_ms is not None and len(b["job_latency_ms"]) < LATENCIES_PER_BUCKET:
                b["job_latency_ms"].append(latency_ms)
        elif outcome == "bad_result":
            b["bad_results"] += 1
        else:
            b["jobs_failed"] += 1

    def reliability(self, rec: HostRecord, now: float) -> dict:
        """Raw aggregates over the last hour, day and week, to the nearest 5 minutes. Not a score:
        scoring, bucketing and decay are step 3's, computed from the same counters."""
        self._account(rec, now)
        bs, hb = self.s.bucket_seconds, self.s.heartbeat_seconds
        out = {}
        for name, window in WINDOWS:
            if window > self.s.history_seconds:
                continue
            first = int((now - window) // bs)
            sel = [b for k, b in rec.buckets.items() if k > first]
            states: Dict[str, float] = {}
            for b in sel:
                for st, secs in b["state_s"].items():
                    states[st] = states.get(st, 0.0) + secs
            observed = sum(states.values())

            def total(key: str) -> float:
                return sum(b[key] for b in sel)

            deltas = [d for b in sel for d in b["challenge_deltas"]]
            uph = [u for b in sel for u in b["microbench_uph"]]
            lat = [x for b in sel for x in b["job_latency_ms"]]
            out[name] = {
                "observed_s": round(observed),
                "state_s": {st: round(v) for st, v in sorted(states.items())},
                "live_fraction": round(states.get("live", 0.0) / observed, 4) if observed else None,
                "heartbeats": {"accepted": total("beats_accepted"), "rejected": total("beats_rejected"),
                               "expected": int(observed // hb)},
                "challenges": {"passed": total("challenges_passed"), "failed": total("challenges_failed"),
                               "mean_delta": round(sum(deltas) / len(deltas), 5) if deltas else None,
                               "max_delta": max(deltas) if deltas else None},
                "microbench": {"runs": len(uph), "misses": total("microbench_misses"),
                               "median_units_per_hour": percentile(uph, 50)},
                "jobs": {"completed": total("jobs_completed"), "failed": total("jobs_failed"),
                         "bad_results": total("bad_results"), "units": round(total("units"), 6),
                         "latency_ms_p50": percentile(lat, 50), "latency_ms_p95": percentile(lat, 95)},
                "minted": total("minted"),
                "host_reported": {"send_failures": total("send_failures"), "engine_restarts": total("engine_restarts"),
                                  "results_undelivered": total("results_undelivered")},
            }
        return out

    # -- re-benchmark (§5) ---------------------------------------------
    def _rebench_held(self, rec: HostRecord) -> List[str]:
        return ["re-benchmark required: " + "; ".join(rec.rebench_reasons)]

    def _require_rebench(self, rec: HostRecord, now: float, reasons: List[str]) -> None:
        """The report no longer describes this host: degraded, no challenges, until a new one is accepted."""
        new = [r for r in reasons if r and r not in rec.rebench_reasons]
        if not new:
            return
        rec.rebench_required = True
        rec.rebench_reasons = rec.rebench_reasons + new
        rec.event(now, "rebench_required", reasons=new)
        if rec.state == "live" or (rec.state == "degraded" and not rec.benchmarking):
            self._set_state(rec, now, "degraded", self._rebench_held(rec))

    def _rebench_triggers(self, rec: HostRecord, eng: dict, gs: dict, now: float) -> List[str]:
        out = change_reasons(rec.report_gpus, rec.report_image, gs.get("uuid"), gs.get("driver_version"), eng.get("image"))
        old = age_reason(rec.report_at, now, self.s.rebench_every_seconds)
        return out + ([old] if old else [])

    def _benchmarking_beat(self, rec: HostRecord, now: float) -> List[str]:
        """A re-benchmarking host has stopped its engine on purpose. Its beat is accepted without
        the engine and GPU checks (the benchmark's own engine is on the GPU), so it does not slide
        to offline, and it stays degraded: no challenge, no jobs, no mint. Not forever, though."""
        if not rec.benchmarking:
            rec.benchmarking, rec.benchmarking_since = True, now
            rec.pending = None
            rec.event(now, "rebench_started", reasons=list(rec.rebench_reasons))
        if now - rec.benchmarking_since > self.s.rebench_max_seconds:
            return [f"re-benchmark running longer than {span(self.s.rebench_max_seconds)}"]
        if rec.state in ("live", "degraded", "offline"):      # offline: it is back, and says what it is doing
            self._set_state(rec, now, "degraded", ["re-benchmarking"])
        return []

    def _beat_problems(self, rec: HostRecord, eng: dict, gs: dict, now: float) -> List[str]:
        reasons: List[str] = []
        if not eng.get("healthy"):
            said = str(eng.get("error") or "")[:240]      # the host's own account, when it has one
            reasons.append("engine unhealthy" + (f": {said}" if said else ""))
        if self.s.require_engine_version and eng.get("version") != self.s.require_engine_version:
            reasons.append(f"engine version {eng.get('version')!r} != lock {self.s.require_engine_version!r}")
        if eng.get("launch_mode") != "docker" and not self.s.allow_bare_metal:
            reasons.append("engine not sandboxed (D4)")
        # A restarted engine is a new engine: it could be serving anything. No more work until it
        # passes a challenge (a host that lies about the restart is left to output verification).
        inst = eng.get("instance")
        if inst and rec.engine_instance and inst != rec.engine_instance and rec.state == "live":
            self._set_state(rec, now, "degraded", ["engine restarted; awaiting a challenge"])
            rec.pending = None
        if inst:
            rec.engine_instance = inst
        if gs.get("available") and (gs.get("foreign_processes") or 0) > 0:
            reasons.append(f"host_contention: {gs['foreign_processes']} foreign process(es) on the GPU")
        return reasons

    def _check_report(self, report: dict, public_key: str) -> List[str]:
        problems: List[str] = []
        if verify_report_signature(report) != public_key:
            problems.append("report signature missing or not by this host")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(report, f)
            path = Path(f.name)
        try:
            ok, verify_problems = verify_report(path, lock=self.lock)
        finally:
            path.unlink(missing_ok=True)
        if not ok:
            problems.extend(verify_problems)
        if not report.get("certified") and not self.s.accept_uncertified:
            problems.append("report is not certified: " + "; ".join(report.get("certified_reasons") or ["no reason given"]))
        mode = (report.get("engine") or {}).get("launch_mode")
        if mode != "docker" and not self.s.allow_bare_metal:
            problems.append(f"engine launch mode {mode!r} is not the Docker sandbox (D4)")
        return problems

    # -- contract ------------------------------------------------------
    def register(self, public_key: str, body: dict) -> dict:
        now = self.clock()
        if body.get("public_key") != public_key:
            raise Rejected(400, "public_key in body does not match the signing key")
        report = body.get("report") or {}
        problems = self._check_report(report, public_key)
        if problems:
            raise Rejected(400, "report rejected: " + " | ".join(problems))
        rate = float(report["score"]["units_per_hour"])
        rec = self.by_public_key(public_key)
        # Reports are public (the results table publishes them) and signing one only proves who
        # signed it, so a report can stand behind one host only.
        other = next((h for h in self.hosts.values()
                      if h.report_sha256 == report["report_sha256"] and h.public_key != public_key), None)
        if other is not None:
            raise Rejected(409, "this benchmark report is already registered to another host")
        if rec is None:
            host_id = "h_" + hashlib.sha256(public_key.encode()).hexdigest()[:12]
            rec = HostRecord(host_id=host_id, public_key=public_key, token=secrets.token_urlsafe(24), rate=rate,
                             bucket=bucket_for(rate), report_sha256=report["report_sha256"],
                             launch_mode=report["engine"]["launch_mode"], registered_at=now,
                             state_since=now, accounted_to=now)
            self.hosts[host_id] = rec
            rec.event(now, "registered", rate=rate)
        else:
            rec.token = secrets.token_urlsafe(24)
            rec.rate, rec.bucket, rec.report_sha256 = rate, bucket_for(rate), report["report_sha256"]
            rec.event(now, "re-registered", rate=rate)
        rec.take_report(report, now)
        return {"host_id": rec.host_id, "token": rec.token, "rate_units_per_hour": rec.rate, "bucket": rec.bucket,
                "config": self.s.wire()}

    def heartbeat(self, host_id: str, body: dict) -> dict:
        now = self.clock()
        rec = self.host(host_id)
        self._sweep(rec, now)
        b = self._bucket(rec, now)
        self._take_telemetry(rec, b, body.get("telemetry"), now)
        eng, gs = body.get("engine") or {}, body.get("gpu_sample") or {}
        if body.get("benchmarking"):
            reasons = self._benchmarking_beat(rec, now)
        else:
            if rec.benchmarking:
                rec.benchmarking, rec.benchmarking_since = False, None
                rec.event(now, "rebench_ended", rebench_required=rec.rebench_required)
            reasons = self._beat_problems(rec, eng, gs, now)
            self._require_rebench(rec, now, self._rebench_triggers(rec, eng, gs, now))

        interval = 0.0
        if rec.last_accepted_beat_at is not None:
            interval = min(now - rec.last_accepted_beat_at, 2 * self.s.heartbeat_seconds)

        minted = 0
        challenge: Optional[Challenge] = None
        if not reasons:
            b["beats_accepted"] += 1
            rec.rejected_reasons = None
            if rec.state == "live" and body.get("wants_mint", True) and interval > 0:
                rec.accrual += rec.rate * interval / 3600.0
                minted = int(rec.accrual)
                if minted:
                    rec.accrual -= minted
                    rec.balance += minted
                    b["minted"] += minted
            rec.last_accepted_beat_at = now
            if rec.rebench_required and not rec.benchmarking:
                # Reachable and healthy, but its report no longer counts: nothing to challenge for.
                held = self._rebench_held(rec)
                if rec.state != "degraded" or rec.reasons != held:
                    self._set_state(rec, now, "degraded", held)
            elif not rec.benchmarking:
                due = rec.pending is None and (
                    rec.state != "live" or rec.last_challenge_at is None
                    or now - rec.last_challenge_at >= self.s.challenge_every_seconds)
                if due:
                    challenge = self.pool.issue()
                    rec.pending = challenge
                    rec.event(now, "challenge", id=challenge.id)
        else:
            b["beats_rejected"] += 1
            if reasons != rec.rejected_reasons:
                rec.event(now, "heartbeat_rejected", reasons=reasons)
                rec.rejected_reasons = reasons
            rec.failures += 1
            if rec.state == "live":
                self._set_state(rec, now, "degraded", reasons)
            elif rec.state == "degraded":
                rec.reasons = reasons
            if rec.failures >= self.s.max_failures and rec.state != "offline":
                self._set_state(rec, now, "offline", reasons + [f"{rec.failures} consecutive failures"])
        rec.last_beat_at = now
        return {"state": rec.state, "reasons": reasons or rec.reasons, "accepted": not reasons,
                "state_since": rec.state_since, "challenge": challenge.to_wire() if challenge else None,
                "minted": minted, "accrual": round(rec.accrual, 4), "balance": rec.balance,
                "rebench_required": rec.rebench_required, "rebench_reasons": rec.rebench_reasons,
                "benchmarking": rec.benchmarking, "config": self.s.wire()}

    def liveness(self, host_id: str, body: dict) -> dict:
        now = self.clock()
        rec = self.host(host_id)
        ch = rec.pending
        if ch is None or body.get("challenge_id") != ch.id:
            raise Rejected(400, "no such pending challenge")
        verdict = judge_challenge(ch, body.get("mean_logprobs"), self.s.tolerance)
        passed, delta = verdict["pass"], verdict["delta"]
        rec.pending = None
        rec.last_challenge_at = now
        rec.last_challenge = {"id": ch.id, "pass": passed, "delta": delta, "deltas": verdict["deltas"],
                              "continuations": [c.id for c in ch.items], "elapsed_ms": body.get("elapsed_ms"), "t": now}
        rec.event(now, "liveness", id=ch.id, passed=passed, delta=delta)
        b = self._bucket(rec, now)
        b["challenges_passed" if passed else "challenges_failed"] += 1
        if delta is not None:
            b["challenge_deltas"].append(delta)
        if passed:
            rec.failures = 0
            rec.passes_needed = max(0, rec.passes_needed - 1)
            healthy_now = rec.last_accepted_beat_at is not None and rec.last_accepted_beat_at == rec.last_beat_at
            if rec.state != "live" and healthy_now:
                if rec.rebench_required:
                    self._set_state(rec, now, "degraded", [f"challenge passed (mean delta {delta}); "
                                                           + self._rebench_held(rec)[0]])
                elif rec.passes_needed:
                    # After a failed challenge one pass is not enough: a substitute model can get lucky
                    # once; it does not get lucky twice in a row (D9).
                    self._set_state(rec, now, "degraded", [f"challenge passed (mean delta {delta}); "
                                                           f"{rec.passes_needed} more in a row before live"])
                else:
                    self._set_state(rec, now, "live", [])
        else:
            rec.failures += 1
            rec.passes_needed = self.s.recovery_passes
            if rec.failures >= self.s.max_failures:
                self._set_state(rec, now, "offline", [f"canary failed, {rec.failures} consecutive failures"])
            else:
                self._set_state(rec, now, "degraded", [f"canary failed (mean delta {delta})"])
        return {**verdict, "state": rec.state, "passes_needed": rec.passes_needed}

    def microbench(self, host_id: str, body: dict) -> dict:
        now = self.clock()
        rec = self.host(host_id)
        uph = float(body.get("units_per_hour") or 0.0)
        within = rec.rate > 0 and abs(uph - rec.rate) / rec.rate <= self.s.microbench_tolerance
        rec.last_microbench = {"units_per_hour": uph, "job_seconds": body.get("job_seconds"), "within": within, "t": now}
        rec.event(now, "microbench", units_per_hour=uph, within=within)
        b = self._bucket(rec, now)
        b["microbench_uph"].append(uph)
        if within:
            rec.microbench_misses = 0
        else:
            rec.microbench_misses += 1
            b["microbench_misses"] += 1
            if rec.microbench_misses >= 2:
                self._require_rebench(rec, now, [MICROBENCH_REASON])
        return {"accepted": True, "within_tolerance": within, "rebench_required": rec.rebench_required}

    def upload_report(self, host_id: str, body: dict) -> dict:
        now = self.clock()
        rec = self.host(host_id)
        report = body.get("report") or {}
        problems = self._check_report(report, rec.public_key)
        if problems:
            raise Rejected(400, "report rejected: " + " | ".join(problems))
        if any(h.report_sha256 == report["report_sha256"] and h is not rec for h in self.hosts.values()):
            raise Rejected(409, "this benchmark report is already registered to another host")
        previous = rec.rate
        rec.rate = float(report["score"]["units_per_hour"])
        rec.bucket, rec.report_sha256 = bucket_for(rec.rate), report["report_sha256"]
        rec.take_report(report, now)
        rec.event(now, "re-benchmarked", rate=rec.rate, previous_rate=previous)
        if rec.state == "degraded" and not rec.benchmarking:
            rec.reasons = ["new report accepted; awaiting a challenge"]
        return {"rate_units_per_hour": rec.rate, "bucket": rec.bucket, "previous_rate_units_per_hour": previous}

    def status(self, host_id: str) -> dict:
        now = self.clock()
        rec = self.host(host_id)
        self._sweep(rec, now)
        out = rec.summary()
        out.update({"now": now, "registered_at": rec.registered_at, "report_at": rec.report_at,
                    "report_accepted_at": rec.report_accepted_at,
                    "rebench_due_at": rec.report_at + self.s.rebench_every_seconds,
                    "host_reported": rec.telemetry, "reliability": self.reliability(rec, now),
                    "config": self.s.wire(), "events": rec.events[-20:]})
        return out

    def events(self, host_id: str, since: Optional[float] = None, limit: int = 200,
               kinds: Optional[List[str]] = None) -> dict:
        rec = self.host(host_id)
        self._sweep(rec, self.clock())
        sel = [e for e in rec.events if (since is None or e["t"] > since) and (not kinds or e["kind"] in kinds)]
        limit = max(1, min(int(limit), EVENTS_KEPT))
        return {"host_id": host_id, "events": sel[-limit:], "more": len(sel) > limit}


# --- router (§7) ----------------------------------------------------------------

class ConnectionLost(RuntimeError):
    pass


class HostConnection:
    """One host's open job channel. `send` writes one text frame."""

    def __init__(self, host_id: str, send: Callable[[str], Any], hello: dict):
        self.host_id = host_id
        self._send = send
        self.hello = hello
        self.max_concurrency = int(hello.get("max_concurrency") or ref.CONCURRENCY)
        self.max_model_len = int(hello.get("max_model_len") or ref.MAX_MODEL_LEN)
        self.pending: Dict[str, asyncio.Future] = {}
        self.inflight_requests = 0
        self.inflight_units = 0.0
        self.closed = False
        self._lock = asyncio.Lock()

    async def send(self, msg: dict) -> None:
        if self.closed:
            raise ConnectionLost(f"{self.host_id}: job channel closed")
        async with self._lock:
            await self._send(json.dumps(msg))

    def close(self, reason: str) -> None:
        self.closed = True
        for fut in self.pending.values():
            if not fut.done():
                fut.set_exception(ConnectionLost(reason))


class Router:
    """Sends a buyer's job to a live host, checks what comes back, re-routes on failure.

    A job goes to the least-loaded live host whose engine fits it; if that host fails,
    rejects, times out, disconnects or returns a result that does not check out, the
    next host gets it, up to `max_attempts`. The buyer sees a failure only when no host
    could deliver before the deadline (primer §6: a slower response, never a failed
    request, as long as capacity exists)."""

    def __init__(self, platform: MockPlatform, scorer: Optional[Scorer] = None, rng: Optional[random.Random] = None):
        self.p = platform
        self.scorer = scorer
        self.rng = rng or random.Random()
        self.conns: Dict[str, HostConnection] = {}
        self.recent: List[dict] = []

    # -- channel lifecycle -------------------------------------------------
    def attach(self, conn: HostConnection) -> None:
        old = self.conns.get(conn.host_id)
        if old is not None and old is not conn:
            old.close("replaced by a new connection")
        self.conns[conn.host_id] = conn
        self.p.host(conn.host_id).event(self.p.clock(), "jobs_connected", max_concurrency=conn.max_concurrency,
                                        max_model_len=conn.max_model_len)

    def detach(self, conn: HostConnection) -> None:
        conn.close("job channel closed")
        if self.conns.get(conn.host_id) is conn:
            del self.conns[conn.host_id]
            rec = self.p.hosts.get(conn.host_id)
            if rec is not None:
                rec.event(self.p.clock(), "jobs_disconnected")

    def on_result(self, conn: HostConnection, result: dict) -> None:
        fut = conn.pending.get(str(result.get("job_id")))
        if fut is not None and not fut.done():
            fut.set_result(result)

    # -- admission and choice ---------------------------------------------
    def candidates(self, exclude: set, context_needed: int, n_requests: int, units: float) -> List[HostConnection]:
        now = self.p.clock()
        out = []
        for hid, conn in self.conns.items():
            rec = self.p.hosts.get(hid)
            if hid in exclude or conn.closed or rec is None:
                continue
            self.p._sweep(rec, now)
            if rec.state != "live" or conn.max_model_len < context_needed:
                continue
            # An idle host always takes one job; a busy one only within its queue and unit window.
            if conn.inflight_requests and \
                    conn.inflight_requests + n_requests > self.p.s.queue_factor * conn.max_concurrency:
                continue
            if conn.inflight_units and \
                    conn.inflight_units + units > rec.rate * self.p.s.unit_window_seconds / 3600.0:
                continue
            out.append(conn)
        self.rng.shuffle(out)
        out.sort(key=lambda c: c.inflight_requests)
        return out

    def check_result(self, conn: HostConnection, job: dict, jhash: str, result: dict) -> List[str]:
        """Everything the platform can check without a model: signature, binding, shape."""
        problems: List[str] = []
        rec = self.p.host(conn.host_id)
        if verify_result_signature(result) != rec.public_key:
            problems.append("result signature missing, invalid or not by this host")
        if result.get("job_id") != job["job_id"] or result.get("job_sha256") != jhash:
            problems.append("result is not bound to the dispatched job")
        if result.get("host_id") != conn.host_id:
            problems.append("result names another host")
        if result.get("status") == "completed":
            outs, reqs = result.get("outputs") or [], job["requests"]
            if [o.get("index") for o in outs] != list(range(len(reqs))):
                problems.append("outputs do not match the requests")
            else:
                for o, r in zip(outs, reqs):
                    ids = o.get("token_ids")
                    if o.get("error") or not isinstance(ids, list) or not all(isinstance(t, int) for t in ids):
                        problems.append(f"output {o['index']}: no token ids")
                        break
                    if len(ids) > r["max_tokens"]:
                        problems.append(f"output {o['index']}: {len(ids)} tokens > max_tokens {r['max_tokens']}")
                        break
            usage = result.get("usage") or {}
            if usage.get("prompt_tokens") != sum(len(r["prompt_token_ids"]) for r in reqs):
                problems.append("usage.prompt_tokens does not match the job")
            if usage.get("completion_tokens") != sum(len(o.get("token_ids") or []) for o in outs):
                problems.append("usage.completion_tokens does not match the outputs")
        return problems

    # -- the buyer's call ---------------------------------------------------
    async def submit(self, requests: List[dict], timeout_s: Optional[float] = None) -> dict:
        if not isinstance(requests, list) or not 1 <= len(requests) <= MAX_REQUESTS_PER_JOB:
            raise JobInvalid(f"requests must be a list of 1..{MAX_REQUESTS_PER_JOB}")
        normalized = [normalize_request(r, i) for i, r in enumerate(requests)]
        n = len(normalized)
        units_reserved = units_for(sum(len(r["prompt_token_ids"]) for r in normalized),
                                   sum(r["max_tokens"] for r in normalized))
        context_needed = max(len(r["prompt_token_ids"]) + r["max_tokens"] for r in normalized)
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + float(timeout_s or self.p.s.job_timeout_seconds)
        base_id = "j_" + secrets.token_hex(8)
        attempts: List[dict] = []
        tried: set = set()
        while len(attempts) < self.p.s.max_attempts:
            remaining = deadline - loop.time()
            if remaining <= 1.0:
                break
            cands = self.candidates(tried, context_needed, n, units_reserved)
            if not cands:
                break
            conn = cands[0]
            tried.add(conn.host_id)
            job = {"job_id": f"{base_id}.{len(attempts) + 1}", "timeout_s": round(remaining - 0.5, 3),
                   "units_reserved": round(units_reserved, 6), "requests": normalized}
            jhash = job_hash(job)
            fut = loop.create_future()
            conn.pending[job["job_id"]] = fut
            conn.inflight_requests += n
            conn.inflight_units += units_reserved
            attempt: Dict[str, Any] = {"job_id": job["job_id"], "host_id": conn.host_id}
            t0 = loop.time()
            result = None
            try:
                await conn.send({"type": "job", "job": job})
                result = await asyncio.wait_for(fut, timeout=remaining)
            except asyncio.TimeoutError:
                attempt["outcome"] = "timeout"
                try:
                    await conn.send({"type": "cancel", "job_id": job["job_id"]})
                except Exception:  # noqa: BLE001
                    pass
            except ConnectionLost:
                attempt["outcome"] = "disconnected"
            finally:
                conn.pending.pop(job["job_id"], None)
                conn.inflight_requests -= n
                conn.inflight_units -= units_reserved
            attempt["latency_ms"] = round((loop.time() - t0) * 1000, 1)
            rec, now = self.p.host(conn.host_id), self.p.clock()
            if result is not None:
                problems = self.check_result(conn, job, jhash, result)
                if problems:
                    attempt.update(outcome="bad_result", problems=problems)
                    rec.bad_results += 1
                    rec.event(now, "bad_result", job_id=job["job_id"], problems=problems)
                    self.p.job_outcome(rec, now, "bad_result")
                    attempts.append(attempt)
                    continue
                if result["status"] == "completed":
                    attempt["outcome"] = "completed"
                    attempts.append(attempt)
                    return await self._deliver(conn, rec, job, result, attempts, units_reserved, started)
                attempt.update(outcome=result["status"], reason=result.get("reason"))
            rec.jobs_failed += 1
            rec.event(now, "job_failed", job_id=job["job_id"], outcome=attempt["outcome"], reason=attempt.get("reason"),
                      latency_ms=attempt["latency_ms"])
            self.p.job_outcome(rec, now, attempt["outcome"], attempt["latency_ms"])
            attempts.append(attempt)
        reason = "no live host with capacity for this job" if not attempts else f"{len(attempts)} attempt(s) failed"
        out = {"job_id": base_id, "status": "failed", "reason": reason, "attempts": attempts,
               "units_reserved": round(units_reserved, 6), "latency_ms": round((loop.time() - started) * 1000, 1)}
        self._remember(out)
        return out

    async def _deliver(self, conn: HostConnection, rec: HostRecord, job: dict, result: dict, attempts: List[dict],
                       units_reserved: float, started: float) -> dict:
        usage = result["usage"]
        units = units_for(usage["prompt_tokens"], usage["completion_tokens"])
        rec.jobs_completed += 1
        rec.delivered_units += units
        # Counted, not logged one by one: a busy host completes thousands a day (§5's counters).
        self.p.job_outcome(rec, self.p.clock(), "completed", attempts[-1]["latency_ms"], units)
        keep = ("index", "text", "token_ids", "finish_reason", "stop_reason", "ttft_ms", "total_ms", "logprobs")
        out = {"job_id": job["job_id"].rsplit(".", 1)[0], "status": "completed", "host_id": conn.host_id,
               "attempts": attempts, "units": round(units, 6), "units_reserved": round(units_reserved, 6),
               "usage": usage, "latency_ms": round((asyncio.get_running_loop().time() - started) * 1000, 1),
               "result_sha256": result["result_sha256"],
               "outputs": [{k: o[k] for k in keep if k in o} for o in result["outputs"]]}
        # Verification happens after delivery and the host is never told which outputs are
        # checked (§7). Inline here for visibility; asynchronous on the real platform.
        if self.scorer is not None and self.p.s.verify_fraction > 0 and self.rng.random() < self.p.s.verify_fraction:
            pairs = [(i, r["prompt_token_ids"], result["outputs"][i]["token_ids"])
                     for i, r in enumerate(job["requests"]) if r["temperature"] == 0.0][: self.p.s.verify_max_requests]
            if pairs:
                v = await verify_greedy_outputs(self.scorer, pairs, self.p.s.verify_tau)
                out["verification"] = v
                if v["pass"] is False:
                    rec.verifications_failed += 1
                rec.event(self.p.clock(), "verification", job_id=job["job_id"], passed=v["pass"], checked=v["checked"])
            else:
                out["verification"] = {"checked": 0, "pass": None,
                                       "note": "no greedy requests; sampled outputs need a statistical test (step 3)"}
        self._remember(out)
        return out

    def _remember(self, out: dict) -> None:
        self.recent.append({k: out.get(k) for k in ("job_id", "status", "host_id", "units", "latency_ms", "reason")}
                           | {"attempts": [a.get("outcome") for a in out.get("attempts", [])],
                              "verified": (out.get("verification") or {}).get("pass")})
        del self.recent[:-200]

    def hosts_view(self) -> List[dict]:
        out = []
        for hid, rec in self.p.hosts.items():
            self.p._sweep(rec, self.p.clock())
            conn = self.conns.get(hid)
            out.append({**rec.summary(), "jobs_channel": conn is not None and not conn.closed,
                        "inflight_requests": conn.inflight_requests if conn else 0})
        return out


# --- tokenization (platform-owned, §7) ----------------------------------------------

class HFTokenizer:
    """The reference model's tokenizer and chat template, via transformers (installed
    wherever vLLM is). Only the dev endpoint uses it; hosts never tokenize."""

    def __init__(self, model_id: str = ref.MODEL_ID, revision: Optional[str] = None):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        # The chat template needs jinja2, which transformers does not install (vLLM does): find
        # out now, so chat requests get a clear refusal instead of a 500 mid-job.
        self.chat_error: Optional[str] = None
        try:
            self.tok.apply_chat_template([{"role": "user", "content": "hi"}], add_generation_prompt=True, tokenize=False)
        except ImportError as e:
            self.chat_error = str(e)

    def encode(self, text: str) -> List[int]:
        return list(self.tok.encode(text, add_special_tokens=True))

    def chat(self, messages: List[dict]) -> List[int]:
        if self.chat_error:
            raise JobInvalid(f"chat messages need the chat template, which is unavailable here: {self.chat_error}")
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if isinstance(ids, dict) or hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return [int(t) for t in ids]


def tokenize_requests(requests: List[dict], tokenizer) -> List[dict]:
    out = []
    for i, r in enumerate(requests):
        r = dict(r)
        if "messages" in r or "prompt" in r:
            if tokenizer is None:
                raise JobInvalid(f"request {i}: this mock platform has no tokenizer; send prompt_token_ids")
            if "messages" in r:
                r["prompt_token_ids"] = tokenizer.chat(r.pop("messages"))
                r.pop("prompt", None)
            else:
                r["prompt_token_ids"] = tokenizer.encode(r.pop("prompt"))
        out.append(r)
    return out


# --- FastAPI wrapper -------------------------------------------------------------

def create_app(platform: MockPlatform, router: Optional[Router] = None, tokenizer: Any = None) -> FastAPI:
    app = FastAPI(title="kWh mock platform", version="0")
    router = router or Router(platform)
    app.state.platform, app.state.router = platform, router

    async def _auth(request: Request, host_id: Optional[str]) -> tuple[str, bytes, dict]:
        body = await request.body()
        expected = None
        if host_id is not None:
            rec = platform.host(host_id)
            expected = rec.public_key
            if request.headers.get("authorization", "") != f"Bearer {rec.token}":
                raise HTTPException(401, "bad or missing bearer token")
        pub = verify_request(dict(request.headers), body, method=request.method, path=request.url.path,
                             expected_public_key=expected, now=platform.clock())
        if pub is None:
            raise HTTPException(401, "bad request signature")
        return pub, body, (json.loads(body) if body else {})

    @app.exception_handler(Rejected)
    async def _rejected(_, exc: Rejected):
        return JSONResponse(status_code=exc.status, content={"detail": exc.detail})

    @app.exception_handler(JobInvalid)
    async def _invalid(_, exc: JobInvalid):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.post("/v1/hosts")
    async def register(request: Request):
        pub, _, body = await _auth(request, None)
        return platform.register(pub, body)

    @app.post("/v1/hosts/{host_id}/heartbeat")
    async def heartbeat(host_id: str, request: Request):
        _, _, body = await _auth(request, host_id)
        return platform.heartbeat(host_id, body)

    @app.post("/v1/hosts/{host_id}/liveness")
    async def liveness(host_id: str, request: Request):
        _, _, body = await _auth(request, host_id)
        return platform.liveness(host_id, body)

    @app.post("/v1/hosts/{host_id}/microbench")
    async def microbench(host_id: str, request: Request):
        _, _, body = await _auth(request, host_id)
        return platform.microbench(host_id, body)

    @app.post("/v1/hosts/{host_id}/reports")
    async def reports(host_id: str, request: Request):
        _, _, body = await _auth(request, host_id)
        return platform.upload_report(host_id, body)

    @app.get("/v1/hosts/{host_id}")
    async def status(host_id: str, request: Request):
        await _auth(request, host_id)
        return platform.status(host_id)

    @app.get("/v1/hosts/{host_id}/events")
    async def events(host_id: str, request: Request, since: Optional[float] = None, limit: int = 200,
                     kinds: Optional[str] = None):
        # Signed like every GET: over the path, without the query, which only narrows what the
        # host may read anyway.
        await _auth(request, host_id)
        return platform.events(host_id, since=since, limit=limit,
                               kinds=[k for k in (kinds or "").split(",") if k] or None)

    @app.websocket("/v1/hosts/{host_id}/jobs")
    async def jobs(websocket: WebSocket, host_id: str):
        rec = platform.hosts.get(host_id)
        ok = (rec is not None
              and websocket.headers.get("authorization", "") == f"Bearer {rec.token}"
              and verify_request(dict(websocket.headers), b"", method="GET", path=websocket.url.path,
                                 expected_public_key=rec.public_key, now=platform.clock()) is not None)
        if not ok:
            await websocket.close(code=1008)          # before accept: the client sees HTTP 403
            return
        await websocket.accept()
        try:
            hello = await asyncio.wait_for(websocket.receive_json(), timeout=15)
        except Exception:  # noqa: BLE001
            await websocket.close(code=1002)
            return
        if not isinstance(hello, dict) or hello.get("type") != "hello":
            await websocket.close(code=1002)
            return
        conn = HostConnection(host_id, websocket.send_text, hello)
        router.attach(conn)
        try:
            while True:
                msg = await websocket.receive_json()
                if isinstance(msg, dict) and msg.get("type") == "result":
                    router.on_result(conn, msg.get("result") or {})
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001 - a malformed frame ends this connection, not the platform
            try:
                await websocket.close(code=1003)
            except Exception:  # noqa: BLE001
                pass
        finally:
            router.detach(conn)

    # -- dev / buyer stand-in (not part of the host contract) ---------------------
    @app.post("/v1/mock/jobs")
    async def mock_jobs(request: Request):
        body = await request.json()
        try:
            reqs = tokenize_requests(body.get("requests") or [], tokenizer)
        except JobInvalid as e:
            return JSONResponse({"job_id": None, "status": "failed", "reason": str(e), "attempts": []}, status_code=400)
        out = await router.submit(reqs, body.get("timeout_s"))
        return JSONResponse(out, status_code=200 if out["status"] == "completed" else 503)

    @app.get("/v1/mock/hosts")
    async def mock_hosts():
        return {"hosts": router.hosts_view(), "recent_jobs": router.recent[-20:]}

    @app.post("/v1/mock/hosts/{host_id}/rebench")
    async def mock_rebench(host_id: str):
        """Hold a host for a re-benchmark now, as if its report had expired (tests and GPU runs)."""
        rec = platform.host(host_id)
        platform._require_rebench(rec, platform.clock(), ["re-benchmark requested on the mock platform"])
        return rec.summary()

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "hosts": len(platform.hosts), "job_channels": len(router.conns),
                "tokenizer": tokenizer is not None,
                "chat": tokenizer is not None and not getattr(tokenizer, "chat_error", None),
                "verifier": router.scorer is not None}

    return app
