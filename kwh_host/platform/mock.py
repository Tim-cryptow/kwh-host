"""Mock platform: the server side of HOST-CLIENT.md §8, in memory.

It exists so the daemon can be built and tested end to end before step 4. The logic
(`MockPlatform`) is framework-free and unit-tested; `create_app` wraps it in FastAPI.
What it enforces is the real thing: signed requests, verified reports, the lifecycle
state machine (§2), challenge canaries (§4), per-heartbeat accrual with integer mints
(§6). What it fakes: the challenge pool. A real platform scores fresh canaries on its
reference node; this one serves the public lock canaries, which is fine for testing
and useless as a guard (the answers are in the lock).
"""

from __future__ import annotations

import hashlib
import json
import random
import secrets
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from kwh_bench import reference as ref
from kwh_bench.lockfile import Lock, load_lock
from kwh_bench.prompts import canonical_prompts
from kwh_bench.verify import verify_report

from ..identity import verify_report_signature, verify_request

STATES = ("registered", "live", "degraded", "offline")


# --- challenges -----------------------------------------------------------

@dataclass
class Challenge:
    id: str
    prompt_text: str
    prompt_tokens: int
    continuation_token_ids: List[int]
    reference_mean_logprob: float          # server-side only

    def to_wire(self) -> dict:
        return {"challenge_id": self.id, "prompt_text": self.prompt_text, "prompt_tokens": self.prompt_tokens,
                "continuation_token_ids": list(self.continuation_token_ids)}


class ChallengePool:
    def __init__(self, challenges: List[Challenge], seed: int = 0):
        if not challenges:
            raise ValueError("empty challenge pool")
        self.challenges = list(challenges)
        self._rng = random.Random(seed)
        self._n = 0

    @classmethod
    def from_lock(cls, lock: Optional[Lock] = None) -> "ChallengePool":
        lock = lock or load_lock()
        if not lock.is_locked:
            raise RuntimeError("kwh-bench lock is incomplete; cannot build a challenge pool from it")
        text = {p.id: p.text for p in canonical_prompts()}
        return cls([Challenge(f"lock-{c.prompt_id}", text[c.prompt_id], ref.PROMPT_TOKENS,
                              c.expected_token_ids, c.reference_mean_logprob) for c in lock.canaries])

    def issue(self) -> Challenge:
        base = self._rng.choice(self.challenges)
        self._n += 1
        # Fresh id per issue so a host cannot replay an earlier answer.
        return Challenge(f"{base.id}-{self._n}-{secrets.token_hex(4)}", base.prompt_text, base.prompt_tokens,
                         base.continuation_token_ids, base.reference_mean_logprob)


# --- settings / records ---------------------------------------------------

@dataclass
class Settings:
    heartbeat_seconds: float = 30.0
    challenge_every_seconds: float = 300.0
    microbench_every_seconds: float = 1800.0
    offline_after_seconds: float = 90.0
    max_failures: int = 5
    tolerance: float = ref.CANARY_MAX_LOGPROB_DELTA
    microbench_tolerance: float = 0.10
    accept_uncertified: bool = False       # tests only: mock-engine reports
    allow_bare_metal: bool = False         # pod testing only (D4)
    require_engine_version: Optional[str] = None   # defaults to the lock's vLLM version

    def wire(self) -> dict:
        return {"heartbeat_seconds": self.heartbeat_seconds, "challenge_every_seconds": self.challenge_every_seconds,
                "microbench_every_seconds": self.microbench_every_seconds,
                "offline_after_seconds": self.offline_after_seconds, "max_failures": self.max_failures}


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
    microbench_misses: int = 0
    last_microbench: Optional[dict] = None
    rebench_required: bool = False
    events: List[dict] = field(default_factory=list)

    def event(self, now: float, kind: str, **data) -> None:
        self.events.append({"t": now, "kind": kind, **data})
        if len(self.events) > 500:
            del self.events[:-500]

    def summary(self) -> dict:
        return {"host_id": self.host_id, "state": self.state, "reasons": self.reasons, "rate_units_per_hour": self.rate,
                "bucket": self.bucket, "accrual": round(self.accrual, 4), "balance": self.balance,
                "failures": self.failures, "last_beat_at": self.last_beat_at, "last_challenge": self.last_challenge,
                "last_microbench": self.last_microbench, "rebench_required": self.rebench_required,
                "report_sha256": self.report_sha256}


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
        """Lazy offline detection: no accepted heartbeat within the timeout."""
        if rec.state in ("live", "degraded") and rec.last_accepted_beat_at is not None \
                and now - rec.last_accepted_beat_at > self.s.offline_after_seconds:
            self._set_state(rec, now, "offline", ["no accepted heartbeat within timeout"])

    def _set_state(self, rec: HostRecord, now: float, state: str, reasons: List[str]) -> None:
        if state == "offline":
            rec.accrual = 0.0          # never minted late (§6)
            rec.pending = None
        if state != rec.state:
            rec.event(now, "state", **{"from": rec.state, "to": state, "reasons": reasons})
        rec.state, rec.reasons = state, list(reasons)

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
        if rec is None:
            host_id = "h_" + hashlib.sha256(public_key.encode()).hexdigest()[:12]
            rec = HostRecord(host_id=host_id, public_key=public_key, token=secrets.token_urlsafe(24), rate=rate,
                             bucket=bucket_for(rate), report_sha256=report["report_sha256"],
                             launch_mode=report["engine"]["launch_mode"], registered_at=now)
            self.hosts[host_id] = rec
            rec.event(now, "registered", rate=rate)
        else:
            rec.token = secrets.token_urlsafe(24)
            rec.rate, rec.bucket, rec.report_sha256 = rate, bucket_for(rate), report["report_sha256"]
            rec.rebench_required = False
            rec.event(now, "re-registered", rate=rate)
        return {"host_id": rec.host_id, "token": rec.token, "rate_units_per_hour": rec.rate, "bucket": rec.bucket,
                "config": self.s.wire()}

    def heartbeat(self, host_id: str, body: dict) -> dict:
        now = self.clock()
        rec = self.host(host_id)
        self._sweep(rec, now)
        reasons: List[str] = []
        eng = body.get("engine") or {}
        if not eng.get("healthy"):
            reasons.append("engine unhealthy")
        if self.s.require_engine_version and eng.get("version") != self.s.require_engine_version:
            reasons.append(f"engine version {eng.get('version')!r} != lock {self.s.require_engine_version!r}")
        if eng.get("launch_mode") != "docker" and not self.s.allow_bare_metal:
            reasons.append("engine not sandboxed (D4)")
        gs = body.get("gpu_sample") or {}
        if gs.get("available") and (gs.get("foreign_processes") or 0) > 0:
            reasons.append(f"host_contention: {gs['foreign_processes']} foreign process(es) on the GPU")

        interval = 0.0
        if rec.last_accepted_beat_at is not None:
            interval = min(now - rec.last_accepted_beat_at, 2 * self.s.heartbeat_seconds)

        minted = 0
        challenge: Optional[Challenge] = None
        if not reasons:
            if rec.state == "live" and body.get("wants_mint", True) and interval > 0:
                rec.accrual += rec.rate * interval / 3600.0
                minted = int(rec.accrual)
                if minted:
                    rec.accrual -= minted
                    rec.balance += minted
                    rec.event(now, "mint", units=minted, balance=rec.balance)
            rec.last_accepted_beat_at = now
            due = rec.pending is None and (
                rec.state != "live" or rec.last_challenge_at is None
                or now - rec.last_challenge_at >= self.s.challenge_every_seconds)
            if due:
                challenge = self.pool.issue()
                rec.pending = challenge
                rec.event(now, "challenge", id=challenge.id)
            if rec.state == "degraded" and rec.reasons and not rec.pending:
                pass  # recovery needs a passed challenge; the one just issued decides
        else:
            rec.failures += 1
            if rec.state == "live":
                self._set_state(rec, now, "degraded", reasons)
            elif rec.state == "degraded":
                rec.reasons = reasons
            if rec.failures >= self.s.max_failures and rec.state != "offline":
                self._set_state(rec, now, "offline", reasons + [f"{rec.failures} consecutive failures"])
        rec.last_beat_at = now
        return {"state": rec.state, "reasons": reasons or rec.reasons, "accepted": not reasons,
                "challenge": challenge.to_wire() if challenge else None, "minted": minted,
                "accrual": round(rec.accrual, 4), "balance": rec.balance,
                "rebench_required": rec.rebench_required, "config": self.s.wire()}

    def liveness(self, host_id: str, body: dict) -> dict:
        now = self.clock()
        rec = self.host(host_id)
        ch = rec.pending
        if ch is None or body.get("challenge_id") != ch.id:
            raise Rejected(400, "no such pending challenge")
        mean = body.get("mean_logprob")
        delta = None if mean is None else round(abs(float(mean) - ch.reference_mean_logprob), 5)
        passed = delta is not None and delta <= self.s.tolerance
        rec.pending = None
        rec.last_challenge_at = now
        rec.last_challenge = {"id": ch.id, "pass": passed, "delta": delta, "elapsed_ms": body.get("elapsed_ms"), "t": now}
        rec.event(now, "liveness", id=ch.id, passed=passed, delta=delta)
        if passed:
            rec.failures = 0
            healthy_now = rec.last_accepted_beat_at is not None and rec.last_accepted_beat_at == rec.last_beat_at
            if rec.state != "live" and healthy_now:
                self._set_state(rec, now, "live", [])
        else:
            rec.failures += 1
            if rec.failures >= self.s.max_failures:
                self._set_state(rec, now, "offline", [f"canary failed, {rec.failures} consecutive failures"])
            else:
                self._set_state(rec, now, "degraded", [f"canary failed (delta {delta})"])
        return {"pass": passed, "delta": delta, "state": rec.state}

    def microbench(self, host_id: str, body: dict) -> dict:
        now = self.clock()
        rec = self.host(host_id)
        uph = float(body.get("units_per_hour") or 0.0)
        within = rec.rate > 0 and abs(uph - rec.rate) / rec.rate <= self.s.microbench_tolerance
        rec.last_microbench = {"units_per_hour": uph, "job_seconds": body.get("job_seconds"), "within": within, "t": now}
        rec.event(now, "microbench", units_per_hour=uph, within=within)
        if within:
            rec.microbench_misses = 0
        else:
            rec.microbench_misses += 1
            if rec.microbench_misses >= 2:
                rec.rebench_required = True
                if rec.state == "live":
                    self._set_state(rec, now, "degraded", [f"micro-benchmark {uph:.2f} u/h outside 10% of {rec.rate:.2f}, twice"])
        return {"accepted": True, "within_tolerance": within, "rebench_required": rec.rebench_required}

    def upload_report(self, host_id: str, body: dict) -> dict:
        now = self.clock()
        rec = self.host(host_id)
        report = body.get("report") or {}
        problems = self._check_report(report, rec.public_key)
        if problems:
            raise Rejected(400, "report rejected: " + " | ".join(problems))
        rec.rate = float(report["score"]["units_per_hour"])
        rec.bucket, rec.report_sha256 = bucket_for(rec.rate), report["report_sha256"]
        rec.rebench_required, rec.microbench_misses = False, 0
        rec.event(now, "re-benchmarked", rate=rec.rate)
        return {"rate_units_per_hour": rec.rate, "bucket": rec.bucket}

    def status(self, host_id: str) -> dict:
        rec = self.host(host_id)
        self._sweep(rec, self.clock())
        out = rec.summary()
        out["events"] = rec.events[-20:]
        return out


# --- FastAPI wrapper -------------------------------------------------------

def create_app(platform: MockPlatform) -> FastAPI:
    app = FastAPI(title="kWh mock platform", version="0")
    app.state.platform = platform

    async def _auth(request: Request, host_id: Optional[str]) -> tuple[str, bytes, dict]:
        body = await request.body()
        expected = None
        if host_id is not None:
            rec = platform.host(host_id)
            expected = rec.public_key
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {rec.token}":
                raise HTTPException(401, "bad or missing bearer token")
        pub = verify_request(dict(request.headers), body, expected_public_key=expected, now=platform.clock())
        if pub is None:
            raise HTTPException(401, "bad request signature")
        return pub, body, (json.loads(body) if body else {})

    @app.exception_handler(Rejected)
    async def _rejected(_, exc: Rejected):
        return JSONResponse(status_code=exc.status, content={"detail": exc.detail})

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

    @app.get("/v1/hosts/{host_id}/jobs")
    async def jobs(host_id: str):
        raise HTTPException(501, "job dispatch is milestone M2")

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "hosts": len(platform.hosts)}

    return app
