"""The `kwh-host run` loop (HOST-CLIENT.md §2, §4, §5, §6, §7): keep the engine up, heartbeat,
answer challenges, micro-benchmark when idle, serve jobs over the platform's WebSocket, and
re-benchmark when the report stops describing the rig. The platform's verdict is mirrored in
state.json, and what happens is logged in events.jsonl (§5's raw events, as the host saw them).

The engine is started and stopped inside the run: for a re-benchmark (§5: every 7 days, and on a
driver, image or GPU change, or two micro-benchmark misses) and when it stops answering. The
heartbeat carries on throughout, so the platform sees a host re-benchmarking or restarting its
engine rather than one that vanished.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
import time
from typing import Awaitable, Callable, Dict, List, Optional

import httpx
from kwh_bench import reference as ref
from kwh_bench.engines import VLLMEngine
from kwh_bench.engines.base import Engine
from kwh_bench.load import PreparedPrompt

from . import __version__, bench, gpu
from .config import HostConfig, read_state, write_state
from .engine import check_version, describe, launch_mode, own_pids, running_image
from .events import EventLog, NullEventLog
from .jobs import Executor, MockExecutor, VLLMExecutor, execute_job, reject_job
from .platform.client import PlatformClient, PlatformError
from .rebench import MICROBENCH_REASON, REBENCH_EVERY_SECONDS, local_reasons

Log = Callable[[str], None]

UNHEALTHY_BEATS_BEFORE_RESTART = 3     # 90 s of an engine that does not answer /health
LATENCIES_KEPT = 1000


async def describe_engine(engine: Engine, cfg: HostConfig) -> dict:
    if isinstance(engine, VLLMEngine):
        return await describe(engine, cfg)
    info = await engine.info()          # mock and other in-process engines (tests)
    return {"healthy": bool(getattr(engine, "healthy", True)), "version": info.version, "served_model": info.model_id,
            "launch_mode": info.launch_mode, "image": None, "port": None}


async def engine_context_len(engine: Engine) -> int:
    """The engine's max_model_len as it reports it (vLLM's /v1/models), else the spec value."""
    if isinstance(engine, VLLMEngine):
        try:
            async with engine.http_client(timeout=10.0) as c:
                r = await c.get("/v1/models")
                card = (r.json().get("data") or [{}])[0]
                if card.get("max_model_len"):
                    return int(card["max_model_len"])
        except (httpx.HTTPError, OSError, ValueError):
            pass
    return ref.MAX_MODEL_LEN


def default_executor(engine: Engine, served_model: Optional[str]) -> Executor:
    if isinstance(engine, VLLMEngine):
        return VLLMExecutor(engine.base_url, served_model or ref.MODEL_ID, transport=engine.make_transport())
    return MockExecutor()


class Telemetry:
    """What the host saw since the last heartbeat the platform received (§5): heartbeats it could
    not send, engine restarts, results it could not deliver, its jobs and their latency. Sent
    with every heartbeat; what reached the platform is subtracted, the rest goes with the next."""

    KEYS = ("send_failures", "engine_restarts", "results_undelivered", "jobs_completed", "jobs_failed", "jobs_rejected")

    def __init__(self):
        self.counts: Dict[str, int] = {k: 0 for k in self.KEYS}
        self.latencies: List[float] = []
        self.last_error: Optional[str] = None

    def add(self, key: str, n: int = 1) -> None:
        self.counts[key] += n

    def job(self, status: str, latency_ms: float) -> None:
        self.counts["jobs_" + status] = self.counts.get("jobs_" + status, 0) + 1
        if status == "completed" and len(self.latencies) < LATENCIES_KEPT:
            self.latencies.append(latency_ms)

    def snapshot(self, uptime_s: float) -> dict:
        lat = sorted(self.latencies)
        return {**self.counts, "last_error": self.last_error, "uptime_s": round(uptime_s),
                "job_latency_n": len(lat), "job_latency_ms_p50": lat[(len(lat) - 1) // 2] if lat else None,
                "job_latency_ms_max": lat[-1] if lat else None}

    def delivered(self, snap: dict) -> None:
        """The platform has `snap`: keep only what happened since it was taken."""
        for k in self.KEYS:
            self.counts[k] = max(0, self.counts[k] - int(snap.get(k) or 0))
        del self.latencies[: int(snap.get("job_latency_n") or 0)]
        if self.last_error == snap.get("last_error"):
            self.last_error = None


class Daemon:
    def __init__(self, cfg: HostConfig, client: PlatformClient, engine: Engine, log: Log = lambda s: print(s, file=sys.stderr),
                 sampler: Callable[..., dict] = gpu.sample, sleep: Callable[[float], Awaitable] = asyncio.sleep,
                 clock: Callable[[], float] = time.monotonic, max_beats: Optional[int] = None,
                 require_lock_version: bool = True, jobs: bool = True,
                 executor_factory: Optional[Callable[[Engine, Optional[str]], Executor]] = None,
                 max_concurrency: int = ref.CONCURRENCY,
                 bench_engine_factory: Optional[Callable[[], Engine]] = None,
                 events: Optional[EventLog] = None, wall_clock: Callable[[], float] = time.time,
                 rebench: bool = True, rebench_runs: int = ref.DEFAULT_MEASURED_JOBS,
                 drain_seconds: float = 120.0, retry_seconds: float = 1800.0, settle_seconds: float = 5.0):
        self.cfg, self.client, self.engine, self.log = cfg, client, engine, log
        self.sampler, self.sleep, self.clock, self.max_beats = sampler, sleep, clock, max_beats
        self.wall = wall_clock
        self.require_lock_version = require_lock_version
        self.jobs_enabled = jobs
        self.executor_factory = executor_factory or default_executor
        self.max_concurrency = max_concurrency
        # A re-benchmark runs on a fresh engine of its own (its own log); without a factory the
        # serving engine object is started again for it, after it has been stopped.
        self.bench_engine_factory = bench_engine_factory or (lambda: self.engine)
        self.events = events or NullEventLog()
        self.rebench_enabled, self.rebench_runs = rebench, rebench_runs
        self.drain_seconds, self.retry_seconds, self.settle_seconds = drain_seconds, retry_seconds, settle_seconds
        self.platform_config: dict = {"heartbeat_seconds": 30.0, "challenge_every_seconds": 300.0,
                                      "microbench_every_seconds": 1800.0, "rebench_every_seconds": REBENCH_EVERY_SECONDS}
        try:
            # What the platform said last time (its re-benchmark interval, its cadence), so the
            # check before the engine first starts uses its numbers rather than the defaults.
            last = (read_state(cfg) or {}).get("platform_config") or {}
            self.platform_config.update({k: v for k, v in last.items() if isinstance(v, (int, float))})
        except (OSError, ValueError):
            pass
        self.report: Optional[dict] = None
        if cfg.report_path.exists():
            try:
                self.report = bench.load_report(cfg.report_path)
            except (OSError, ValueError):
                self.report = None
        self.state: dict = {"client_version": __version__, "pid": os.getpid(), "started_at": self.wall(),
                            "state": "starting", "reasons": [], "state_since": None, "beats": 0, "accepted_beats": 0,
                            "send_failures": 0, "accrual": 0.0, "balance": 0, "minted_total": 0,
                            "last_challenge": None, "last_microbench": None, "rebench_required": False,
                            "rebench_reasons": [], "benchmarking": False, "last_rebench": None,
                            "engine": None, "engine_restarts": 0, "gpu": None, "errors": 0, "jobs_channel": "closed",
                            "jobs": {"completed": 0, "failed": 0, "rejected": 0, "undelivered": 0,
                                     "completion_tokens": 0},
                            "report": self._report_info(), "platform_config": self.platform_config}
        self.tel = Telemetry()
        self.in_flight = 0
        self.executor: Optional[Executor] = None
        self.max_model_len = ref.MAX_MODEL_LEN
        self._engine_desc: dict = {}
        self._prepared: list[PreparedPrompt] = []
        self._last_micro_at: Optional[float] = None
        self._jobs_started = 0
        self._job_tasks: Dict[str, asyncio.Task] = {}
        self._sem = asyncio.Semaphore(max_concurrency)
        self._send_lock = asyncio.Lock()
        self._beat_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._leave = asyncio.Event()          # the serving engine has to stop: re-benchmark or restart
        self._leave_why: Optional[str] = None  # "rebench" | "restart"
        self._hb_task: Optional[asyncio.Task] = None
        self._t0 = self.clock()
        self._engine_up = False
        self._engine_starts = 0
        self._instance = secrets.token_hex(8)       # this engine start; the platform re-challenges on a new one
        self._benchmarking = False
        self._rebench_reasons: List[str] = []
        self._retry_at: Optional[float] = None
        self._unhealthy = 0
        self._restarts_in_a_row = 0

    def stop(self) -> None:
        self._stop.set()

    def _persist(self) -> None:
        self.state["updated_at"] = self.wall()
        self.state["in_flight"] = self.in_flight
        try:
            write_state(self.cfg, self.state)
        except OSError:
            pass

    def _report_info(self) -> Optional[dict]:
        r = self.report
        if not r:
            return None
        return {"units_per_hour": (r.get("score") or {}).get("units_per_hour"), "finished_at": r.get("finished_at"),
                "report_sha256": r.get("report_sha256"), "certified": r.get("certified")}

    def _image(self) -> Optional[str]:
        return running_image(self.cfg)

    # -- one heartbeat cycle --------------------------------------------
    async def beat(self) -> dict:
        async with self._beat_lock:
            return await self._beat()

    async def _send_heartbeat(self, eng: dict, gs: dict) -> dict:
        snap = self.tel.snapshot(self.clock() - self._t0)
        try:
            resp = await self.client.heartbeat(eng, gs, in_flight=self.in_flight, benchmarking=self._benchmarking,
                                               telemetry=snap)
        except (httpx.HTTPError, PlatformError) as e:
            if isinstance(e, httpx.HTTPError) or e.status >= 500:
                # The platform never saw it (or could not take it): the next heartbeat says so.
                self.tel.add("send_failures")
                self.tel.last_error = f"{type(e).__name__}: {e}"[:200]
                self.state["send_failures"] += 1
                self.events.add("heartbeat_failed", error=self.tel.last_error)
            raise
        self.tel.delivered(snap)
        return resp

    async def _beat(self) -> dict:
        up = self._engine_up
        if up:
            eng = {**await describe_engine(self.engine, self.cfg), "instance": self._instance}
            pids = own_pids(self.engine, self.cfg) if isinstance(self.engine, VLLMEngine) else set()
        else:
            # Stopped on purpose (re-benchmark) or starting again: nothing to ask it.
            eng = {"healthy": False, "version": None, "served_model": None, "image": self._image(),
                   "launch_mode": self._engine_desc.get("launch_mode") or launch_mode(self.cfg),
                   "instance": self._instance, "running": False}
            pids = None
        gs = self.sampler(self.cfg.gpu_index, pids)
        t0 = time.perf_counter()
        resp = await self._send_heartbeat(eng, gs)
        rtt_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.platform_config.update(resp.get("config") or {})
        before = (self.state["state"], self.state["reasons"])
        self.state.update({"state": resp["state"], "reasons": resp.get("reasons") or [],
                           "state_since": resp.get("state_since"), "accrual": resp.get("accrual"),
                           "balance": resp.get("balance"), "rebench_required": resp.get("rebench_required", False),
                           "rebench_reasons": resp.get("rebench_reasons") or [], "engine": eng, "gpu": gs})
        self.state["beats"] += 1
        accepted = bool(resp.get("accepted"))
        if accepted:
            self.state["accepted_beats"] += 1
        if resp.get("minted"):
            self.state["minted_total"] += resp["minted"]
            self.log(f"minted {resp['minted']} unit(s), balance {resp['balance']}")
        self.events.add("heartbeat", state=resp["state"], accepted=accepted, minted=resp.get("minted") or 0,
                        rtt_ms=rtt_ms, **({} if accepted else {"reasons": resp.get("reasons") or []}))
        self._note_state(before)

        if up and not self._benchmarking:
            if eng.get("healthy"):
                self._unhealthy = 0
                if accepted:
                    self._restarts_in_a_row = 0
            else:
                self._unhealthy += 1
                if self._unhealthy >= UNHEALTHY_BEATS_BEFORE_RESTART:
                    self._want_restart(f"engine unhealthy for {self._unhealthy} heartbeats")

        ch = resp.get("challenge")
        if ch and up:
            await self._answer(ch)

        self._check_rebench(resp, gs)

        every = float(self.platform_config.get("microbench_every_seconds") or 1800.0)
        due = self._last_micro_at is None or self.clock() - self._last_micro_at >= every
        if not up or self._benchmarking or self._leave.is_set():
            pass
        elif due and self.in_flight == 0 and self.state["state"] in ("live", "degraded") and self._prepared:
            await self._micro_benchmark(gs)
        elif due and self.in_flight:
            self._last_micro_at = self.clock()      # skipped, not failed: busy hosts prove liveness by delivering
            self.events.add("microbench", skipped="jobs in flight")
        self._persist()
        return resp

    def _note_state(self, before: tuple) -> None:
        state, reasons = self.state["state"], self.state["reasons"]
        if state != before[0]:
            self.events.add("state", **{"from": before[0], "to": state, "reasons": reasons})
        if (state, reasons) != before and reasons:
            self.log(f"platform: {state} ({'; '.join(reasons)})")

    async def _answer(self, ch: dict) -> None:
        try:
            ans = await bench.score_challenge(self.engine, ch)
        except Exception as e:  # noqa: BLE001
            ans = {"challenge_id": ch["challenge_id"], "mean_logprobs": [None] * len(ch.get("items") or []),
                   "elapsed_ms": 0}
            self.log(f"challenge scoring failed: {type(e).__name__}: {e}")
        verdict = await self.client.liveness(**ans)
        before = (self.state["state"], self.state["reasons"])
        self.state["last_challenge"] = {**ans, **verdict, "t": self.wall()}
        self.state["state"] = verdict.get("state", self.state["state"])
        owed = verdict.get("passes_needed") or 0
        self.events.add("challenge", id=ch["challenge_id"], passed=verdict["pass"], delta=verdict["delta"],
                        deltas=verdict.get("deltas"), elapsed_ms=ans["elapsed_ms"], state=self.state["state"],
                        passes_needed=owed)
        if self.state["state"] != before[0]:
            self.events.add("state", **{"from": before[0], "to": self.state["state"], "reasons": []})
        self.log(f"challenge {ch['challenge_id']}: {'pass' if verdict['pass'] else 'FAIL'} "
                 f"(mean delta {verdict['delta']} over {len(ans['mean_logprobs'])}, {ans['elapsed_ms']} ms) "
                 f"-> {self.state['state']}" + (f", {owed} more pass(es) in a row needed" if owed else ""))

    async def _micro_benchmark(self, gs: dict) -> None:
        started_before = self._jobs_started
        mb = await bench.micro_benchmark(self.engine, self._prepared)
        self._last_micro_at = self.clock()
        if self._jobs_started != started_before:
            # A job shared the GPU with it: the number says nothing about the rig. Try again later.
            self.log("micro-benchmark discarded: a job arrived while it ran")
            self.events.add("microbench", discarded="a job arrived while it ran", units_per_hour=mb["units_per_hour"])
            self._last_micro_at = None
            return
        verdict = await self.client.microbench(mb["units_per_hour"], mb["job_seconds"], gs)
        self.state["last_microbench"] = {**mb, **verdict, "t": self.wall()}
        self.state["rebench_required"] = verdict.get("rebench_required", False)
        self.events.add("microbench", units_per_hour=mb["units_per_hour"], job_seconds=mb["job_seconds"],
                        within=verdict.get("within_tolerance"), rebench_required=self.state["rebench_required"])
        self.log(f"micro-benchmark: {mb['units_per_hour']:.2f} u/h equivalent in {mb['job_seconds']:.2f}s "
                 f"-> {'within' if verdict['within_tolerance'] else 'OUTSIDE'} tolerance")
        if self.state["rebench_required"]:
            self._check_rebench({"rebench_reasons": [MICROBENCH_REASON]}, gs)

    # -- re-benchmark and restart (§5) -------------------------------------
    def _check_rebench(self, resp: dict, gs: dict) -> None:
        reasons = list(resp.get("rebench_reasons") or ([] if not resp.get("rebench_required") else
                                                       ["required by the platform"]))
        if self.report:
            every = float(self.platform_config.get("rebench_every_seconds") or REBENCH_EVERY_SECONDS)
            reasons += [r for r in local_reasons(self.report, gs or {}, self._image(), self.wall(), every)
                        if r not in reasons]
        if reasons:
            self._want_rebench(reasons)

    def _want_rebench(self, reasons: List[str]) -> None:
        if not self.rebench_enabled or self._benchmarking or self._leave.is_set():
            return
        if self._retry_at is not None and self.clock() < self._retry_at:
            return
        self._rebench_reasons = list(reasons)
        self._benchmarking = True          # from here on new jobs are turned away and heartbeats say so
        self.state["benchmarking"] = True
        self._leave_why = "rebench"
        self._leave.set()

    def _want_restart(self, why: str) -> None:
        if self._leave.is_set():
            return
        self.log(f"{why}; restarting it")
        self.events.add("engine_restart", reason=why)
        self._leave_why = "restart"
        self._leave.set()

    async def _drain(self) -> None:
        """Before the engine stops for a re-benchmark: tell the platform now (it stops routing here),
        and let the jobs in flight finish, for up to `drain_seconds`."""
        try:
            await self.beat()
        except (PlatformError, httpx.HTTPError) as e:
            self.log(f"platform not told of the re-benchmark yet ({e}); it will be on the next heartbeat")
        deadline = self.clock() + self.drain_seconds
        while self.in_flight and self.clock() < deadline and not self._stop.is_set():
            await self.sleep(1.0)
            await asyncio.sleep(0)
        if self.in_flight:
            self.log(f"re-benchmark: {self.in_flight} job(s) still running after {self.drain_seconds:.0f}s; "
                     "cancelling them (the platform re-routes)")

    async def _rebenchmark(self) -> None:
        """A full certified benchmark on a fresh engine, then the new report to the platform. On any
        failure the host serves on as before (the platform decides whether that is degraded) and
        tries again after `retry_seconds`."""
        reasons = self._rebench_reasons or ["re-benchmark"]
        self._benchmarking = True
        self.state["benchmarking"] = True
        self._persist()
        self._start_heartbeats()
        self.log("re-benchmarking: " + "; ".join(reasons))
        self.events.add("rebench", phase="start", reasons=reasons)
        new = self.cfg.report_path.with_name("report.new.json")
        t0 = self.clock()
        previous = (self._report_info() or {}).get("units_per_hour")
        outcome: dict = {"reasons": reasons, "started_at": self.wall()}
        try:
            await self.sleep(self.settle_seconds)          # let the serving engine's VRAM go before the pre-flight
            report = await bench.full_benchmark(self.bench_engine_factory(), self.client.identity, new,
                                                runs=self.rebench_runs, log=self.log)
            out = None
            for attempt in range(4):
                try:
                    out = await self.client.upload_report(report)
                    break
                except PlatformError:
                    # Refused: keep it for a look, never send it again.
                    os.replace(new, self.cfg.report_path.with_name("report-rejected.json"))
                    raise
                except httpx.HTTPError as e:
                    # Unreachable for a moment: a good benchmark is worth a few tries.
                    if attempt == 3:
                        raise
                    self.log(f"report upload failed ({type(e).__name__}); trying again in 15 s")
                    await self.sleep(15.0)
            os.replace(new, self.cfg.report_path)
            self.report = report
            self.state["report"] = self._report_info()
            rate = out.get("rate_units_per_hour")
            outcome.update(ok=True, units_per_hour=rate, previous_units_per_hour=previous,
                           seconds=round(self.clock() - t0, 1))
            self.events.add("rebench", phase="done", units_per_hour=rate, previous_units_per_hour=previous,
                            seconds=outcome["seconds"])
            self.log(f"re-benchmark accepted: {rate} units/hour (was {previous})")
            self._retry_at = None
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - a failed re-benchmark must not take the host down
            err = f"{type(e).__name__}: {e}"[:500]
            self._retry_at = self.clock() + self.retry_seconds
            outcome.update(ok=False, error=err, seconds=round(self.clock() - t0, 1))
            self.events.add("rebench", phase="failed", error=err)
            self.log(f"re-benchmark failed: {err}; serving on the current report, next try in "
                     f"{self.retry_seconds / 60:.0f} min")
        self.state["last_rebench"] = outcome
        self._rebench_reasons = []
        self._persist()
        # Still benchmarking, as far as the platform is concerned, until the engine serves again.

    def _restart_delay(self) -> float:
        self._restarts_in_a_row += 1
        return min(30.0 * 2 ** (self._restarts_in_a_row - 1), 600.0)

    # -- heartbeat loop -------------------------------------------------------
    def _start_heartbeats(self) -> None:
        if self._hb_task is None:
            self._hb_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        backoff = 1.0
        try:
            while not self._stop.is_set():
                try:
                    await self.beat()
                    backoff = 1.0
                except PlatformError as e:
                    self.state["errors"] += 1
                    if e.status in (401, 404):
                        self.log(f"platform rejected us ({e.detail}); re-register with `kwh-host register`")
                        self.state["state"] = "rejected"
                        self.events.add("rejected", detail=e.detail)
                        self._persist()
                        return
                    self.log(f"platform error: {e.detail}")
                except httpx.HTTPError as e:
                    self.state["errors"] += 1
                    self.log(f"platform unreachable: {type(e).__name__}: {e}; retrying in {backoff:.0f}s")
                    self._persist()
                    await self.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)
                    continue
                if self.max_beats is not None and self.state["beats"] >= self.max_beats:
                    return
                await self.sleep(float(self.platform_config.get("heartbeat_seconds") or 30.0))
        finally:
            self._stop.set()               # no heartbeats, no hosting

    # -- job channel (§7) -------------------------------------------------
    async def _job_loop(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with self.client.connect_jobs() as ws:
                    self.state["jobs_channel"] = "open"
                    backoff = 1.0
                    await ws.send(json.dumps({"type": "hello", "client_version": __version__,
                                              "max_concurrency": self.max_concurrency,
                                              "max_model_len": self.max_model_len, "in_flight": self.in_flight}))
                    self.log(f"job channel open (concurrency {self.max_concurrency}, context {self.max_model_len})")
                    self.events.add("jobs_channel", open=True, context=self.max_model_len)
                    async for raw in ws:
                        msg = json.loads(raw)
                        kind = msg.get("type") if isinstance(msg, dict) else None
                        if kind == "job":
                            self._spawn_job(ws, msg.get("job") or {})
                        elif kind == "cancel":
                            task = self._job_tasks.get(str(msg.get("job_id")))
                            if task:
                                task.cancel()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - handshake refused, network, platform restart
                self.log(f"job channel: {type(e).__name__}: {str(e)[:200]}")
                self.events.add("jobs_channel", open=False, error=f"{type(e).__name__}: {str(e)[:200]}")
            finally:
                self.state["jobs_channel"] = "closed"
                # The platform re-routes whatever was in flight; finishing it would be wasted work.
                for task in list(self._job_tasks.values()):
                    task.cancel()
            if self._stop.is_set():
                return
            await self.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    def _spawn_job(self, ws, job: dict) -> None:
        job_id = str(job.get("job_id"))
        if self._benchmarking or not self._engine_up:
            task = asyncio.create_task(self._turn_away(ws, job, "host is re-benchmarking; not taking jobs"
                                                       if self._benchmarking else "engine not running"))
        else:
            task = asyncio.create_task(self._run_job(ws, job))
        self._job_tasks[job_id] = task
        task.add_done_callback(lambda _t, jid=job_id: self._job_tasks.pop(jid, None))

    async def _turn_away(self, ws, job: dict, reason: str) -> None:
        result = reject_job(job, reason, identity=self.client.identity, host_id=self.client.host_id,
                            engine=self._engine_desc)
        await self._finish(ws, job, result, 0.0)

    async def _run_job(self, ws, job: dict) -> None:
        self.in_flight += 1
        self._jobs_started += 1
        t0 = time.perf_counter()
        try:
            result = await execute_job(job, self.executor, self._sem, identity=self.client.identity,
                                       host_id=self.client.host_id, engine=self._engine_desc,
                                       max_model_len=self.max_model_len)
        finally:
            self.in_flight -= 1
        await self._finish(ws, job, result, round((time.perf_counter() - t0) * 1000, 1))

    async def _finish(self, ws, job: dict, result: dict, latency_ms: float) -> None:
        status = result["status"]
        self.state["jobs"][status] = self.state["jobs"].get(status, 0) + 1
        self.state["jobs"]["completion_tokens"] += result["usage"]["completion_tokens"]
        self.tel.job(status, latency_ms)
        n = len(job.get("requests") or [])
        self.events.add("job", job_id=result["job_id"], status=status, requests=n,
                        completion_tokens=result["usage"]["completion_tokens"], latency_ms=latency_ms,
                        **({"reason": result["reason"]} if result.get("reason") else {}))
        self.log(f"job {result['job_id']}: {status} ({n} request(s), {result['usage']['completion_tokens']} tokens)"
                 + (f" - {result['reason']}" if result.get("reason") else ""))
        try:
            async with self._send_lock:
                await ws.send(json.dumps({"type": "result", "result": result}))
        except Exception as e:  # noqa: BLE001
            self.state["jobs"]["undelivered"] += 1
            self.tel.add("results_undelivered")
            self.events.add("job_undelivered", job_id=result["job_id"], error=type(e).__name__)
            self.log(f"job {result['job_id']}: result not delivered ({type(e).__name__})")

    # -- serving ------------------------------------------------------------
    async def _serve(self) -> None:
        """Start the engine and serve until stopped, or until the engine has to go (re-benchmark,
        restart). The heartbeat loop is not tied to it and runs on."""
        self._leave.clear()
        self._leave_why = None
        self._instance = secrets.token_hex(8)
        self._unhealthy = 0
        async with self.engine:
            eng = await describe_engine(self.engine, self.cfg)
            if self.require_lock_version:
                why = check_version(eng.get("version"))
                if why:
                    raise RuntimeError(f"refusing to serve: {why}")
            self._engine_starts += 1
            if self._engine_starts > 1:
                self.tel.add("engine_restarts")
                self.state["engine_restarts"] += 1
            self.log(f"engine {eng.get('version')} serving {eng.get('served_model')} ({eng['launch_mode']})")
            self._engine_desc = {k: eng.get(k) for k in ("version", "served_model", "launch_mode")}
            self._prepared = await bench.prepare_micro_prompts(self.engine)
            self.max_model_len = await engine_context_len(self.engine)
            self.executor = self.executor_factory(self.engine, eng.get("served_model"))
            self.events.add("engine_up", instance=self._instance, version=eng.get("version"),
                            served_model=eng.get("served_model"), image=eng.get("image"), context=self.max_model_len)
            self._engine_up = True
            if not self._leave.is_set():       # a re-benchmark decided while it started keeps its flag
                self._benchmarking = False
                self.state["benchmarking"] = False
            if self.state["state"] == "starting":
                self.state["state"] = "registered"
            self._persist()
            self._start_heartbeats()
            waits = [asyncio.create_task(self._stop.wait()), asyncio.create_task(self._leave.wait())]
            job_task = asyncio.create_task(self._job_loop()) if self.jobs_enabled else None
            try:
                await asyncio.wait(waits + ([job_task] if job_task else []), return_when=asyncio.FIRST_COMPLETED)
                if self._leave_why == "rebench" and not self._stop.is_set():
                    await self._drain()
            finally:
                self._engine_up = False
                tasks = waits + ([job_task] if job_task else [])
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                for t in list(self._job_tasks.values()):
                    t.cancel()
                await asyncio.gather(*self._job_tasks.values(), return_exceptions=True)
                if hasattr(self.executor, "aclose"):
                    await self.executor.aclose()
                self.events.add("engine_down", instance=self._instance,
                                reason="stop" if self._stop.is_set() else (self._leave_why or "unknown"))

    async def _until_stopped(self, coro: Awaitable) -> None:
        task = asyncio.ensure_future(coro)
        stop = asyncio.ensure_future(self._stop.wait())
        try:
            await asyncio.wait([task, stop], return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in (task, stop):
                if not t.done():
                    t.cancel()
            await asyncio.gather(task, stop, return_exceptions=True)
        if not task.cancelled() and task.exception() is not None:
            raise task.exception()

    # -- main loop ---------------------------------------------------------
    async def run(self) -> dict:
        self.log(f"kwh-host {__version__} starting engine ({self.cfg.engine_mode})")
        self.events.add("start", version=__version__, engine_mode=self.cfg.engine_mode, image=self._image(),
                        host_id=self.client.host_id)
        error: Optional[BaseException] = None
        try:
            # A report that no longer describes the rig is redone before the engine first starts.
            self._check_rebench({}, self.sampler(self.cfg.gpu_index, None))
            while not self._stop.is_set():
                if self._leave_why == "rebench":
                    await self._until_stopped(self._rebenchmark())
                elif self._leave_why == "restart":
                    await self._until_stopped(self.sleep(self._restart_delay()))
                if self._stop.is_set():
                    break
                await self._serve()
        except BaseException as e:  # noqa: BLE001
            error = e
        finally:
            self._stop.set()
            if self._hb_task is not None:
                if not self._hb_task.done():
                    self._hb_task.cancel()
                res = (await asyncio.gather(self._hb_task, return_exceptions=True))[0]
                if error is None and isinstance(res, BaseException) and not isinstance(res, asyncio.CancelledError):
                    error = res
        if error is not None:
            self.events.add("stop", error=f"{type(error).__name__}: {error}"[:500])
            raise error
        self.state["state"] = "stopped" if self.state["state"] != "rejected" else "rejected"
        self.events.add("stop", state=self.state["state"])
        self._persist()
        return self.state
