"""The `kwh-host run` loop (HOST-CLIENT.md §2, §4, §6, §7): keep the engine up, heartbeat,
answer challenges, micro-benchmark when idle, and serve jobs over the platform's
WebSocket, mirroring the platform's verdict locally."""

from __future__ import annotations

import asyncio
import json
import secrets
import sys
import time
from typing import Callable, Dict, Optional

import httpx
from kwh_bench import reference as ref
from kwh_bench.engines import VLLMEngine
from kwh_bench.engines.base import Engine
from kwh_bench.load import PreparedPrompt

from . import __version__, bench, gpu
from .config import HostConfig, write_state
from .engine import check_version, describe, own_pids
from .jobs import Executor, MockExecutor, VLLMExecutor, execute_job
from .platform.client import PlatformClient, PlatformError

Log = Callable[[str], None]


async def describe_engine(engine: Engine, cfg: HostConfig) -> dict:
    if isinstance(engine, VLLMEngine):
        return await describe(engine, cfg)
    info = await engine.info()          # mock and other in-process engines (tests)
    return {"healthy": True, "version": info.version, "served_model": info.model_id,
            "launch_mode": info.launch_mode, "port": None}


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


class Daemon:
    def __init__(self, cfg: HostConfig, client: PlatformClient, engine: Engine, log: Log = lambda s: print(s, file=sys.stderr),
                 sampler: Callable[..., dict] = gpu.sample, sleep: Callable[[float], "asyncio.Future"] = asyncio.sleep,
                 clock: Callable[[], float] = time.monotonic, max_beats: Optional[int] = None,
                 require_lock_version: bool = True, jobs: bool = True,
                 executor_factory: Optional[Callable[[Engine, Optional[str]], Executor]] = None,
                 max_concurrency: int = ref.CONCURRENCY):
        self.cfg, self.client, self.engine, self.log = cfg, client, engine, log
        self.sampler, self.sleep, self.clock, self.max_beats = sampler, sleep, clock, max_beats
        self.require_lock_version = require_lock_version
        self.jobs_enabled = jobs
        self.executor_factory = executor_factory or default_executor
        self.max_concurrency = max_concurrency
        self.platform_config: dict = {"heartbeat_seconds": 30.0, "challenge_every_seconds": 300.0,
                                      "microbench_every_seconds": 1800.0}
        self.state: dict = {"client_version": __version__, "state": "starting", "reasons": [], "beats": 0,
                            "accepted_beats": 0, "accrual": 0.0, "balance": 0, "minted_total": 0,
                            "last_challenge": None, "last_microbench": None, "rebench_required": False,
                            "engine": None, "gpu": None, "errors": 0, "jobs_channel": "closed",
                            "jobs": {"completed": 0, "failed": 0, "rejected": 0, "undelivered": 0,
                                     "completion_tokens": 0}}
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
        self._stop = asyncio.Event()
        self._instance = secrets.token_hex(8)       # this engine start; the platform re-challenges on a new one

    def stop(self) -> None:
        self._stop.set()

    def _persist(self) -> None:
        self.state["updated_at"] = time.time()
        self.state["in_flight"] = self.in_flight
        try:
            write_state(self.cfg, self.state)
        except OSError:
            pass

    # -- one heartbeat cycle --------------------------------------------
    async def beat(self) -> dict:
        eng = {**await describe_engine(self.engine, self.cfg), "instance": self._instance}
        pids = own_pids(self.engine, self.cfg) if isinstance(self.engine, VLLMEngine) else set()
        gs = self.sampler(self.cfg.gpu_index, pids)
        resp = await self.client.heartbeat(eng, gs, in_flight=self.in_flight)
        self.platform_config.update(resp.get("config") or {})
        self.state.update({"state": resp["state"], "reasons": resp.get("reasons") or [], "accrual": resp.get("accrual"),
                           "balance": resp.get("balance"), "rebench_required": resp.get("rebench_required", False),
                           "engine": eng, "gpu": gs})
        self.state["beats"] += 1
        if resp.get("accepted"):
            self.state["accepted_beats"] += 1
        if resp.get("minted"):
            self.state["minted_total"] += resp["minted"]
            self.log(f"minted {resp['minted']} unit(s), balance {resp['balance']}")
        if resp.get("reasons"):
            self.log(f"platform: {resp['state']} ({'; '.join(resp['reasons'])})")

        ch = resp.get("challenge")
        if ch:
            try:
                ans = await bench.score_challenge(self.engine, ch)
            except Exception as e:  # noqa: BLE001
                ans = {"challenge_id": ch["challenge_id"], "mean_logprobs": [None] * len(ch.get("items") or []),
                       "elapsed_ms": 0}
                self.log(f"challenge scoring failed: {type(e).__name__}: {e}")
            verdict = await self.client.liveness(**ans)
            self.state["last_challenge"] = {**ans, **verdict}
            self.state["state"] = verdict.get("state", self.state["state"])
            owed = verdict.get("passes_needed") or 0
            self.log(f"challenge {ch['challenge_id']}: {'pass' if verdict['pass'] else 'FAIL'} "
                     f"(mean delta {verdict['delta']} over {len(ans['mean_logprobs'])}, {ans['elapsed_ms']} ms) "
                     f"-> {self.state['state']}" + (f", {owed} more pass(es) in a row needed" if owed else ""))

        every = float(self.platform_config.get("microbench_every_seconds") or 1800.0)
        due = self._last_micro_at is None or self.clock() - self._last_micro_at >= every
        if due and self.in_flight == 0 and self.state["state"] in ("live", "degraded") and self._prepared:
            started_before = self._jobs_started
            mb = await bench.micro_benchmark(self.engine, self._prepared)
            self._last_micro_at = self.clock()
            if self._jobs_started != started_before:
                # A job shared the GPU with it: the number says nothing about the rig. Try again later.
                self.log("micro-benchmark discarded: a job arrived while it ran")
                self._last_micro_at = None
            else:
                verdict = await self.client.microbench(mb["units_per_hour"], mb["job_seconds"], gs)
                self.state["last_microbench"] = {**mb, **verdict}
                self.state["rebench_required"] = verdict.get("rebench_required", False)
                self.log(f"micro-benchmark: {mb['units_per_hour']:.2f} u/h equivalent in {mb['job_seconds']:.2f}s "
                         f"-> {'within' if verdict['within_tolerance'] else 'OUTSIDE'} tolerance")
        elif due and self.in_flight:
            self._last_micro_at = self.clock()      # skipped, not failed: busy hosts prove liveness by delivering
        self._persist()
        return resp

    async def _heartbeat_loop(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                await self.beat()
                backoff = 1.0
            except PlatformError as e:
                self.state["errors"] += 1
                if e.status in (401, 404):
                    self.log(f"platform rejected us ({e.detail}); re-register with `kwh-host register`")
                    self.state["state"] = "rejected"
                    self._persist()
                    return
                self.log(f"platform error: {e.detail}")
            except httpx.HTTPError as e:
                self.state["errors"] += 1
                self.log(f"platform unreachable: {type(e).__name__}: {e}; retrying in {backoff:.0f}s")
                await self.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            if self.max_beats is not None and self.state["beats"] >= self.max_beats:
                return
            await self.sleep(float(self.platform_config.get("heartbeat_seconds") or 30.0))

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
        task = asyncio.create_task(self._run_job(ws, job))
        self._job_tasks[job_id] = task
        task.add_done_callback(lambda _t, jid=job_id: self._job_tasks.pop(jid, None))

    async def _run_job(self, ws, job: dict) -> None:
        self.in_flight += 1
        self._jobs_started += 1
        try:
            result = await execute_job(job, self.executor, self._sem, identity=self.client.identity,
                                       host_id=self.client.host_id, engine=self._engine_desc,
                                       max_model_len=self.max_model_len)
        finally:
            self.in_flight -= 1
        status = result["status"]
        self.state["jobs"][status] = self.state["jobs"].get(status, 0) + 1
        self.state["jobs"]["completion_tokens"] += result["usage"]["completion_tokens"]
        n = len(job.get("requests") or [])
        self.log(f"job {result['job_id']}: {status} ({n} request(s), {result['usage']['completion_tokens']} tokens)"
                 + (f" - {result['reason']}" if result.get("reason") else ""))
        try:
            async with self._send_lock:
                await ws.send(json.dumps({"type": "result", "result": result}))
        except Exception as e:  # noqa: BLE001
            self.state["jobs"]["undelivered"] += 1
            self.log(f"job {result['job_id']}: result not delivered ({type(e).__name__})")

    # -- main loop ---------------------------------------------------------
    async def run(self) -> dict:
        self.log(f"kwh-host {__version__} starting engine ({self.cfg.engine_mode})")
        async with self.engine:
            eng = await describe_engine(self.engine, self.cfg)
            if self.require_lock_version:
                why = check_version(eng.get("version"))
                if why:
                    raise RuntimeError(f"refusing to serve: {why}")
            self.log(f"engine {eng.get('version')} serving {eng.get('served_model')} ({eng['launch_mode']})")
            self._engine_desc = {k: eng.get(k) for k in ("version", "served_model", "launch_mode")}
            self._prepared = await bench.prepare_micro_prompts(self.engine)
            self.max_model_len = await engine_context_len(self.engine)
            self.executor = self.executor_factory(self.engine, eng.get("served_model"))
            self.state["state"] = "registered"
            self._persist()
            tasks = [asyncio.create_task(self._heartbeat_loop()), asyncio.create_task(self._stop.wait())]
            if self.jobs_enabled:
                tasks.append(asyncio.create_task(self._job_loop()))
            try:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                self._stop.set()
                for t in tasks:
                    t.cancel()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for t in list(self._job_tasks.values()):
                    t.cancel()
                await asyncio.gather(*self._job_tasks.values(), return_exceptions=True)
                if hasattr(self.executor, "aclose"):
                    await self.executor.aclose()
            for r in results:
                if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError):
                    raise r
        self.state["state"] = "stopped" if self.state["state"] != "rejected" else "rejected"
        self._persist()
        return self.state
