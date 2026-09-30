"""The `kwh-host run` loop (HOST-CLIENT.md §2, §4, §6): keep the engine up, heartbeat, answer
challenges, micro-benchmark when idle, and mirror the platform's verdict locally."""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Callable, List, Optional

import httpx
from kwh_bench.engines import VLLMEngine
from kwh_bench.engines.base import Engine
from kwh_bench.load import PreparedPrompt

from . import __version__, bench, gpu
from .config import HostConfig, write_state
from .engine import check_version, describe, own_pids
from .platform.client import PlatformClient, PlatformError

Log = Callable[[str], None]


async def describe_engine(engine: Engine, cfg: HostConfig) -> dict:
    if isinstance(engine, VLLMEngine):
        return await describe(engine, cfg)
    info = await engine.info()          # mock and other in-process engines (tests)
    return {"healthy": True, "version": info.version, "served_model": info.model_id,
            "launch_mode": info.launch_mode, "port": None}


class Daemon:
    def __init__(self, cfg: HostConfig, client: PlatformClient, engine: Engine, log: Log = lambda s: print(s, file=sys.stderr),
                 sampler: Callable[..., dict] = gpu.sample, sleep: Callable[[float], "asyncio.Future"] = asyncio.sleep,
                 clock: Callable[[], float] = time.monotonic, max_beats: Optional[int] = None,
                 require_lock_version: bool = True):
        self.cfg, self.client, self.engine, self.log = cfg, client, engine, log
        self.sampler, self.sleep, self.clock, self.max_beats = sampler, sleep, clock, max_beats
        self.require_lock_version = require_lock_version
        self.platform_config: dict = {"heartbeat_seconds": 30.0, "challenge_every_seconds": 300.0,
                                      "microbench_every_seconds": 1800.0}
        self.state: dict = {"client_version": __version__, "state": "starting", "reasons": [], "beats": 0,
                            "accepted_beats": 0, "accrual": 0.0, "balance": 0, "minted_total": 0,
                            "last_challenge": None, "last_microbench": None, "rebench_required": False,
                            "engine": None, "gpu": None, "errors": 0}
        self.in_flight = 0
        self._prepared: List[PreparedPrompt] = []
        self._last_micro_at: Optional[float] = None
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    def _persist(self) -> None:
        self.state["updated_at"] = time.time()
        try:
            write_state(self.cfg, self.state)
        except OSError:
            pass

    # -- one heartbeat cycle --------------------------------------------
    async def beat(self) -> dict:
        eng = await describe_engine(self.engine, self.cfg)
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
                ans = {"challenge_id": ch["challenge_id"], "mean_logprob": None, "elapsed_ms": 0}
                self.log(f"challenge scoring failed: {type(e).__name__}: {e}")
            verdict = await self.client.liveness(**ans)
            self.state["last_challenge"] = {**ans, **verdict}
            self.state["state"] = verdict.get("state", self.state["state"])
            self.log(f"challenge {ch['challenge_id']}: {'pass' if verdict['pass'] else 'FAIL'} "
                     f"(delta {verdict['delta']}, {ans['elapsed_ms']} ms) -> {self.state['state']}")

        every = float(self.platform_config.get("microbench_every_seconds") or 1800.0)
        due = self._last_micro_at is None or self.clock() - self._last_micro_at >= every
        if due and self.in_flight == 0 and self.state["state"] in ("live", "degraded") and self._prepared:
            mb = await bench.micro_benchmark(self.engine, self._prepared)
            self._last_micro_at = self.clock()
            verdict = await self.client.microbench(mb["units_per_hour"], mb["job_seconds"], gs)
            self.state["last_microbench"] = {**mb, **verdict}
            self.state["rebench_required"] = verdict.get("rebench_required", False)
            self.log(f"micro-benchmark: {mb['units_per_hour']:.2f} u/h equivalent in {mb['job_seconds']:.2f}s "
                     f"-> {'within' if verdict['within_tolerance'] else 'OUTSIDE'} tolerance")
        elif due and self.in_flight:
            self._last_micro_at = self.clock()      # skipped, not failed: busy hosts prove liveness by delivering
        self._persist()
        return resp

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
            self._prepared = await bench.prepare_micro_prompts(self.engine)
            self.state["state"] = "registered"
            self._persist()
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
                        break
                    self.log(f"platform error: {e.detail}")
                except httpx.HTTPError as e:
                    self.state["errors"] += 1
                    self.log(f"platform unreachable: {type(e).__name__}: {e}; retrying in {backoff:.0f}s")
                    await self.sleep(backoff)
                    backoff = min(backoff * 2, 60.0)
                    continue
                if self.max_beats is not None and self.state["beats"] >= self.max_beats:
                    break
                await self.sleep(float(self.platform_config.get("heartbeat_seconds") or 30.0))
        self.state["state"] = "stopped" if self.state["state"] != "rejected" else "rejected"
        self._persist()
        return self.state
