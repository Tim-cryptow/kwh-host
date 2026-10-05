"""M4 in the daemon (HOST-CLIENT.md §5): re-benchmarking when the report stops counting, restarting an
engine that stops answering, reporting what the platform could not see, and logging all of it. The
daemon, a mock engine and the mock platform in process, on a fake clock."""

import asyncio
import json

import httpx
from kwh_bench.engines import MockEngine

from conftest import challenge_from, idle_gpu
from kwh_host.daemon import Daemon
from kwh_host.events import EventLog
from kwh_host.identity import verify_result_signature
from kwh_host.platform.client import PlatformClient
from kwh_host.platform.mock import ChallengePool, MockPlatform, Settings, create_app
from kwh_host.rebench import MICROBENCH_REASON

MAX_FAKE_SECONDS = 6 * 3600        # a test that has not finished by then is stuck


def fast_engine():
    return MockEngine(step_ms=0.01, prefill_ms_per_1k=0.02)


class FlakyEngine(MockEngine):
    """A mock engine that can stop answering /health; a fresh start answers again."""

    def __init__(self):
        super().__init__(step_ms=0.01, prefill_ms_per_1k=0.02)
        self.healthy, self.starts = True, 0

    async def start(self):
        self.starts += 1
        self.healthy = True
        await super().start()


async def platform_for(engine, **kw):
    s = dict(heartbeat_seconds=30, challenge_every_seconds=300, microbench_every_seconds=3600,
             microbench_tolerance=10.0, accept_uncertified=True, allow_bare_metal=True, require_engine_version=None,
             rebench_max_seconds=86400)
    s.update(kw)
    ref_node = fast_engine()            # the platform's reference node: the same model, its own engine
    pool = ChallengePool([await challenge_from(ref_node, prompt_id=i) for i in range(6)])
    return pool, Settings(**s)


async def run(cfg, identity, report, engine, platform, clock, until, transport=None, sampler=idle_gpu, **kw):
    """Register, then run the daemon until `until(daemon)` holds (checked between sleeps)."""
    app = create_app(platform)
    asgi = httpx.ASGITransport(app=app)
    async with PlatformClient("http://mock", identity, transport=asgi, clock=clock) as reg:
        await reg.register(report)
        cfg.host_id, cfg.token = reg.host_id, reg.token
        cfg.save()
    start = clock.t
    log = []
    holder = {}

    async def fake_sleep(s):
        # 1 ms of real time per fake second: the other loops run meanwhile, and a mock benchmark
        # (seconds of real CPU) passes as an hour or so of heartbeats, not a day.
        clock.advance(s)
        await asyncio.sleep(s / 1000)
        if until(holder["d"]) or clock.t - start > MAX_FAKE_SECONDS:
            holder["d"].stop()

    async with PlatformClient("http://mock", identity, token=cfg.token, host_id=cfg.host_id,
                              transport=transport or asgi, clock=clock) as client:
        d = Daemon(cfg, client, engine, log=log.append, sampler=sampler, sleep=fake_sleep, clock=clock,
                   wall_clock=clock, require_lock_version=False, jobs=False,
                   events=EventLog(cfg.events_path, clock=clock), **kw)
        holder["d"] = d
        final = await asyncio.wait_for(d.run(), timeout=120)
    assert clock.t - start <= MAX_FAKE_SECONDS, "the daemon never got there:\n" + "\n".join(log[-30:])
    return d, final, log


def platform_kinds(platform, host_id):
    return [e["kind"] for e in platform.hosts[host_id].events]


def live_after(platform, kind):
    """The host is live again after the platform logged `kind`."""
    def check(d):
        rec = platform.hosts.get(d.client.host_id)
        if rec is None or rec.state != "live":
            return False
        return any(e["kind"] == kind for e in rec.events)
    return check


async def test_rebenchmarks_when_the_platform_requires_it(cfg, identity, report, clock):
    """Two micro-benchmark misses: degraded, the engine stops, a fresh one benchmarks, the platform
    takes the new report, the engine comes back, a challenge, live."""
    engine = fast_engine()
    pool, s = await platform_for(engine, microbench_every_seconds=30, microbench_tolerance=0.0)
    platform = MockPlatform(pool, s, clock=clock)
    benched = []

    def bench_engine():
        platform.s.microbench_tolerance = 10.0          # the rig is fine again; the new rate will hold
        benched.append(fast_engine())
        return benched[-1]

    d, final, log = await run(cfg, identity, report, engine, platform, clock, live_after(platform, "rebench_ended"),
                              bench_engine_factory=bench_engine, rebench_runs=3)
    hid = d.client.host_id
    kinds = platform_kinds(platform, hid)
    order = [kinds.index(k) for k in ("rebench_required", "rebench_started", "re-benchmarked", "rebench_ended")]
    assert order == sorted(order) and platform.hosts[hid].state == "live"
    assert len(benched) == 1 and final["state"] == "stopped" and final["engine_restarts"] == 1
    assert final["last_rebench"]["ok"] and final["last_rebench"]["reasons"] == [MICROBENCH_REASON]
    new_report = json.loads(cfg.report_path.read_text())          # report.json is the one the platform took
    assert new_report["report_sha256"] != report["report_sha256"]
    assert platform.hosts[hid].report_sha256 == new_report["report_sha256"]
    local = EventLog(cfg.events_path).read()
    seq = [(e["kind"], e.get("phase") or e.get("reason")) for e in local
           if e["kind"] in ("engine_up", "engine_down", "rebench")]
    assert seq[:5] == [("engine_up", None), ("engine_down", "rebench"), ("rebench", "start"), ("rebench", "done"),
                       ("engine_up", None)]
    # the host told the platform about its restarted engine
    day = platform.reliability(platform.hosts[hid], clock.t)["24h"]
    assert day["host_reported"]["engine_restarts"] == 1


async def test_a_report_that_no_longer_fits_is_redone_before_serving(cfg, identity, report, clock):
    """The GPU is not the one in the report: benchmark first, then start the serving engine once."""
    engine = fast_engine()
    pool, s = await platform_for(engine)
    platform = MockPlatform(pool, s, clock=clock)
    stale = {**report, "hardware": {**report["hardware"], "gpus": [{"uuid": "GPU-old", "driver_version": "570.1"}]}}
    cfg.report_path.write_text(json.dumps(stale))      # what the daemon reads; the platform has the original

    def sampler(*_):
        return {**idle_gpu(), "uuid": "GPU-new", "driver_version": "580.95.05"}

    d, final, log = await run(cfg, identity, report, engine, platform, clock, live_after(platform, "re-benchmarked"),
                              sampler=sampler, bench_engine_factory=fast_engine, rebench_runs=3)
    local = [(e["kind"], e.get("phase")) for e in EventLog(cfg.events_path).read()
             if e["kind"] in ("start", "engine_up", "engine_down", "rebench")]
    assert local[:4] == [("start", None), ("rebench", "start"), ("rebench", "done"), ("engine_up", None)]
    assert final["last_rebench"]["reasons"] == ["GPU changed: GPU-new is not the GPU benchmarked"]
    assert final["engine_restarts"] == 0                # the serving engine started once, after the benchmark
    assert "rebench_started" in platform_kinds(platform, d.client.host_id)


async def test_an_engine_that_stops_answering_is_restarted(cfg, identity, report, clock):
    engine = FlakyEngine()
    pool, s = await platform_for(engine)
    platform = MockPlatform(pool, s, clock=clock)
    beats = []

    def until(d):
        beats.append(d.state["beats"])
        if engine.starts == 1 and d.state["state"] == "live" and d.state["beats"] >= 3:
            engine.healthy = False                       # the engine hangs
        return engine.starts == 2 and platform.hosts[d.client.host_id].state == "live"

    d, final, log = await run(cfg, identity, report, engine, platform, clock, until)
    assert engine.starts == 2 and final["engine_restarts"] == 1
    assert any("engine unhealthy for 3 heartbeats; restarting it" in line for line in log)
    local = [e for e in EventLog(cfg.events_path).read() if e["kind"] in ("engine_restart", "engine_down", "engine_up")]
    assert [e["kind"] for e in local] == ["engine_up", "engine_restart", "engine_down", "engine_up", "engine_down"]
    assert local[2]["reason"] == "restart" and local[4]["reason"] == "stop"
    rec = platform.hosts[d.client.host_id]
    assert platform.reliability(rec, clock.t)["24h"]["host_reported"]["engine_restarts"] == 1
    assert "heartbeat_rejected" in platform_kinds(platform, d.client.host_id)     # the platform saw it unhealthy


async def test_a_failed_rebenchmark_keeps_serving_and_tries_again_later(cfg, identity, report, clock):
    engine = fast_engine()
    pool, s = await platform_for(engine, microbench_every_seconds=30, microbench_tolerance=0.0)
    platform = MockPlatform(pool, s, clock=clock)
    calls = []

    def bench_engine():
        calls.append(clock.t)
        if len(calls) == 1:
            raise RuntimeError("pre-flight: the GPU is busy")
        platform.s.microbench_tolerance = 10.0
        return fast_engine()

    d, final, log = await run(cfg, identity, report, engine, platform, clock, live_after(platform, "re-benchmarked"),
                              bench_engine_factory=bench_engine, rebench_runs=3, retry_seconds=1800)
    assert len(calls) == 2 and calls[1] - calls[0] >= 1800
    phases = [e.get("phase") for e in EventLog(cfg.events_path).read() if e["kind"] == "rebench"]
    assert phases == ["start", "failed", "start", "done"]
    assert final["last_rebench"]["ok"] and final["engine_restarts"] == 2
    # between the two, the old report still did not count: degraded, never live, nothing minted
    rec = platform.hosts[d.client.host_id]
    held = [e for e in rec.events if e["kind"] == "state" and calls[0] < e["t"] < calls[1]]
    assert all(e["to"] != "live" for e in held)


async def test_jobs_are_turned_away_while_rebenchmarking(cfg, identity):
    class WS:
        def __init__(self):
            self.sent = []

        async def send(self, m):
            self.sent.append(json.loads(m))

    client = PlatformClient("http://mock", identity, token="t", host_id="h_test")
    d = Daemon(cfg, client, fast_engine(), require_lock_version=False, events=EventLog(cfg.events_path))
    d._engine_up, d._benchmarking = True, True
    ws = WS()
    job = {"job_id": "j_1.1", "timeout_s": 5.0, "units_reserved": 0.01,
           "requests": [{"prompt_token_ids": [1, 2], "max_tokens": 4}]}
    d._spawn_job(ws, job)
    await asyncio.gather(*list(d._job_tasks.values()))
    res = ws.sent[0]["result"]
    assert res["status"] == "rejected" and res["reason"] == "host is re-benchmarking; not taking jobs"
    assert verify_result_signature(res) == identity.public_key_hex and res["job_id"] == "j_1.1"
    assert d.state["jobs"]["rejected"] == 1 and d.tel.counts["jobs_rejected"] == 1
    await client.aclose()


class Drops(httpx.AsyncBaseTransport):
    """The network eats the first `n` requests to paths ending in `suffix`."""

    def __init__(self, inner, suffix, n):
        self.inner, self.suffix, self.n = inner, suffix, n

    async def handle_async_request(self, request):
        if request.url.path.endswith(self.suffix) and self.n > 0:
            self.n -= 1
            raise httpx.ConnectError("connection refused", request=request)
        return await self.inner.handle_async_request(request)


async def test_heartbeats_that_never_arrived_are_reported_with_the_next_one(cfg, identity, report, clock):
    engine = fast_engine()
    pool, s = await platform_for(engine)
    platform = MockPlatform(pool, s, clock=clock)
    transport = Drops(httpx.ASGITransport(app=create_app(platform)), "/heartbeat", 2)
    d, final, log = await run(cfg, identity, report, engine, platform, clock,
                              lambda d: d.state["accepted_beats"] >= 3, transport=transport)
    rec = platform.hosts[d.client.host_id]
    reported = [e for e in rec.events if e["kind"] == "host_reported"]
    assert len(reported) == 1 and reported[0]["send_failures"] == 2
    assert reported[0]["last_error"].startswith("ConnectError")
    assert final["send_failures"] == 2 and d.tel.counts["send_failures"] == 0          # delivered, then cleared
    assert [e["kind"] for e in EventLog(cfg.events_path).read()].count("heartbeat_failed") == 2


async def test_a_new_report_survives_a_moment_without_the_platform(cfg, identity, report, clock):
    """The benchmark is the expensive part: an upload that cannot get through is tried again, not redone."""
    engine = fast_engine()
    pool, s = await platform_for(engine, microbench_every_seconds=30, microbench_tolerance=0.0)
    platform = MockPlatform(pool, s, clock=clock)
    transport = Drops(httpx.ASGITransport(app=create_app(platform)), "/reports", 2)
    benched = []

    def bench_engine():
        platform.s.microbench_tolerance = 10.0
        benched.append(fast_engine())
        return benched[-1]

    d, final, log = await run(cfg, identity, report, engine, platform, clock, live_after(platform, "re-benchmarked"),
                              transport=transport, bench_engine_factory=bench_engine, rebench_runs=3)
    assert len(benched) == 1 and final["last_rebench"]["ok"]
    assert sum("report upload failed (ConnectError); trying again in 15 s" in line for line in log) == 2


MISMATCH = ("engine process exited early with code 125; the engine's log ends: : Error response from daemon: failed "
            "to create task for container: ... nvidia-container-cli: initialization error: nvml error: driver/library "
            "version mismatch: unknown")


class UpdatedDriver(MockEngine):
    """Cannot start while `driver['updated']` is set: an NVIDIA driver updated under the running machine, so
    nothing new gets the GPU until a reboot (a rented VM, 2026-10-05). The machine "reboots" after
    `driver['reboot_after']` failed starts of the serving engine."""

    def __init__(self, driver, serving=False):
        super().__init__(step_ms=0.01, prefill_ms_per_1k=0.02)
        self.driver, self.serving = driver, serving

    async def start(self):
        if self.driver["updated"]:
            if self.serving:
                self.driver["failed_starts"] += 1
                if self.driver["failed_starts"] >= self.driver["reboot_after"]:
                    self.driver["updated"] = False        # rebooted: the next start works
            raise RuntimeError(MISMATCH)
        await super().start()


async def test_an_engine_that_cannot_start_is_reported_and_retried_not_fatal(cfg, identity, report, clock):
    """The M4 GPU run, replayed: the driver is updated just as a re-benchmark starts. The benchmark
    fails, the serving engine cannot start either, and the daemon, which used to exit there, keeps
    heartbeating with the reason and retries until the machine is rebooted. Then the re-benchmark
    it still owes runs and the host is live again."""
    driver = {"updated": False, "failed_starts": 0, "reboot_after": 3}
    engine = UpdatedDriver(driver, serving=True)
    pool, s = await platform_for(engine, microbench_every_seconds=30, microbench_tolerance=0.0)
    platform = MockPlatform(pool, s, clock=clock)

    def bench_engine():
        if not driver["failed_starts"]:
            driver["updated"] = True                    # the update lands as the first re-benchmark starts
        else:
            platform.s.microbench_tolerance = 10.0      # the second try: the rig is fine again
        return UpdatedDriver(driver)

    d, final, log = await run(cfg, identity, report, engine, platform, clock, live_after(platform, "re-benchmarked"),
                              bench_engine_factory=bench_engine, rebench_runs=3, retry_seconds=1800)
    local = EventLog(cfg.events_path).read()
    seq = [e["kind"] + (f":{e['phase']}" if e["kind"] == "rebench" else "") for e in local
           if e["kind"] in ("engine_up", "engine_failed", "rebench", "stop")]
    assert seq[:3] == ["engine_up", "rebench:start", "rebench:failed"]
    assert seq.count("engine_failed") == 3 and seq[3:6] == ["engine_failed"] * 3
    assert seq[6:] == ["engine_up", "rebench:start", "rebench:done", "engine_up", "stop"]
    failed = next(e for e in local if e["kind"] == "engine_failed")
    assert "reboot to load the new one" in failed["error"]
    assert any("engine failed to start" in line and "trying again in 30 s" in line for line in log)
    # the platform heard why, instead of a host gone silent
    rejected = [e for e in platform.hosts[d.client.host_id].events if e["kind"] == "heartbeat_rejected"]
    assert rejected and rejected[0]["reasons"][0] == ("engine unhealthy: the NVIDIA driver was updated while the "
                                                      "machine was running; reboot to load the new one")
    assert final["state"] == "stopped" and final["engine_error"] is None and final["last_rebench"]["ok"]


async def test_a_version_the_lock_does_not_certify_still_stops_the_daemon(cfg, identity, report, clock):
    from kwh_host.daemon import EngineRefused
    engine = fast_engine()                               # reports version "0"; the lock says 0.30.0
    pool, s = await platform_for(engine)
    platform = MockPlatform(pool, s, clock=clock)
    async with PlatformClient("http://mock", identity, transport=httpx.ASGITransport(app=create_app(platform)),
                              clock=clock) as client:
        await client.register(report)
        d = Daemon(cfg, client, engine, log=lambda s: None, sampler=idle_gpu, clock=clock, wall_clock=clock,
                   require_lock_version=True, jobs=False)
        try:
            await asyncio.wait_for(d.run(), timeout=30)
            raise AssertionError("the daemon served an uncertified engine version")
        except EngineRefused as e:
            assert "refusing to serve" in str(e)
