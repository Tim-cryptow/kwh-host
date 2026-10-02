"""Job dispatch end to end (HOST-CLIENT.md §7): a real uvicorn server running the mock platform,
daemons with the benchmark's mock engine connecting over the WebSocket, and the router
delivering, re-routing, cancelling and refusing."""

import asyncio
import random
import time
from types import SimpleNamespace

import pytest
import uvicorn
from kwh_bench.engines import MockEngine

from conftest import challenge_from, idle_gpu, unsigned_mock_report
from kwh_host.config import HostConfig
from kwh_host.daemon import Daemon
from kwh_host.identity import Identity
from kwh_host.jobs import MockExecutor
from kwh_host.jobspec import units_for
from kwh_host.mockmodel import ToyLM
from kwh_host.platform.client import PlatformClient
from kwh_host.platform.mock import ChallengePool, MockPlatform, Router, Settings, create_app
from kwh_host.platform.verifier import ToyScorer


def settings(**kw):
    base = dict(heartbeat_seconds=0.05, challenge_every_seconds=0.5, microbench_every_seconds=3600,
                offline_after_seconds=30, accept_uncertified=True, allow_bare_metal=True,
                require_engine_version=None, job_timeout_seconds=10)
    base.update(kw)
    return Settings(**base)


@pytest.fixture
async def live(home):
    """Start the platform; `live.start_host(...)` adds a daemon to it."""
    ch = await challenge_from(MockEngine(step_ms=0.01, prefill_ms_per_1k=0.02))
    platform = MockPlatform(ChallengePool([ch]), settings())
    router = Router(platform, rng=random.Random(0))
    server = uvicorn.Server(uvicorn.Config(create_app(platform, router), host="127.0.0.1", port=0,
                                           log_level="warning", lifespan="off", ws="websockets-sansio"))
    serve = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    url = f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    hosts = []

    async def start_host(executor=None, wrong_model=False, max_concurrency=32, wait_live=True):
        ident = Identity.generate()
        report = ident.sign_report(await unsigned_mock_report())
        client = PlatformClient(url, ident)
        await client.register(report)
        engine = MockEngine(step_ms=0.01, prefill_ms_per_1k=0.02)
        if wrong_model:
            # a different model scores the platform's continuation differently
            orig = engine.score_continuation

            async def off(p, c):
                return [x - 0.5 for x in await orig(p, c)]
            engine.score_continuation = off
        log = []
        d = Daemon(HostConfig(platform_url=url, engine_mode="bare-metal"), client, engine, log=log.append,
                   sampler=idle_gpu, require_lock_version=False, max_concurrency=max_concurrency,
                   executor_factory=lambda e, m: executor or MockExecutor())
        task = asyncio.create_task(d.run())
        h = SimpleNamespace(id=client.host_id, daemon=d, task=task, client=client, log=log, ident=ident)
        hosts.append(h)
        if wait_live:
            await until(lambda: platform.hosts[h.id].state == "live" and h.id in router.conns)
        return h

    yield SimpleNamespace(platform=platform, router=router, url=url, start_host=start_host)
    for h in hosts:
        h.daemon.stop()
    await asyncio.gather(*(h.task for h in hosts), return_exceptions=True)
    for h in hosts:
        await h.client.aclose()
    server.should_exit = True
    await serve


async def until(cond, timeout=10.0):
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.02)


def greedy(prompt, n=16):
    return {"prompt_token_ids": list(prompt), "max_tokens": n, "temperature": 0.0}


async def test_job_runs_on_a_live_host_and_is_metered_and_verified(live):
    h = await live.start_host()
    live.router.scorer = ToyScorer(ToyLM())
    live.platform.s.verify_fraction = 1.0
    out = await live.router.submit([greedy([1, 2, 3], 20), greedy([50, 60], 8),
                                    {"prompt_token_ids": [4, 5], "max_tokens": 10, "temperature": 0.8, "seed": 7}])
    assert out["status"] == "completed" and out["host_id"] == h.id and [a["outcome"] for a in out["attempts"]] == ["completed"]
    model = ToyLM()
    assert out["outputs"][0]["token_ids"] == model.generate([1, 2, 3], 20)[0]
    assert out["outputs"][2]["token_ids"] == model.generate([4, 5], 10, temperature=0.8, seed=7)[0]
    usage = out["usage"]
    assert usage["prompt_tokens"] == 7 and out["units"] == pytest.approx(units_for(7, usage["completion_tokens"]), abs=1e-6)
    v = out["verification"]
    assert v["pass"] is True and v["checked"] == 2           # the two greedy requests; the sampled one is step 3's
    rec = live.platform.hosts[h.id]
    assert rec.jobs_completed == 1 and rec.delivered_units == pytest.approx(out["units"], abs=1e-6)
    await until(lambda: h.daemon.state["jobs"]["completed"] == 1)


async def test_a_failing_host_is_routed_around(live):
    bad = await live.start_host(executor=MockExecutor(fail=True))
    good = await live.start_host()
    live.router.rng.shuffle = lambda conns: conns.sort(key=lambda c: c.host_id != bad.id)   # try the bad one first
    out = await live.router.submit([greedy([7, 8, 9])])
    assert out["status"] == "completed" and out["host_id"] == good.id
    assert [(a["host_id"], a["outcome"]) for a in out["attempts"]] == [(bad.id, "failed"), (good.id, "completed")]
    assert live.platform.hosts[bad.id].jobs_failed == 1 and live.platform.hosts[good.id].jobs_completed == 1


async def test_a_wrong_model_never_gets_work(live):
    h = await live.start_host(wrong_model=True, wait_live=False)
    await until(lambda: live.platform.hosts[h.id].last_challenge is not None and h.id in live.router.conns)
    assert live.platform.hosts[h.id].state == "degraded"
    out = await live.router.submit([greedy([1])], timeout_s=3)
    assert out["status"] == "failed" and out["reason"] == "no live host with capacity for this job" and out["attempts"] == []


async def test_handshake_needs_the_right_key_and_token(live):
    from websockets.exceptions import InvalidStatus
    h = await live.start_host()
    imposter = PlatformClient(live.url, Identity.generate(), token=h.client.token, host_id=h.id)
    wrong_token = PlatformClient(live.url, h.ident, token="nope", host_id=h.id)
    for c in (imposter, wrong_token):
        with pytest.raises(InvalidStatus) as e:
            async with c.connect_jobs():
                pass
        assert e.value.response.status_code == 403
        await c.aclose()


async def test_disconnect_mid_job_fails_over_and_the_host_stops_working(live):
    h = await live.start_host(executor=MockExecutor(step_s=0.05))
    job = asyncio.create_task(live.router.submit([{**greedy([1, 2], 60), "ignore_eos": True}], timeout_s=8))
    await until(lambda: h.daemon.in_flight == 1)
    h.daemon.stop()
    out = await job
    assert out["status"] == "failed" and [a["outcome"] for a in out["attempts"]] == ["disconnected"]
    await asyncio.wait_for(h.task, 5)
    assert h.daemon.in_flight == 0 and not h.daemon._job_tasks


async def test_the_host_enforces_its_own_deadline(live):
    h = await live.start_host(executor=MockExecutor(step_s=0.1))
    out = await live.router.submit([{**greedy([3], 200), "ignore_eos": True}], timeout_s=2.0)
    # The envelope gives the host half a second less than the router waits, so a slow host
    # reports its own failure instead of going silent.
    assert out["status"] == "failed" and [a["outcome"] for a in out["attempts"]] == ["failed"]
    assert out["attempts"][0]["reason"].startswith("deadline")
    await until(lambda: h.daemon.in_flight == 0, timeout=3)
    assert live.platform.hosts[h.id].jobs_failed == 1


async def test_cancel_stops_the_work_on_the_host(live):
    h = await live.start_host(executor=MockExecutor(step_s=0.1))
    job = asyncio.create_task(live.router.submit([{**greedy([3], 200), "ignore_eos": True}], timeout_s=3.0))
    await until(lambda: h.daemon.in_flight == 1)
    conn = live.router.conns[h.id]
    await conn.send({"type": "cancel", "job_id": next(iter(conn.pending))})
    await until(lambda: h.daemon.in_flight == 0, timeout=2)   # stopped well before its 2.5 s deadline
    out = await job
    assert [a["outcome"] for a in out["attempts"]] == ["timeout"]   # a cancelled job sends nothing back


async def test_busy_hosts_and_oversized_jobs_are_not_offered_work(live):
    await live.start_host(executor=MockExecutor(step_s=0.05), max_concurrency=2)
    live.platform.s.queue_factor = 1.0
    first = asyncio.create_task(live.router.submit([{**greedy([1], 40), "ignore_eos": True}] * 2, timeout_s=8))
    await until(lambda: any(c.inflight_requests for c in live.router.conns.values()))
    busy = await live.router.submit([greedy([2])], timeout_s=2)
    assert busy["status"] == "failed" and busy["attempts"] == []      # 2 in flight + 1 > 1 x concurrency 2
    assert (await first)["status"] == "completed"
    too_long = await live.router.submit([greedy([1] * 1000, 100)], timeout_s=2)   # 1,100 > the host's 1,024
    assert too_long["status"] == "failed" and too_long["attempts"] == []
