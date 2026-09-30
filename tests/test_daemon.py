"""End to end: the daemon, a mock engine and the mock platform (over HTTP, in process) reach
`live`, answer challenges, mint, and micro-benchmark. This is M1's definition of done minus
the real GPU."""

import httpx
import pytest
from kwh_bench import reference as ref
from kwh_bench.prompts import canonical_prompts

from kwh_host.daemon import Daemon
from kwh_host.platform.client import PlatformClient, PlatformError
from kwh_host.platform.mock import Challenge, ChallengePool, MockPlatform, Settings, create_app


def idle_gpu(*_):
    return {"available": True, "util_pct": 0.0, "power_w": 15.0, "mem_used_mib": 1.0, "mem_total_mib": 24564.0,
            "compute_processes": 1, "foreign_processes": 0, "foreign_pids": []}


async def challenge_from(engine, prompt_id=0, continuation=None):
    """A fresh challenge whose reference value comes from the engine under test (the real platform's
    reference node does the same thing with the real model)."""
    prompt = canonical_prompts()[prompt_id]
    continuation = continuation or list(range(100, 100 + ref.CANARY_TOKENS))
    async with engine as e:
        ids = (await e.tokenize(prompt.text))[:ref.PROMPT_TOKENS]
        lps = await e.score_continuation(ids, continuation)
    return Challenge(f"fresh-{prompt_id}", prompt.text, ref.PROMPT_TOKENS, continuation, sum(lps) / len(lps))


async def test_daemon_reaches_live_mints_and_microbenches(cfg, identity, report, mock_engine, clock):
    ch = await challenge_from(mock_engine)
    settings = Settings(heartbeat_seconds=30, challenge_every_seconds=60, microbench_every_seconds=90,
                        accept_uncertified=True, allow_bare_metal=True, require_engine_version=None)
    platform = MockPlatform(ChallengePool([ch]), settings, clock=clock)
    app = create_app(platform)
    log = []
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)
        clock.advance(s)

    async with PlatformClient("http://mock", identity, transport=httpx.ASGITransport(app=app), clock=clock) as client:
        out = await client.register(report)
        assert out["rate_units_per_hour"] == report["score"]["units_per_hour"]
        cfg.host_id, cfg.token = client.host_id, client.token
        cfg.save()
        d = Daemon(cfg, client, mock_engine, log=log.append, sampler=idle_gpu, sleep=fake_sleep, clock=clock,
                   max_beats=8, require_lock_version=False)
        final = await d.run()

    rec = platform.hosts[client.host_id]
    assert final["state"] == "stopped" and final["beats"] == 8 and final["accepted_beats"] == 8
    assert final["last_challenge"]["pass"] and final["last_challenge"]["delta"] <= 1e-4
    assert rec.state == "live" and rec.balance + rec.accrual > 0 and final["minted_total"] == rec.balance
    assert final["last_microbench"] is not None and final["last_microbench"]["accepted"]
    assert final["last_microbench"]["generated_tokens"] == 32 * ref.GENERATED_TOKENS
    assert sleeps == [30.0] * 7                       # heartbeat cadence from the platform's config
    kinds = [e["kind"] for e in rec.events]
    assert kinds[:4] == ["registered", "challenge", "liveness", "state"] and "microbench" in kinds
    assert any("challenge" in line and "pass" in line for line in log)
    # local state mirrors the platform for `kwh-host status --local`
    from kwh_host.config import read_state
    assert read_state(cfg)["state"] == "stopped" and read_state(cfg)["balance"] == rec.balance


async def test_wrong_model_is_degraded_not_live(cfg, identity, report, mock_engine, clock):
    """A host serving something else scores the platform's continuation differently: never live, never mints."""
    ch = await challenge_from(mock_engine)
    ch.reference_mean_logprob += 0.5                  # what a different model would have scored
    platform = MockPlatform(ChallengePool([ch]), Settings(heartbeat_seconds=30, challenge_every_seconds=30,
                                                          accept_uncertified=True, allow_bare_metal=True,
                                                          require_engine_version=None, max_failures=3), clock=clock)
    app = create_app(platform)

    async def fake_sleep(s):
        clock.advance(s)

    async with PlatformClient("http://mock", identity, transport=httpx.ASGITransport(app=app), clock=clock) as client:
        await client.register(report)
        cfg.host_id, cfg.token = client.host_id, client.token
        d = Daemon(cfg, client, mock_engine, log=lambda s: None, sampler=idle_gpu, sleep=fake_sleep, clock=clock,
                   max_beats=4, require_lock_version=False)
        final = await d.run()
    rec = platform.hosts[client.host_id]
    assert final["last_challenge"]["pass"] is False and rec.balance == 0 and rec.accrual == 0
    assert rec.state in ("degraded", "offline") and "live" not in [e.get("to") for e in rec.events]


async def test_unsigned_or_wrong_token_is_refused(cfg, identity, report, mock_engine, clock):
    from kwh_host.identity import Identity
    platform = MockPlatform(ChallengePool([await challenge_from(mock_engine)]),
                            Settings(accept_uncertified=True, allow_bare_metal=True, require_engine_version=None), clock=clock)
    app = create_app(platform)
    transport = httpx.ASGITransport(app=app)
    async with PlatformClient("http://mock", identity, transport=transport, clock=clock) as client:
        await client.register(report)
        hid = client.host_id
    # another key with the right token
    async with PlatformClient("http://mock", Identity.generate(), token=platform.hosts[hid].token, host_id=hid,
                              transport=transport, clock=clock) as imposter:
        with pytest.raises(PlatformError) as e:
            await imposter.heartbeat({"healthy": True}, idle_gpu())
        assert e.value.status == 401
    # the right key with a wrong token
    async with PlatformClient("http://mock", identity, token="nope", host_id=hid, transport=transport, clock=clock) as c:
        with pytest.raises(PlatformError) as e:
            await c.status()
        assert e.value.status == 401
    # the right key and token: GET is signed over its (empty) body and succeeds
    async with PlatformClient("http://mock", identity, token=platform.hosts[hid].token, host_id=hid,
                              transport=transport, clock=clock) as c:
        s = await c.status()
        assert s["host_id"] == hid and s["state"] == "registered"
