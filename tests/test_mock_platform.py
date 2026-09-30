"""The platform state machine, challenges and accrual (HOST-CLIENT.md §2, §4, §6), with a fake clock."""

import pytest

from kwh_host.platform.mock import Challenge, ChallengePool, MockPlatform, Rejected, Settings

REF = -1.2345


def pool():
    return ChallengePool([Challenge("c1", "word " * 800, 512, list(range(32)), REF)])


def settings(**kw):
    base = dict(heartbeat_seconds=30, challenge_every_seconds=300, microbench_every_seconds=1800,
                offline_after_seconds=90, max_failures=3, accept_uncertified=True, allow_bare_metal=True,
                require_engine_version=None)
    base.update(kw)
    return Settings(**base)


def healthy(version="0", mode="docker"):
    return {"healthy": True, "version": version, "launch_mode": mode, "served_model": "x"}


def idle_gpu(foreign=0):
    return {"available": True, "util_pct": 0.0, "power_w": 15.0, "foreign_processes": foreign}


def register(platform, identity, report):
    out = platform.register(identity.public_key_hex, {"public_key": identity.public_key_hex, "report": report, "client_version": "t"})
    return out["host_id"]


def answer(platform, host_id, resp, mean=REF):
    assert resp["challenge"], "expected a challenge"
    return platform.liveness(host_id, {"challenge_id": resp["challenge"]["challenge_id"], "mean_logprob": mean, "elapsed_ms": 5})


def test_register_requires_signed_verified_report(identity, report, clock):
    p = MockPlatform(pool(), settings(), clock=clock)
    hid = register(p, identity, report)
    assert hid.startswith("h_") and p.hosts[hid].rate == report["score"]["units_per_hour"]
    # tampered report: hash no longer matches content
    bad = dict(report); bad["score"] = dict(report["score"], units_per_hour=999.0)
    with pytest.raises(Rejected) as e:
        p.register(identity.public_key_hex, {"public_key": identity.public_key_hex, "report": bad})
    assert "report_sha256" in e.value.detail
    # uncertified refused when the platform is strict
    strict = MockPlatform(pool(), settings(accept_uncertified=False), clock=clock)
    with pytest.raises(Rejected) as e:
        register(strict, identity, report)
    assert "not certified" in e.value.detail
    # bare metal refused when the platform enforces D4
    d4 = MockPlatform(pool(), settings(allow_bare_metal=False), clock=clock)
    with pytest.raises(Rejected) as e:
        register(d4, identity, report)
    assert "Docker sandbox" in e.value.detail


def test_lifecycle_registered_to_live_and_accrual(identity, report, clock):
    p = MockPlatform(pool(), settings(), clock=clock)
    hid = register(p, identity, report)
    rate = p.hosts[hid].rate
    r1 = p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})
    assert r1["state"] == "registered" and r1["challenge"] is not None   # first beat: challenge issued, not live yet
    v = answer(p, hid, r1)
    assert v["pass"] and v["state"] == "live"
    # live beats accrue rate * interval; whole units mint
    clock.advance(30)
    r2 = p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})
    assert r2["state"] == "live" and r2["challenge"] is None
    expected = rate * 30 / 3600
    assert abs(r2["accrual"] + r2["minted"] - expected) < 1e-3
    total = r2["accrual"] + r2["minted"]
    for _ in range(int(3600 / 30)):
        clock.advance(30)
        r = p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})
        total += r["minted"]
    # one hour of live beats mints about `rate` units (remainder carries as accrual)
    assert abs(p.hosts[hid].balance + p.hosts[hid].accrual - expected - rate) < 0.05


def test_gap_is_capped_and_offline_discards_accrual(identity, report, clock):
    p = MockPlatform(pool(), settings(), clock=clock)
    hid = register(p, identity, report)
    answer(p, hid, p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()}))
    clock.advance(30)
    p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})
    rec = p.hosts[hid]
    assert rec.state == "live" and rec.accrual > 0
    # a 5-minute silence: offline on the next contact, accrual discarded, and the gap never accrues
    clock.advance(300)
    r = p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})
    assert rec.state == "offline" and rec.accrual == 0 and r["minted"] == 0 and r["challenge"] is not None
    v = answer(p, hid, r)
    assert v["state"] == "live"


def test_contention_and_bad_engine_degrade_then_offline(identity, report, clock):
    p = MockPlatform(pool(), settings(max_failures=3), clock=clock)
    hid = register(p, identity, report)
    answer(p, hid, p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()}))
    clock.advance(30)
    r = p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu(foreign=1)})
    assert r["state"] == "degraded" and r["minted"] == 0 and "host_contention" in r["reasons"][0]
    clock.advance(30)
    r = p.heartbeat(hid, {"engine": {"healthy": False, "version": "0", "launch_mode": "docker"}, "gpu_sample": idle_gpu()})
    assert r["state"] == "degraded" and "engine unhealthy" in r["reasons"]
    clock.advance(30)
    r = p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu(foreign=2)})
    assert r["state"] == "offline"
    # recovery: clean beat + passed challenge
    clock.advance(30)
    r = p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})
    assert r["challenge"] is not None
    assert answer(p, hid, r)["state"] == "live"


def test_failed_challenge_degrades_and_wrong_model_never_goes_live(identity, report, clock):
    p = MockPlatform(pool(), settings(), clock=clock)
    hid = register(p, identity, report)
    r = p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})
    v = answer(p, hid, r, mean=REF - 0.2)        # a 4-bit substitute: far outside 0.05
    assert not v["pass"] and v["state"] == "degraded" and v["delta"] == 0.2
    clock.advance(30)
    r = p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})
    assert r["state"] == "degraded" and r["minted"] == 0 and r["challenge"] is not None
    assert not answer(p, hid, r, mean=None)["pass"]           # scoring failed counts as a failure
    # a replayed / unknown challenge id is refused
    with pytest.raises(Rejected):
        p.liveness(hid, {"challenge_id": "nope", "mean_logprob": REF})


def test_engine_version_enforced_when_strict(identity, report, clock):
    p = MockPlatform(pool(), settings(require_engine_version="0.30.0"), clock=clock)
    hid = register(p, identity, report)
    r = p.heartbeat(hid, {"engine": healthy(version="0.29.0"), "gpu_sample": idle_gpu()})
    assert not r["accepted"] and "engine version" in r["reasons"][0] and r["challenge"] is None


def test_microbench_two_misses_require_rebench(identity, report, clock):
    p = MockPlatform(pool(), settings(), clock=clock)
    hid = register(p, identity, report)
    answer(p, hid, p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()}))
    rate = p.hosts[hid].rate
    ok = p.microbench(hid, {"units_per_hour": rate * 0.95, "job_seconds": 1.0, "gpu_sample": idle_gpu()})
    assert ok["within_tolerance"] and not ok["rebench_required"]
    p.microbench(hid, {"units_per_hour": rate * 0.7, "job_seconds": 1.0, "gpu_sample": idle_gpu()})
    bad = p.microbench(hid, {"units_per_hour": rate * 0.7, "job_seconds": 1.0, "gpu_sample": idle_gpu()})
    assert bad["rebench_required"] and p.hosts[hid].state == "degraded"
    # a new accepted report clears it
    out = p.upload_report(hid, {"report": report})
    assert out["rate_units_per_hour"] == rate and not p.hosts[hid].rebench_required


def test_status_and_events(identity, report, clock):
    p = MockPlatform(pool(), settings(), clock=clock)
    hid = register(p, identity, report)
    answer(p, hid, p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()}))
    s = p.status(hid)
    assert s["state"] == "live" and [e["kind"] for e in s["events"]] == ["registered", "challenge", "liveness", "state"]
