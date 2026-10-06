"""The platform state machine, challenges and accrual (HOST-CLIENT.md §2, §4, §6), with a fake clock."""

import pytest

from kwh_host.platform.mock import ChallengePool, Continuation, MockPlatform, Rejected, Settings

REF = -1.2345


def pool(n=1):
    return ChallengePool([Continuation(f"c{i}", "word " * 800, 512, list(range(32)), REF) for i in range(n)])


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


def answer(platform, host_id, resp, mean=REF, means=None):
    """Answer the challenge in `resp`: every continuation scored `mean`, or the given list."""
    assert resp["challenge"], "expected a challenge"
    n = len(resp["challenge"]["items"])
    body = {"challenge_id": resp["challenge"]["challenge_id"], "mean_logprobs": means if means is not None else [mean] * n,
            "elapsed_ms": 5}
    return platform.liveness(host_id, body)


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
    assert not v["pass"] and v["state"] == "degraded" and v["delta"] == 0.2 and v["passes_needed"] == 2
    clock.advance(30)
    r = p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})
    assert r["state"] == "degraded" and r["minted"] == 0 and r["challenge"] is not None
    assert not answer(p, hid, r, mean=None)["pass"]           # scoring failed counts as a failure
    # a replayed / unknown challenge id is refused
    with pytest.raises(Rejected):
        p.liveness(hid, {"challenge_id": "nope", "mean_logprobs": [REF]})


def test_engine_version_enforced_when_strict(identity, report, clock):
    p = MockPlatform(pool(), settings(require_engine_version="0.30.0"), clock=clock)
    hid = register(p, identity, report)
    r = p.heartbeat(hid, {"engine": healthy(version="0.29.0"), "gpu_sample": idle_gpu()})
    assert not r["accepted"] and "engine version" in r["reasons"][0] and r["challenge"] is None
    # an engine that does not answer has an unknown version, not a wrong one: one reason, not two
    clock.advance(30)
    r = p.heartbeat(hid, {"engine": {"healthy": False, "version": None, "launch_mode": "docker"}, "gpu_sample": idle_gpu()})
    assert r["reasons"] == ["engine unhealthy"]


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


def test_engine_restart_requires_a_new_challenge(identity, report, clock):
    p = MockPlatform(pool(), settings(), clock=clock)
    hid = register(p, identity, report)
    answer(p, hid, p.heartbeat(hid, {"engine": {**healthy(), "instance": "a"}, "gpu_sample": idle_gpu()}))
    clock.advance(30)
    assert p.heartbeat(hid, {"engine": {**healthy(), "instance": "a"}, "gpu_sample": idle_gpu()})["state"] == "live"
    clock.advance(30)
    r = p.heartbeat(hid, {"engine": {**healthy(), "instance": "b"}, "gpu_sample": idle_gpu()})
    assert r["state"] == "degraded" and r["minted"] == 0 and r["challenge"] is not None   # nothing routed or minted until re-proven
    assert "engine restarted" in r["reasons"][0]
    assert answer(p, hid, r)["state"] == "live"


def test_challenge_carries_four_continuations_and_passes_on_their_mean(identity, report, clock):
    """D9: one noisy continuation does not decide; a substitute's mean does."""
    p = MockPlatform(pool(8), settings(), clock=clock)
    hid = register(p, identity, report)
    r = p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})
    items = r["challenge"]["items"]
    assert len(items) == 4 and "reference_mean_logprob" not in items[0]           # the reference stays server-side
    assert len({c.id for c in p.hosts[hid].pending.items}) == 4                    # four distinct continuations
    v = answer(p, hid, r, means=[REF, REF, REF, REF - 0.15])                        # one off by 0.15, mean 0.0375
    assert v["pass"] and v["delta"] == 0.0375 and v["deltas"] == [0.0, 0.0, 0.0, 0.15] and v["state"] == "live"
    clock.advance(300)
    r = p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})
    a40 = [0.02521, 0.04788, 0.01496, 0.00524]                                      # honest A40 canaries: pass
    assert answer(p, hid, r, means=[REF + d for d in a40])["pass"]
    clock.advance(300)
    r = p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})
    awq = [0.03582, 0.094, 0.05838, 0.16482]                                        # the 4-bit control: fail
    v = answer(p, hid, r, means=[REF - d for d in awq])
    assert not v["pass"] and v["delta"] == round(sum(awq) / 4, 5) and v["state"] == "degraded"


def test_unscored_or_malformed_answers_fail(identity, report, clock):
    p = MockPlatform(pool(8), settings(), clock=clock)
    hid = register(p, identity, report)
    r = p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})
    v = answer(p, hid, r, means=[REF, REF, None, REF])                              # one continuation not scored
    assert not v["pass"] and v["delta"] is None and v["deltas"][2] is None
    clock.advance(30)
    r = p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})
    assert not answer(p, hid, r, means=[REF])["pass"]                              # wrong length: nothing judged
    clock.advance(30)
    r = p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})
    assert not answer(p, hid, r, means=["x", REF, REF, REF])["pass"]


def test_after_a_failed_challenge_two_passes_in_a_row_restore_live(identity, report, clock):
    """D9: a substitute can get lucky once; recovery needs two passes in a row."""
    p = MockPlatform(pool(8), settings(), clock=clock)
    hid = register(p, identity, report)
    assert answer(p, hid, p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()}))["state"] == "live"
    clock.advance(300)
    r = p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})
    assert answer(p, hid, r, mean=REF + 0.2)["state"] == "degraded"                # failed: two passes owed
    clock.advance(30)
    r = p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})
    v = answer(p, hid, r)
    assert v["pass"] and v["state"] == "degraded" and v["passes_needed"] == 1      # one pass is not enough
    assert "1 more in a row" in p.hosts[hid].reasons[0]
    clock.advance(30)
    r = p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})
    assert r["challenge"] is not None and r["minted"] == 0                         # re-challenged on the next beat
    v = answer(p, hid, r, mean=REF + 0.2)                                           # fails again: back to two
    assert v["passes_needed"] == 2 and v["state"] == "degraded"
    for owed in (1, 0):
        clock.advance(30)
        v = answer(p, hid, p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()}))
        assert v["passes_needed"] == owed
    assert v["state"] == "live"
    # an engine restart is not a failed challenge: one pass restores live
    clock.advance(30)
    p.heartbeat(hid, {"engine": {**healthy(), "instance": "a"}, "gpu_sample": idle_gpu()})
    clock.advance(30)
    r = p.heartbeat(hid, {"engine": {**healthy(), "instance": "b"}, "gpu_sample": idle_gpu()})
    assert r["state"] == "degraded" and answer(p, hid, r)["state"] == "live"


class DryPool(ChallengePool):
    """A pool that can run out, as a platform's does between reference-model bursts."""

    def __init__(self, n):
        super().__init__([Continuation(f"c{i}", "word " * 800, 512, list(range(32)), REF) for i in range(4)])
        self.left = n

    def issue(self):
        if self.left <= 0:
            return None
        self.left -= 1
        return super().issue()


def test_empty_pool_means_no_challenge_not_an_error(identity, report, clock):
    pool_ = DryPool(1)
    p = MockPlatform(pool_, settings(), clock=clock)
    hid = register(p, identity, report)
    r1 = p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})
    assert answer(p, hid, r1)["state"] == "live"
    for _ in range(11):                                 # beating as a host does, until a challenge is due ...
        clock.advance(30)
        r2 = p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})
    assert r2["accepted"] and r2["state"] == "live" and r2["challenge"] is None   # ... and the pool is empty
    clock.advance(30)
    p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})
    assert [e["kind"] for e in p.hosts[hid].events].count("challenge_unavailable") == 1   # said once, not every beat
    # a new host cannot go live without a challenge, and goes live once the pool is refilled
    dry = DryPool(0)
    p2 = MockPlatform(dry, settings(), clock=clock)
    hid2 = register(p2, identity, report)
    r = p2.heartbeat(hid2, {"engine": healthy(), "gpu_sample": idle_gpu()})
    assert r["state"] == "registered" and r["challenge"] is None
    dry.left = 1
    clock.advance(30)
    r = p2.heartbeat(hid2, {"engine": healthy(), "gpu_sample": idle_gpu()})
    assert answer(p2, hid2, r)["state"] == "live"


def test_snapshot_restores_hosts_as_they_were(identity, report, clock):
    import json
    p = MockPlatform(pool(), settings(), clock=clock)
    hid = register(p, identity, report)
    r1 = p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})
    answer(p, hid, r1)
    for _ in range(12):
        clock.advance(30)
        p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})
    clock.advance(300)
    pending = p.heartbeat(hid, {"engine": healthy(), "gpu_sample": idle_gpu()})["challenge"]
    assert pending is not None                          # a challenge in flight across the restart
    data = json.loads(json.dumps(p.snapshot()))         # as it would come back from a database
    q = MockPlatform(pool(), settings(), clock=clock)
    assert q.restore(data) == 1
    a, b = p.hosts[hid], q.hosts[hid]
    assert b.to_dict() == a.to_dict()
    assert b.pending.id == a.pending.id and b.pending.items[0].reference_mean_logprob == REF
    assert set(b.buckets) == set(a.buckets) and all(isinstance(k, int) for k in b.buckets)
    # the restored platform carries on: the pending challenge can be answered, the host stays live
    assert q.reliability(b, clock()) == p.reliability(a, clock())
    assert answer(q, hid, {"challenge": pending})["state"] == "live"


def test_app_without_dev_routes_has_no_mock_endpoints(identity, report, clock):
    from fastapi.testclient import TestClient
    from kwh_host.platform.mock import create_app
    p = MockPlatform(pool(), settings(), clock=clock)
    paths = {r.path for r in create_app(p, dev_routes=False).routes}
    assert "/v1/hosts/{host_id}/heartbeat" in paths and "/healthz" in paths
    assert not any(x.startswith("/v1/mock") for x in paths)
    assert any(x.startswith("/v1/mock") for x in {r.path for r in create_app(p).routes})
    assert TestClient(create_app(p, dev_routes=False)).get("/v1/mock/hosts").status_code == 404
