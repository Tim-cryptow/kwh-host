"""M4 on the platform side (HOST-CLIENT.md §5): when a report stops counting, how a re-benchmarking
host is treated, the state time and raw counters kept per host, and the events endpoint. Fake clock."""

import time

import httpx
import pytest

from kwh_host import rebench
from kwh_host.config import CUDA12_DOCKER_IMAGE, DEFAULT_DOCKER_IMAGE
from kwh_host.platform.client import PlatformClient
from kwh_host.platform.mock import MockPlatform, create_app
from test_mock_platform import REF, answer, healthy, idle_gpu, pool, register, settings

BEAT = {"engine": healthy(), "gpu_sample": idle_gpu()}


def go_live(p, hid):
    v = answer(p, hid, p.heartbeat(hid, BEAT))
    assert v["state"] == "live"


def beat_until_challenge(p, hid, clock):
    """Heartbeats every 30 s (a live host keeps beating) until the next challenge is issued."""
    for _ in range(20):
        clock.advance(30)
        r = p.heartbeat(hid, BEAT)
        if r["challenge"]:
            return r
    raise AssertionError("no challenge issued")


# --- the rules ------------------------------------------------------------------

def test_what_makes_a_report_stop_counting():
    gpus = {"GPU-a": "580.95.05"}
    same = rebench.change_reasons(gpus, DEFAULT_DOCKER_IMAGE, "GPU-a", "580.95.05", DEFAULT_DOCKER_IMAGE)
    assert same == []
    # a config written before the digest pin names the tag: the same build, not a change
    assert rebench.change_reasons(gpus, DEFAULT_DOCKER_IMAGE, "GPU-a", "580.95.05", "vllm/vllm-openai:v0.30.0") == []
    assert rebench.change_reasons(gpus, DEFAULT_DOCKER_IMAGE, "GPU-b", "580.95.05", DEFAULT_DOCKER_IMAGE) == \
        ["GPU changed: GPU-b is not the GPU benchmarked"]
    assert rebench.change_reasons(gpus, DEFAULT_DOCKER_IMAGE, "GPU-a", "590.10", DEFAULT_DOCKER_IMAGE) == \
        ["driver changed: 580.95.05 benchmarked, 590.10 now"]
    image = rebench.change_reasons(gpus, DEFAULT_DOCKER_IMAGE, "GPU-a", "580.95.05", CUDA12_DOCKER_IMAGE)
    assert len(image) == 1 and image[0].startswith("engine image changed: vllm/vllm-openai:v0.30.0 (sha256:")
    # a report that recorded no GPU or image (in-process engines) has nothing to compare against
    assert rebench.change_reasons({}, None, "GPU-x", "1.0", "any") == []
    assert rebench.age_reason(0.0, 7 * 86400 - 1) is None
    assert rebench.age_reason(0.0, 7 * 86400) == "report older than 7 days"
    assert rebench.age_reason(0.0, 3600, every=600) == "report older than 10 minutes"
    # a real report's shape (the M3 run on Vast): tag-named image, one GPU
    report = {"hardware": {"gpus": [{"uuid": "GPU-473c38ee", "driver_version": "580.95.05"}]},
              "engine": {"launch_mode": "docker", "extra": {"docker_image": "vllm/vllm-openai:v0.30.0"}},
              "finished_at": "2026-10-05T09:35:39Z"}
    assert rebench.report_gpus(report) == {"GPU-473c38ee": "580.95.05"}
    assert rebench.report_image(report) == DEFAULT_DOCKER_IMAGE
    t = rebench.iso_ts(report["finished_at"])
    sample = {"uuid": "GPU-473c38ee", "driver_version": "580.95.05"}
    assert rebench.local_reasons(report, sample, DEFAULT_DOCKER_IMAGE, t + 6 * 86400) == []
    assert rebench.local_reasons(report, sample, DEFAULT_DOCKER_IMAGE, t + 7 * 86400) == ["report older than 7 days"]


# --- holding a host until it has a new report --------------------------------------

def test_a_required_rebench_holds_the_host_through_a_passed_challenge(identity, report, clock):
    """The M4 bug: a host owing a re-benchmark went back to live on its next passed challenge."""
    p = MockPlatform(pool(8), settings(), clock=clock)
    hid = register(p, identity, report)
    go_live(p, hid)
    rate = p.hosts[hid].rate
    pending = beat_until_challenge(p, hid, clock)                 # issued, not yet answered
    for _ in range(2):
        p.microbench(hid, {"units_per_hour": rate * 0.7, "job_seconds": 1.0, "gpu_sample": idle_gpu()})
    rec = p.hosts[hid]
    assert rec.rebench_required and rec.state == "degraded"
    assert rec.reasons == ["re-benchmark required: " + rebench.MICROBENCH_REASON]
    v = answer(p, hid, pending)                                   # passes, but the report no longer counts
    assert v["pass"] and v["state"] == "degraded" and "re-benchmark required" in rec.reasons[0]
    for _ in range(3):                                            # reachable and healthy: accepted, held, unpaid
        clock.advance(30)
        r = p.heartbeat(hid, BEAT)
        assert r["accepted"] and r["state"] == "degraded" and r["challenge"] is None and r["minted"] == 0
        assert r["rebench_required"] and r["rebench_reasons"] == [rebench.MICROBENCH_REASON]
    out = p.upload_report(hid, {"report": report})
    assert out["rate_units_per_hour"] == rate and out["previous_rate_units_per_hour"] == rate
    assert not rec.rebench_required and rec.reasons == ["new report accepted; awaiting a challenge"]
    clock.advance(30)
    r = p.heartbeat(hid, BEAT)
    assert r["challenge"] is not None and answer(p, hid, r)["state"] == "live"
    kinds = [e["kind"] for e in rec.events]
    assert kinds.count("rebench_required") == 1 and "re-benchmarked" in kinds


def test_driver_image_gpu_and_age_require_a_rebench(identity, report, clock):
    p = MockPlatform(pool(8), settings(), clock=clock)
    hid = register(p, identity, report)
    rec = p.hosts[hid]
    # mock-engine reports record no GPU; give this host what a real report records (M3 on Vast)
    rec.report_gpus, rec.report_image = {"GPU-a": "580.95.05"}, DEFAULT_DOCKER_IMAGE

    def beat(uuid="GPU-a", driver="580.95.05", image="vllm/vllm-openai:v0.30.0"):
        return p.heartbeat(hid, {"engine": {**healthy(), "image": image},
                                 "gpu_sample": {**idle_gpu(), "uuid": uuid, "driver_version": driver}})

    assert answer(p, hid, beat())["state"] == "live"             # the tag in an old config is the pinned build
    clock.advance(30)
    r = beat(driver="590.10")
    assert r["state"] == "degraded" and r["rebench_reasons"] == ["driver changed: 580.95.05 benchmarked, 590.10 now"]
    clock.advance(30)
    r = beat(uuid="GPU-b", driver="590.10", image=CUDA12_DOCKER_IMAGE)   # more changes add to the reasons
    assert len(r["rebench_reasons"]) == 3 and r["challenge"] is None
    p.upload_report(hid, {"report": report})                     # the new report was made on what runs now
    assert rec.rebench_reasons == [] and rec.report_gpus == {}
    clock.advance(30)
    assert answer(p, hid, beat(uuid="GPU-b", driver="590.10"))["state"] == "live"
    # a week of silence: offline when the timeout ran out, and the report is too old to come back on
    clock.advance(7 * 86400)
    r = beat(uuid="GPU-b", driver="590.10")
    assert r["accepted"] and r["state"] == "degraded" and r["challenge"] is None
    assert r["reasons"] == ["re-benchmark required: report older than 7 days"]
    went_offline = next(e for e in rec.events if e["kind"] == "state" and e["to"] == "offline")
    assert went_offline["t"] == clock.t - 7 * 86400 + p.s.offline_after_seconds


def test_a_rebenchmarking_host_stays_reachable_but_unpaid(identity, report, clock):
    p = MockPlatform(pool(8), settings(max_failures=3, rebench_max_seconds=600), clock=clock)
    hid = register(p, identity, report)
    go_live(p, hid)
    rec = p.hosts[hid]
    benching = {"engine": {"healthy": False, "launch_mode": "docker"}, "gpu_sample": idle_gpu(foreign=1),
                "benchmarking": True}
    for _ in range(10):                       # 5 minutes with the engine down and another process on the GPU
        clock.advance(30)
        r = p.heartbeat(hid, benching)
        assert r["accepted"] and r["state"] == "degraded" and r["reasons"] == ["re-benchmarking"]
        assert r["challenge"] is None and r["minted"] == 0 and r["benchmarking"]
    assert [e["kind"] for e in rec.events].count("rebench_started") == 1
    for _ in range(14):                       # ... but not forever: past the limit, the beats count as failures
        clock.advance(30)
        r = p.heartbeat(hid, benching)
    assert r["state"] == "offline" and "re-benchmark running longer than 10 minutes" in r["reasons"][0]
    assert [e["kind"] for e in rec.events].count("heartbeat_rejected") == 1      # a run of the same rejection: one event
    # a host that comes back from an outage re-benchmarking is reachable: degraded, not offline
    rec.benchmarking_since = clock.t
    clock.advance(30)
    assert p.heartbeat(hid, benching)["state"] == "degraded"
    clock.advance(30)
    r = p.heartbeat(hid, BEAT)                # serving again
    assert r["challenge"] is not None and not rec.benchmarking and answer(p, hid, r)["state"] == "live"
    assert "rebench_ended" in [e["kind"] for e in rec.events]


# --- state time and counters -------------------------------------------------------

def test_state_time_and_counters_over_windows(identity, report, clock):
    clock.t = (int(time.time() // 3600) + 1) * 3600.0        # on a bucket edge, so the numbers are exact
    t0 = clock.t
    p = MockPlatform(pool(8), settings(), clock=clock)
    hid = register(p, identity, report)
    go_live(p, hid)
    rec = p.hosts[hid]
    p.job_outcome(rec, t0 + 60, "completed", 900.0, 0.5)     # before the last hour
    for i in range(1, 120):                                   # an hour of beats, challenges answered
        clock.advance(30)
        body = dict(BEAT)
        if i == 40:
            body["telemetry"] = {"send_failures": 2, "last_error": "ConnectError: refused", "uptime_s": 1200}
        r = p.heartbeat(hid, body)
        if r["challenge"]:
            answer(p, hid, r)
        if i == 100:
            p.job_outcome(rec, clock.t, "completed", 100.0, 0.25)
            p.job_outcome(rec, clock.t, "completed", 300.0, 0.25)
            p.job_outcome(rec, clock.t, "timeout", 5000.0)
    clock.advance(600)                                        # then silence
    s = p.status(hid)
    assert s["state"] == "offline" and s["state_since"] == t0 + 3570 + 90
    hour, day = s["reliability"]["1h"], s["reliability"]["24h"]
    assert set(s["reliability"]) == {"1h", "24h", "7d"}
    # the last hour, to the nearest bucket: from t0+600 to now (t0+4170)
    assert hour["state_s"] == {"live": 3060, "offline": 510} and hour["observed_s"] == 3570
    assert hour["live_fraction"] == round(3060 / 3570, 4)
    assert hour["heartbeats"] == {"accepted": 100, "rejected": 0, "expected": 119}
    assert hour["challenges"]["passed"] == 10 and hour["challenges"]["failed"] == 0 and hour["challenges"]["max_delta"] == 0.0
    assert hour["jobs"] == {"completed": 2, "failed": 1, "bad_results": 0, "units": 0.5,
                            "latency_ms_p50": 100.0, "latency_ms_p95": 300.0}
    assert hour["host_reported"] == {"send_failures": 2, "engine_restarts": 0, "results_undelivered": 0}
    # the day: everything since registration
    assert day["state_s"] == {"live": 3660, "offline": 510} and day["heartbeats"]["accepted"] == 120
    assert day["challenges"]["passed"] == 12 and day["jobs"]["completed"] == 3
    assert day["minted"] == rec.balance > 0
    assert s["host_reported"]["last_error"] == "ConnectError: refused"
    assert any(e["kind"] == "host_reported" and e["send_failures"] == 2 for e in rec.events)
    assert s["rebench_due_at"] == s["report_at"] + p.s.rebench_every_seconds


def test_counters_keep_a_week(identity, report, clock):
    p = MockPlatform(pool(), settings(), clock=clock)
    hid = register(p, identity, report)
    go_live(p, hid)
    for _ in range(3):
        clock.advance(3 * 86400)
        p.heartbeat(hid, BEAT)
    rec = p.hosts[hid]
    oldest = min(rec.buckets) * p.s.bucket_seconds
    assert clock.t - oldest <= p.s.history_seconds + p.s.bucket_seconds
    week = p.reliability(rec, clock.t)["7d"]
    assert week["observed_s"] <= 7 * 86400 and week["state_s"]["offline"] > week["state_s"].get("live", 0)


# --- the events endpoint --------------------------------------------------------------

async def test_events_endpoint_filters_and_is_signed(identity, report, clock):
    p = MockPlatform(pool(8), settings(challenge_every_seconds=30), clock=clock)
    app = create_app(p)
    transport = httpx.ASGITransport(app=app)
    async with PlatformClient("http://mock", identity, transport=transport, clock=clock) as c:
        await c.register(report)
        hid = c.host_id
        r = await c.heartbeat(healthy(), idle_gpu())
        await c.liveness(r["challenge"]["challenge_id"], [REF] * len(r["challenge"]["items"]), 3)
        t_live = clock.t
        for _ in range(3):
            clock.advance(30)
            r = await c.heartbeat(healthy(), idle_gpu())
            await c.liveness(r["challenge"]["challenge_id"], [REF + 0.3] * 4, 3)     # failures: state changes
        states = await c.events(kinds=["state"])
        assert [e["to"] for e in states["events"]] == ["live", "degraded", "offline"] and not states["more"]
        later = await c.events(since=t_live)
        assert all(e["t"] > t_live for e in later["events"]) and later["events"]
        last2 = await c.events(limit=2)
        assert len(last2["events"]) == 2 and last2["more"]
        status = await c.status()
        assert status["reliability"]["1h"]["challenges"] == {"passed": 1, "failed": 3, "mean_delta": 0.225,
                                                             "max_delta": 0.3}
    # someone else's key cannot read them, query or not
    from kwh_host.identity import Identity
    async with PlatformClient("http://mock", Identity.generate(), token=p.hosts[hid].token, host_id=hid,
                              transport=transport, clock=clock) as other:
        with pytest.raises(Exception) as e:
            await other.events(kinds=["state"])
        assert getattr(e.value, "status", None) == 401
