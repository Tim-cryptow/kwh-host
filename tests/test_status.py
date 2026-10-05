"""`kwh-host status` and `kwh-host events` (M4): the event log, and what a host reads."""

import json
import time

from click.testing import CliRunner

from kwh_host.cli import main
from kwh_host.config import DEFAULT_DOCKER_IMAGE, write_state
from kwh_host.events import EventLog
from kwh_host.status import ago, describe_event, parse_since, render_events, render_status

NOW = 1_791_200_000.0


def test_event_log_appends_rotates_and_filters(tmp_path):
    t = [NOW]
    log = EventLog(tmp_path / "events.jsonl", max_bytes=400, clock=lambda: t[0])
    for i in range(20):
        t[0] += 10
        log.add("heartbeat" if i % 2 else "challenge", n=i)
    assert log.rotated.exists() and log.path.stat().st_size < 400 + 100
    every = log.read()
    assert [e["n"] for e in every] == sorted(e["n"] for e in every) and every[-1]["n"] == 19
    assert all(e["kind"] == "challenge" for e in log.read(kinds=["challenge"]))
    assert [e["n"] for e in log.read(since=NOW + 180)] == [18, 19]
    assert len(log.read(limit=3)) == 3
    (tmp_path / "events.jsonl").open("a").write('{"t": 1, "kind": "cut sho')     # a crash mid-line
    assert log.read()[-1]["n"] == 19


def test_small_formatters():
    assert ago(59) == "59 s" and ago(90) == "1 min" and ago(7980) == "2 h 13 min" and ago(3 * 86400) == "3 days"
    assert parse_since("2h", NOW) == NOW - 7200 and parse_since("30m", NOW) == NOW - 1800 and parse_since(None, NOW) is None
    assert describe_event({"kind": "challenge", "passed": False, "delta": 0.1285, "elapsed_ms": 412, "state": "degraded",
                           "passes_needed": 2}) == \
        "FAILED, mean delta 0.1285 (412 ms) -> degraded, 2 more pass(es) in a row needed"
    assert describe_event({"kind": "rebench", "phase": "done", "units_per_hour": 101.61,
                           "previous_units_per_hour": 101.51, "seconds": 412}) == \
        "done: 101.61 units/hour (was 101.51), 6 min"
    assert describe_event({"kind": "state", "from": "live", "to": "degraded", "reasons": ["engine unhealthy"]}) == \
        "live -> degraded: engine unhealthy"
    # accepted heartbeats are routine; a rejected one is not
    text = render_events([{"t": NOW, "kind": "heartbeat", "state": "live", "accepted": True, "minted": 1},
                          {"t": NOW + 30, "kind": "heartbeat", "state": "degraded", "accepted": False,
                           "reasons": ["engine unhealthy"]}])
    assert "accepted" not in text and "REJECTED: engine unhealthy" in text


def platform_view():
    window = {"observed_s": 3600, "state_s": {"live": 3600}, "live_fraction": 1.0,
              "heartbeats": {"accepted": 120, "rejected": 0, "expected": 120},
              "challenges": {"passed": 12, "failed": 0, "mean_delta": 0.0, "max_delta": 0.0},
              "microbench": {"runs": 2, "misses": 0, "median_units_per_hour": 101.2},
              "jobs": {"completed": 40, "failed": 1, "bad_results": 0, "units": 3.2, "latency_ms_p50": 1200.0,
                       "latency_ms_p95": 3400.0},
              "minted": 101, "host_reported": {"send_failures": 0, "engine_restarts": 0, "results_undelivered": 0}}
    return {"host_id": "h_abc", "state": "live", "reasons": [], "state_since": NOW - 7980, "now": NOW,
            "rate_units_per_hour": 101.508, "bucket": "I-1/100", "accrual": 0.62, "balance": 214,
            "last_challenge": {"pass": True, "delta": 0.0, "t": NOW - 180},
            "last_microbench": {"units_per_hour": 101.2, "within": True, "t": NOW - 720},
            "rebench_required": False, "rebench_reasons": [], "report_at": NOW - 2 * 86400,
            "rebench_due_at": NOW + 5 * 86400, "reliability": {"1h": window, "24h": window, "7d": window},
            "events": [{"t": NOW - 7980, "kind": "state", "from": "degraded", "to": "live", "reasons": []},
                       {"t": NOW - 180, "kind": "liveness", "passed": True, "delta": 0.0}]}


def daemon_state(**kw):
    s = {"pid": 4242, "started_at": NOW - 86400, "updated_at": NOW - 10, "state": "live", "benchmarking": False,
         "platform_config": {"heartbeat_seconds": 30.0},
         "engine": {"healthy": True, "version": "0.30.0", "launch_mode": "docker", "image": DEFAULT_DOCKER_IMAGE},
         "gpu": {"available": True, "name": "NVIDIA GeForce RTX 4090", "util_pct": 31.0, "power_w": 287.0,
                 "mem_used_mib": 21811.0, "mem_total_mib": 24564.0, "temp_c": 61.0, "foreign_processes": 0},
         "jobs": {"completed": 1203, "failed": 2, "rejected": 0, "undelivered": 0}, "jobs_channel": "open",
         "report": {"units_per_hour": 101.508}}
    s.update(kw)
    return s


def test_status_reads_like_a_page():
    host = {"version": "0.2.0", "host_id": "h_abc", "platform": "https://platform.example"}
    text = render_status(host, daemon_state(), platform_view(), now=NOW)
    for line in ("Daemon       running (pid 4242), up 24 h",
                 "State        live for 2 h 13 min",
                 "Rate         101.51 units/hour (bucket I-1/100)",
                 "Earnings     214 units minted, 0.62 accruing",
                 "Engine       vLLM 0.30.0, in Docker, vllm/vllm-openai:v0.30.0 (sha256:8a69ffad…), healthy",
                 "GPU          NVIDIA GeForce RTX 4090: 31% busy, 287 W, 21.3 of 24.0 GB, 61 °C, no other processes",
                 "Last checks  challenge 3 min ago: passed, mean delta 0.0000",
                 "             micro-benchmark 12 min ago: 101.20 units/hour, within 10%",
                 "             report measured 2 days ago, re-benchmark due in 5 days",
                 "Jobs         1,203 completed, 2 failed, 0 turned away since the daemon started; job channel open"):
        assert line in text, f"missing: {line}\n---\n{text}"
    assert "  live                  100.0%        100.0%        100.0%" in text
    assert "degraded -> live" in text and "liveness" not in text            # passed challenges are not news


def test_status_without_the_platform_says_so_and_shows_the_last_word():
    host = {"version": "0.2.0", "host_id": "h_abc", "platform": "https://platform.example"}
    local = daemon_state(state="degraded", reasons=["re-benchmarking"], benchmarking=True, state_since=NOW - 300,
                         balance=214, accrual=0.1)
    text = render_status(host, local, {"error": "ConnectError: refused"}, now=NOW)
    assert "Platform     unreachable: ConnectError: refused; below is what the daemon last heard" in text
    assert "re-benchmarking" in text and "Reliability" not in text
    stopped = render_status(host, daemon_state(state="stopped", updated_at=NOW - 7200), None, now=NOW)
    assert "Daemon       stopped 2 h ago" in stopped
    silent = render_status(host, daemon_state(updated_at=NOW - 3600), None, now=NOW)
    assert "not running? its last update was 1 h ago" in silent
    assert "has not run yet" in render_status(host, None, None, now=NOW)


def test_status_and_events_commands(cfg, identity):
    write_state(cfg, daemon_state(updated_at=time.time()))
    log = EventLog(cfg.events_path)
    log.add("start", version="0.2.0", engine_mode="docker", image=DEFAULT_DOCKER_IMAGE)
    log.add("heartbeat", state="live", accepted=True, minted=1, rtt_ms=12.0)
    log.add("rebench", phase="start", reasons=["report older than 7 days"])
    runner = CliRunner()
    out = runner.invoke(main, ["status", "--local"])
    assert out.exit_code == 0, out.output
    assert "not registered" in out.output and "NVIDIA GeForce RTX 4090" in out.output
    raw = runner.invoke(main, ["status", "--local", "--json"])
    assert json.loads(raw.output)["local"]["pid"] == 4242
    ev = runner.invoke(main, ["events"])
    assert ev.exit_code == 0 and "started: report older than 7 days" in ev.output and "accepted" not in ev.output
    every = runner.invoke(main, ["events", "--all", "--json"])
    assert [json.loads(line)["kind"] for line in every.output.splitlines()] == ["start", "heartbeat", "rebench"]
    only = runner.invoke(main, ["events", "--kind", "heartbeat"])
    assert "live, accepted, minted 1 (12 ms)" in only.output
    bad = runner.invoke(main, ["events", "--since", "yesterday"])
    assert bad.exit_code != 0 and "30m" in bad.output
