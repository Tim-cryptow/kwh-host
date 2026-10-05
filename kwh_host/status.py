"""What `kwh-host status` and `kwh-host events` print (HOST-CLIENT.md §5, M4).

Two sources, kept apart: the platform's view (its verdict, rate, balance, reliability counters
and event log, which is what counts) and the daemon's (state.json and events.jsonl: the engine,
the GPU, the jobs it ran, and the last thing it heard from the platform, which is all there is
when the platform cannot be reached). Times shown as "ago" come from the platform's clock when
the platform said them, so a host whose clock is off still reads its own state right.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from .config import image_label

STATE_WORDS = {"live": "live", "degraded": "degraded", "offline": "offline", "registered": "registered, not yet live",
               "rejected": "rejected by the platform", "stopped": "stopped", "starting": "starting"}


# --- small formatters ----------------------------------------------------------

def ago(seconds: Optional[float]) -> str:
    """90 -> '1 min'; 7980 -> '2 h 13 min'; 200000 -> '2 days'."""
    if seconds is None:
        return "?"
    s = max(0, int(seconds))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min"
    if s < 2 * 86400:
        h, m = divmod(s // 60, 60)
        return f"{h} h" + (f" {m} min" if m else "")
    return f"{s // 86400} days"


def clock_time(t: Optional[float], with_date: bool = False) -> str:
    if t is None:
        return "?"
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S" if with_date else "%H:%M")


def pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def ms(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x:.0f} ms" if x < 1000 else f"{x / 1000:.1f} s"


def num(x: Any, digits: int = 2) -> str:
    if x is None:
        return "-"
    if isinstance(x, int):
        return f"{x:,}"
    return f"{x:,.{digits}f}"


def parse_since(s: Optional[str], now: float) -> Optional[float]:
    """'90m', '2h', '3d', '45s' or a bare number of seconds -> a unix time that long before now."""
    if not s:
        return None
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*", s)
    if not m:
        raise ValueError(f"cannot read {s!r}; use e.g. 30m, 2h or 3d")
    n = float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
    return now - n


# --- events ---------------------------------------------------------------------

def describe_event(ev: Dict[str, Any]) -> str:
    """One event, local or the platform's, in a line a host can read."""
    k = ev.get("kind")
    if k == "heartbeat":
        s = f"{ev.get('state')}, {'accepted' if ev.get('accepted') else 'REJECTED'}"
        if ev.get("minted"):
            s += f", minted {ev['minted']}"
        if ev.get("reasons"):
            s += ": " + "; ".join(ev["reasons"])
        return s + (f" ({ev['rtt_ms']:.0f} ms)" if isinstance(ev.get("rtt_ms"), (int, float)) else "")
    if k == "heartbeat_failed":
        return f"not sent: {ev.get('error')}"
    if k == "heartbeat_rejected":
        return "rejected: " + "; ".join(ev.get("reasons") or [])
    if k == "state":
        s = f"{ev.get('from')} -> {ev.get('to')}"
        return s + (": " + "; ".join(ev["reasons"]) if ev.get("reasons") else "")
    if k in ("challenge", "liveness") and "passed" in ev:
        s = f"{'passed' if ev['passed'] else 'FAILED'}, mean delta {num(ev.get('delta'), 4)}"
        if ev.get("elapsed_ms") is not None:
            s += f" ({ms(ev['elapsed_ms'])})"
        if ev.get("state"):
            s += f" -> {ev['state']}"
        if ev.get("passes_needed"):
            s += f", {ev['passes_needed']} more pass(es) in a row needed"
        return s
    if k == "challenge":
        return f"issued {ev.get('id')}"
    if k == "microbench":
        if ev.get("skipped"):
            return f"skipped: {ev['skipped']}"
        if ev.get("discarded"):
            return f"discarded: {ev['discarded']}"
        s = f"{num(ev.get('units_per_hour'))} units/hour"
        if ev.get("job_seconds") is not None:
            s += f" in {ev['job_seconds']:.2f} s"
        within = ev.get("within")
        return s + ("" if within is None else (", within tolerance" if within else ", OUTSIDE tolerance"))
    if k == "job":
        s = f"{ev.get('job_id')} {ev.get('status')}, {ev.get('requests')} request(s), " \
            f"{num(ev.get('completion_tokens'))} tokens, {ms(ev.get('latency_ms'))}"
        return s + (f": {ev['reason']}" if ev.get("reason") else "")
    if k == "job_failed":
        return f"{ev.get('job_id')} {ev.get('outcome')}" + (f": {ev['reason']}" if ev.get("reason") else "")
    if k == "rebench":
        phase = ev.get("phase")
        if phase == "start":
            return "started: " + "; ".join(ev.get("reasons") or [])
        if phase == "done":
            return f"done: {num(ev.get('units_per_hour'))} units/hour (was {num(ev.get('previous_units_per_hour'))}), " \
                   f"{ago(ev.get('seconds'))}"
        return f"FAILED: {ev.get('error')}"
    if k == "rebench_required":
        return "; ".join(ev.get("reasons") or [])
    if k == "re-benchmarked":
        return f"new report: {num(ev.get('rate'))} units/hour (was {num(ev.get('previous_rate'))})"
    if k == "engine_up":
        return f"{ev.get('version')} serving, context {num(ev.get('context'))}" + \
               (f", {image_label(ev['image'])}" if ev.get("image") else "")
    if k == "engine_down":
        return f"stopped ({ev.get('reason')})"
    if k == "start":
        return f"kwh-host {ev.get('version')} started ({ev.get('engine_mode')})" + \
               (f", {image_label(ev['image'])}" if ev.get("image") else "")
    if k == "stop":
        return f"stopped: {ev['error']}" if ev.get("error") else f"stopped ({ev.get('state')})"
    if k in ("registered", "re-registered"):
        return f"rate {num(ev.get('rate'))} units/hour"
    if k == "engine_restart":
        return str(ev.get("reason"))
    if k == "jobs_channel":
        return "open" + (f", context {num(ev.get('context'))}" if ev.get("context") else "") if ev.get("open") \
            else f"closed: {ev.get('error')}"
    if k == "rebench_started":
        return "the host stopped its engine to re-benchmark" + \
               (": " + "; ".join(ev["reasons"]) if ev.get("reasons") else "")
    if k == "rebench_ended":
        return "the host is serving again" + ("; a new report is still required" if ev.get("rebench_required") else "")
    if k == "host_reported":
        labels = {"send_failures": "heartbeat(s) not delivered", "engine_restarts": "engine restart(s)",
                  "results_undelivered": "job result(s) not delivered"}
        parts = [f"{ev[x]} {label}" for x, label in labels.items() if ev.get(x)]
        return "the host saw " + ", ".join(parts) + (f"; last error: {ev['last_error']}" if ev.get("last_error") else "")
    if k == "jobs_connected":
        return f"job channel open (concurrency {ev.get('max_concurrency')}, context {num(ev.get('max_model_len'))})"
    if k == "jobs_disconnected":
        return "job channel closed"
    rest = {key: v for key, v in ev.items() if key not in ("t", "kind")}
    return ", ".join(f"{key}={v}" for key, v in rest.items())


def is_routine(ev: Dict[str, Any]) -> bool:
    """Accepted heartbeats with nothing to say, issued challenges (the answer follows): hidden unless asked."""
    if ev.get("kind") == "heartbeat":
        return bool(ev.get("accepted")) and not ev.get("reasons")
    return ev.get("kind") == "challenge" and "passed" not in ev


def is_notable(ev: Dict[str, Any]) -> bool:
    """For the short list in `kwh-host status`: changes and failures, not the steady drumbeat."""
    k = ev.get("kind")
    if k in ("liveness", "challenge"):
        return ev.get("passed") is False
    if k == "microbench":
        return ev.get("within") is False
    return k not in ("heartbeat", "job", "jobs_channel", "jobs_connected", "jobs_disconnected", "verification")


def render_events(events: Iterable[Dict[str, Any]], show_all: bool = False) -> str:
    lines = []
    for ev in events:
        if not show_all and is_routine(ev):
            continue
        lines.append(f"{clock_time(ev.get('t'), with_date=True)}  {str(ev.get('kind')):<17} {describe_event(ev)}")
    return "\n".join(lines) if lines else "(no events)"


# --- status ---------------------------------------------------------------------

def _daemon_line(local: Optional[dict], now: float) -> str:
    if not local:
        return "has not run yet (kwh-host run, or kwh-host service install)"
    age = now - (local.get("updated_at") or 0)
    every = float((local.get("platform_config") or {}).get("heartbeat_seconds") or 30.0)
    if local.get("state") == "stopped":
        return f"stopped {ago(age)} ago"
    if local.get("state") == "rejected":
        return f"stopped {ago(age)} ago: the platform refused this host (kwh-host register)"
    if age > 3 * every + 60:
        return f"not running? its last update was {ago(age)} ago"
    s = f"running (pid {local.get('pid')}), up {ago(now - local['started_at'])}" if local.get("started_at") else "running"
    if local.get("benchmarking"):
        s += ", re-benchmarking"
    return s


def render_status(host: dict, local: Optional[dict], remote: Optional[dict], now: Optional[float] = None) -> str:
    """`host`: version, host_id, platform URL, engine mode and image from the config."""
    now = time.time() if now is None else now
    out: List[str] = [f"kwh-host {host.get('version')} · {host.get('host_id') or 'not registered'} · {host.get('platform')}", ""]

    def row(label: str, text: str) -> None:
        first, *more = text.split("\n")
        out.append(f"{label:<13}{first}")
        out.extend(f"{'':<13}{m}" for m in more)

    error = (remote or {}).get("error")
    view = remote if remote and not error else None
    pnow = (view or {}).get("now") or now               # the platform's clock, for its own timestamps

    row("Daemon", _daemon_line(local, now))
    if view is None and remote is not None:
        row("Platform", f"unreachable: {error}" + ("; below is what the daemon last heard" if local else ""))

    src = view or local or {}
    state = src.get("state")
    if state:
        s = STATE_WORDS.get(state, state)
        since = src.get("state_since")
        if since:
            s += f" for {ago(pnow - since)}" if view else f" since {clock_time(since)}"
        if src.get("reasons"):
            s += "\n" + "; ".join(src["reasons"])
        row("State", s)

    rate = (view or {}).get("rate_units_per_hour") or ((local or {}).get("report") or {}).get("units_per_hour")
    if rate:
        row("Rate", f"{num(float(rate))} units/hour" + (f" (bucket {view['bucket']})" if view and view.get("bucket") else ""))
    if src.get("balance") is not None:
        row("Earnings", f"{num(src.get('balance'))} units minted, {num(src.get('accrual') or 0.0, 2)} accruing")

    eng = (local or {}).get("engine") or {}
    if eng:
        mode = {"docker": "in Docker", "subprocess": "bare metal"}.get(eng.get("launch_mode"), eng.get("launch_mode"))
        bits = [f"vLLM {eng.get('version')}" if eng.get("version") else "engine", mode or ""]
        if eng.get("image"):
            bits.append(image_label(eng["image"]))
        health = "healthy" if eng.get("healthy") else ("stopped" if eng.get("running") is False else "NOT answering")
        row("Engine", ", ".join(b for b in bits if b) + f", {health}")
    gs = (local or {}).get("gpu") or {}
    if gs.get("available"):
        parts = [f"{gs['util_pct']:.0f}% busy" if gs.get("util_pct") is not None else None,
                 f"{gs['power_w']:.0f} W" if gs.get("power_w") is not None else None,
                 f"{gs['mem_used_mib'] / 1024:.1f} of {gs['mem_total_mib'] / 1024:.1f} GB"
                 if gs.get("mem_used_mib") is not None and gs.get("mem_total_mib") else None,
                 f"{gs['temp_c']:.0f} °C" if gs.get("temp_c") is not None else None]
        foreign = gs.get("foreign_processes")
        parts.append("no other processes" if foreign == 0 else
                     (f"{foreign} OTHER process(es) on it" if foreign else None))
        row("GPU", f"{gs.get('name') or 'GPU'}: " + ", ".join(p for p in parts if p))

    checks: List[str] = []
    ch = src.get("last_challenge")
    if ch:
        when = (pnow - ch["t"]) if view and ch.get("t") else ((now - ch["t"]) if ch.get("t") else None)
        checks.append(f"challenge {ago(when)} ago: {'passed' if ch.get('pass') else 'FAILED'}, "
                      f"mean delta {num(ch.get('delta'), 4)}")
    mb = src.get("last_microbench")
    if mb:
        when = (pnow - mb["t"]) if view and mb.get("t") else ((now - mb["t"]) if mb.get("t") else None)
        within = mb.get("within", mb.get("within_tolerance"))
        checks.append(f"micro-benchmark {ago(when)} ago: {num(mb.get('units_per_hour'))} units/hour, "
                      + ("within 10%" if within else "OUTSIDE 10%"))
    if view and view.get("report_at"):
        due = view.get("rebench_due_at")
        line = f"report measured {ago(pnow - view['report_at'])} ago"
        if view.get("rebench_required"):
            line += "; re-benchmark required: " + "; ".join(view.get("rebench_reasons") or [])
        elif due:
            line += f", re-benchmark due in {ago(due - pnow)}"
        checks.append(line)
    last = (local or {}).get("last_rebench")
    if last:
        checks.append(f"last re-benchmark {clock_time(last.get('started_at'), with_date=True)}: "
                      + (f"{num(last.get('units_per_hour'))} units/hour" if last.get("ok") else f"FAILED ({last.get('error')})"))
    if checks:
        row("Last checks", "\n".join(checks))

    jobs = (local or {}).get("jobs")
    if jobs:
        row("Jobs", f"{num(jobs.get('completed', 0))} completed, {num(jobs.get('failed', 0))} failed, "
                    f"{num(jobs.get('rejected', 0))} turned away"
                    + (f", {num(jobs['undelivered'])} results not delivered" if jobs.get("undelivered") else "")
                    + " since the daemon started"
                    + (f"; job channel {local['jobs_channel']}" if local.get("jobs_channel") else ""))

    rel = (view or {}).get("reliability") or {}
    if rel:
        names = [n for n in ("1h", "24h", "7d") if n in rel]
        heads = {"1h": "last hour", "24h": "last day", "7d": "last week"}
        out += ["", f"{'Reliability':<16}" + "".join(f"{heads[n]:>14}" for n in names)]

        def rrow(label: str, cell) -> None:
            out.append(f"  {label:<14}" + "".join(f"{cell(rel[n]):>14}" for n in names))
        rrow("observed", lambda w: ago(w.get("observed_s")))
        rrow("live", lambda w: pct(w.get("live_fraction")))
        rrow("beats ok", lambda w: f"{w['heartbeats']['accepted']}/{w['heartbeats']['accepted'] + w['heartbeats']['rejected']}")
        rrow("beats missed", lambda w: num(max(0, w['heartbeats']['expected'] - w['heartbeats']['accepted']
                                               - w['heartbeats']['rejected'])))
        rrow("challenges", lambda w: f"{w['challenges']['passed']}/{w['challenges']['passed'] + w['challenges']['failed']}")
        rrow("jobs done", lambda w: f"{w['jobs']['completed']}")
        rrow("jobs failed", lambda w: f"{w['jobs']['failed'] + w['jobs']['bad_results']}")
        rrow("latency p50", lambda w: ms(w["jobs"].get("latency_ms_p50")))
        rrow("latency p95", lambda w: ms(w["jobs"].get("latency_ms_p95")))
        rrow("micro-bench", lambda w: num(w["microbench"].get("median_units_per_hour")))
        rrow("minted", lambda w: num(w.get("minted")))

    evs = [e for e in (view or {}).get("events") or [] if is_notable(e)][-6:]
    if evs:
        out += ["", "Recent (the platform's log; more: kwh-host events --remote)"]
        out += [f"  {clock_time(e.get('t'))}  {str(e.get('kind')):<17} {describe_event(e)}" for e in evs]
    return "\n".join(out)
