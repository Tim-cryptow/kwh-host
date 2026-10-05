#!/usr/bin/env bash
# M4 on a rented GPU machine: what happens to a host over time, on a real card. The one-line
# installer, a certified benchmark in the sandbox, the daemon live with buyer jobs, then:
#
#   5. the GPU slowed down (graphics clock locked low): the micro-benchmarks miss, the platform
#      holds the host, the daemon re-benchmarks on a fresh engine and is live again at the lower rate;
#   6. the clock released: the same again, back up to the full rate;
#   7. the engine frozen (docker pause): the daemon notices and restarts it;
#   8. kwh-host status and kwh-host events, as a host would read them.
#
#   curl -fsSL https://raw.githubusercontent.com/Tim-cryptow/kwh-host/main/scripts/vm-m4.sh -o vm-m4.sh && bash vm-m4.sh
#
# `bash vm-m4.sh again` runs it once more on the same machine: the last run's results are kept aside,
# and the report it certified is registered as it is, so a daemon on a machine that has changed
# since (a driver updated and the machine rebooted, say) has to notice by itself and re-benchmark.
#
# Like vm-m3.sh: as root it creates a normal user, kwh, with passwordless sudo and runs everything as
# that user; the test runs in the background and the command follows its output, so a dropped
# connection does not stop it and the same command follows it again. Everything lands in ~/m4 of the
# user running it (summary.json last); m4-out.tgz, with all of it, also goes to the login user's home.
set -uo pipefail
REF="${KWH_REF:-main}"
SELF="$(readlink -f "$0")"

if [ "$(id -u)" -eq 0 ]; then
  if ! id kwh >/dev/null 2>&1; then
    useradd -m -s /bin/bash kwh || exit 1
    echo "kwh ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/kwh && chmod 0440 /etc/sudoers.d/kwh
  fi
  # a new file, renamed into place: a test already running from the old one keeps reading it intact
  install -o kwh -g kwh -m 0755 "$SELF" /home/kwh/vm-m4.sh.new && mv -f /home/kwh/vm-m4.sh.new /home/kwh/vm-m4.sh || exit 1
  exec sudo -iu kwh env KWH_REF="$REF" KWH_M4_COPY_TO="$HOME" bash /home/kwh/vm-m4.sh "$@"
fi

OUT="$HOME/m4"
if [ "${KWH_M4_RUN:-}" != 1 ]; then
  sudo -n true 2>/dev/null || { echo "this needs passwordless sudo; run it as root and it sets up a user that has it"; exit 1; }
  mkdir -p "$OUT"
  running() { [ -n "$1" ] && grep -qs vm-m4 "/proc/$1/cmdline"; }    # a zombie or a reused pid is not the test
  pid="$(cat "$OUT/pid" 2>/dev/null)"
  if [ "${1:-}" = again ] && ! running "$pid" && [ -s "$OUT/console.log" ]; then
    stamp="$(date -u +%Y%m%d-%H%M%S)"
    mv "$OUT" "$OUT.run-$stamp" && mkdir -p "$OUT"
    [ -f "$HOME/m4-out.tgz" ] && mv "$HOME/m4-out.tgz" "$HOME/m4-out.run-$stamp.tgz"
    [ -f "$HOME/.kwh-host/events.jsonl" ] && mv "$HOME/.kwh-host/events.jsonl" "$HOME/.kwh-host/events.run-$stamp.jsonl"
    echo "the last run is kept in $OUT.run-$stamp; starting again"
    pid=""
  fi
  if running "$pid"; then
    echo "the test is already running; following it (Ctrl+C stops following, not the test)"
  elif [ -f "$HOME/m4-out.tgz" ]; then
    tail -n 30 "$OUT/console.log"; echo "the test has finished: ~/m4-out.tgz"; exit 0
  else
    if [ -s "$OUT/console.log" ]; then           # an earlier attempt that stopped part way: keep it aside
      mv "$OUT" "$OUT.stopped-$(date +%s)" && mkdir -p "$OUT"
      rm -f "$HOME/.kwh-host/config.json" "$HOME/.kwh-host/state.json" "$HOME/.kwh-host/events.jsonl"
    fi
    rm -f "$OUT/pid"
    KWH_M4_RUN=1 setsid nohup bash "$SELF" > "$OUT/console.log" 2>&1 < /dev/null &
    for _ in $(seq 1 20); do [ -s "$OUT/pid" ] && break; sleep 0.5; done
    echo "started; following it (Ctrl+C stops following, not the test; the same command follows it again)"
  fi
  pid="$(cat "$OUT/pid" 2>/dev/null)"
  [ -n "$pid" ] || { cat "$OUT/console.log"; echo "the test did not start"; exit 1; }
  tail -n +1 --pid="$pid" -f "$OUT/console.log"
  exit 0
fi
echo $$ > "$OUT/pid"
PORT=9000
PLATFORM="http://127.0.0.1:$PORT"
KH="${KWH_HOST_HOME:-$HOME/.kwh-host}"
export PATH="$HOME/.local/bin:$PATH"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/steps.log"; }
# as_docker CMD...: run with the docker group, which the installer may have granted after this login began
as_docker() { if docker info >/dev/null 2>&1; then "$@"; else sg docker -c "$(printf '%q ' "$@")"; fi; }
hosts() { curl -fs "$PLATFORM/v1/mock/hosts"; }
host_is() { hosts | python3 -c "import json,sys; d=json.load(sys.stdin)['hosts']; h=d[0] if d else {}; sys.exit(0 if ($1) else 1)" 2>/dev/null; }
host_get() { hosts | python3 -c "import json,sys; d=json.load(sys.stdin)['hosts']; h=d[0] if d else {}; print($1)" 2>/dev/null; }
wait_host() { for _ in $(seq 1 "${2:-300}"); do host_is "$1" && return 0; sleep 3; done; return 1; }
wait_http() { for _ in $(seq 1 120); do curl -fs "$1" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
container() { as_docker docker inspect -f '{{.Id}} {{.State.Status}}' kwh-engine-gpu0 2>/dev/null | cut -c1-12,65-; }
# The daemon runs under sg or a function's subshell, so $! is a wrapper: stop it by name.
stop_daemon() {
  pkill -TERM -u "$(id -u)" -f 'kwh-hos[t] run' 2>/dev/null
  for _ in $(seq 1 120); do pgrep -u "$(id -u)" -f 'kwh-hos[t] run' > /dev/null || return 0; sleep 1; done
  pkill -KILL -u "$(id -u)" -f 'kwh-hos[t] run' 2>/dev/null
}
# The platform holds the host for a re-benchmark, the daemon does it, the host is live again. The
# micro-benchmarks should be what notices; if they have not after 8 minutes, ask for it outright so
# the run still shows the re-benchmark itself.
rebench_cycle() {   # $1: name
  if wait_host "h.get('rebench_required')" 160; then
    log "$1: platform holds the host: $(host_get "'; '.join(h.get('rebench_reasons') or [])")"
  else
    log "$1: the micro-benchmarks did not miss twice (last: $(host_get "h.get('last_microbench')")); asking for a re-benchmark"
    curl -fs -X POST "$PLATFORM/v1/mock/hosts/$(host_get "h['host_id']")/rebench" > /dev/null
  fi
  wait_host "h.get('benchmarking')" 100 && log "$1: daemon re-benchmarking (engine stopped)"
  kwh-host submit --platform "$PLATFORM" --max-tokens 16 --timeout 20 --prompt "Say hello." \
    --out "$OUT/jobs-during-$1.json" > "$OUT/jobs-during-$1.txt" 2>&1
  log "$1: a buyer job meanwhile: exit $? ($(head -n 1 "$OUT/jobs-during-$1.txt" | cut -c1-160))"
  if wait_host "h.get('state') == 'live' and not h.get('rebench_required') and not h.get('benchmarking')" 600; then
    log "$1: live again at $(host_get "h['rate_units_per_hour']") units/hour, engine $(container)"
  else
    log "$1: NOT live again after 30 min: $(host_get "(h.get('state'), h.get('reasons'))")"
  fi
  cp "$KH/report.json" "$OUT/report-$1.json"
  local t; t=$(date +%s)
  wait_host "(h.get('last_microbench') or {}).get('t', 0) > $t" 100 \
    && log "$1: next micro-benchmark $(host_get "round(h['last_microbench']['units_per_hour'], 2)") units/hour, within: $(host_get "h['last_microbench']['within']")"
}

# leftovers of an earlier attempt, if any: its platform, daemon, engine container and GPU settings
pkill -f 'kwh-hos[t] mock-platform' 2>/dev/null; pkill -f 'kwh-hos[t] run' 2>/dev/null
as_docker docker rm -f kwh-engine-gpu0 >/dev/null 2>&1
sudo nvidia-smi -rgc >/dev/null 2>&1
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,clocks.max.sm,power.default_limit --format=csv | tee "$OUT/gpu.txt"
nvidia-smi > "$OUT/nvidia-smi.txt" 2>&1
grep -o "CUDA Version: [0-9.]*" "$OUT/nvidia-smi.txt" | tee -a "$OUT/gpu.txt"
df -h / "$HOME" | tee "$OUT/disk.txt"; free -g | tee -a "$OUT/disk.txt"

log "0. installer (--yes: Docker, NVIDIA Container Toolkit, kwh-host as needed)"
curl -fsSL "https://raw.githubusercontent.com/Tim-cryptow/kwh-host/$REF/install.sh" | bash -s -- --yes --ref "$REF" 2>&1 | tee "$OUT/install.log"
log "installer exit ${PIPESTATUS[1]}; $(kwh-host --version 2>&1)"
"$KH/venv/bin/pip" install -q transformers jinja2 2>&1 | tail -1   # the mock platform's tokenizer and chat template (buyer side)

log "1. init, fetch, doctor"
as_docker kwh-host init --platform "$PLATFORM" | tee "$OUT/init.json"
T0=$(date +%s)
as_docker kwh-host fetch > "$OUT/fetch.json" 2> "$OUT/fetch.log"; log "fetch exit $? in $(( $(date +%s) - T0 ))s"
as_docker kwh-host doctor | tee "$OUT/doctor.txt"; log "doctor exit ${PIPESTATUS[0]}"

log "2. mock platform: Docker hosts only (D4), certified reports only; heartbeat 10 s, challenge 1 min, micro-benchmark 2 min"
HF_HOME="$KH/hf" HF_HUB_OFFLINE=1 setsid nohup kwh-host mock-platform --port "$PORT" --heartbeat 10 \
  --challenge-every 60 --microbench-every 120 > "$OUT/mock-platform.log" 2>&1 < /dev/null &
wait_http "$PLATFORM/healthz" && curl -fs "$PLATFORM/healthz" | tee -a "$OUT/steps.log"; echo
nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu,memory.used \
  --format=csv -l 5 > "$OUT/gpu-trace.csv" 2>&1 &
TRACE=$!

# report_says FIELD: a field of the certified report already here (none on a fresh machine)
report_says() { python3 -c "import json,sys; r=json.load(open(sys.argv[1])); g=(r['hardware']['gpus'] or [{}])[0]; print({'certified': r.get('certified'), 'driver': g.get('driver_version'), 'finished': r.get('finished_at'), 'rate': r['score']['units_per_hour']}[sys.argv[2]])" "$KH/report.json" "$1" 2>/dev/null; }
if [ "$(report_says certified)" = True ]; then
  log "3. the certified report already here, registered as it is: $(report_says rate) units/hour, driver $(report_says driver), measured $(report_says finished)"
  log "   this machine's driver now: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
  log "3. certified benchmark inside the sandbox, then register"
  as_docker kwh-host bench > "$OUT/bench.txt" 2>&1; log "bench exit $?"
  grep -E "units/hour|canary|certified" "$OUT/bench.txt" | tee -a "$OUT/steps.log"
fi
cp "$KH/report.json" "$OUT/report-first.json" 2>/dev/null
as_docker kwh-host register > "$OUT/register.json" 2>&1; log "register: $(tr -d '\n ' < "$OUT/register.json")"

log "4. the daemon, live, and buyer jobs"
as_docker kwh-host run > "$OUT/run.log" 2>&1 &
# up to 15 minutes: a report that no longer fits this machine is redone before the engine first starts
wait_host "h.get('state') == 'live' and h.get('jobs_channel')" 300 && log "live, job channel open, engine $(container)" || log "NOT live"
kwh-host events --kind rebench > "$OUT/rebench-at-start.txt" 2>&1
grep -q "rebench" "$OUT/rebench-at-start.txt" && sed 's/^/   at start: /' "$OUT/rebench-at-start.txt" | tee -a "$OUT/steps.log"
cp "$KH/report.json" "$OUT/report-start.json" 2>/dev/null
kwh-host submit --platform "$PLATFORM" --max-tokens 128 --temperature 0 \
  --prompt "Explain how photosynthesis works to a ten-year-old." \
  --prompt "Write a Python function that checks whether a string is a palindrome." \
  --out "$OUT/jobs-first.json" > "$OUT/jobs-first.txt" 2>&1; log "buyer jobs: exit $?"
wait_host "(h.get('last_microbench') or {}).get('within') is not None" 100 \
  && log "micro-benchmark: $(host_get "round(h['last_microbench']['units_per_hour'], 2)") units/hour, within: $(host_get "h['last_microbench']['within']")"

log "5. the GPU slowed down: graphics clock locked low (a stand-in for throttling: heat, a power cap, a failing fan)"
MAX_SM=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -1 | tr -d ' ')
LOW=$(( ${MAX_SM:-2500} * 3 / 10 ))
sudo nvidia-smi -pm 1 > /dev/null 2>&1
# nvidia-smi can refuse with "not supported ... treating as warning" and still exit 0: read what it said
said="$(sudo nvidia-smi -lgc "$LOW,$LOW" 2>&1)"; rc=$?
echo "$said" | tee -a "$OUT/steps.log"
if [ "$rc" -ne 0 ] || echo "$said" | grep -qiE "not supported|insufficient|warning|failed"; then
  MIN_PL=$(nvidia-smi --query-gpu=power.min_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
  log "clock lock refused; capping power at the card's minimum, $MIN_PL W, instead"
  sudo nvidia-smi -pl "$MIN_PL" 2>&1 | tee -a "$OUT/steps.log"
else
  log "graphics clock locked at $LOW MHz (the card's maximum is $MAX_SM)"
fi
# If neither slows the card past the 10% tolerance, rebench_cycle asks for the re-benchmark after
# 8 minutes, so the run still shows one.
rebench_cycle slow

log "6. the clock released"
sudo nvidia-smi -rgc 2>&1 | tee -a "$OUT/steps.log"
DEF_PL=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits | head -1 | cut -d. -f1)
sudo nvidia-smi -pl "$DEF_PL" > /dev/null 2>&1
rebench_cycle full

log "7. the engine frozen (docker pause): the daemon should restart it"
BEFORE=$(container | cut -c1-12)
as_docker docker pause kwh-engine-gpu0 > /dev/null && T_PAUSE=$(date +%s) && echo "$T_PAUSE" > "$OUT/t-pause"
back=""
for _ in $(seq 1 200); do
  now=$(container)
  if [ -n "$now" ] && [ "${now:0:12}" != "$BEFORE" ] && [ "${now#* }" = running ] && host_is "h.get('state') == 'live'"; then
    back=1; break
  fi
  sleep 3
done
[ -n "$back" ] && log "engine restarted and live $(( $(date +%s) - T_PAUSE ))s after the freeze, engine $(container)" \
  || log "NOT restarted after 10 min: $(host_get "(h.get('state'), h.get('reasons'))")"
as_docker docker inspect kwh-engine-gpu0 > "$OUT/engine-inspect.json" 2>/dev/null
python3 - "$OUT" <<'PY' | tee -a "$OUT/steps.log"
import json, sys
try:
    i = json.load(open(f"{sys.argv[1]}/engine-inspect.json"))[0]
    hc = i["HostConfig"]
    print("restarted engine:", json.dumps({"readonly_rootfs": hc["ReadonlyRootfs"], "network": hc["NetworkMode"],
                                          "cap_drop": hc["CapDrop"], "user": i["Config"]["User"]}))
except Exception as e:
    print("restarted engine: no inspect", e)
PY

log "8. what a host reads"
sleep 20
kwh-host status > "$OUT/status.txt" 2>&1; cat "$OUT/status.txt"
kwh-host status --json > "$OUT/status.json" 2>&1
kwh-host events --limit 200 > "$OUT/events.txt" 2>&1
kwh-host events --all --json --limit 100000 > "$OUT/events-host.jsonl" 2>&1
kwh-host events --remote --limit 200 > "$OUT/events-platform.txt" 2>&1
kwh-host events --remote --all --json --limit 1250 > "$OUT/events-platform.jsonl" 2>&1
hosts > "$OUT/hosts-end.json"
stop_daemon
kill "$TRACE" 2>/dev/null
as_docker docker inspect kwh-engine-gpu0 >/dev/null 2>&1 && log "ENGINE CONTAINER LEFT BEHIND" || log "daemon stopped, engine container removed"
cp "$KH/engine.log" "$KH/engine-bench.log" "$OUT/" 2>/dev/null

log "9. summary"
python3 - "$OUT" <<'PY' | tee "$OUT/summary.json"
import json, os, sys
o = sys.argv[1]
def load(n):
    try:
        return json.load(open(os.path.join(o, n)))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
def lines(n):
    try:
        return [json.loads(x) for x in open(os.path.join(o, n)) if x.strip().startswith("{")]
    except OSError:
        return []
def text(n):
    try:
        return open(os.path.join(o, n)).read()
    except OSError:
        return ""
def report(n):
    r = load(n)
    gpus = (r.get("hardware") or {}).get("gpus") or [{}]
    return {"units_per_hour": (r.get("score") or {}).get("units_per_hour"), "certified": r.get("certified"),
            "reasons": r.get("certified_reasons"), "canary_mean": (r.get("canary") or {}).get("mean_delta"),
            "launch_mode": (r.get("engine") or {}).get("launch_mode"),
            "image": ((r.get("engine") or {}).get("extra") or {}).get("docker_image"),
            "gpu": gpus[0].get("uuid"), "driver": gpus[0].get("driver_version"), "finished_at": r.get("finished_at")}
pev = lines("events-platform.jsonl")
hev = lines("events-host.jsonl")
cycles, cur = [], None
for e in pev:
    k = e["kind"]
    if k == "rebench_required" and cur is None:
        cur = {"trigger": "platform", "reasons": e.get("reasons"), "held_at": e["t"]}
        cycles.append(cur)
    elif k == "rebench_started":
        if cur is None:          # the host decided by itself (its report no longer fit the machine)
            mine = next((h for h in hev if h["kind"] == "rebench" and h.get("phase") == "start"
                         and abs(h["t"] - e["t"]) < 120), {})
            cur = {"trigger": "host", "reasons": mine.get("reasons"), "held_at": e["t"]}
            cycles.append(cur)
        cur.setdefault("started_at", e["t"])
    elif cur is not None and k == "re-benchmarked":
        cur.update(reported_at=e["t"], rate=e.get("rate"), previous_rate=e.get("previous_rate"))
    elif cur is not None and k == "state" and e.get("to") == "live" and "reported_at" in cur:
        cur["live_at"] = e["t"]
        cur = None
for c in cycles:
    t0 = c.pop("held_at")
    for key in ("started_at", "reported_at", "live_at"):
        if key in c:
            c[key.replace("_at", "_after_s")] = round(c.pop(key) - t0, 1)
micro = [(round(e["t"]), e.get("units_per_hour"), e.get("within")) for e in pev if e["kind"] == "microbench"]
try:
    t_pause = float(text("t-pause").strip())
except ValueError:
    t_pause = None
freeze = {}
if t_pause:
    after = [e for e in hev if e["t"] >= t_pause]
    first = lambda kind, **kw: next((e for e in after if e["kind"] == kind and all(e.get(a) == b for a, b in kw.items())), None)
    for name, e in (("restart_decided", first("engine_restart")), ("engine_down", first("engine_down", reason="restart")),
                    ("engine_up", first("engine_up"))):
        if e:
            freeze[name + "_after_s"] = round(e["t"] - t_pause, 1)
    live = next((e for e in pev if e["kind"] == "state" and e["t"] >= t_pause and e.get("to") == "live"), None)
    if live:
        freeze["live_after_s"] = round(live["t"] - t_pause, 1)
    freeze["platform_states"] = [(e.get("from"), e.get("to")) for e in pev if e["kind"] == "state" and e["t"] >= t_pause]
status = load("status.json")
pview = status.get("platform_view") or {}
print(json.dumps({
    "gpu": text("gpu.txt").strip().splitlines()[1:],
    "version": [x for x in text("steps.log").splitlines() if "installer exit" in x],
    "doctor": text("doctor.txt").strip().splitlines(),
    "reports": {n: report(f"report-{n}.json") for n in ("first", "start", "slow", "full")},
    "host_rebench": [{k: v for k, v in e.items() if k != "kind"} for e in hev if e["kind"] in ("rebench", "engine_failed")],
    "rebench_cycles": cycles,
    "microbench_platform": micro,
    "jobs_during_rebench": {n: [{k: r.get(k) for k in ("status", "reason")} for r in
                                (load(f"jobs-during-{n}.json") if isinstance(load(f"jobs-during-{n}.json"), list) else [])]
                            for n in ("slow", "full")},
    "freeze": freeze,
    "reliability_1h": (pview.get("reliability") or {}).get("1h"),
    "host_reported": pview.get("host_reported"),
    "end_state": {k: pview.get(k) for k in ("state", "reasons", "rate_units_per_hour", "balance", "jobs")},
    "event_counts": {"host": len(hev), "platform": len(pev)},
    "steps": text("steps.log").strip().splitlines(),
}, indent=1))
PY
sudo nvidia-smi -rgc > /dev/null 2>&1
tar czf "$HOME/m4-out.tgz.part" -C "$OUT" .
if [ -n "${KWH_M4_COPY_TO:-}" ] && [ "$KWH_M4_COPY_TO" != "$HOME" ]; then
  sudo cp "$HOME/m4-out.tgz.part" "$KWH_M4_COPY_TO/m4-out.tgz" && log "copied to $KWH_M4_COPY_TO/m4-out.tgz"
fi
mv "$HOME/m4-out.tgz.part" "$HOME/m4-out.tgz" && log "done: ~/m4 and ~/m4-out.tgz"
