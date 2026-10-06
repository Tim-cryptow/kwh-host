#!/usr/bin/env bash
# The Windows path (HOST-CLIENT.md §12, docs/windows-wsl2.md) on a real Windows PC, as far as it goes
# without an NVIDIA card:
#   - Ubuntu in WSL2 with systemd;
#   - the one-line installer: Docker Engine inside Ubuntu (not Docker Desktop), the docker group,
#     kwh-host;
#   - the engine sandbox, with the stand-in engine CI uses: certify, register, live, a buyer job, a
#     re-benchmark;
#   - the host as a background service, then keeping Ubuntu running with no window open (the guide's
#     step 5), checked from the Windows side by scripts/wsl-keepalive.ps1.
# What it cannot test is the GPU itself: a card reaching Docker through the Windows driver, and a
# real benchmark.
#
# In Ubuntu (WSL2), not in PowerShell:
#   curl -fsSL https://raw.githubusercontent.com/Tim-cryptow/kwh-host/main/scripts/wsl-check.sh -o ~/wsl-check.sh && bash ~/wsl-check.sh
#
# It runs in stages and says what to do between them:
#   bash ~/wsl-check.sh   1. checks WSL and systemd, runs the installer (asks for your Linux password)
#   (wsl --shutdown in PowerShell, then open Ubuntu again: the docker group takes effect)
#   bash ~/wsl-check.sh   2. the sandbox end to end, then the host as a service, left running. It
#                            prints the PowerShell command for the keep-alive check, which ends by
#                            collecting everything into wsl-check.tgz on the Windows side.
# The keep-alive check calls this script back from PowerShell: `probe LABEL [SECONDS]` (the host's
# state, with a last line PowerShell reads) and `collect` (results into a tarball; removes the service).
set -uo pipefail
REF="${KWH_REF:-main}"
OUT="$HOME/wsl-check"
SRC="$HOME/kwh-host-src"
KH="${KWH_HOST_HOME:-$HOME/.kwh-host}"
PORT=9000
FAKE_PORT=18999
PLATFORM="http://127.0.0.1:$PORT"
ME="$(id -un)"                 # not USER: a session that wsl.exe --exec starts may not set it
mkdir -p "$OUT"
export PATH="$HOME/.local/bin:$PATH"
# systemctl --user needs the user manager's runtime dir, which a `wsl.exe --exec` session may not set
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/steps.log"; }
die() { log "STOP: $*"; exit 1; }
hosts() { curl -fs --max-time 10 "$PLATFORM/v1/mock/hosts"; }
host_is() { hosts | python3 -c "import json,sys; d=json.load(sys.stdin)['hosts']; h=d[0] if d else {}; sys.exit(0 if ($1) else 1)" 2>/dev/null; }
wait_host() { for _ in $(seq 1 "${2:-100}"); do host_is "$1" && return 0; sleep 3; done; return 1; }
wait_http() { for _ in $(seq 1 120); do curl -fs --max-time 5 "$1" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
windows_dir() {   # where files for the Windows side go: C:\Users\<you>\code if there is one, else C:\Users\<you>
  local u home=""
  u="$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d '\r')"
  if [ -n "$u" ] && [ -d "/mnt/c/Users/$u" ]; then
    home="/mnt/c/Users/$u"
  else              # no Windows interop: the one user folder with a kwh-host clone in code\
    home="$(dirname "$(dirname "$(find /mnt/c/Users -mindepth 3 -maxdepth 3 -path '*/code/kwh-host' 2>/dev/null | head -1)")")"
    [ "$home" = . ] && return 1
  fi
  if [ -d "$home/code" ]; then echo "$home/code"; else echo "$home"; fi
}

grep -qi microsoft /proc/version 2>/dev/null || [ -n "${KWH_WSL_CHECK_ANYWHERE:-}" ] \
  || { echo "This is for Ubuntu inside WSL2 on Windows; /proc/version does not mention Microsoft."; exit 1; }

# --- probe: the host's state, for wsl-keepalive.ps1 -----------------------------------------------
#   probe LABEL [SECONDS [WINDOWS_MS]]
# SECONDS: also judge the heartbeats of the last SECONDS. WINDOWS_MS: Windows' clock when PowerShell
# asked (Unix ms), to measure Ubuntu's clock against it (on the first test PC it ran about 5% slow).
# Saves what it saw in probe-NN-LABEL/ and ends with one line, RESULT key=value ..., for PowerShell.
# Exit 0 when the service is active and its engine container is running.
if [ "${1:-}" = probe ]; then
  read -r up0 _ < /proc/uptime; rt0="$(date +%s.%N)"      # first, before anything slow
  label="${2:-now}"; window="${3:-0}"; winms="${4:-0}"
  n="$(find "$OUT" -maxdepth 1 -name 'probe-*' | wc -l)"
  d="$OUT/probe-$(printf %02d $((n + 1)))-$label"
  mkdir -p "$d"
  {
    echo "== $(date -u +%FT%TZ) probe $label $window $winms"
    echo "== ps -p 1"; ps -o pid=,etimes=,comm= -p 1
    echo "== clocksource"; cat /sys/devices/system/clocksource/clocksource0/current_clocksource
    echo "== systemctl --user status kwh-host"; timeout 20 systemctl --user status --no-pager kwh-host 2>&1 | head -12
    echo "== docker ps -a"; timeout 20 docker ps -a --format '{{.Names}}  {{.Status}}' 2>&1
  } > "$d/state.txt" 2>&1
  timeout 30 kwh-host status > "$d/status.txt" 2>&1
  if hosts > "$d/platform-hosts.json" 2>/dev/null; then
    timeout 30 kwh-host events --remote --all --json --limit 1000 > "$d/events-platform.jsonl" 2>&1
  else
    rm -f "$d/platform-hosts.json"
  fi
  "$KH/venv/bin/python" - "$d" "$label" "$window" "$KH" "$up0" "$rt0" "$winms" "$OUT/clock.csv" <<'PY'
import json, os, subprocess, sys, time

d, label, window, kh = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
up0, rt0, winms, clock_csv = float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]), sys.argv[8]
now = time.time()
HZ = os.sysconf("SC_CLK_TCK")


def sh(*argv):
    try:
        return subprocess.run(list(argv), capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def started(pid):
    """Seconds after boot that a process started, by Ubuntu's own uptime clock (immune to clock jumps)."""
    try:
        stat = open(f"/proc/{pid}/stat").read()
        return int(stat[stat.rindex(")") + 2:].split()[19]) / HZ
    except (OSError, ValueError, IndexError):
        return None


def span(s):
    s = int(round(s))
    return f"{s // 60} min {s % 60} s" if s >= 60 else f"{s} s"


# Everything below is measured on Ubuntu's uptime clock, never its wall clock: under WSL2 the wall
# clock is pulled to Windows' time in jumps, so differences of wall times can come out negative.
uptime = float(open("/proc/uptime").read().split()[0])
boot_s = started(1) or 0.0                     # systemd, Ubuntu's PID 1
up = uptime - boot_s
service = sh("systemctl", "--user", "is-active", "kwh-host") or "unknown"
mono = sh("systemctl", "--user", "show", "kwh-host", "-p", "ActiveEnterTimestampMonotonic", "--value")
service_s = int(mono) / 1e6 if service == "active" and mono.isdigit() and int(mono) > 0 else None
eng = sh("docker", "inspect", "-f", "{{.State.Running}} {{.State.Pid}}", "kwh-engine-gpu0").split()
engine_running = len(eng) == 2 and eng[0] == "true"
engine_s = started(eng[1]) if engine_running else None


def after_boot(t):
    return int(round(t - boot_s)) if t is not None else -1


out = {"label": label, "up": int(up), "service": service, "svc_boot": after_boot(service_s),
       "engine": "running" if engine_running else "down", "eng_boot": after_boot(engine_s)}
print(f"Ubuntu up {span(up)}; kwh-host service {service}"
      + (f" (since {span(service_s - boot_s)} after Ubuntu started)" if service_s is not None else "")
      + f"; engine container {'running' if engine_running else 'not running'}"
      + (f" (since {span(engine_s - boot_s)} after)" if engine_s is not None else ""))

events = []
try:
    with open(os.path.join(kh, "events.jsonl")) as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
except OSError:
    pass
if window > 0:
    lo = now - window
    beats = [e["t"] for e in events if e.get("kind") in ("heartbeat", "heartbeat_failed") and e.get("t", 0) >= lo]
    accepted = sum(1 for e in events if e.get("kind") == "heartbeat" and e.get("t", 0) >= lo)
    gaps = [b - a for a, b in zip(beats, beats[1:])] + ([now - beats[-1]] if beats else [])
    gap = max(gaps) if beats else -1
    lead = beats[0] - lo if beats else -1
    out.update(beats=len(beats), accepted=accepted, failed=len(beats) - accepted, gap=int(round(gap)),
               lead=int(round(lead)))
    if beats:
        print(f"heartbeats in the last {span(window)}: {len(beats)} sent, {accepted} accepted; the first "
              f"{lead:.0f} s in, the longest gap {gap:.0f} s")
    else:
        print(f"heartbeats in the last {span(window)}: none")

platform = "down"
plat_line = "platform: not answering"
try:
    hs = json.load(open(os.path.join(d, "platform-hosts.json")))["hosts"]
    if hs:
        platform = hs[0]["state"]
        last = hs[0].get("last_beat_at")
        plat_line = f"platform: {platform}" + (f", last heartbeat {now - last:.0f} s ago" if last else "")
except (OSError, ValueError, KeyError):
    pass
out["platform"] = platform
print(plat_line)

# Ubuntu's clock against Windows': this sample, and the first one taken since this boot.
if winms > 0:
    boot_id = open("/proc/sys/kernel/random/boot_id").read().strip()
    first = None
    try:
        for row in open(clock_csv):
            f = row.strip().split(",")
            if len(f) == 5 and f[4] == boot_id and first is None:
                first = (float(f[1]), float(f[3]))
    except OSError:
        pass
    with open(clock_csv, "a") as f:
        f.write(f"{label},{winms / 1000:.3f},{rt0:.3f},{up0:.3f},{boot_id}\n")
    if first and winms / 1000 - first[0] >= 60:
        win_s, up_s = winms / 1000 - first[0], up0 - first[1]
        pct = (1 - up_s / win_s) * 100
        out.update(clock_pct=f"{pct:.1f}", clock_span=int(win_s))
        print(f"Ubuntu's clock: {up_s:.0f} s in {win_s:.0f} s of Windows time "
              f"({abs(pct):.1f}% {'slow' if pct >= 0 else 'fast'})")
print("RESULT " + " ".join(f"{k}={v}" for k, v in out.items()))
sys.exit(0 if service == "active" and engine_running else 1)
PY
  exit $?
fi

# --- collect: everything into one tarball, and the test host's service removed --------------------
if [ "${1:-}" = collect ]; then
  log "collect"
  {
    echo "== $(date -u +%FT%TZ)"
    echo "== ps -p 1"; ps -o pid=,etimes=,comm= -p 1
    echo "== systemctl --user status kwh-host"; timeout 20 systemctl --user status --no-pager kwh-host 2>&1 | head -20
    echo "== loginctl show-user"; loginctl show-user "$ME" -p Linger 2>&1
    echo "== docker ps -a"; timeout 20 docker ps -a 2>&1
  } > "$OUT/final-state.txt" 2>&1
  journalctl --user -u kwh-host --no-pager 2>/dev/null | tail -400 > "$OUT/service-journal.txt"
  timeout 30 kwh-host status > "$OUT/status-final.txt" 2>&1
  cp "$KH/events.jsonl" "$OUT/events-host.jsonl" 2>/dev/null
  cp "$KH/state.json" "$OUT/state-host.json" 2>/dev/null
  if hosts > "$OUT/platform-hosts-final.json" 2>/dev/null; then
    timeout 30 kwh-host events --remote --all --json --limit 1000 > "$OUT/events-platform-final.jsonl" 2>&1
  else
    rm -f "$OUT/platform-hosts-final.json"
  fi
  kwh-host service uninstall >> "$OUT/service.txt" 2>&1 && log "the test host's service is removed"
  pkill -f 'kwh-hos[t] mock-platform' 2>/dev/null; pkill -f 'fake_engine/serve[r].py' 2>/dev/null
  timeout 60 docker rm -f kwh-engine-gpu0 >/dev/null 2>&1
  win="$(windows_dir)"
  [ -n "$win" ] && [ -f "$win/wsl-keepalive.log" ] && cp "$win/wsl-keepalive.log" "$OUT/"
  tar czf "$HOME/wsl-check.tgz" -C "$OUT" .
  if [ -n "$win" ] && cp "$HOME/wsl-check.tgz" "$win/"; then
    log "results: $(wslpath -w "$win/wsl-check.tgz")"
  else
    log "results: ~/wsl-check.tgz"
  fi
  exit 0
fi

# --- 1. WSL, systemd, the installer ----------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1 || [ ! -x "$KH/venv/bin/kwh-host" ]; then
  log "1. WSL, systemd and the installer"
  {
    echo "== distro"; echo "${WSL_DISTRO_NAME:-?}"
    echo "== /proc/version"; cat /proc/version
    echo "== wslinfo"; wslinfo --version 2>&1 || echo "(no wslinfo)"
    echo "== os"; grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release
    echo "== /etc/wsl.conf"; cat /etc/wsl.conf 2>&1
    echo "== cpus and memory"; nproc; free -g
    echo "== clocksource"; cat /sys/devices/system/clocksource/clocksource0/{current,available}_clocksource
    echo "== pid 1"; ps -o comm= -p 1
    echo "== systemd"; systemctl is-system-running 2>&1
    echo "== nvidia-smi"; nvidia-smi 2>&1 | head -3
  } | tee "$OUT/wsl.txt"
  [ "$(ps -o comm= -p 1)" = systemd ] \
    || die "systemd is not running. Turn it on: printf '[boot]\\nsystemd=true\\n' | sudo tee -a /etc/wsl.conf ; then wsl --shutdown in PowerShell, open Ubuntu again and run this again"
  log "systemd: $(systemctl is-system-running 2>&1)"
  echo "The installer needs your Linux password for the system changes (Docker, the docker group, packages)."
  sudo -v || die "sudo did not work"
  curl -fsSL "https://raw.githubusercontent.com/Tim-cryptow/kwh-host/$REF/install.sh" \
    | bash -s -- --yes --skip-gpu-check --ref "$REF" 2>&1 | tee "$OUT/install.log"
  log "installer exit ${PIPESTATUS[1]}"
  if command -v docker >/dev/null 2>&1 && [ -x "$KH/venv/bin/kwh-host" ]; then
    log "installed. Now restart WSL so the docker group takes effect: in PowerShell run   wsl --shutdown"
    log "then open Ubuntu again and run   bash ~/wsl-check.sh   once more"
  else
    die "the installer did not finish; its output is in $OUT/install.log"
  fi
  exit 0
fi

# --- 2. the sandbox, then the service ---------------------------------------------------------------
id -nG | grep -qw docker \
  || die "this session is not in the docker group yet: run   wsl --shutdown   in PowerShell, open Ubuntu again, and run this again"
docker info >/dev/null 2>&1 || die "Docker is installed but not reachable: $(docker info 2>&1 | tail -1)"
log "2. the sandbox end to end, then the service"
log "Docker: $(docker info --format '{{.OperatingSystem}}, server {{.ServerVersion}}, cgroup v{{.CgroupVersion}}' 2>&1)"
echo "${WSL_DISTRO_NAME:-Ubuntu-24.04}" > "$OUT/distro"

# an earlier attempt: its service stops, its host state moves aside
if [ -f "$HOME/.config/systemd/user/kwh-host.service" ]; then
  kwh-host service uninstall >> "$OUT/service.txt" 2>&1; log "an earlier attempt's service removed"
fi
if [ -e "$KH/state.json" ] || [ -e "$KH/events.jsonl" ] || compgen -G "$OUT/probe-*" >/dev/null; then
  prev="$OUT/earlier-$(date -u +%H%M%S)"; mkdir -p "$prev"
  mv "$KH/state.json" "$KH"/events.jsonl* "$OUT"/probe-* "$prev/" 2>/dev/null
  log "an earlier attempt's host state and probes moved to $prev"
fi

if [ -d "$SRC/.git" ]; then
  git -C "$SRC" fetch -q --depth 1 origin "$REF" && git -C "$SRC" reset -q --hard FETCH_HEAD
else
  git clone -q --depth 1 -b "$REF" "https://github.com/Tim-cryptow/kwh-host" "$SRC"
fi
log "source: $(git -C "$SRC" log --oneline -1)"
# --network host: the build's pip install uses Ubuntu's own network. (Containers' DNS on WSL2 is not
# what this checks; the engine runs with no network at all.)
docker build --network host -q -t kwh-fake-engine:test "$SRC/tests/fake_engine" > "$OUT/fake-engine-build.txt" 2>&1 \
  || die "stand-in engine image did not build: $(tail -3 "$OUT/fake-engine-build.txt")"
log "stand-in engine image built"

# CI's script, unchanged: init, bench, register, run, the lockdown inspected, a buyer job, a
# re-benchmark, another job, stop. The venv's python has what the stand-in engine's server needs.
PATH="$KH/venv/bin:$PATH" bash "$SRC/scripts/ci-sandbox.sh" > "$OUT/sandbox.log" 2>&1
rc=$?
grep -E "^\[ci-sandbox\]|readonly=|rebench +(started|done)|^State" "$OUT/sandbox.log" | tee -a "$OUT/steps.log"
if [ "$rc" = 0 ]; then log "sandbox end to end: ok"; else log "sandbox end to end: FAILED (exit $rc; $OUT/sandbox.log)"; fi

log "the service: a host that stays up after this script ends"
pkill -f 'kwh-hos[t] mock-platform' 2>/dev/null; pkill -f 'fake_engine/serve[r].py' 2>/dev/null
cd "$SRC" || exit 1
setsid nohup "$KH/venv/bin/python" tests/fake_engine/server.py RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 \
  --port "$FAKE_PORT" --max-model-len 8192 > "$OUT/fake-engine.log" 2>&1 < /dev/null &
wait_http "http://127.0.0.1:$FAKE_PORT/health" || die "stand-in reference engine did not start"
setsid nohup kwh-host mock-platform --port "$PORT" --heartbeat 10 --challenge-every 60 --microbench-every 3600 \
  --accept-uncertified --no-tokenizer --challenges-from "http://127.0.0.1:$FAKE_PORT" > "$OUT/mock-platform.log" 2>&1 < /dev/null &
wait_http "$PLATFORM/healthz" || die "mock platform did not start"
kwh-host init --platform "$PLATFORM" --docker-image kwh-fake-engine:test --no-gpu --engine-memory 2g > "$OUT/init.json" 2>&1
kwh-host doctor > "$OUT/doctor.txt" 2>&1
grep -E "platform|Docker|GPU in containers|engine network" "$OUT/doctor.txt" | tee -a "$OUT/steps.log"
kwh-host bench > "$OUT/bench.txt" 2>&1 || die "bench failed: $(tail -3 "$OUT/bench.txt")"
kwh-host register > "$OUT/register.json" 2>&1 || die "register failed: $(cat "$OUT/register.json")"
echo "Keeping your user's services running without an open window (loginctl enable-linger) needs your password:"
sudo loginctl enable-linger "$ME" && log "lingering on"
kwh-host service install > "$OUT/service.txt" 2>&1; log "service install exit $?"
if wait_host "h.get('state') == 'live'" 60; then
  log "service: live"
else
  die "the service is not live after 3 minutes ($(systemctl --user is-active kwh-host 2>&1)); see $OUT/service.txt"
fi
kwh-host status | tee "$OUT/status-before.txt"

win="$(windows_dir)" || die "cannot find your Windows user folder from here (is Windows interop off?)"
cp "$SRC/scripts/wsl-keepalive.ps1" "$win/" || die "could not copy wsl-keepalive.ps1 to $win"
ps1="$(wslpath -w "$win/wsl-keepalive.ps1")"
log "stage 2 done; the keep-alive check is next: $ps1"
cat <<EOF

Stage 2 is done: the test host runs as a service. Keep this window open for now.

Next, the keep-alive check. Open a new PowerShell window (not as administrator) and run:

  powershell -ExecutionPolicy Bypass -File "$ps1" -Distro ${WSL_DISTRO_NAME:-Ubuntu-24.04}

It takes about 15 minutes, tells you when to close this window, and collects the results at the end.
EOF
