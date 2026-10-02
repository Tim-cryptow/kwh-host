#!/usr/bin/env bash
# M3 on a rented GPU machine (a whole VM, so Docker works; RunPod pods cannot run Docker):
# the one-line installer, then the Docker path for real. Fetch (hash-checked), a certified
# benchmark in the sandbox, registration with a platform that accepts Docker hosts only, the
# daemon live with buyer jobs, the sandbox inspected from outside, a 4-bit substitute caught,
# and the systemd service.
#
#   curl -fsSL https://raw.githubusercontent.com/Tim-cryptow/kwh-host/main/scripts/vm-m3.sh -o vm-m3.sh && bash vm-m3.sh
#
# As root (Vast.ai logs you in as root) it creates a normal user, kwh, with passwordless sudo and
# runs everything as that user, the way a host runs the client; as a normal user it needs
# passwordless sudo. The test runs in the background and the command follows its output: if the
# connection drops, the test carries on, and the same command picks up following it again.
# Everything lands in ~/m3 of the user running it (summary.json last); m3-out.tgz, with all of
# it, also goes to the login user's home.
set -uo pipefail
REF="${KWH_REF:-main}"
SELF="$(readlink -f "$0")"

if [ "$(id -u)" -eq 0 ]; then
  if ! id kwh >/dev/null 2>&1; then
    useradd -m -s /bin/bash kwh || exit 1
    echo "kwh ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/kwh && chmod 0440 /etc/sudoers.d/kwh
  fi
  # a new file, renamed into place: a test already running from the old one keeps reading it intact
  install -o kwh -g kwh -m 0755 "$SELF" /home/kwh/vm-m3.sh.new && mv -f /home/kwh/vm-m3.sh.new /home/kwh/vm-m3.sh || exit 1
  exec sudo -iu kwh env KWH_REF="$REF" KWH_M3_COPY_TO="$HOME" bash /home/kwh/vm-m3.sh
fi

OUT="$HOME/m3"
if [ "${KWH_M3_RUN:-}" != 1 ]; then
  sudo -n true 2>/dev/null || { echo "this needs passwordless sudo; run it as root and it sets up a user that has it"; exit 1; }
  mkdir -p "$OUT"
  running() { [ -n "$1" ] && grep -qs vm-m3 "/proc/$1/cmdline"; }    # a zombie or a reused pid is not the test
  pid="$(cat "$OUT/pid" 2>/dev/null)"
  if running "$pid"; then
    echo "the test is already running; following it (Ctrl+C stops following, not the test)"
  elif [ -f "$HOME/m3-out.tgz" ]; then
    tail -n 30 "$OUT/console.log"; echo "the test has finished: ~/m3-out.tgz"; exit 0
  else
    if [ -s "$OUT/console.log" ]; then           # an earlier attempt that stopped part way: keep it aside
      mv "$OUT" "$OUT.stopped-$(date +%s)" && mkdir -p "$OUT"
      rm -f "$HOME/.kwh-host/config.json" "$HOME/.kwh-host/state.json"
    fi
    rm -f "$OUT/pid"
    KWH_M3_RUN=1 setsid nohup bash "$SELF" > "$OUT/console.log" 2>&1 < /dev/null &
    for _ in $(seq 1 20); do [ -s "$OUT/pid" ] && break; sleep 0.5; done
    echo "started; following it (Ctrl+C stops following, not the test; the same command follows it again)"
  fi
  pid="$(cat "$OUT/pid" 2>/dev/null)"
  [ -n "$pid" ] || { cat "$OUT/console.log"; echo "the test did not start"; exit 1; }
  tail -n +1 --pid="$pid" -f "$OUT/console.log"
  exit 0
fi
echo $$ > "$OUT/pid"
AWQ=hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
PORT=9000
PLATFORM="http://127.0.0.1:$PORT"
export PATH="$HOME/.local/bin:$PATH"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/steps.log"; }
# as_docker CMD...: run with the docker group, which the installer may have granted after this login began
as_docker() { if docker info >/dev/null 2>&1; then "$@"; else sg docker -c "$(printf '%q ' "$@")"; fi; }
hosts() { curl -fs "$PLATFORM/v1/mock/hosts"; }
host_is() { hosts | python3 -c "import json,sys; d=json.load(sys.stdin)['hosts']; h=d[0] if d else {}; sys.exit(0 if ($1) else 1)" 2>/dev/null; }
wait_host() { for _ in $(seq 1 "${2:-300}"); do host_is "$1" && return 0; sleep 3; done; return 1; }
wait_http() { for _ in $(seq 1 120); do curl -fs "$1" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }
stop_pid() { kill -TERM "$1" 2>/dev/null; for _ in $(seq 1 120); do kill -0 "$1" 2>/dev/null || return 0; sleep 1; done; kill -KILL "$1" 2>/dev/null; }

# leftovers of an earlier attempt, if any: its platform, daemon and engine container
pkill -f "kwh-host mock-platform" 2>/dev/null; pkill -f "kwh-host run" 2>/dev/null
as_docker docker rm -f kwh-engine-gpu0 >/dev/null 2>&1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee "$OUT/gpu.txt"
nvidia-smi > "$OUT/nvidia-smi.txt" 2>&1
grep -o "CUDA Version: [0-9.]*" "$OUT/nvidia-smi.txt" | tee -a "$OUT/gpu.txt"
df -h / "$HOME" | tee "$OUT/disk.txt"; free -g | tee -a "$OUT/disk.txt"

log "0. installer (--yes: Docker, NVIDIA Container Toolkit, kwh-host as needed)"
curl -fsSL "https://raw.githubusercontent.com/Tim-cryptow/kwh-host/$REF/install.sh" | bash -s -- --yes --ref "$REF" 2>&1 | tee "$OUT/install.log"
log "installer exit ${PIPESTATUS[1]}"
"$HOME/.kwh-host/venv/bin/pip" install -q transformers 2>&1 | tail -1   # the mock platform's tokenizer (buyer-side, not the host)

log "1. init, doctor, fetch"
as_docker kwh-host init --platform "$PLATFORM" | tee "$OUT/init.json"
as_docker kwh-host doctor | tee "$OUT/doctor-before-fetch.txt"
T0=$(date +%s)
as_docker kwh-host fetch > "$OUT/fetch.json" 2> "$OUT/fetch.log"; log "fetch exit $? in $(( $(date +%s) - T0 ))s"
as_docker kwh-host doctor | tee "$OUT/doctor.txt"; log "doctor exit ${PIPESTATUS[0]}"

log "2. mock platform: Docker hosts only (D4), certified reports only, the engine's tokenizer for buyers"
HF_HOME="$HOME/.kwh-host/hf" HF_HUB_OFFLINE=1 setsid nohup kwh-host mock-platform --port "$PORT" --heartbeat 10 \
  --challenge-every 60 --microbench-every 120 > "$OUT/mock-platform.log" 2>&1 < /dev/null &
wait_http "$PLATFORM/healthz" && curl -fs "$PLATFORM/healthz" | tee -a "$OUT/steps.log"; echo

log "3. certified benchmark inside the sandbox"
as_docker kwh-host bench > "$OUT/bench.txt" 2>&1; log "bench exit $?"
cp "$HOME/.kwh-host/report.json" "$OUT/report.json" 2>/dev/null
grep -E "units/hour|canary|certified" "$OUT/bench.txt" | tee -a "$OUT/steps.log"
as_docker kwh-host register > "$OUT/register.json" 2>&1; log "register: $(tr -d '\n ' < "$OUT/register.json")"

log "4. the daemon, live in the sandbox"
as_docker kwh-host run > "$OUT/run.log" 2>&1 &
RUN=$!
wait_host "h.get('state') == 'live' and h.get('jobs_channel')" 300 && log "live, job channel open" || log "NOT live"
docker inspect kwh-engine-gpu0 > "$OUT/engine-inspect.json" 2>/dev/null || as_docker docker inspect kwh-engine-gpu0 > "$OUT/engine-inspect.json"
python3 - "$OUT" <<'PY' | tee -a "$OUT/steps.log"
import json, sys
i = json.load(open(f"{sys.argv[1]}/engine-inspect.json"))[0]
hc = i["HostConfig"]
print(json.dumps({"readonly_rootfs": hc["ReadonlyRootfs"], "network": hc["NetworkMode"], "cap_drop": hc["CapDrop"],
                  "security_opt": hc["SecurityOpt"], "pids_limit": hc["PidsLimit"], "memory_gib": round(hc["Memory"] / 2**30, 1),
                  "user": i["Config"]["User"], "devices": [d.get("DeviceIDs") for d in hc.get("DeviceRequests") or []],
                  "mounts": {m["Destination"]: ("rw" if m["RW"] else "ro") for m in i["Mounts"]}}))
PY
probe() { as_docker docker exec kwh-engine-gpu0 python3 -c "$1" 2>&1 | tail -1; }
{
  echo "network out:  $(probe "import socket; socket.create_connection(('1.1.1.1', 53), 3); print('CONNECTED')")"
  echo "dns:          $(probe "import socket; print(socket.gethostbyname('huggingface.co'))")"
  echo "write /usr:   $(probe "open('/usr/x','w'); print('WROTE')")"
  echo "write /hf:    $(probe "open('/hf/x','w'); print('WROTE')")"
  echo "uid, CapEff:  $(probe "import os; print(os.getuid(), open('/proc/self/status').read().split('CapEff:')[1].split()[0])")"
  echo "gpu inside:   $(as_docker docker exec kwh-engine-gpu0 nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | tail -1)"
} | tee "$OUT/sandbox-probes.txt" | tee -a "$OUT/steps.log"

log "5. buyer jobs"
kwh-host submit --platform "$PLATFORM" --max-tokens 128 --temperature 0 \
  --prompt "Explain how photosynthesis works to a ten-year-old." \
  --prompt "Write a Python function that checks whether a string is a palindrome." \
  --out "$OUT/jobs-greedy.json" > "$OUT/jobs-greedy.txt" 2>&1; log "greedy job: exit $?"
kwh-host submit --platform "$PLATFORM" --max-tokens 64 --temperature 0.8 --seed 7 \
  --prompt "Write a haiku about rain in Lagos." --out "$OUT/jobs-sampled.json" > "$OUT/jobs-sampled.txt" 2>&1; log "sampled job: exit $?"
python3 -c "import json; json.dump({'requests': [{'prompt_token_ids': [128000] + [791] * 6000, 'max_tokens': 64, 'temperature': 0.0}]}, open('$OUT/long.json', 'w'))"
kwh-host submit --platform "$PLATFORM" --file "$OUT/long.json" --out "$OUT/jobs-long.json" > "$OUT/jobs-long.txt" 2>&1
log "6,065-token job (needs D8's context): exit $?"
python3 -c "import json; json.dump({'requests': [{'prompt_token_ids': [128000] + [791] * 8191, 'max_tokens': 100, 'temperature': 0.0}]}, open('$OUT/too-long.json', 'w'))"
kwh-host submit --platform "$PLATFORM" --file "$OUT/too-long.json" --timeout 10 --out "$OUT/jobs-too-long.json" > "$OUT/jobs-too-long.txt" 2>&1
log "too-long job (8,292 > 8,192, expected to fail): exit $?"
sleep 15
hosts > "$OUT/hosts-after-jobs.json"
stop_pid "$RUN"
as_docker docker inspect kwh-engine-gpu0 >/dev/null 2>&1 && log "ENGINE CONTAINER LEFT BEHIND" || log "daemon stopped, engine container removed"

log "6. a 4-bit substitute in the same sandbox"
as_docker kwh-host fetch --model "$AWQ" --no-image > "$OUT/fetch-awq.json" 2>> "$OUT/fetch.log"; log "fetch AWQ exit $?"
as_docker kwh-host run --model "$AWQ" > "$OUT/run-awq.log" 2>&1 &
RUN=$!
wait_host "(h.get('last_challenge') or {}).get('pass') is False" 600 && log "substitute failed its challenge" || log "no failed challenge seen"
hosts > "$OUT/hosts-wrong-model.json"
stop_pid "$RUN"

log "7. systemd user service"
sudo loginctl enable-linger "$USER"
XDG_RUNTIME_DIR="/run/user/$(id -u)"
export XDG_RUNTIME_DIR
for _ in $(seq 1 20); do [ -S "$XDG_RUNTIME_DIR/bus" ] && break; sleep 1; done
as_docker kwh-host service install > "$OUT/service.txt" 2>&1; log "service install exit $?"
wait_host "h.get('state') == 'live'" 300 && log "service: live" || log "service: NOT live"
systemctl --user status --no-pager kwh-host.service > "$OUT/service-status.txt" 2>&1
kwh-host service uninstall >> "$OUT/service.txt" 2>&1; log "service uninstalled"

log "8. summary"
python3 - "$OUT" <<'PY' | tee "$OUT/summary.json"
import json, os, sys
o = sys.argv[1]
def load(n):
    try:
        return json.load(open(os.path.join(o, n)))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
def text(n):
    try:
        return open(os.path.join(o, n)).read()
    except OSError:
        return ""
rep = load("report.json")
jobs = {n: [{k: r.get(k) for k in ("status", "units", "latency_ms", "reason")} | {"tokens": (r.get("usage") or {}).get("completion_tokens")}
            for r in (load(f"jobs-{n}.json") if isinstance(load(f"jobs-{n}.json"), list) else [])]
        for n in ("greedy", "sampled", "long", "too-long")}
print(json.dumps({
    "gpu": text("gpu.txt").strip().splitlines()[1:],
    "engine_image": load("init.json").get("engine_image"),
    "fetch": load("fetch.json"),
    "doctor": text("doctor.txt").strip().splitlines(),
    "bench": {"units_per_hour": (rep.get("score") or {}).get("units_per_hour"), "certified": rep.get("certified"),
              "reasons": rep.get("certified_reasons"), "canary_mean": (rep.get("canary") or {}).get("mean_delta"),
              "launch_mode": (rep.get("engine") or {}).get("launch_mode"), "report_sha256": rep.get("report_sha256")},
    "sandbox_probes": text("sandbox-probes.txt").strip().splitlines(),
    "jobs": jobs,
    "host_after_jobs": [{k: h.get(k) for k in ("state", "jobs", "last_challenge")} for h in load("hosts-after-jobs.json").get("hosts", [])],
    "wrong_model": [{k: h.get(k) for k in ("state", "reasons", "last_challenge")} for h in load("hosts-wrong-model.json").get("hosts", [])],
    "service": text("service.txt").strip().splitlines(),
}, indent=1))
PY
tar czf "$HOME/m3-out.tgz.part" -C "$OUT" .
if [ -n "${KWH_M3_COPY_TO:-}" ] && [ "$KWH_M3_COPY_TO" != "$HOME" ]; then
  sudo cp "$HOME/m3-out.tgz.part" "$KWH_M3_COPY_TO/m3-out.tgz" && log "copied to $KWH_M3_COPY_TO/m3-out.tgz"
fi
mv "$HOME/m3-out.tgz.part" "$HOME/m3-out.tgz" && log "done: ~/m3 and ~/m3-out.tgz"
