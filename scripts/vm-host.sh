#!/usr/bin/env bash
# A rented GPU machine made a host of a kWh Exchange platform in one command. Given the platform's
# burst token, it also becomes the platform's reference node (`kwh-host burst`, HOST-CLIENT.md §4).
#
#   curl -fsSL https://raw.githubusercontent.com/Tim-cryptow/kwh-host/main/scripts/vm-host.sh -o vm-host.sh
#   bash vm-host.sh https://<platform>                           # a host
#   KWH_BURST_TOKEN=... bash vm-host.sh https://<platform>       # a host that also runs the bursts
#
# In order: the installer (Docker, the NVIDIA Container Toolkit, kwh-host), the model, the first burst
# (with a token), the certified benchmark, registration (it prints this host's key and waits while the
# operator invites it), the service, and (with a token) a daily burst at 04:10 machine time that
# pauses the service for the minutes it takes. Like vm-m3.sh: as root it creates a user, kwh, with
# passwordless sudo and runs as that user; the work runs in the background and the command follows
# its output, so a dropped connection does not stop it and the same command follows it again. Run
# it again later and it does what is left: a registered host is not benchmarked again.
set -uo pipefail
REF="${KWH_REF:-main}"
SELF="$(readlink -f "$0")"
PLATFORM="${1:-${KWH_PLATFORM_URL:-}}"
usage() { echo "usage: [KWH_BURST_TOKEN=...] bash vm-host.sh https://<platform>"; exit 2; }

if [ "$(id -u)" -eq 0 ]; then
  [ -n "$PLATFORM" ] || usage
  if ! id kwh >/dev/null 2>&1; then
    useradd -m -s /bin/bash kwh || exit 1
    echo "kwh ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/kwh && chmod 0440 /etc/sudoers.d/kwh
  fi
  if [ -n "${KWH_BURST_TOKEN:-}" ]; then          # into a file only kwh can read, not onto a command line
    install -d -o kwh -g kwh /home/kwh/.kwh-host
    (umask 077 && printf '%s\n' "$KWH_BURST_TOKEN" > /home/kwh/.kwh-host/burst-token) || exit 1
    chown kwh:kwh /home/kwh/.kwh-host/burst-token
  fi
  # a new file, renamed into place: a setup already running from the old one keeps reading it intact
  install -o kwh -g kwh -m 0755 "$SELF" /home/kwh/vm-host.sh.new && mv -f /home/kwh/vm-host.sh.new /home/kwh/vm-host.sh || exit 1
  exec sudo -iu kwh env KWH_REF="$REF" bash /home/kwh/vm-host.sh "$PLATFORM"
fi

OUT="$HOME/kwh-setup"
TOKEN_FILE="$HOME/.kwh-host/burst-token"
if [ "${KWH_SETUP_RUN:-}" != 1 ]; then
  [ -n "$PLATFORM" ] || usage
  sudo -n true 2>/dev/null || { echo "this needs passwordless sudo; run it as root and it sets up a user that has it"; exit 1; }
  mkdir -p "$OUT" "$HOME/.kwh-host"
  if [ -n "${KWH_BURST_TOKEN:-}" ]; then
    (umask 077 && printf '%s\n' "$KWH_BURST_TOKEN" > "$TOKEN_FILE") || exit 1
  fi
  running() { [ -n "$1" ] && grep -qs vm-host "/proc/$1/cmdline"; }    # a reused pid is not the setup
  pid="$(cat "$OUT/pid" 2>/dev/null)"
  if running "$pid"; then
    echo "the setup is already running; following it (Ctrl+C stops following, not the setup)"
  else
    rm -f "$OUT/pid"
    [ -s "$OUT/console.log" ] && mv "$OUT/console.log" "$OUT/console.$(date +%s).log"
    KWH_SETUP_RUN=1 setsid nohup env -u KWH_BURST_TOKEN bash "$SELF" "$PLATFORM" > "$OUT/console.log" 2>&1 < /dev/null &
    for _ in $(seq 1 20); do [ -s "$OUT/pid" ] && break; sleep 0.5; done
    echo "started; following it (Ctrl+C stops following, not the setup; the same command follows it again)"
  fi
  pid="$(cat "$OUT/pid" 2>/dev/null)"
  [ -n "$pid" ] || { cat "$OUT/console.log"; echo "the setup did not start"; exit 1; }
  tail -n +1 --pid="$pid" -f "$OUT/console.log"
  exit 0
fi
echo $$ > "$OUT/pid"
export PATH="$HOME/.local/bin:$PATH"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
fail() { log "STOPPED: $*"; exit 1; }
# as_docker CMD...: run with the docker group, which the installer may have granted after this login began
as_docker() { if docker info >/dev/null 2>&1; then "$@"; else sg docker -c "$(printf '%q ' "$@")"; fi; }
registered() { kwh-host status --local --json 2>/dev/null | python3 -c "import json,sys; sys.exit(0 if json.load(sys.stdin).get('host_id') else 1)"; }
# The user's systemd runs the service. If it started before the installer added this user to the
# docker group, it lacks the group and the engine cannot start: then it is restarted.
manager_has_docker() {
  local gid pid
  gid="$(getent group docker | cut -d: -f3)"; pid="$(pgrep -u "$(id -u)" -x systemd | head -1)"
  [ -n "$gid" ] && [ -n "$pid" ] && awk -v g="$gid" '/^Groups:/ { for (i = 2; i <= NF; i++) if ($i == g) f = 1 } END { exit !f }' "/proc/$pid/status"
}

log "kwh-host setup on $(hostname) for $PLATFORM"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || fail "no NVIDIA GPU visible"
df -h "$HOME" | tail -1

log "1. installer: Docker, the NVIDIA Container Toolkit and kwh-host, as needed"
curl -fsSL "https://raw.githubusercontent.com/Tim-cryptow/kwh-host/$REF/install.sh" | bash -s -- --yes --ref "$REF" \
  || fail "the installer failed (above)"

log "2. init"
as_docker kwh-host init --platform "$PLATFORM" > "$OUT/init.json" || fail "init failed"
KEY="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["public_key"])' "$OUT/init.json")"
log "this host's public key: $KEY"

log "3. the model at the locked revision, hash-checked, and the engine image"
as_docker kwh-host fetch > "$OUT/fetch.json" || fail "fetch failed (above)"
as_docker kwh-host doctor || fail "doctor found a problem (above)"

if [ -s "$TOKEN_FILE" ]; then
  log "4. a burst: fresh challenges for the platform's pool (no host goes live without them)"
  as_docker kwh-host burst --pause-service || log "the burst did not finish (above); continuing, the daily one tries again"
fi

if registered; then
  log "5. registered already: no new benchmark"
else
  log "5. the certified benchmark, in the sandbox"
  as_docker kwh-host bench > "$OUT/bench.txt" 2>&1
  tail -n 20 "$OUT/bench.txt"
  python3 - "$HOME/.kwh-host/report.json" <<'PY' || fail "the benchmark did not certify (reasons above)"
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except (OSError, ValueError) as e:
    sys.exit(f"no report: {e}")
print("certified:", r.get("certified"), "at", (r.get("score") or {}).get("units_per_hour"), "units/hour")
for reason in r.get("certified_reasons") or []:
    print("  -", reason)
sys.exit(0 if r.get("certified") else 1)
PY
  log "6. register; if the platform takes invited hosts only, this waits for the operator to invite $KEY"
  kwh-host register --wait 1440 || fail "registration failed (above)"
fi

log "7. the service: starts with the machine, restarts after a failure"
sudo loginctl enable-linger "$USER"
XDG_RUNTIME_DIR="/run/user/$(id -u)"
export XDG_RUNTIME_DIR
for _ in $(seq 1 20); do [ -S "$XDG_RUNTIME_DIR/bus" ] && break; sleep 1; done
if ! manager_has_docker; then
  sudo systemctl restart "user@$(id -u).service"
  for _ in $(seq 1 20); do [ -S "$XDG_RUNTIME_DIR/bus" ] && manager_has_docker && break; sleep 1; done
fi
kwh-host service install || fail "service install failed (above)"

if [ -s "$TOKEN_FILE" ]; then
  log "8. a burst every day at 04:10 machine time ($(date +%Z)); the service pauses for it"
  command -v crontab >/dev/null || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cron >/dev/null
  line="10 4 * * * $HOME/.local/bin/kwh-host burst --pause-service >> $HOME/.kwh-host/burst.log 2>&1"
  if { crontab -l 2>/dev/null | grep -v 'kwh-host burst'; echo "$line"; } | crontab -; then
    crontab -l | grep 'kwh-host burst'
  else
    log "could not install the daily burst"
  fi
fi

log "9. waiting for the platform to put this host live"
st=""
for _ in $(seq 1 90); do
  st="$(kwh-host status --json 2>/dev/null | python3 -c "import json,sys; print((json.load(sys.stdin).get('platform_view') or {}).get('state') or '')" 2>/dev/null)"
  [ "$st" = live ] && break
  sleep 10
done
kwh-host status
if [ "$st" = live ]; then
  log "done: live on $PLATFORM"
else
  log "not live yet (state: ${st:-unknown}); kwh-host status and kwh-host events show why"
fi
