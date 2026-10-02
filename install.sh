#!/usr/bin/env bash
# kWh Exchange host client installer, for Ubuntu (or Debian) on x86-64, natively or inside WSL2.
#
#   curl -fsSL https://raw.githubusercontent.com/Tim-cryptow/kwh-host/main/install.sh | bash
#
# It checks the machine, asks before changing anything system-wide, and installs kwh-host for the
# current user in ~/.kwh-host/venv. What it may install, each only with your yes (or --yes):
#   - Docker Engine (Docker's own script, get.docker.com) and docker-group membership for you
#   - the NVIDIA Container Toolkit (NVIDIA's apt repository), so Docker can hand the GPU to the engine
#   - python3-venv and git, if missing
# It never installs a GPU driver: on Linux that needs a reboot, and on WSL2 the Windows driver does it.
#
# Options (pass through the pipe as: ... | bash -s -- --yes):
#   --yes             answer yes to every system change
#   --check           check only; change nothing
#   --ref REF         install this branch or tag of kwh-host (default: main)
#   --skip-gpu-check  continue without a usable NVIDIA GPU (CI, or preparing a machine)
set -euo pipefail

REPO="Tim-cryptow/kwh-host"
REF="main"
YES=0
CHECK=0
SKIP_GPU=0
KWH_DIR="${KWH_HOST_HOME:-$HOME/.kwh-host}"
VENV="$KWH_DIR/venv"
MIN_VRAM_MIB=15872

while [ $# -gt 0 ]; do
  case "$1" in
    --yes|-y) YES=1 ;;
    --check) CHECK=1 ;;
    --ref) REF="${2:?--ref needs a value}"; shift ;;
    --skip-gpu-check) SKIP_GPU=1 ;;
    -h|--help) sed -n '2,20p' "$0" 2>/dev/null || true; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
ok()   { printf '    ok   %s\n' "$*"; }
warn() { printf '    WARN %s\n' "$*"; }
die()  { printf '    FAIL %s\n' "$*" >&2; exit 1; }
need_change=()
failed=()

# ask "question": yes with --yes; no in --check mode or with no terminal to ask on
ask() {
  [ "$CHECK" = 1 ] && return 1
  [ "$YES" = 1 ] && return 0
  local reply=""
  { printf '    %s [y/N] ' "$1" > /dev/tty && read -r reply < /dev/tty; } 2>/dev/null || return 1
  case "$reply" in y|Y|yes|YES) return 0 ;; esac
  return 1
}

# change "description" cmd...: run it if allowed; otherwise record it as a step left to do
change() {
  local what="$1"; shift
  if ! ask "$what?"; then
    need_change+=("$what")
    return 1
  fi
  if "$@"; then
    return 0
  fi
  failed+=("$what")
  warn "failed: $what"
  return 1
}

say "kWh host client installer"
[ "$(id -u)" -ne 0 ] || die "run this as your normal user, not root (it uses sudo only for system packages)"
[ "$(uname -m)" = "x86_64" ] || die "x86-64 only (this machine is $(uname -m))"
. /etc/os-release 2>/dev/null || die "cannot read /etc/os-release"
case "${ID:-}" in ubuntu|debian) ok "$PRETTY_NAME" ;; *) warn "$PRETTY_NAME is untested; Ubuntu 22.04 or 24.04 is the supported path" ;; esac
WSL=0
if grep -qi microsoft /proc/version 2>/dev/null; then WSL=1; ok "running inside WSL2"; fi

# --- GPU ---------------------------------------------------------------------------------
say "NVIDIA GPU"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  line="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | head -1)"
  gpu_name="$(echo "$line" | cut -d, -f1 | xargs)"
  vram="$(echo "$line" | cut -d, -f2 | xargs)"
  ok "$gpu_name, $((vram / 1024)) GB, driver $(echo "$line" | cut -d, -f3 | xargs)"
  if [ "${vram%.*}" -lt "$MIN_VRAM_MIB" ]; then
    [ "$SKIP_GPU" = 1 ] && warn "under 16 GB: this card cannot serve the reference model" \
      || die "the reference model needs a GPU with 16 GB or more (this one has $((vram / 1024)) GB)"
  fi
elif [ "$SKIP_GPU" = 1 ]; then
  warn "no NVIDIA GPU visible (continuing: --skip-gpu-check)"
elif [ "$WSL" = 1 ]; then
  die "no NVIDIA GPU visible in WSL2. Install the current NVIDIA driver for Windows (it brings the GPU into WSL2; install nothing inside Linux), then run this again"
else
  die "no NVIDIA driver. Install it with: sudo ubuntu-drivers install, reboot, then run this again"
fi

# --- Docker ------------------------------------------------------------------------------
say "Docker"
USE_SG=0
# docker_cmd ARGS...: docker as this user, through the docker group if it was granted during this run
docker_cmd() { if [ "$USE_SG" = 1 ]; then sg docker -c "docker $(printf '%q ' "$@")"; else docker "$@"; fi; }
docker_state() {   # ok | denied | down
  local out
  if out="$(docker_cmd info 2>&1)"; then echo ok; return; fi
  if echo "$out" | grep -qi "permission denied"; then echo denied; else echo down; fi
}
has_systemd() { command -v systemctl >/dev/null 2>&1 && [ -d /run/systemd/system ]; }
start_docker() { if has_systemd; then sudo systemctl enable --now docker; else sudo service docker start; fi; }
restart_docker() { if has_systemd; then sudo systemctl restart docker; else sudo service docker restart; fi; }
if ! command -v docker >/dev/null 2>&1; then
  install_docker() { curl -fsSL https://get.docker.com | sudo sh; }
  change "Install Docker Engine (Docker's official script, needs sudo)" install_docker && ok "Docker installed" || true
fi
DESKTOP=""
if command -v docker >/dev/null 2>&1; then
  state="$(docker_state)"
  if [ "$state" = down ]; then
    change "Start the Docker daemon (needs sudo)" start_docker || true
    state="$(docker_state)"
  fi
  if [ "$state" = denied ]; then
    add_group() { sudo usermod -aG docker "$USER"; }
    if change "Let $USER use Docker without sudo (adds you to the docker group)" add_group; then
      USE_SG=1
      ok "added to the docker group: new logins have it, and this run uses it already"
      state="$(docker_state)"
    fi
  fi
  if [ "$state" = ok ]; then
    os_name="$(docker_cmd info --format '{{.OperatingSystem}}' 2>/dev/null || true)"
    ok "Docker $(docker_cmd version --format '{{.Server.Version}}' 2>/dev/null || echo '?') on $os_name"
    if echo "$os_name" | grep -q "Docker Desktop"; then
      DESKTOP=1
      warn "Docker Desktop cannot seal the engine off behind a Unix socket."
      warn "Recommended: Docker Engine inside WSL2 (HOST-CLIENT.md §12); otherwise init with --engine-transport tcp."
    fi

    say "NVIDIA Container Toolkit"
    runtimes="$(docker_cmd info --format '{{json .Runtimes}}' 2>/dev/null || true)"
    if echo "$runtimes" | grep -q nvidia || [ -e /etc/cdi/nvidia.yaml ] || [ -n "$DESKTOP" ]; then
      ok "Docker can hand the GPU to containers"
    else
      install_toolkit() {
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
          | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg &&
        curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
          | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
          | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null &&
        sudo apt-get update -q &&
        sudo apt-get install -y -q nvidia-container-toolkit &&
        sudo nvidia-ctk runtime configure --runtime=docker &&
        restart_docker
      }
      change "Install the NVIDIA Container Toolkit (NVIDIA's apt repository, needs sudo)" install_toolkit \
        && ok "toolkit installed; Docker configured for the GPU" || true
    fi
  else
    warn "Docker is not usable by $USER yet ($state)"
  fi
fi

# --- Python, git ---------------------------------------------------------------------------
say "Python and git"
missing=()
command -v python3 >/dev/null 2>&1 || missing+=(python3)
python3 -c 'import venv, ensurepip' >/dev/null 2>&1 || missing+=(python3-venv)
command -v git >/dev/null 2>&1 || missing+=(git)
if [ ${#missing[@]} -gt 0 ]; then
  install_py() { sudo apt-get update -q && sudo apt-get install -y -q "${missing[@]}"; }
  change "Install ${missing[*]} (apt, needs sudo)" install_py || true
fi
if command -v python3 >/dev/null 2>&1; then
  pyv="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || die "Python $pyv is too old; 3.10 or newer is needed"
  ok "Python $pyv"
fi

# --- kwh-host --------------------------------------------------------------------------
say "kwh-host"
if [ "$CHECK" = 1 ]; then
  [ -x "$VENV/bin/kwh-host" ] && ok "installed: $("$VENV/bin/kwh-host" --version)" || warn "not installed yet (would go to $VENV)"
elif python3 -c 'import venv, ensurepip' >/dev/null 2>&1 && command -v git >/dev/null 2>&1; then
  mkdir -p "$KWH_DIR"
  [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q --upgrade "kwh-host @ git+https://github.com/$REPO@$REF"
  mkdir -p "$HOME/.local/bin"
  ln -sf "$VENV/bin/kwh-host" "$HOME/.local/bin/kwh-host"
  ok "$("$VENV/bin/kwh-host" --version) in $VENV, linked from ~/.local/bin/kwh-host"
  case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) warn "$HOME/.local/bin is not on PATH in this shell; open a new terminal (or log in again)" ;; esac
else
  need_change+=("Install kwh-host (needs python3-venv and git first)")
fi

# --- summary ---------------------------------------------------------------------------
echo
if [ ${#failed[@]} -gt 0 ]; then
  say "Failed (see the output above):"
  for c in "${failed[@]}"; do echo "    - $c"; done
fi
if [ ${#need_change[@]} -gt 0 ]; then
  say "Not done (rerun with --yes, or answer y):"
  for c in "${need_change[@]}"; do echo "    - $c"; done
fi
if [ ${#failed[@]} -gt 0 ] || [ ${#need_change[@]} -gt 0 ]; then
  [ "$CHECK" = 1 ] && [ ${#failed[@]} -eq 0 ] && exit 0
  exit 1
fi
say "Ready. Next:"
cat <<EOF
    kwh-host init --platform <platform URL>${DESKTOP:+ --engine-transport tcp}
    kwh-host doctor            # every check should say ok
    kwh-host fetch             # the model (~9 GB, hash-checked) and the engine image (~10 GB)
    kwh-host bench             # the certified benchmark, ~10 minutes
    kwh-host register
    kwh-host service install   # runs in the background from now on, restarts on failure
EOF
if [ "$WSL" = 1 ]; then
  echo "    WSL2: keep the Ubuntu window (or a WSL session) open while hosting; see HOST-CLIENT.md §12."
fi
