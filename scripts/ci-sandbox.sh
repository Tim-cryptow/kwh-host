#!/usr/bin/env bash
# The Docker path end to end without a GPU: init (docker) -> bench -> register -> run -> a buyer
# job, with tests/fake_engine standing in for vLLM inside the real sandbox. The platform here
# enforces D4 (no --allow-bare-metal). CI runs this on every push; so can anyone with Docker:
#
#   docker build -t kwh-fake-engine:test tests/fake_engine && scripts/ci-sandbox.sh
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${KWH_TEST_ENGINE_IMAGE:-kwh-fake-engine:test}"
WORK="$(mktemp -d)"
export KWH_HOST_HOME="$WORK/h"
PLATFORM_PORT=19431
FAKE_PORT=18999
PIDS=()
log() { echo "[ci-sandbox] $*"; }
cleanup() {
  for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done
  docker rm -f kwh-engine-gpu0 >/dev/null 2>&1 || true
}
trap cleanup EXIT
wait_http() { for _ in $(seq 1 120); do curl -fs "$1" >/dev/null 2>&1 && return 0; sleep 1; done; return 1; }

log "a fake engine on the host scores the platform's challenge continuations (the reference node's job)"
python3 tests/fake_engine/server.py RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 --port "$FAKE_PORT" \
  --max-model-len 8192 > "$WORK/fake.log" 2>&1 &
PIDS+=($!)
wait_http "http://127.0.0.1:$FAKE_PORT/health"

log "mock platform: Docker hosts only (D4), uncertified reports accepted (the fake fails its canaries)"
kwh-host mock-platform --port "$PLATFORM_PORT" --heartbeat 2 --challenge-every 6 --microbench-every 3600 \
  --accept-uncertified --no-tokenizer --challenges-from "http://127.0.0.1:$FAKE_PORT" > "$WORK/platform.log" 2>&1 &
PIDS+=($!)
wait_http "http://127.0.0.1:$PLATFORM_PORT/healthz"

log "init: docker engine, Unix socket, no GPU (CI)"
kwh-host init --platform "http://127.0.0.1:$PLATFORM_PORT" --docker-image "$IMAGE" --no-gpu --engine-memory 2g

log "bench in the sandbox"
kwh-host bench > "$WORK/bench.txt" 2>&1 || { cat "$WORK/bench.txt"; exit 1; }
python3 - "$KWH_HOST_HOME/report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
args = r["engine"]["launch_args"]
assert r["engine"]["launch_mode"] == "docker", r["engine"]["launch_mode"]
for flag in ("--read-only", "--uds", "--max-model-len"):
    assert flag in args, flag
assert args[args.index("--network") + 1] == "none" and args[args.index("--cap-drop") + 1] == "ALL"
assert not any(a.startswith("/home/") for a in args), "home directory leaked into the report"
assert r["signature"] and r["score"]["units_per_hour"] > 0
print("report: docker, sandboxed, signed;", r["score"]["units_per_hour"], "u/h (fake engine)")
PY

log "register"
kwh-host register

log "run: the daemon starts the engine in the sandbox, answers challenges, opens the job channel"
kwh-host run > "$WORK/run.log" 2>&1 &
RUN=$!
PIDS+=("$RUN")
for _ in $(seq 1 120); do
  if curl -fs "http://127.0.0.1:$PLATFORM_PORT/v1/mock/hosts" | python3 -c \
    "import json,sys; h=json.load(sys.stdin)['hosts']; sys.exit(0 if h and h[0]['state']=='live' and h[0]['jobs_channel'] else 1)"; then
    break
  fi
  sleep 1
done
curl -fs "http://127.0.0.1:$PLATFORM_PORT/v1/mock/hosts" | python3 -c \
  "import json,sys; h=json.load(sys.stdin)['hosts'][0]; print('host', h['host_id'], h['state'], 'last challenge', h['last_challenge']); assert h['state']=='live'"

log "the running engine, as Docker sees it"
docker inspect kwh-engine-gpu0 --format \
  'readonly={{.HostConfig.ReadonlyRootfs}} network={{.HostConfig.NetworkMode}} capdrop={{.HostConfig.CapDrop}} user={{.Config.User}} pids={{.HostConfig.PidsLimit}}' \
  | tee "$WORK/inspect.txt"
grep -q "readonly=true network=none capdrop=\[ALL\]" "$WORK/inspect.txt"

log "a buyer job"
echo '{"requests": [{"prompt_token_ids": [128000, 9906, 1917], "max_tokens": 16, "temperature": 0.0}]}' > "$WORK/job.json"
kwh-host submit --platform "http://127.0.0.1:$PLATFORM_PORT" --file "$WORK/job.json" --out "$WORK/job-out.json"

log "stop: the container goes with the daemon"
kill -TERM "$RUN"
for _ in $(seq 1 60); do kill -0 "$RUN" 2>/dev/null || break; sleep 1; done
if docker inspect kwh-engine-gpu0 >/dev/null 2>&1; then echo "engine container still there"; exit 1; fi
cat "$WORK/run.log"
log "ok"
