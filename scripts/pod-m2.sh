#!/usr/bin/env bash
# M2 on a RunPod pod (HOST-CLIENT.md §10): jobs on real hardware, a wrong-model host, the
# verifier calibration (reference vs a 4-bit substitute) and the D8 context-length rate check.
# Bare-metal engine and the mock platform on the same pod: how the loop is proven, not the
# production shape. Everything lands in /workspace/m2/ (summary.json at the end).
#
#   bash <(curl -fsSL https://raw.githubusercontent.com/Tim-cryptow/kwh-host/main/scripts/pod-m2.sh)
set -uo pipefail
OUT=/workspace/m2
mkdir -p "$OUT"
export KWH_HOST_HOME="${KWH_HOST_HOME:-/workspace/kwh-host-home}"
VLLM_VERSION="${VLLM_VERSION:-0.30.0}"
REF=RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8
REV=024e24cbe4153670f747383ea3265d0fb197c727
AWQ=hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
PLATFORM=http://127.0.0.1:9000
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/steps.log"; }

wait_http() { for _ in $(seq 1 "${2:-300}"); do curl -fs "$1" >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }
wait_gpu_free() {
  for _ in $(seq 1 90); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 1500 ] && return 0; sleep 2
  done; return 1
}
stop_group() { kill -TERM -- "-$1" 2>/dev/null; for _ in $(seq 1 90); do kill -0 "$1" 2>/dev/null || break; sleep 1; done
               kill -KILL -- "-$1" 2>/dev/null; wait_gpu_free; }
start_vllm() {  # model port logfile [extra args]; MML overrides --max-model-len; prints the process-group id
  local model=$1 port=$2 logf=$3; shift 3
  setsid nohup vllm serve "$model" --dtype auto --max-model-len "${MML:-1024}" --max-num-seqs 32 \
    --no-enable-prefix-caching --seed 0 --gpu-memory-utilization 0.90 --port "$port" --host 127.0.0.1 "$@" \
    > "$logf" 2>&1 < /dev/null &
  echo $!
}
host_json() { curl -fs "$PLATFORM/v1/mock/hosts"; }
host_is() {  # python condition on h (the first host)
  host_json | python3 -c "import json,sys; d=json.load(sys.stdin)['hosts']; h=d[0] if d else {}; sys.exit(0 if ($1) else 1)" 2>/dev/null
}
wait_host() { for _ in $(seq 1 "${2:-300}"); do host_is "$1" && return 0; sleep 3; done; return 1; }

log "0. install"
cd /workspace
DRIVER_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\)\.\([0-9]*\).*/\1.\2/p' | head -1)"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu,power.draw --format=csv > "$OUT/gpu.txt"
if [ -n "$DRIVER_CUDA" ] && [ "${DRIVER_CUDA%%.*}" -lt 13 ]; then
  log "driver supports CUDA ${DRIVER_CUDA}; installing vllm==${VLLM_VERSION}+cu129"
  pip install -q uv
  uv pip install --system --break-system-packages -q "vllm==${VLLM_VERSION}+cu129" huggingface_hub \
    --torch-backend=cu129 --extra-index-url "https://wheels.vllm.ai/${VLLM_VERSION}/cu129" --index-strategy unsafe-best-match
else
  pip install -q "vllm==${VLLM_VERSION}" huggingface_hub
fi
pip install -q "git+https://github.com/Tim-cryptow/kwh-host@main"
python3 -c "import vllm, kwh_bench, kwh_host; print('vllm', vllm.__version__, 'kwh-bench', kwh_bench.__version__, 'kwh-host', kwh_host.__version__)" | tee -a "$OUT/steps.log"

log "1. mock platform (lock canaries, reference tokenizer, verifier against the host's own engine)"
setsid nohup kwh-host mock-platform --port 9000 --allow-bare-metal --heartbeat 10 --challenge-every 60 \
  --microbench-every 120 --verify-url http://127.0.0.1:8000 --verify-fraction 1.0 > "$OUT/mock-platform.log" 2>&1 < /dev/null &
wait_http "$PLATFORM/healthz" 60 && curl -fs "$PLATFORM/healthz" | tee -a "$OUT/steps.log"; echo

log "2. host: init, certified benchmark, register"
kwh-host init --platform "$PLATFORM" --engine bare-metal --allow-bare-metal > "$OUT/init.json"
kwh-host bench --engine-log "$OUT/engine-bench.log" > "$OUT/bench.txt" 2>&1; log "bench exit $?"
cp "$KWH_HOST_HOME/report.json" "$OUT/report.json" 2>/dev/null
kwh-host register > "$OUT/register.json" 2>&1; log "register: $(tr -d '\n ' < "$OUT/register.json")"

log "3. daemon with the reference model"
setsid nohup kwh-host run --engine-log "$OUT/engine-run.log" > "$OUT/run.log" 2>&1 < /dev/null &
RUN=$!
wait_host "h.get('state') == 'live' and h.get('jobs_channel')" 300 && log "live, job channel open" || log "NOT live"

log "4. jobs"
python3 - "$OUT" <<'PY'
import json, sys
from kwh_host.experiments import CHAT_PROMPTS
out = sys.argv[1]
burst = [{"requests": [{"messages": [{"role": "user", "content": CHAT_PROMPTS[(4 * j + k) % len(CHAT_PROMPTS)]}],
                        "max_tokens": 64, "temperature": 0.0} for k in range(4)]} for j in range(12)]
json.dump(burst, open(f"{out}/burst.json", "w"))
json.dump({"requests": [{"prompt_token_ids": [128000] + [791] * 999, "max_tokens": 100, "temperature": 0.0}]},
          open(f"{out}/too-long.json", "w"))
PY
kwh-host submit --platform "$PLATFORM" --max-tokens 128 --temperature 0 \
  --prompt "Explain how photosynthesis works to a ten-year-old." \
  --prompt "Write a Python function that checks whether a string is a palindrome." \
  --prompt "What is the difference between TCP and UDP?" \
  --prompt "Give me a simple recipe for jollof rice." --out "$OUT/jobs-greedy.json" > "$OUT/jobs-greedy.txt" 2>&1
log "greedy job: exit $?"
kwh-host submit --platform "$PLATFORM" --max-tokens 96 --temperature 0.8 --seed 7 \
  --prompt "Write a haiku about rain in Lagos." --prompt "Write a limerick about a cat who loves coffee." \
  --out "$OUT/jobs-sampled.json" > "$OUT/jobs-sampled.txt" 2>&1
log "sampled job: exit $?"
kwh-host submit --platform "$PLATFORM" --raw --max-tokens 48 --temperature 0 --prompt "The three primary colours are" \
  --out "$OUT/jobs-raw.json" > "$OUT/jobs-raw.txt" 2>&1
log "raw-text job: exit $?"
kwh-host submit --platform "$PLATFORM" --file "$OUT/burst.json" --concurrent 6 --out "$OUT/jobs-burst.json" > "$OUT/jobs-burst.txt" 2>&1
log "burst (12 jobs x 4 requests, 6 at a time): exit $?"
kwh-host submit --platform "$PLATFORM" --file "$OUT/too-long.json" --timeout 10 --out "$OUT/jobs-too-long.json" > "$OUT/jobs-too-long.txt" 2>&1
log "too-long job (expected to fail: 1,100 tokens > 1,024): exit $?"
sleep 15
host_json > "$OUT/hosts-after-jobs.json"
kill -TERM "$RUN"; for _ in $(seq 1 90); do kill -0 "$RUN" 2>/dev/null || break; sleep 1; done; wait_gpu_free
log "daemon stopped"

log "5. the same host, now serving a 4-bit substitute"
setsid nohup kwh-host run --model "$AWQ" --engine-log "$OUT/engine-awq.log" > "$OUT/run-awq.log" 2>&1 < /dev/null &
RUN=$!
wait_host "(h.get('last_challenge') or {}).get('pass') is False and h.get('jobs_channel')" 600 \
  && log "substitute failed its challenge" || log "no failed challenge seen"
host_json > "$OUT/hosts-wrong-model.json"
kwh-host submit --platform "$PLATFORM" --timeout 20 --prompt "Hello" --out "$OUT/jobs-wrong-model.json" > "$OUT/jobs-wrong-model.txt" 2>&1
log "job while only the substitute is connected (expected to fail): exit $?"
kill -TERM "$RUN"; for _ in $(seq 1 90); do kill -0 "$RUN" 2>/dev/null || break; sleep 1; done; wait_gpu_free
log "daemon stopped"

log "6. verifier calibration: reference vs substitute, scored by the reference"
G=$(start_vllm "$REF" 8100 "$OUT/vllm-ref-1.log" --revision "$REV"); wait_http http://127.0.0.1:8100/health 300
kwh-host experiment generate --url http://127.0.0.1:8100 --out "$OUT/gen-ref.json" --label reference 2>&1 | tee -a "$OUT/steps.log"
kwh-host experiment score --url http://127.0.0.1:8100 --in "$OUT/gen-ref.json" --out "$OUT/score-ref.json" 2>&1 | tee -a "$OUT/steps.log"
stop_group "$G"
G=$(start_vllm "$AWQ" 8200 "$OUT/vllm-awq.log"); wait_http http://127.0.0.1:8200/health 300
kwh-host experiment generate --url http://127.0.0.1:8200 --prompts-from "$OUT/gen-ref.json" --out "$OUT/gen-awq.json" --label awq-int4 2>&1 | tee -a "$OUT/steps.log"
stop_group "$G"
G=$(start_vllm "$REF" 8100 "$OUT/vllm-ref-2.log" --revision "$REV"); wait_http http://127.0.0.1:8100/health 300
kwh-host experiment score --url http://127.0.0.1:8100 --in "$OUT/gen-awq.json" --out "$OUT/score-awq.json" 2>&1 | tee -a "$OUT/steps.log"
stop_group "$G"

log "7. D8: reference job rate at --max-model-len 1024 vs 8192 (attached, so uncertified; same session)"
for M in 1024 8192; do
  G=$(MML=$M start_vllm "$REF" 8300 "$OUT/vllm-mml$M.log" --revision "$REV"); wait_http http://127.0.0.1:8300/health 300
  kwh-bench run --engine vllm --server-url http://127.0.0.1:8300 --ignore-preflight --runs 3 --out "$OUT/mml$M.json" \
    > "$OUT/mml$M.txt" 2>&1
  log "max-model-len $M: $(grep -m1 'units/hour' "$OUT/mml$M.txt")"
  stop_group "$G"
done

log "8. summary"
python3 - "$OUT" <<'PY' | tee -a "$OUT/steps.log"
import json, sys, os
o = sys.argv[1]
def load(name):
    try:
        return json.load(open(os.path.join(o, name)))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
def jobs(name):
    d = load(name)
    if isinstance(d, dict):
        return d
    return [{"status": r.get("status"), "host": r.get("host_id"), "units": r.get("units"), "ms": r.get("latency_ms"),
             "attempts": [a.get("outcome") for a in r.get("attempts", [])], "reason": r.get("reason"),
             "tokens": (r.get("usage") or {}).get("completion_tokens"),
             "verified": (r.get("verification") or {}).get("pass"), "checked": (r.get("verification") or {}).get("checked"),
             "max_gap": max([q.get("max_gap") or 0 for q in (r.get("verification") or {}).get("requests", [])] or [None])}
            for r in d]
rep = load("report.json")
s = {
    "bench": {"units_per_hour": (rep.get("score") or {}).get("units_per_hour"), "certified": rep.get("certified"),
              "canary_max_delta": max([x["delta"] or 0 for x in (rep.get("canary") or {}).get("results", [])] or [None]),
              "report_sha256": rep.get("report_sha256")},
    "jobs": {n: jobs(f"jobs-{n}.json") for n in ("greedy", "sampled", "raw", "burst", "too-long", "wrong-model")},
    "host_after_jobs": [{k: h.get(k) for k in ("host_id", "state", "jobs", "balance")} for h in load("hosts-after-jobs.json").get("hosts", [])],
    "host_wrong_model": [{k: h.get(k) for k in ("host_id", "state", "reasons", "last_challenge", "jobs")} for h in load("hosts-wrong-model.json").get("hosts", [])],
    "verifier": {name: {k: d.get(k) for k in ("generated_by", "scored_by", "all", "by_kind")} for name, d in
                 (("reference", load("score-ref.json")), ("awq_int4", load("score-awq.json")))},
    "d8": {m: {"units_per_hour": (load(f"mml{m}.json").get("score") or {}).get("units_per_hour"),
               "stability": (load(f"mml{m}.json").get("score") or {}).get("stability")} for m in ("1024", "8192")},
}
json.dump(s, open(os.path.join(o, "summary.json"), "w"), indent=1)
print(json.dumps(s, indent=1)[:6000])
PY
log "done"
