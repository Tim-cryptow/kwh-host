#!/usr/bin/env bash
# M1 on a RunPod pod: bare-metal engine + the mock platform on the same pod (HOST-CLIENT.md §10, M1).
# Not the production shape (D4 wants Docker, D6 puts the platform in the cloud); it is how the loop
# is proven on real hardware.
#
#   bash <(curl -fsSL https://raw.githubusercontent.com/Tim-cryptow/kwh-host/main/scripts/pod-m1.sh) [--beats N]
#
# Leaves: /workspace/kwh-host-home/report.json (signed, certified), /workspace/kwh-host-run.log,
# /workspace/kwh-host-status.json, engine logs, mock-platform.log.
set -euo pipefail
BEATS=12
while [ $# -gt 0 ]; do case "$1" in --beats) BEATS="$2"; shift;; esac; shift; done
VLLM_VERSION="${VLLM_VERSION:-0.30.0}"
export KWH_HOST_HOME="${KWH_HOST_HOME:-/workspace/kwh-host-home}"
cd /workspace

# Same rule as kwh-benchmark/scripts/runpod.sh: PyPI vLLM needs a CUDA 13 driver; older drivers get cu129.
DRIVER_CUDA="$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\)\.\([0-9]*\).*/\1.\2/p' | head -1)"
if [ -n "$DRIVER_CUDA" ] && [ "${DRIVER_CUDA%%.*}" -lt 13 ]; then
  echo "driver supports CUDA ${DRIVER_CUDA}; installing vllm==${VLLM_VERSION}+cu129"
  pip install -q uv
  uv pip install --system --break-system-packages -q "vllm==${VLLM_VERSION}+cu129" huggingface_hub \
    --torch-backend=cu129 --extra-index-url "https://wheels.vllm.ai/${VLLM_VERSION}/cu129" --index-strategy unsafe-best-match
else
  pip install -q "vllm==${VLLM_VERSION}" huggingface_hub
fi
pip install -q "git+https://github.com/Tim-cryptow/kwh-host@main"
python -c "import vllm, kwh_bench, kwh_host; print('vllm', vllm.__version__, 'kwh-bench', kwh_bench.__version__, 'kwh-host', kwh_host.__version__)"

mkdir -p "$KWH_HOST_HOME"
if ! curl -fs http://127.0.0.1:9000/healthz >/dev/null 2>&1; then
  nohup kwh-host mock-platform --port 9000 --allow-bare-metal --heartbeat 15 --challenge-every 60 --microbench-every 300 \
    > /workspace/mock-platform.log 2>&1 &
  sleep 3
fi
curl -fs http://127.0.0.1:9000/healthz && echo

kwh-host init --platform http://127.0.0.1:9000 --engine bare-metal --allow-bare-metal
kwh-host bench --engine-log /workspace/engine-bench.log
kwh-host register
kwh-host run --engine-log /workspace/engine-run.log --beats "$BEATS" 2>&1 | tee /workspace/kwh-host-run.log
kwh-host status --json | tee /workspace/kwh-host-status.json
