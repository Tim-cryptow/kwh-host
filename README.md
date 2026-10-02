# kWh Host Client

The program a GPU owner installs to sell work on the kWh Exchange. It benchmarks the rig with [kwh-benchmark](https://github.com/Tim-cryptow/kwh-benchmark), registers the certified rate with the platform, proves the rig is live (heartbeat, platform-issued canaries, micro-benchmarks), and executes buyer jobs in a sandboxed copy of the certified engine. Units are minted by the platform against that liveness; the client never mints on its own.

Build step 2 of 7. **[HOST-CLIENT.md](HOST-CLIENT.md)** is the scope: lifecycle, liveness, minting, jobs and their verification, the platform API contract, decisions D1–D6 and the open ones (D7 metering, D8 context length).

## Status

- [x] M0 — scope document, decisions recorded
- [x] M1 — daemon skeleton + mock platform: `init → bench → register → run` reaches **live**, answers challenges, mints, micro-benchmarks. Proven on a RunPod RTX 3090 on 2026-09-30 (see below).
- [ ] M2 — jobs: the router dispatches to live hosts over their WebSocket and re-routes on failure; hosts return signed results with generated token ids; greedy outputs are verified after delivery by teacher-forced scoring under the reference model; a wrong-model host never receives work. Built and tested end to end in process (45 tests); the real-GPU run is what closes it.
- [ ] M3 — Docker sandbox hardening, one-line install, WSL2 path
- [ ] M4 — reliability telemetry, `kwh-host status`
- [ ] M5 — real platform (step 4), stake deposit

## M1 on real hardware (RTX 3090, RunPod, 2026-09-30)

`scripts/pod-m1.sh` on a fresh community 3090 (driver 570, CUDA 12.8, $0.22/hr), bare-metal engine, mock platform on the same pod:

```
pre-flight: 100% VRAM free, GPU 0.0% busy, 28.5 W -> idle
units/hour: 78.287   median job: 45.9847 s   stability: 0.01994     canary: PASS (8/8)   certified: YES
registered: host h_5dc9b78d5e10, bucket I-1/60
challenge lock-208-1-89173137: pass (delta 0.0, 279 ms) -> live
micro-benchmark: 79.60 u/h equivalent in 5.65s -> within tolerance
minted 1 unit(s), balance 1 … balance 3          (12 heartbeats at 15 s, 3 challenges passed, 12/12 accepted)
```

The certified report it produced is the RTX 3090 row in the benchmark's `results/`, signed by the host's key. Two things the run taught: inside a container `nvidia-smi` reports host-namespace PIDs, so the GPU sample now treats a PID it cannot see in `/proc` as *unattributable* rather than *foreign* (a foreign process is one that is visible and not ours); and `kwh-host status` signed `{}` for a GET whose body is empty, which the platform correctly rejected — fixed, with a test.

## Try it without a GPU

```bash
pip install -e ".[dev]"
python -m pytest -q          # 45 tests: signing, envelope, execution, the vLLM stream parser, verifier,
                             # platform state machine, and jobs end to end over a real local server

# terminal 1: the mock platform; --mock-challenges lets a mock-engine host pass its challenges
kwh-host mock-platform --port 9000 --heartbeat 5 --challenge-every 10 --microbench-every 300 \
  --accept-uncertified --allow-bare-metal --mock-challenges --no-tokenizer

# terminal 2: a host with the benchmark's mock engine
export KWH_HOST_HOME=/tmp/kwh-host-demo
kwh-host init --platform http://127.0.0.1:9000 --engine bare-metal --allow-bare-metal
kwh-host bench --mock-engine
kwh-host register
kwh-host run --mock-engine

# terminal 3: be the buyer (token ids, since the mock has no tokenizer without transformers)
echo '{"requests": [{"prompt_token_ids": [128000, 9906], "max_tokens": 24, "temperature": 0.0}]}' > /tmp/job.json
kwh-host submit --platform http://127.0.0.1:9000 --file /tmp/job.json
kwh-host status
```

Without `--mock-challenges` the mock platform serves the real lock canaries, the mock engine fails them, and the host sits in `degraded` with nothing minted and no jobs: the wrong-model guard doing its job.

## On a real GPU (RunPod pod, bare metal)

```bash
pip install git+https://github.com/Tim-cryptow/kwh-host           # pulls kwh-bench
kwh-host mock-platform --port 9000 --allow-bare-metal &            # stand-in until step 4
kwh-host init --platform http://127.0.0.1:9000 --engine bare-metal --allow-bare-metal
kwh-host bench          # kwh-bench run, signed; ~5 min on a 4090
kwh-host register       # certified report -> host_id, rate, bucket
kwh-host run            # engine up, heartbeat every 30 s, challenges, micro-benchmark every 30 min, jobs
kwh-host submit --prompt "Explain photosynthesis in two sentences."     # as a buyer, from another shell
```

On a pod the mock platform finds the reference tokenizer (transformers comes with vLLM), so `submit --prompt` works with plain text; `--verify-url http://127.0.0.1:8000` on the mock platform verifies every delivered greedy output against a reference engine.

Docker mode (`--engine docker`, the default and the only mode the real platform accepts, D4) launches `vllm/vllm-openai:v0.30.0` with the pinned flags. Hosts with a CUDA 12.x driver need the cu129 build in bare-metal mode; see the benchmark's `scripts/runpod.sh`.

## What talks to what

```
kwh-host run ──▶ engine (vLLM, pinned flags)            ← kwh_bench.engines.VLLMEngine
      │  ├── heartbeat: engine health + GPU sample       POST /v1/hosts/{id}/heartbeat
      │  ├── challenge: score platform continuation      POST /v1/hosts/{id}/liveness
      │  ├── micro-benchmark when idle (1/8 job)         POST /v1/hosts/{id}/microbench
      │  └── jobs: token ids in, signed token ids out    WS   /v1/hosts/{id}/jobs
      ▼
platform (mock in kwh_host/platform/mock.py; real one is step 4)
      ├── verifies signatures + report (kwh-bench verify)
      ├── state machine: registered → live ⇄ degraded → offline
      ├── accrual per accepted live heartbeat, integer mints, nothing ahead, nothing late
      ├── router: tokenize, dispatch to a live host that fits, re-route on failure, meter
      └── verifier (step 3 prototype): teacher-force delivered greedy outputs through the reference
```

## Layout

```
HOST-CLIENT.md          scope, lifecycle, liveness, API contract, decisions
kwh_host/
  config.py             ~/.kwh-host: config, identity, report, state
  identity.py           ed25519 keypair; request, report and result signing
  engine.py             daemon-owned vLLM (docker | bare-metal), health, own PIDs
  gpu.py                nvidia-smi sample + foreign-process detection
  bench.py              full run, micro-benchmark, challenge scoring (all via kwh_bench)
  jobspec.py            job envelope, validation, provisional metering (shared with the platform)
  jobs.py               job execution: vLLM streaming executor, toy executor, signed results
  daemon.py             the run loop: heartbeat + job channel
  mockmodel.py          deterministic toy language model (tests, GPU-free demo)
  experiments.py        verifier calibration on real cards
  platform/client.py    the §8 contract, client side
  platform/mock.py      the §8 contract, server side (in memory, FastAPI), router
  platform/verifier.py  teacher-forced greedy verification (step 3 prototype)
  cli.py                init | bench | register | run | status | mock-platform | submit | experiment
tests/                  identity, jobs, verifier, platform state machine, router end to end
```

Apache-2.0, same as the benchmark.
