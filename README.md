# kWh Host Client

The program a GPU owner installs to sell work on the kWh Exchange. It benchmarks the rig with [kwh-benchmark](https://github.com/Tim-cryptow/kwh-benchmark), registers the certified rate with the platform, proves the rig is live (heartbeat, platform-issued canaries, micro-benchmarks), and executes buyer jobs in a sandboxed copy of the certified engine. Units are minted by the platform against that liveness; the client never mints on its own.

## Install (Ubuntu, or Windows with WSL2)

```bash
curl -fsSL https://raw.githubusercontent.com/Tim-cryptow/kwh-host/main/install.sh | bash
```

It checks for an NVIDIA card with 16 GB or more, asks before installing Docker Engine and the NVIDIA Container Toolkit, runs a test container on the GPU (and repairs the toolkit if that fails), and installs `kwh-host` for your user. It also offers to keep the NVIDIA driver out of Ubuntu's automatic updates: an update under a running GPU stops it working until a reboot. Then:

```bash
kwh-host init --platform <platform URL>
kwh-host doctor            # driver, GPU, Docker, GPU in containers, image, checkpoint
kwh-host fetch             # the model at the locked revision, hash-checked, and the engine image
kwh-host bench             # the certified benchmark, in the sandbox
kwh-host register
kwh-host service install   # runs in the background, restarts on failure
kwh-host status            # state, rate, earnings, engine, GPU, last checks, reliability
kwh-host events            # what happened: challenges, jobs, re-benchmarks, state changes
```

On Windows, start with [docs/windows-wsl2.md](docs/windows-wsl2.md). The engine runs in a locked-down container: no network (the daemon reaches it through a Unix socket), read-only, no privileges, bounded memory and processes (HOST-CLIENT.md §3).

Build step 2 of 7. **[HOST-CLIENT.md](HOST-CLIENT.md)** is the scope: lifecycle, liveness, minting, jobs and their verification, the platform API contract, decisions D1–D6, D8 (context length) and D9 (challenge format), and the open one (D7 metering).

## Status

- [x] M0 — scope document, decisions recorded
- [x] M1 — daemon skeleton + mock platform: `init → bench → register → run` reaches **live**, answers challenges, mints, micro-benchmarks. Proven on a RunPod RTX 3090 on 2026-09-30 (see below).
- [x] M2 — jobs: the router dispatches to live hosts over their WebSocket and re-routes on failure; hosts return signed results with generated token ids; greedy outputs are verified after delivery by teacher-forced scoring under the reference model; a wrong-model host never receives work. Proven on a RunPod A40 on 2026-10-02 (see below).
- [ ] M3 — Docker sandbox, one-line install, WSL2 path. Proven on Linux on a rented RTX 4090 VM on 2026-10-05 (see below): the one-line install, then a certified benchmark inside the sandbox, live, buyer jobs, a 4-bit substitute caught, and the background service. CI runs the same flow without a GPU on every push. The Windows path without a GPU, on a Windows 11 laptop on 2026-10-06 (see below): the installer inside WSL2, the sandbox, the service, and keeping it running with no window open. Left: the GPU through WSL2.
- [x] M4 — reliability telemetry and re-benchmarking. Proven on a rented RTX 4090 VM on 2026-10-05 (see below): a driver change found by the daemon itself, a slowed card held and re-rated within 5 minutes and re-rated back when it recovered, and a frozen engine replaced in under 2 minutes. CI runs a re-benchmark in the real sandbox on every push.
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

## M2 on real hardware (A40, RunPod, 2026-10-02)

`scripts/pod-m2.sh` on a RunPod Secure A40 (48 GB, driver 580, CUDA 13; no 3090 or 4090 was available), bare-metal engine, mock platform on the same pod, its verifier pointed at the host's own engine. Everything it wrote is in [results/m2-a40-2026-10-02](results/m2-a40-2026-10-02/).

```
units/hour: 60.187   median job: 59.8132 s   stability: 0.00163   canary: PASS (8/8, largest delta 0.0479)   certified: YES
registered: host h_3da628a7cbe8, bucket I-1/60
challenge lock-208-1-41f63a90: pass (delta 0.02227, 433 ms) -> live              39 s after `kwh-host run`
micro-benchmark discarded: a job arrived while it ran
15 jobs completed (greedy, sampled, raw text, 12 × 4 requests six at a time), 0 failed, 3,578 tokens, 0.0507 units
job of 1,100 tokens against a 1,024-token engine: refused, no host fits
micro-benchmark: 59.81 u/h equivalent in 7.52s -> within tolerance
--- same host, engine restarted on a 4-bit substitute (AWQ-INT4) ---
platform: degraded (engine restarted; awaiting a challenge)
challenge lock-208-2-7f8af1f3: FAIL (delta 0.16455, 649 ms) -> degraded
buyer job while only the substitute is connected: failed, no live host
```

What it settled:

- **The job path works on a real engine.** Token ids in, signed token ids out, metered, nothing lost or retried.
- **A wrong model got no work.** The restart rule held the host in `degraded` before any challenge, the challenge failed at more than three times the tolerance, and the buyer's job found no host.
- **Per-request verification cannot be the rule.** The reference model, scoring its own greedy output on the same card and engine, disagrees with 2.9% of the tokens, so 13 of the 14 honest jobs verified at τ = 0.1 would have failed, and two jobs with identical prompts produced different text. Judged over a window of requests, the honest host and a 4-bit substitute separate cleanly. HOST-CLIENT.md §7 has the numbers and the rule.
- **The canary margin is thin on this card.** Its deltas reach 0.048 against a 0.05 tolerance, where the 3090, 4090 and A5000 scored 0.0000. The substitute still failed. Since D9 a challenge carries four continuations and passes on their mean delta, and the benchmark judges its canaries the same way (rc.6).
- **Context length does not change the rate (D8):** 60.230 units/hour at 1,024, 60.228 at 8,192. Hosts now certify and serve at 8,192 by default.
- One install snag: Ubuntu 24.04 ships `cryptography` 41.0.7 through apt, which pip cannot upgrade; kwh-host now accepts it.

## M3 on real hardware (RTX 4090, Vast.ai VM, 2026-10-05)

`scripts/vm-m3.sh` on a rented VM (Ubuntu 22.04, driver 580.95.05, CUDA 13.0, $0.51/hr; a whole VM, since RunPod pods cannot run Docker). It ran the one-line installer, then the Docker path for real: no bare metal anywhere, and a mock platform that accepts only Docker hosts with certified reports. Everything it wrote is in [results/m3-vast-4090-2026-10-05](results/m3-vast-4090-2026-10-05/).

```
installer: a test container sees the GPU        init: engine build vllm/vllm-openai:v0.30.0 for a CUDA 13.0 driver
units/hour: 101.508   median job: 35.4652 s   stability: 0.00155   canary: PASS (mean delta 0.0000)   certified: YES
live, job channel open                                                            42 s after `kwh-host run`
engine container: read-only, network none, cap_drop ALL, no-new-privileges, user 1002:1002, /hf read-only
  from inside: Network is unreachable / DNS fails / Read-only file system / CapEff 0000000000000000 / sees the RTX 4090
jobs: chat greedy (256 tokens), chat sampled, a 6,065-token prompt: completed; an 8,292-token job: refused
daemon stopped, engine container removed
--- same sandbox, the 4-bit substitute (AWQ-INT4) ---
platform: degraded (engine restarted; awaiting a challenge)
challenge: FAIL (mean delta 0.12852 over 4) -> degraded
service install -> live in 58 s; service uninstall -> removed
```

What it settled:

- **The sandbox costs nothing.** Bare metal, the same card model scored 100.56 units/hour on RunPod. Inside a container with no network, a read-only root, no capabilities and an ordinary uid, it scored 101.51 here (101.59 on a second run), at the same 315 units per electric kWh. The report is the benchmark's second RTX 4090 row and its first certified run in Docker mode.
- **The lockdown holds as built (§3),** checked from outside with `docker inspect` and from inside with `docker exec`.
- **The installer has to try the GPU, not read settings.** This VM's template promised the NVIDIA Container Toolkit, but none of its programs were installed, while Docker still listed an `nvidia` runtime. The first attempt's engine died with `could not select device driver`. The installer now runs a test container on the GPU and installs the toolkit when that fails, which it did here. `doctor` runs `nvidia-smi` inside the engine image. An engine that dies at launch now shows its own error.
- Two faults in the test script, not the client, cost a second attempt. Both are written up in the results folder.

## M4 on real hardware (RTX 4090, Vast.ai VM, 2026-10-05)

A report stops counting after seven days, or at once when the driver, the engine image or the GPU changes, or after two micro-benchmark misses. The platform then holds the host in `degraded` (no jobs, no mint) until a new report arrives, and the daemon produces one:

1. It tells the platform it is re-benchmarking and turns away new jobs.
2. It lets the jobs in flight finish.
3. It benchmarks on a fresh engine and uploads the signed report.
4. It serves again; the next passed challenge makes it live at the new rate.

Heartbeats continue throughout, so the host is never seen as offline. An engine that stops answering, or will not start, is retried. The platform counts what happens to each host, in 5-minute buckets kept for a week, and step 3 will score from those counts. HOST-CLIENT.md §5 has the rules.

`scripts/vm-m4.sh` put all of it to a real card: a rented VM in Israel with an RTX 4090 and Ubuntu 22.04, $1.45 for both runs. The files are in [results/m4-vast-4090-2026-10-05](results/m4-vast-4090-2026-10-05/).

```
at start   driver changed: 580.95.05 benchmarked, 580.178.04 now -> re-benchmarked before serving: 101.88 units/hour
slow       graphics clock locked at 945 MHz; micro-benchmarks 72.06, 71.81 -> held; buyer job refused (no live host)
           re-benchmarked: 70.99 units/hour, live again 4 min 52 s after the hold; next micro-benchmark 71.48
full       clock released; micro-benchmarks 100.95, 100.94 -> held; re-benchmarked: 101.52, live again after 3 min 52 s
freeze     docker pause on the engine -> restart decided after 44 s; live again in a new locked-down container at 115 s
```

What it settled:

- **The re-benchmark loop works on a real card, both ways.** A card slowed by 30% was caught, held, and re-rated in under 5 minutes. When it sped back up, it was re-rated upward. Buyers were refused while the host was held, not given a host whose rate no longer held.
- **A driver update is caught by the daemon itself.** It compared its report with `nvidia-smi` before serving anything, and re-benchmarked first. The rate came back unchanged (101.83 to 101.88); the rule is there for when it would not.
- **A slower clock makes cheaper units.** At 945 MHz the card made 30% fewer units per hour but 28% more per electric kWh: 415 against 324. Which of the two a host should be rated on is step 3's question.
- **Ubuntu's automatic updates broke the first attempt.** Ten minutes after boot they replaced the NVIDIA driver (580.95.05 to 580.178.04) under the running VM. The old kernel module stayed loaded, so nothing new could use the GPU until a reboot. The engine, stopped for its re-benchmark, could not start again, and the daemon exited, which left the host silent. The fixes:
  - The daemon now reports and retries an engine that will not start.
  - `doctor` and `kwh-host status` say to reboot.
  - The installer offers to keep the driver out of the automatic updates.

## The Windows path on a real PC (no GPU, 2026-10-06)

`scripts/wsl-check.sh` (in Ubuntu) and `scripts/wsl-keepalive.ps1` (in PowerShell) ran on a Windows 11 laptop with no NVIDIA card, starting from a PC without WSL. The files are in [results/wsl-windows11-2026-10-06](results/wsl-windows11-2026-10-06/).

```
installer      Docker Engine 29.8.2 inside Ubuntu 24.04 (WSL 3.0.1), the docker group, kwh-host: exit 0, 2 min 23 s
sandbox        CI's script end to end: bench, register, live, lockdown, a buyer job, a re-benchmark: ok
service        live 3 s after kwh-host service install
guide's task   "Access is denied" for an ordinary user; registers with -User. Defaults: no start on battery, 72 h limit
no window      with the task: 3 min, 17/17 heartbeats accepted. With instanceIdleTimeout=-1 and no task: kept running
logon          after wsl --shutdown, the task alone started Ubuntu; the host's engine was running ~6 s later
default        WSL stopped Ubuntu 17 s after its last session ended
```

What it settled:

- **Everything but the GPU works on Windows.** That covers the installer inside WSL2, the sandbox, the service, and a host that survives a closed window and comes back at logon.
- **The guide's keep-alive step did not, as written.** An ordinary user cannot register the task without `-User`. Its defaults would also stop it on a laptop. Step 5 is rewritten: a `.wslconfig` setting keeps Ubuntu running, and a task only starts it at logon.
- **Ubuntu's clock ran about 5% slow there.** Windows pulls it back every half minute or so, by a jump of about 2 seconds. The benchmark times itself on that clock, so it would read about 5% high in WSL2 on that laptop. That has to be checked on the PC with an NVIDIA card before Windows hosts are rated (HOST-CLIENT.md §12).

## Try it without a GPU

```bash
pip install -e ".[dev]"
python -m pytest -q          # 94 tests: signing, envelope, execution, the vLLM stream parser, verifier,
                             # platform state machine, re-benchmarking and reliability counters, status,
                             # sandbox, and jobs end to end over a real local server

# the sandbox for real, with a stand-in engine (needs Docker; what CI runs on every push)
docker build -t kwh-fake-engine:test tests/fake_engine
KWH_TEST_ENGINE_IMAGE=kwh-fake-engine:test python -m pytest -q tests/test_sandbox_docker.py
scripts/ci-sandbox.sh        # init, bench, register, run, a buyer job and a re-benchmark, all through the sandbox

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
kwh-host events
```

`mock-platform --rebench-every 120` makes reports expire after two minutes, so the re-benchmark cycle can be watched (`kwh-host events --kind rebench`).

Without `--mock-challenges` the mock platform serves the real lock canaries, the mock engine fails them, and the host sits in `degraded` with nothing minted and no jobs: the wrong-model guard doing its job.

## On a real GPU (RunPod pod, bare metal)

```bash
pip install git+https://github.com/Tim-cryptow/kwh-host           # pulls kwh-bench
kwh-host mock-platform --port 9000 --allow-bare-metal &            # stand-in until step 4
kwh-host init --platform http://127.0.0.1:9000 --engine bare-metal --allow-bare-metal   # serves 8,192-token context (D8)
kwh-host bench          # kwh-bench run at that context, signed; ~5 min on a 4090
kwh-host register       # certified report -> host_id, rate, bucket
kwh-host run            # engine up, heartbeat every 30 s, challenges, micro-benchmark every 30 min, jobs
kwh-host submit --prompt "Explain photosynthesis in two sentences."     # as a buyer, from another shell
```

On a pod the mock platform finds the reference tokenizer (transformers comes with vLLM), so `submit --prompt` works with plain text; `--verify-url http://127.0.0.1:8000` on the mock platform verifies every delivered greedy output against a reference engine.

Docker mode (`--engine docker`, the default and the only mode the real platform accepts, D4) launches `vllm/vllm-openai:v0.30.0` with the pinned flags. That build needs a CUDA 13 driver (580 or newer); on an older driver `init` picks the same version's `v0.30.0-cu129` build, and `doctor` flags a mismatch. Both are pinned by registry digest (`sha256:8a69ffad…` and `sha256:a67f8f18…`), so a tag pushed again cannot change the engine under a host. In bare-metal mode the same applies to the vLLM wheel; see the benchmark's `scripts/runpod.sh`.

## What talks to what

```
kwh-host run ──▶ engine in the sandbox, over a Unix socket ← kwh_host.sandbox (vLLM, pinned flags)
      │  ├── heartbeat: engine health + GPU sample       POST /v1/hosts/{id}/heartbeat
      │  │   + what the host saw since the last one
      │  ├── challenge: score 4 platform continuations   POST /v1/hosts/{id}/liveness
      │  ├── micro-benchmark when idle (1/8 job)         POST /v1/hosts/{id}/microbench
      │  ├── re-benchmark when the report expires         POST /v1/hosts/{id}/reports
      │  ├── jobs: token ids in, signed token ids out    WS   /v1/hosts/{id}/jobs
      │  └── events.jsonl (kwh-host events)               GET  /v1/hosts/{id}, /events (kwh-host status)
      ▼
platform (mock in kwh_host/platform/mock.py; real one is step 4)
      ├── verifies signatures + report (kwh-bench verify)
      ├── state machine: registered → live ⇄ degraded → offline; holds a host whose report expired
      ├── accrual per accepted live heartbeat, integer mints, nothing ahead, nothing late
      ├── reliability counters per host, 5-minute buckets, a week of them
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
  sandbox.py            the engine container: no network, read-only, no privileges, bounded
  fetch.py              the checkpoint at the locked revision, hash-checked; the engine image
  doctor.py             readiness checks with a fix for each failure
  service.py            systemd user service
  gpu.py                nvidia-smi sample + foreign-process detection
  bench.py              full run, micro-benchmark, challenge scoring (all via kwh_bench)
  rebench.py            when a report stops counting (shared with the platform)
  jobspec.py            job envelope, validation, provisional metering (shared with the platform)
  jobs.py               job execution: vLLM streaming executor, toy executor, signed results
  daemon.py             the run loop: heartbeat, job channel, re-benchmark, engine restart
  events.py             events.jsonl, the daemon's own log
  status.py             what kwh-host status and kwh-host events print
  mockmodel.py          deterministic toy language model (tests, GPU-free demo)
  experiments.py        verifier calibration on real cards
  platform/client.py    the §8 contract, client side
  platform/mock.py      the §8 contract, server side (in memory, FastAPI), router
  platform/verifier.py  teacher-forced greedy verification (step 3 prototype)
  cli.py                init | fetch | doctor | bench | register | run | status | events | service | mock-platform | submit
install.sh              the one-line installer (Ubuntu, WSL2)
docs/windows-wsl2.md    hosting on Windows
tests/                  identity, jobs, verifier, platform state machine, re-benchmark and reliability, status,
                        router, sandbox (+ fake_engine/ for Docker)
scripts/                real-GPU milestone runs (pod-m1.sh, pod-m2.sh, vm-m3.sh, vm-m4.sh), the Windows check
                        (wsl-check.sh in Ubuntu, then wsl-keepalive.ps1 in PowerShell) and ci-sandbox.sh
results/                what those runs wrote, one folder per run
```

Apache-2.0, same as the benchmark.
