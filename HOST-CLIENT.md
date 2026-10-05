# kWh Host Client — scope (v0)

**Build step:** 2 of 7 (primer §10). **Upstream:** [kwh-benchmark](https://github.com/Tim-cryptow/kwh-benchmark) 1.0.0-rc.6, series I-1.
**Definition of done (primer):** Linux + Windows. Docker sandbox, benchmark run, liveness heartbeat, stake deposit, mint. Reports units/hour and reliability.

This document draws the line for the first cut. The host client is the program a GPU owner installs to turn a rig into supply. Everything it does is in service of one rule from the primer: *units are minted only against live capacity*. The client's job is to prove capacity is live, continuously, cheaply, and in a way the platform can check.

---

## 1. Scope line

Two of the five nouns in the definition of done, **stake deposit** and **mint**, depend on the unit rails (step 4), which are not decided. v0 builds everything that does not depend on that decision, and builds the mint path up to the platform's door.

| In v0 | Deferred |
| --- | --- |
| Install on Linux; Windows via WSL2 (§8) | Native Windows service / tray app |
| Run the Grade I benchmark, produce a certified report | — |
| Register the rig with the platform (signed report) | Stake deposit (needs rails, §9 D1) |
| Engine sandbox: the certified vLLM image serving the reference model | Any model other than the Grade I reference |
| Heartbeat + platform-issued liveness challenges | — |
| Periodic micro-benchmark and full re-benchmark | Score computation (step 3; client reports raw) |
| Mint *requests* in the heartbeat; platform mints | Wallet, token, resale, expiry, USDC payout (step 4) |
| Job execution over an outbound WebSocket, signed results with token ids | Production verifier, slashing (step 3) |
| Reliability telemetry (uptime, liveness, job outcomes) | Host dashboard (step 7) |

v0 runs end to end against a **mock platform** that ships in this repo. When step 4 exists, the mock is replaced by a base URL.

## 2. Host lifecycle

```
installed ─▶ benchmarked ─▶ registered ─▶ live ⇄ degraded ─▶ offline
                                            ▲                    │
                                            └── heartbeat ok ────┘
```

| From → to | On |
| --- | --- |
| installed → benchmarked | certified `kwh-bench` report written and verified locally |
| benchmarked → registered | platform accepts the signed report, issues `host_id` + token |
| registered → live | first accepted heartbeat with the engine healthy and a passed challenge |
| live → degraded | failed challenge, micro-benchmark outside tolerance, foreign process on the GPU, engine unhealthy, engine restarted (a new instance id in the heartbeat: a restarted engine could be serving anything, so it is re-challenged before it gets more work) |
| degraded → live | next passed challenge with a clean GPU sample; after a failed challenge, two passed challenges in a row (D9) |
| degraded → offline | N consecutive failures (platform config, default 5) |
| live/degraded → offline | no accepted heartbeat within the timeout (default 90 s) |
| offline → live | accepted heartbeat + passed challenge (two in a row if the last one failed); accrual restarts from zero |

- **installed**: binary present, keypair generated, Docker reachable, GPU visible.
- **benchmarked**: a certified `kwh-bench` report exists for this GPU (UUID-bound). Re-done on a schedule (§5) and whenever the driver, engine image or GPU changes.
- **registered**: the platform has the signed report and has issued a `host_id` and token. The rig has a rate (units/hour) and a provisional bucket.
- **live**: heartbeats accepted, last liveness challenge passed, engine serving. **Only in this state does the platform mint for this host.**
- **degraded**: a check failed but the host is still reachable. No minting. Recovers to live on the next passing challenge; drops to offline after N consecutive failures.
- **offline**: no accepted heartbeat for the timeout. Unminted accrual is discarded (never minted ahead, never minted late).

## 3. Components

```
kwh-host (daemon, Python)
├── engine/      start/stop/health the vLLM container; OpenAI-compatible client to it
├── bench/       thin wrapper over kwh_bench: full run, micro-benchmark, canary scoring
├── liveness/    heartbeat loop, challenge scoring, GPU sampling
├── jobs/        WebSocket consumer, request execution, signed results
├── platform/    API client (signed requests), mock server for local dev
├── identity/    keypair, token storage
└── cli          kwh-host init | bench | register | run | status
```

**Language.** Python, importing `kwh_bench` as a library. The benchmark already contains the load generator, the canary scorer, the GPU probe/pre-flight, the report hasher and the verifier; a client in another language would reimplement all of it and drift from the spec. Packaging as a single file (PyInstaller) comes with the tray app, not before.

**Engine sandbox.** One long-lived Docker container per GPU running the certified engine image at the locked version (`vllm/vllm-openai:v0.30.0`, built on CUDA 13 and so needing driver 580 or newer, or the same version's `v0.30.0-cu129` build on a driver that only supports CUDA 12.x, which `kwh-host init` picks from the driver; both are pinned by registry digest, and the lock pins the version string, which is what certification checks) with the pinned §5 flags from `kwh_bench.reference`, serving the Grade I reference model and nothing else. Grade I is one model, so a "job" is a batch of OpenAI-compatible requests to that container, not arbitrary code. The sandbox therefore isolates the host from **buyer inputs**, not from buyer programs. As built in M3 (`kwh_host/sandbox.py`):

| | How |
| --- | --- |
| No network | `--network none`. The engine serves on a Unix socket (vLLM's `--uds`) in a host directory mounted at `/run/kwh`, and the daemon talks to it there. Nothing reaches it from outside and it reaches nothing. |
| Read-only | `--read-only` root filesystem; the checkpoint mounted read-only and loaded offline (`HF_HUB_OFFLINE=1`). Writable: `/tmp` (a 4 GB tmpfs) and `/cache` (torch, Triton and CUDA compile caches, so a restart skips recompiling). |
| No privileges | `--cap-drop ALL`, `no-new-privileges`, and the daemon's own uid; a daemon running as root gets an engine running as `nobody`. |
| Bounded | `--memory` at 3/4 of RAM (at most 64 GiB, no swap), `--pids-limit 4096`, `--shm-size 2g`, one GPU (`--gpus device=N`). |
| The checkpoint it certified | `kwh-host fetch` downloads the locked revision and checks every file against the lock's SHA-256 before the engine ever loads it. |
| The engine it certified | `bench` and `run` use the same launch, image digest included, and the report records it (with the home directory redacted, since reports are public). |

Docker Desktop (Windows, macOS) runs containers in its own VM, and a Unix socket cannot cross from there to the host. With it, `kwh-host init --engine-transport tcp` publishes the engine on 127.0.0.1 instead: still unreachable from the network, but no longer cut off from it. `kwh-host doctor` says which applies. The daemon itself runs as an ordinary user.

**Why the same image as the benchmark.** The rate the host registered was measured on this exact engine build with these exact flags. Serving on anything else makes the rate a lie. The daemon refuses to go live if the running container's version differs from `reference/lock.json`.

## 4. Liveness

The primer asks for "continuous heartbeat plus periodic micro-benchmarks". Three mechanisms, cheapest first:

| Check | Cadence | Cost | What it catches |
| --- | --- | --- | --- |
| **Heartbeat** with a GPU sample (util, power, VRAM, foreign compute processes) and engine `/health` | every 30 s | ~0 | offline, engine down, another tenant on the GPU |
| **Challenge** issued by the platform: four fresh continuations | every 5 min, and on every recovery | 4 prefills, one at a time (~2 s on an A40) | engine serving a different/cheaper model, host faking the engine |
| **Micro-benchmark**: 1/8 of a reference job (32 requests, 512→256, concurrency 32), timed | every 30 min, only when idle | ~5 s on a 4090 | throttling, thermal decay, background load, VRAM contention |

**Challenge canaries, not the locked ones.** `reference/lock.json` is public: its eight canaries and their reference log-probabilities are known, so a host could answer them from a table without running a model. For liveness the platform sends **fresh** continuations: each a prompt and the 32 tokens the platform's own reference node produced after it, scored there moments ago and never seen by this host. A challenge carries **four** of them (D9). The host scores each under teacher forcing, one request at a time (the same `score_continuation` call the benchmark uses), and returns the four mean log-probabilities; the platform takes each one's delta to its reference value and passes the challenge when the **mean of the four deltas** is within the benchmark's tolerance (0.05 nats, the same rule as the benchmark's canaries since rc.6). An unscored continuation fails the challenge. The host cannot precompute, and the check costs four prefills. This is also the first job for the platform reference node that step 6 needs anyway.

**Why four, and why the mean (M2, 2026-10-02).** Three certified cards score the locked canaries at delta 0.0000. The A40 scores them at 0.005–0.048, and a 4-bit substitute at 0.036–0.219, so continuation by continuation the honest and substitute ranges overlap. Under the first draft's one-continuation challenge at 0.05, that substitute would clear about one challenge in eight (it clears one of the eight locked canaries), and since a degraded host is re-challenged on every heartbeat until it passes, it would keep finding its way back to live, five minutes at a time. The means do not overlap (on the locked eight: A40 0.023, the substitute 0.108 on a 4090), and a host that failed a challenge now needs two passes in a row to return to live: a substitute can get lucky once, not twice running.

**Micro-benchmark rule.** Run only when no jobs are in flight; skipped, not failed, when busy (a host delivering jobs is proving liveness the expensive way). A micro-benchmark that a job arrived during is discarded and retried later, since the number would describe the job, not the rig. Result must be within 10% of the registered rate; outside it twice in a row → degraded, and a full re-benchmark is scheduled.

**Contention.** The heartbeat's GPU sample is `kwh_bench.hardware`'s pre-flight sampler adapted for a running engine: our container's VRAM is expected, anything else holding VRAM or driving utilization is a foreign process. The Saturday RTX 5090 in the benchmark's field notes is exactly this case.

## 5. Benchmark and re-benchmark

- **First run:** `kwh-host bench` = `kwh-bench run` against the sandbox container (launched by the daemon with the pinned flags, so the report is certified rather than "attached"). Output is the standard report; `kwh-bench verify` must pass locally before registration is attempted.
- **Re-benchmark:** every 7 days, and immediately on: driver change, engine image change, GPU UUID change, or two consecutive micro-benchmark misses. Uploaded as a new report; the platform replaces the rate. The rig is **degraded** (no minting) until the new report is accepted.
- **Reliability baseline:** the client does not compute a score. It reports raw events (heartbeat sent/accepted, challenge pass/fail with delta, micro-benchmark rate, job accepted/completed/failed with latency). Scoring, bucketing and decay are step 3 and live on the platform, where they cannot be edited by the host.

## 6. Minting

The primer says the host client mints. In v0 the client **requests** and the platform **mints**, because a mint performed by software the host controls is a mint the host can forge. The no-mint-ahead rule is enforced in one place, the platform's mint function, against the platform's own view of liveness:

- Every accepted heartbeat from a **live** host accrues `rate × interval` units to that host (a 4090 at 100.56 u/h accrues 0.84 units per 30 s beat).
- Whole units are minted into the host's wallet as the accrual crosses each integer; the remainder carries.
- Accrual is never computed for intervals not yet elapsed, and a missed or rejected heartbeat accrues nothing. A host that comes back after an outage starts from zero for the gap.
- Minted units expire 72 h after mint (step 4 enforces; the client never sees this).

The client's part is small: the heartbeat carries `wants_mint: true` and the engine state; the response carries the current accrual and wallet balance so `kwh-host status` can show them.

## 7. Jobs

**Transport.** Hosts are residential and behind NAT, so dispatch is **pull** (D3): the daemon keeps an outbound WebSocket to the platform and receives job envelopes on it. No inbound ports, no port forwarding, works on CGNAT. The handshake is signed like every other request (§8). After it, the host sends `hello` with its concurrency (32) and its engine's context length, and the platform routes it nothing that does not fit.

**The platform owns tokenization.** A buyer's chat or text request is turned into prompt token ids by the platform, with the reference model's tokenizer and chat template, before it is dispatched; hosts run completions on token ids. This is what makes the rest checkable: the platform, the host and the verifier agree on exactly which tokens were the prompt, a host cannot alter it, and metering counts tokens the platform counted itself.

**Envelope (platform → host).**

```json
{"type": "job", "job": {
  "job_id": "j_9f2c41a0b7d3e815.1",
  "timeout_s": 59.5,
  "units_reserved": 0.0123,
  "requests": [{
    "prompt_token_ids": [128000, 9906],  "max_tokens": 256,
    "temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0,
    "seed": null, "stop": [], "stop_token_ids": [], "ignore_eos": false, "logprobs": null
  }]}}
```

`.1` is the first attempt; a re-route is `.2`. `timeout_s` is relative to receipt, so host clock skew does not matter. `units_reserved` is the upper bound at `max_tokens`, for the host's information. Every sampling field is present: vLLM fills any field a request leaves unset from the model's `generation_config` (Llama 3.1 Instruct ships temperature 0.6 and top_p 0.9), which would silently turn the buyer's job into something else, so the host rejects an envelope with a missing field rather than let that happen.

**Result (host → platform),** one per job:

```json
{"type": "result", "result": {
  "job_id": "j_9f2c41a0b7d3e815.1", "job_sha256": "…", "host_id": "h_…",
  "status": "completed", "reason": null,
  "outputs": [{"index": 0, "token_ids": [9906, 1917, 128009], "text": "Hello world",
               "finish_reason": "stop", "stop_reason": 128009, "ttft_ms": 41.2, "total_ms": 380.5, "error": null}],
  "usage": {"prompt_tokens": 2, "completion_tokens": 3},
  "engine": {"version": "0.30.0", "served_model": "RedHatAI/…", "launch_mode": "docker"},
  "received_at": 1790812345.1, "finished_at": 1790812345.5,
  "result_sha256": "…", "signature": {"alg": "ed25519", "signed": "kwh-result-v1:result_sha256", "…": "…"}}}
```

`status` is `completed` (every request finished), `failed` (deadline, or a request errored) or `rejected` (invalid envelope, or it does not fit this engine; nothing ran). Generated **token ids** come back, not just text: text cannot be re-tokenized reliably, and the verifier scores the exact tokens the host produced. The vLLM call uses `return_token_ids`, so logprobs are computed only when the buyer asked for them. `job_sha256` (of the canonical envelope) and the signature make the output non-repudiable: what the verifier later finds in it is attributable to this host and this job, which is what step 3 slashes against.

**What the platform checks on every result, without a model:** signed by the registered key; bound to the dispatched job; one output per request, with token ids; none longer than its `max_tokens`; usage equal to what the platform counts. A result that fails any of these is a protocol violation (`bad_result`), counted apart from honest failures.

**Verification.** The first draft of this section put a `sample_spec` in the envelope naming the positions whose logprobs the host should return. That told the host in advance which outputs would be checked: it could have run the reference model on those and something cheaper on the rest. It is gone. The platform now decides **after delivery**, on its own, which requests to verify, and the host never learns which:

- *Greedy requests* are verified by teacher forcing. The reference node runs one prefill over prompt + delivered output with `prompt_logprobs=1` and reads, at every position, the rank of the token the host produced and its gap to the reference's top choice; a substitute model makes choices the reference ranks well below its top, and positions with gap > τ are confident disagreements. One prefill costs a few percent of the generation it checks (the metering weight assumes 1/16 per token), not the 100% of re-running it.
- *Sampled requests* (temperature > 0) cannot be checked token by token. They need a statistical test over many tokens: the delivered tokens' log-likelihood under the reference against what sampling from the reference would give. That is step 3.

**A request is evidence; a host is judged.** The first draft of this section said an honest output has no confident disagreements. M2 measured otherwise ([results/m2-a40-2026-10-02](results/m2-a40-2026-10-02/)). On an A40, the reference model scoring its own greedy output, on the same card and the same engine, disagrees with 2.9% of the tokens, and 13 of the 14 honest jobs the mock router verified at τ = 0.1 would have failed. Decode and prefill take different kernel paths, and batch composition changes the arithmetic: two burst jobs with identical prompts produced different text. W8A8's per-token INT8 activations probably amplify both, since a tiny difference can move an activation across a quantization step. So the platform does not rule on a request. It keeps, per host, a rolling window of verified natural-text greedy requests and counts those whose largest gap exceeds τ:

- On the A40 data (128-token outputs), τ = 1.0 nats separates the two cleanly: 2 of 32 honest chat requests exceed it, against 24 of 32 from a 4-bit AWQ substitute. Over a window of 12 such requests, flagging at 5 or more would misjudge an honest host about 0.1% of the time and miss the substitute about 0.3% of the time (binomial from those rates: one card, small samples, a first estimate).
- Random-word prompts, like the benchmark's, do not separate them usefully (4 of 32 against 8 of 32): their next-token distributions are too flat. Verification samples natural-text requests only.
- Thresholds are per GPU model and per output length, from honest baselines the platform collects on its own reference runs. `verifier.py` defaults to τ = 1.0 until a second card says otherwise.

The prototype is `kwh_host/platform/verifier.py`, and the mock router can run it inline (`--verify-url`); it counts failed verdicts per host and acts on none. On the real platform it runs asynchronously, after the buyer has the response.

**Failure, re-routing, cancellation.** The envelope gives the host half a second less than the router waits, so a slow host reports `failed: deadline` instead of going silent. On `failed`, `rejected`, a bad result or a dropped connection, the router re-routes to the next live host, up to three attempts; the buyer sees a failure only if no host could deliver in time. If the router gives up on a host anyway it sends `cancel`, and the host stops the work. If the WebSocket drops, the host cancels everything in flight, since the platform has already re-routed it. The client never retries silently.

**Capacity and admission.** The router offers a job only to live hosts whose engine context fits every request (`prompt + max_tokens ≤ max_model_len`), whose in-flight requests stay within twice their concurrency, and whose in-flight units stay within `rate × 5 min`. An idle host always takes one job. Least-loaded first; score-based routing comes with step 3.

**Metering (provisional, D7).** `units = (completion_tokens + prompt_tokens / 16) / 73,728`. One reference job (65,536 generated + 131,072 prompt tokens) is exactly 1.0 unit under any prefill weight, so the weight only moves price between prompt-heavy and output-heavy buyers. Units are reserved at `max_tokens` when a job is dispatched and burned at the actual count when it completes; the delivering host is credited with what was burned.

## 8. Platform API contract (v0)

Base URL from config; JSON bodies; semantic versioning on the path. Every request carries `X-Kwh-Timestamp`, `X-Kwh-Public-Key` and `X-Kwh-Signature`, an ed25519 signature over `kwh-req-v1\n{timestamp}\n{METHOD}\n{path}\n{sha256(body)}`, plus `Authorization: Bearer <token>` after registration. GETs and the WebSocket handshake sign an empty body. Binding the method and path means a captured request cannot be replayed against another endpoint inside the five-minute clock window, and the purpose prefix means no request signature can pass as a result signature.

| Method | Path | Body → Response | Notes |
| --- | --- | --- | --- |
| `POST` | `/v1/hosts` | `{report, public_key, client_version}` → `{host_id, token, rate_units_per_hour, bucket, config}` | `report` is a certified kwh-bench report with `signature` filled (the schema already reserves the field). Platform runs `kwh-bench verify` on ingest. A report registers one host only: reports are public, and signing one proves who signed it, not who ran it. |
| `POST` | `/v1/hosts/{id}/heartbeat` | `{engine, gpu_sample, in_flight, wants_mint, client_version}` → `{state, challenge?, accrual, balance, config}` | 30 s. `state` is the platform's verdict (live/degraded/offline). `challenge` is `{challenge_id, items: [{prompt_text, prompt_tokens, continuation_token_ids}] × 4}` when one is due (§4). |
| `POST` | `/v1/hosts/{id}/liveness` | `{challenge_id, mean_logprobs: [4], elapsed_ms}` → `{pass, delta, deltas, state, passes_needed}` | Answer to a challenge, one mean log-probability per continuation in the order sent. `delta` is the mean of `deltas`; `passes_needed` counts the passes still owed after a failure. |
| `POST` | `/v1/hosts/{id}/microbench` | `{units_per_hour, job_seconds, gpu_sample}` → `{accepted, within_tolerance}` | |
| `POST` | `/v1/hosts/{id}/reports` | `{report}` → `{rate_units_per_hour, bucket}` | Re-benchmark upload. |
| `WS` | `/v1/hosts/{id}/jobs` | server → `job`, `cancel`; client → `hello` (once), `result` | Outbound from the host; handshake signed as a GET. §7. Reconnect with backoff; jobs in flight at a disconnect are re-routed by the platform and cancelled by the host. |
| `GET` | `/v1/hosts/{id}` | → `{state, rate, bucket, accrual, balance, last_challenge, last_microbench}` | For `kwh-host status`. |

Not in this contract, by design: wallet operations, listing/asks, stake, payouts, anything a buyer does. Those are steps 4 and 5 and get their own contracts. The mock platform also serves `POST /v1/mock/jobs` and `GET /v1/mock/hosts` as a stand-in for the buyer API; they are not part of the host contract.

**Identity.** An ed25519 keypair generated at `kwh-host init`, private key in the OS keyring where available, else a `0600` file. The public key is the host's durable identity across reinstalls; the token is a session credential the platform can revoke.

## 9. Decisions

**Decided 2026-09-30.** All five taken as recommended below, plus D6. **Decided 2026-10-02:** D8 and D9, as recommended after the M2 run.

| | Decision | Consequence for v0 |
| --- | --- | --- |
| D1 | Off-chain ledger with on-chain USDC settlement | Minting, resale, expiry and burn are ledger rows on the platform; one USDC payout per host per day. Whether $KWH becomes a token is a later question the ledger does not foreclose. |
| D2 | Mint authority is the platform | §6 as written. The primer's "host client mints" is read as "host client causes minting". |
| D3 | Outbound WebSocket for dispatch | §7 as written. No inbound ports on hosts. |
| D4 | Docker only for the engine sandbox | Bare-metal mode exists for testing on pods that cannot run Docker and is refused for registration. |
| D5 | Per-heartbeat accrual, integer mints | §6 as written. |
| D6 | Platform runs on rented cloud; the exchange owns no hardware | API + ledger on a managed host with managed Postgres (Render/Fly first, AWS when it matters; the client only sees a base URL). One **dedicated** 24 GB GPU rented by the hour (RunPod Secure or equivalent, not a shared community host) as the reference node for challenge canaries and step 6's public endpoint: ~$300–600/month, the platform's largest fixed cost until volume. Seed supply, if needed before real hosts arrive, is the host client itself running on rented cards. Buyer inference never runs on platform-rented GPUs; if it has to, the unit economics have already failed. |
| D8 | Serving context length: 1024 to 8192, the host's choice, default 8192 | Benchmark rc.6 certifies any `--max-model-len` from 1024 to 8192 and records it. The host client certifies and serves at `max_model_len` from its config (`kwh-host init --max-model-len`, default 8192), so a host takes buyer requests up to 8,192 tokens instead of 1,024. The unit is unchanged: the A40 ran the reference job at 60.230 units/hour at 1,024 and 60.228 at 8,192 in the same session. The canary deltas do move with the setting (largest 0.048 at 1,024, 0.035 at 8,192), which is why a host serves at the value it certified with and is challenged by the engine it serves with. |
| D9 | Challenges carry four continuations and pass on the mean; two passes in a row after a failure | §4 and §2 as written. The benchmark's canary check judges the mean of its eight canaries the same way since rc.6. |

**Open (2026-10-02).**

| | Question | Recommendation |
| --- | --- | --- |
| D7 | Metering: how much a prompt token counts against a unit | Keep the provisional prefill weight of 1/16 (§7) until a calibration run measures it: a prefill-heavy and a decode-heavy variant of the reference job on the table cards. It is a platform constant, not part of the I-1 spec, so changing it never changes the unit; it only moves price between prompt-heavy and output-heavy buyers. |

The rationale for D1–D5, as recorded before the decision:

**D1 — Unit rails: off-chain ledger with on-chain USDC settlement, or an L2 token from day one.**
*Recommendation: off-chain ledger for v1.* A 4090 mints ~100 units an hour; one host is ~2,400 mints a day, and resale, expiry and burn are each another event. On an L2 that is either a gas bill or a batching layer that reinvents the ledger anyway. The 72-hour expiry and the no-mint-ahead rule are a few lines in a ledger the platform runs, and the platform is already the trusted verifier and router in v1, so the ledger adds no trust the design does not already assume. Settlement stays on-chain: one USDC payout per host per day. If units later earn their own token, the ledger is the source of truth it mints from. *Affects the client only in §6: the client does not care which, as long as minting is a platform call.*

**D2 — Mint authority: platform, not client.**
*Recommendation: platform, as written in §6.* This is a departure from the primer's wording ("host client mints") and is worth stating in the primer's next revision, because "enforced in code" only means something if the code is not on the host's machine.

**D3 — Dispatch transport: outbound WebSocket (pull) vs inbound endpoint (push).**
*Recommendation: WebSocket.* Residential hosts behind NAT are the whole supply side. An inbound endpoint would exclude most of them or require a relay, which is a WebSocket with extra steps.

**D4 — Sandbox: Docker only, or also a bare-metal mode.**
*Recommendation: Docker only in v0.* Bare metal (the pip path the benchmark uses on RunPod) cannot guarantee the engine build or isolate buyer inputs. RunPod-style pods that cannot run Docker are not the target host; they are how we test.

**D5 — Minting granularity: per heartbeat accrual with integer mints (as in §6) vs hourly mints.**
*Recommendation: per-heartbeat accrual.* Hourly mints mean an hour of live capacity that goes offline at minute 59 minted nothing, which punishes flaky residential connections harder than the score already does.

## 10. Milestones

| # | Deliverable | Done when |
| --- | --- | --- |
| M0 | This document | Pushed; D1–D6 recorded in §9 (2026-09-30). |
| M1 | Daemon skeleton + mock platform | `kwh-host init → bench → register → run` reaches **live** against the in-repo mock, with challenges answered and accrual ticking, on a RunPod 4090 (bare-metal mode for the test only). |
| M2 | Jobs | Mock router dispatches jobs over the WebSocket to live hosts and re-routes on failure; signed results with token ids come back and greedy outputs are verified after delivery; a wrong-model host fails the challenge and never receives work. **Done 2026-10-02:** built and tested end to end in process, then proven on a RunPod A40 (README, [results/m2-a40-2026-10-02](results/m2-a40-2026-10-02/)). The same run showed that verification must judge hosts over a window, not requests (§7), and led to D8 and D9. |
| M3 | Docker sandbox + install | One-line install on Ubuntu; engine container pinned to the lock; resource limits; WSL2 path documented and tested. **Built 2026-10-02:** the sandbox (§3), `fetch`, `doctor`, `service`, `install.sh`; proven end to end without a GPU on every push (CI runs a stand-in engine inside the real sandbox: init, bench, register, live, a buyer job). The engine image follows the driver: CUDA 13 build on 580+, the cu129 build on CUDA 12.x drivers. **Proven on a rented RTX 4090 VM 2026-10-05** (Vast.ai, `scripts/vm-m3.sh`, [results](results/m3-vast-4090-2026-10-05/)): the one-line install, a certified benchmark inside the sandbox at 101.51 units/hour (bare metal on the same card model: 100.56), the lockdown checked from outside and inside, live, buyer jobs up to the 8,192-token context, a 4-bit substitute caught (challenge mean delta 0.129), and the systemd service. The VM's toolkit was missing behind a registered `nvidia` runtime, so the installer and `doctor` now try the GPU in a real container instead of reading Docker's settings. Left: WSL2 on a real Windows PC. |
| M4 | Reliability telemetry | Every event in §5 reported; `kwh-host status` shows state, rate, accrual, last checks. |
| M5 | Real platform | Base URL swap when step 4's ledger exists; stake deposit added to `kwh-host register`. |

## 11. Out of scope for v0

Multi-GPU rigs (one daemon per GPU is fine, one daemon for many is later); any model but the Grade I reference; the host dashboard; pricing rules (asks are set at the platform, from the wallet, not from the rig); TEE attestation; native Windows without WSL2; macOS.

## 12. Windows

Windows support in v0 means **WSL2 with Docker Engine installed inside the Ubuntu distribution, plus the NVIDIA Container Toolkit**, the path NVIDIA documents for containers on WSL2. The only driver is the normal NVIDIA driver for Windows; nothing GPU-related is installed inside Linux. The daemon runs inside WSL2 exactly as on Linux, and the one-line installer works there unchanged: it decides whether the toolkit works by running a test container on the GPU, which is the same check on both. Step by step: [docs/windows-wsl2.md](docs/windows-wsl2.md).

Two things differ from Linux:

- **WSL2 stops Ubuntu about a minute after its last window closes**, services or not, and hosting stops with it. Until the tray app exists, a host keeps an Ubuntu window open or starts one at logon (the guide has a scheduled task for it).
- **`nvidia-smi` lists no processes under WSL2**, so the heartbeat cannot see another program using the GPU. Contention then shows up only as a slower micro-benchmark.

Docker Desktop with its WSL2 integration also works, with `--engine-transport tcp` (§3). A native tray app that starts the WSL2 daemon, keeps it running and shows status is the first thing after M5, because that is the install experience gamers will judge.
