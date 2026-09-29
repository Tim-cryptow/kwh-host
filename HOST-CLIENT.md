# kWh Host Client — scope (v0)

**Build step:** 2 of 7 (primer §10). **Upstream:** [kwh-benchmark](https://github.com/Tim-cryptow/kwh-benchmark) 1.0.0-rc.3, series I-1.
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
| Job execution over an outbound WebSocket, results returned with verification samples | Redundant-sampling verifier, slashing (step 3) |
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
| live → degraded | failed challenge, micro-benchmark outside tolerance, foreign process on the GPU, engine unhealthy |
| degraded → live | next passed challenge with a clean GPU sample |
| degraded → offline | N consecutive failures (platform config, default 5) |
| live/degraded → offline | no accepted heartbeat within the timeout (default 90 s) |
| offline → live | accepted heartbeat + passed challenge; accrual restarts from zero |

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
├── jobs/        WebSocket consumer, request execution, result + sample upload
├── platform/    API client (signed requests), mock server for local dev
├── identity/    keypair, token storage
└── cli          kwh-host init | bench | register | run | status
```

**Language.** Python, importing `kwh_bench` as a library. The benchmark already contains the load generator, the canary scorer, the GPU probe/pre-flight, the report hasher and the verifier; a client in another language would reimplement all of it and drift from the spec. Packaging as a single file (PyInstaller) comes with the tray app, not before.

**Engine sandbox.** One long-lived Docker container per GPU running the certified engine image at the locked version (`vllm/vllm-openai:v0.30.0`; the lock pins the version string, which is what certification checks) with the pinned §5 flags from `kwh_bench.reference`, serving the Grade I reference model and nothing else. Grade I is one model, so a "job" is a batch of OpenAI-compatible requests to that container, not arbitrary code. The sandbox therefore isolates the host from **buyer inputs**, not from buyer programs: no host mounts except a read-only HF cache, no network except the loopback the daemon uses, memory and PID limits, non-root. The daemon itself runs as an ordinary user.

**Why the same image as the benchmark.** The rate the host registered was measured on this exact engine build with these exact flags. Serving on anything else makes the rate a lie. The daemon refuses to go live if the running container's version differs from `reference/lock.json`.

## 4. Liveness

The primer asks for "continuous heartbeat plus periodic micro-benchmarks". Three mechanisms, cheapest first:

| Check | Cadence | Cost | What it catches |
| --- | --- | --- | --- |
| **Heartbeat** with a GPU sample (util, power, VRAM, foreign compute processes) and engine `/health` | every 30 s | ~0 | offline, engine down, another tenant on the GPU |
| **Challenge canary** issued by the platform | every 5 min, and on every recovery | 1 forward pass (~0.2 s) | engine serving a different/cheaper model, host faking the engine |
| **Micro-benchmark**: 1/8 of a reference job (32 requests, 512→256, concurrency 32), timed | every 30 min, only when idle | ~5 s on a 4090 | throttling, thermal decay, background load, VRAM contention |

**Challenge canaries, not the locked ones.** `reference/lock.json` is public: its eight canaries and their reference log-probabilities are known, so a host could answer them from a table without running a model. For liveness the platform sends **fresh** canaries: a prompt and a 32-token continuation the platform's own reference node scored moments ago, never seen by this host. The host scores it under teacher forcing (the same `score_continuation` call the benchmark uses) and returns the mean log-probability; the platform compares under the §7 tolerance. The host cannot precompute, and the check costs one forward pass. This is also the first job for the platform reference node that step 6 needs anyway.

**Micro-benchmark rule.** Run only when no jobs are in flight; skipped, not failed, when busy (a host delivering jobs is proving liveness the expensive way). Result must be within 10% of the registered rate; outside it twice in a row → degraded, and a full re-benchmark is scheduled.

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

- **Transport.** Hosts are residential and behind NAT, so dispatch is **pull**: the daemon keeps an outbound WebSocket to the platform and receives job envelopes on it. No inbound ports, no port forwarding, works on CGNAT.
- **Envelope.** `{job_id, unit_count, requests: [OpenAI chat/completions bodies], deadline, sample_spec}`. The daemon executes the requests against the sandbox container at the spec's concurrency and returns `{job_id, outputs, timings, samples}`.
- **Samples for verification.** `sample_spec` names a random subset of (request, token position) pairs; the daemon returns the top-k logprobs at those positions from its own output. Step 3's verifier re-executes the same shards elsewhere and compares. The client cannot know in advance which positions will be checked.
- **Failure.** A job the daemon cannot complete by the deadline is returned as `failed` with a reason; the platform re-routes. The client never retries silently.
- **Capacity.** In-flight units per host are capped by the platform at the registered rate × a window, so a host cannot accept more work than it can deliver.

## 8. Platform API contract (v0)

Base URL from config; all requests carry `Authorization: Bearer <token>` after registration and an `X-Kwh-Signature` (ed25519 over the body) always. JSON bodies. Semantic versioning on the path.

| Method | Path | Body → Response | Notes |
| --- | --- | --- | --- |
| `POST` | `/v1/hosts` | `{report, public_key, client_version}` → `{host_id, token, rate_units_per_hour, bucket, config}` | `report` is a certified kwh-bench report with `signature` filled (the schema already reserves the field). Platform runs `kwh-bench verify` on ingest. |
| `POST` | `/v1/hosts/{id}/heartbeat` | `{engine, gpu_sample, in_flight, wants_mint, client_version}` → `{state, challenge?, accrual, balance, config}` | 30 s. `state` is the platform's verdict (live/degraded/offline). `challenge` is a fresh canary when one is due. |
| `POST` | `/v1/hosts/{id}/liveness` | `{challenge_id, mean_logprob, elapsed_ms}` → `{pass, delta}` | Answer to a challenge. |
| `POST` | `/v1/hosts/{id}/microbench` | `{units_per_hour, job_seconds, gpu_sample}` → `{accepted, within_tolerance}` | |
| `POST` | `/v1/hosts/{id}/reports` | `{report}` → `{rate_units_per_hour, bucket}` | Re-benchmark upload. |
| `WS` | `/v1/hosts/{id}/jobs` | server → `job envelope`; client → `job result` | Outbound from the host. Reconnect with backoff; jobs in flight at disconnect are returned as failed. |
| `GET` | `/v1/hosts/{id}` | → `{state, rate, bucket, accrual, balance, last_challenge, last_microbench}` | For `kwh-host status`. |

Not in this contract, by design: wallet operations, listing/asks, stake, payouts, anything a buyer does. Those are steps 4 and 5 and get their own contracts.

**Identity.** An ed25519 keypair generated at `kwh-host init`, private key in the OS keyring where available, else a `0600` file. The public key is the host's durable identity across reinstalls; the token is a session credential the platform can revoke.

## 9. Decisions to make

These are the calls that shape v0. Each has a recommendation; none is made yet.

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
| M0 | This document | Pushed; D1–D5 answered in a follow-up commit. |
| M1 | Daemon skeleton + mock platform | `kwh-host init → bench → register → run` reaches **live** against the in-repo mock, with challenges answered and accrual ticking, on a RunPod 4090 (bare-metal mode for the test only). |
| M2 | Jobs | Mock router dispatches jobs over the WebSocket; outputs and verification samples come back; a deliberately wrong-model container fails the challenge. |
| M3 | Docker sandbox + install | One-line install on Ubuntu; engine container pinned to the lock; resource limits; WSL2 path documented and tested. |
| M4 | Reliability telemetry | Every event in §5 reported; `kwh-host status` shows state, rate, accrual, last checks. |
| M5 | Real platform | Base URL swap when step 4's ledger exists; stake deposit added to `kwh-host register`. |

## 11. Out of scope for v0

Multi-GPU rigs (one daemon per GPU is fine, one daemon for many is later); any model but the Grade I reference; the host dashboard; pricing rules (asks are set at the platform, from the wallet, not from the rig); TEE attestation; native Windows without WSL2; macOS.

## 12. Windows

Windows support in v0 means **WSL2 + Docker Desktop with the WSL2 backend and NVIDIA GPU passthrough**. The daemon runs inside the WSL2 distribution exactly as on Linux; the benchmark repo already lists Windows this way. A native tray app that starts the WSL2 daemon and shows status is the first thing after M5, because that is the install experience gamers will judge.
