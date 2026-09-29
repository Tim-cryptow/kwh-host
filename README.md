# kWh Host Client

The program a GPU owner installs to sell work on the kWh Exchange. It benchmarks the rig with [kwh-benchmark](https://github.com/Tim-cryptow/kwh-benchmark), registers the certified rate with the platform, proves the rig is live (heartbeat, platform-issued canaries, micro-benchmarks), and executes buyer jobs in a sandboxed copy of the certified engine. Units are minted by the platform against that liveness; the client never mints on its own.

Build step 2 of 7. Nothing here runs yet; **[HOST-CLIENT.md](HOST-CLIENT.md)** is the scope for the first cut, including the platform API contract the client talks to and the five decisions that shape it.

## Status

- [x] M0 — scope document
- [ ] M1 — daemon skeleton against the in-repo mock platform, end to end on a RunPod 4090
- [ ] M2 — job dispatch over WebSocket, verification samples
- [ ] M3 — Docker sandbox, one-line install, WSL2 path
- [ ] M4 — reliability telemetry, `kwh-host status`
- [ ] M5 — real platform (step 4), stake deposit

## Layout (planned)

```
HOST-CLIENT.md        scope, lifecycle, liveness, API contract, decisions
kwh_host/
  engine/             vLLM container lifecycle + OpenAI-compatible client
  bench/              kwh_bench wrapper: full run, micro-benchmark, canaries
  liveness/           heartbeat, challenges, GPU sampling
  jobs/               WebSocket consumer, execution, samples
  platform/           API client + mock server
  identity/           keypair, token
  cli.py              kwh-host init | bench | register | run | status
```

Apache-2.0, same as the benchmark.
