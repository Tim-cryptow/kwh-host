# M4 on a rented RTX 4090 VM (Vast.ai, 2026-10-05)

`scripts/vm-m4.sh` on a Vast.ai VM in Israel: one RTX 4090, Ubuntu 22.04, $0.56 an hour; both runs together cost $1.45. Like M3, it used the one-line installer and the Docker path only. The mock platform accepted Docker hosts with certified reports and nothing else. The question was what happens to a host over time:

- does a card that changes get re-rated,
- does the platform stop paying for it in the meantime,
- does the daemon recover on its own?

The first run broke in a way the tests had not imagined. The fixes went in, and the second run on the same machine passed every step.

## The second run (the files in this folder)

| | What happened | Rate after |
| --- | --- | --- |
| Start | The report from the first run was certified on driver 580.95.05; after a reboot the VM ran 580.178.04. The daemon noticed before its engine first started, re-benchmarked in 3 minutes and went live (`driver changed: 580.95.05 benchmarked, 580.178.04 now`). | 101.88 |
| Slowed card | Graphics clock locked at 945 MHz (of 3,150). Two micro-benchmarks missed (72.06, 71.81 against 101.88). The platform held the host, and a buyer job was refused (`no live host`), not sent to it. The daemon re-benchmarked; the host was live again 4 min 52 s after the hold. The next micro-benchmark agreed: 71.48. | 70.99 |
| Clock released | The same, the other way: two micro-benchmarks at 100.95 and 100.94 against 70.99, a hold, a re-benchmark. Live again 3 min 52 s after the hold; the next micro-benchmark was 100.4. | 101.52 |
| Frozen engine | `docker pause` on the engine. The daemon decided to restart it after three unanswered heartbeats (44 s) and replaced the container. Live again 115 s after the freeze, and still locked down: read-only root, no network, no capabilities, uid 1002. | 101.52 |

```
[13:07:28] 4. the daemon, live, and buyer jobs
   at start: 2026-10-05 13:07:28  rebench   started: driver changed: 580.95.05 benchmarked, 580.178.04 now
   at start: 2026-10-05 13:10:41  rebench   done: 101.88 units/hour (was 101.83), 3 min
[13:11:20] live, job channel open
[13:15:56] slow: platform holds the host: micro-benchmark outside 10% of the rate twice in a row
[13:15:57] slow: a buyer job meanwhile: exit 1 (... no live host with capacity for this job)
[13:20:49] slow: live again at 70.993 units/hour
[13:20:55] slow: next micro-benchmark 71.48 units/hour, within: True
[13:25:05] full: platform holds the host: micro-benchmark outside 10% of the rate twice in a row
[13:28:57] full: live again at 101.515 units/hour
[13:29:03] 7. the engine frozen (docker pause): the daemon should restart it
[13:30:59] engine restarted and live 116s after the freeze
```

The four reports are all certified, with all eight canaries exact (mean delta 0.0000). The pinned image digest and the GPU UUID match in every one.

| Report | Driver | Clock | Units/hour | Mean power | Units per electric kWh |
| --- | --- | --- | --- | --- | --- |
| first | 580.95.05 | full | 101.83 | 314 W | 324 |
| start | 580.178.04 | full | 101.88 | 314 W | 324 |
| slow | 580.178.04 | 945 MHz | 70.99 | 171 W | **415** |
| full | 580.178.04 | full | 101.52 | 315 W | 322 |

`status.txt` is `kwh-host status` at the end, after 23 minutes of all this. The platform's counters show:

- 40% of the time live;
- 129 of 138 heartbeats accepted;
- 12 of 12 challenges passed;
- 8 micro-benchmarks, 4 of them misses;
- 12 units minted.

The host itself reported three engine restarts.

## The first run (`attempts/1-driver-updated/`)

1. **11:05 UTC:** the VM booted.
2. **11:14:** the installer and benchmark finished; the first report came out at 101.83 units/hour. The host went live and served buyer jobs.
3. **11:15–11:19:** the clock was locked at 945 MHz. The micro-benchmarks fell to 71.7, twice; the platform held the host and the daemon started its re-benchmark, as designed.
4. **11:16–11:29:** meanwhile, Ubuntu's `unattended-upgrade` installed 268 updates in the background, its list of exclusions empty, as it ships. They included the NVIDIA server driver, 580.95.05 to 580.178.04 (`vm-diag.txt` has the apt history). The new libraries were in place by 11:19:29. The kernel module still loaded was 580.95.05, so from then on nothing new could use the GPU (`Failed to initialize NVML: Driver/library version mismatch`).
5. **11:19:31:** the benchmark engine failed to start (`nvidia-container-cli: nvml error: driver/library version mismatch`). So did the serving engine after it. The daemon then exited, and the platform saw only a host gone silent.

The fixes:

- **The daemon no longer exits when its engine will not start.** It keeps heartbeating, with the reason, and retries (30 s doubling to 10 minutes). The one exception is an engine version the lock does not certify, which no retry can fix.
- **`doctor`, `kwh-host status` and the heartbeat name the case.** For a driver/library mismatch they say: the driver was updated while the machine was running; reboot.
- **The installer offers to keep NVIDIA packages out of Ubuntu's automatic updates** (`/etc/apt/apt.conf.d/52kwh-host-nvidia`), so the owner updates the driver and reboots when hosting allows. The second run's installer applied it. The rule was checked against the updater's own matching code: it skips every package that broke this VM.
- **A frozen engine no longer gets two reasons at the platform.** It used to read "engine version None != lock" next to "engine unhealthy": an engine that does not answer has an unknown version, not a wrong one.

## What it settled

- **The re-benchmark loop works on a real card, in both directions.**
  - A card that slowed by 30% was caught by two micro-benchmarks in just over 4 minutes, held, and re-rated in under 5 minutes.
  - When the card sped back up it was re-rated upward, in under 4 minutes.
  - While a host was held, buyers were refused rather than given a host the platform could not vouch for.
- **A driver update is a re-benchmark, and the daemon finds it by itself.**
  - Nothing told it the driver had changed: it compared its report with `nvidia-smi` at start, before serving anything.
  - The rate came back unchanged (101.83 to 101.88). The rule exists because it might not have.
- **A slower clock is a cheaper unit to make.** At 945 MHz the card made 30% fewer units per hour but 28% more per electric kWh (415 against 324). The benchmark measures both, and the exchange is named after the second. Whether hosts should be rated on one or both is a question for step 3.
- **Automatic updates are a hazard for any GPU host, not just this VM.** Ubuntu installs NVIDIA driver updates by default, and an update without a reboot stops the GPU taking new work. A running engine carries on until it next restarts, which is exactly when a host needs it. The installer now asks to keep the driver out of the automatic updates.
