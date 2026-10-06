# The Windows path on a real PC, without a GPU (2026-10-06)

`scripts/wsl-check.sh` ran in Ubuntu, then `scripts/wsl-keepalive.ps1` in PowerShell, on a laptop with no NVIDIA card:

- Windows 11 Pro (build 26200), 8 CPUs, 15 GB of memory for WSL;
- nothing installed beforehand: the first `wsl` command installed WSL itself (3.0.1, kernel 6.18), and `wsl --install -d Ubuntu-24.04` installed the guide's Ubuntu (24.04.5, with systemd on);
- PowerShell ran as an ordinary user, as a host's would.

The question was everything on the Windows path that does not need the card. Does the installer work inside WSL2? Does the engine sandbox? Does the host run as a service, and does it keep running when nobody has a window open?

## What happened

| Step | Result |
| --- | --- |
| Ubuntu and systemd | systemd running. Its state is "degraded" because `systemd-binfmt` fails under WSL, which is harmless. |
| The one-line installer | Exit 0 after 2 min 23 s. It installed Docker Engine 29.8.2 inside Ubuntu (cgroup v2), the docker group, the NVIDIA Container Toolkit (it installs without a card) and kwh-host. Docker's own script stopped for 20 seconds to recommend Docker Desktop instead. |
| The sandbox, end to end | CI's script, unchanged: bench, register, live, the lockdown (read-only, no network, no capabilities, not root), a buyer job, a re-benchmark, another job. All passed. |
| The background service | Live 3 s after `kwh-host service install`. |
| The guide's scheduled task, as written | **Refused** for an ordinary user: "Access is denied". It registered once the trigger was limited to the user (`-User`). Its defaults would also have hurt a laptop host: no start on battery, a stop when the laptop goes on battery, and a stop after 72 hours. Starting it opens a console window. |
| That task, and no Ubuntu window, for 3 minutes | Ubuntu kept running and the host stayed live: 17 heartbeats accepted out of 17, and 3 challenges passed. |
| Logon | After `wsl --shutdown`, the task alone started Ubuntu in 2 s. With no window opened, the service started with Ubuntu, and its engine was running about 6 s after the task started. |
| `instanceIdleTimeout=-1` in `.wslconfig`, no task | Ubuntu was started by a command that ended at once, and kept running for 3 minutes with nothing holding it. So did the host: the same service process throughout, still heartbeating (into nothing, since the test platform had stopped with the restart before). The script's own verdict said the host had not; that was the clock, below. |
| Nothing holding Ubuntu | WSL stopped it 17 s after its last session ended. The guide said about a minute. |
| `wsl --shutdown` | It stops the host cleanly: systemd stops the service, which removes its engine and writes its last state. |

## The clock

Ubuntu's clock ran about 5% slow against Windows':

- Between two checks 306 s apart by Windows' clock, Ubuntu's uptime grew by 292 s.
- Ubuntu's wall clock was kept to Windows' time by jumps. The daemon heartbeats every 10 s by its own monotonic clock. In wall time the beats were 10.0 s apart, except every third to fifth one, which was 11.5 to 12.4 s apart. That is a jump of about 2 seconds every half minute; the average interval was 10.53 s.
- In the `instanceIdleTimeout` step, Ubuntu counted 177 s of uptime in at least 181 s of Windows time.

So on this laptop, a duration timed inside WSL2 comes out about 5% short. The benchmark times its jobs that way (`perf_counter`) and integrates power over the same clock. A benchmark run here would therefore report about 5% more units per hour, and per kWh, than the card delivered.

This needs checking on the PC with an NVIDIA card before Windows hosts are rated. Two fixes would work whatever the clock does:

- the benchmark compares its elapsed time with the wall clock, and does not certify when they disagree;
- the platform times micro-benchmarks itself.

The check now measures Ubuntu's clock against Windows' on every run.

Since then the benchmark has taken the first: from rc.7 it times its measured runs on both clocks and does not certify when they disagree by more than 1%.

## What it changed

- **docs/windows-wsl2.md, step 5, rewritten.** `instanceIdleTimeout=-1` keeps Ubuntu running with no window open (WSL 2.5.4 and later). A scheduled task starts Ubuntu at logon. The task is registered for the user (`-User`), set to run on battery and without a time limit, and only starts Ubuntu, so its window closes by itself. Older WSL keeps the task that holds a window open, with the same fixes. Step 3 now says `wsl --shutdown` after the installer rather than "open a new window". A new window is not enough for the background service: systemd's user manager runs it with the groups the manager started with.
- **The installer, on WSL2.** It says to let Docker's script carry on, and to run `wsl --shutdown` when it has just added the docker group.
- **The check itself.** It measures time on Ubuntu's uptime clock, never its wall clock, and allows 10% for that clock. It also measures that clock against Windows'.

## What it could not test

- **The GPU.** The NVIDIA driver for Windows reaching Docker inside WSL2, `nvidia-smi` there, and a real benchmark. `doctor`'s GPU checks failed, as they should with no card.
- **A real logon, and sleep or hibernate.** The task was started by hand after `wsl --shutdown`, which is what Windows does at logon.
- **The new step 5 as one piece.** Each part was tested: the setting with a command that ends at once, the task starting Ubuntu as at logon, and the `-User` trigger. The task with `--exec true` and the battery and time-limit settings were not run together.

## Files

- `wsl-keepalive.log`: the PowerShell side, step by step, with its summary.
- Stages 1 and 2 in Ubuntu: `steps.log`, `wsl.txt`, `install.log`, `sandbox.log`, `doctor.txt`, `bench.txt`, `status-before.txt`.
- `probe-01-start` to `probe-04-setting-hold`: Ubuntu at each check (the platform's view too, while it was up).
- The daemon's own record: `events-host.jsonl`, `service-journal.txt`, `state-host.json`, `final-state.txt`, `status-final.txt`.
