# M3 on a rented RTX 4090 VM, 2026-10-05

`scripts/vm-m3.sh` on a Vast.ai VM: instance 54287449 in Chile, built from the "Ubuntu 22.04 VM" template (Ubuntu 22.04.5, kernel 6.8). The hardware was an AMD Ryzen 7 9700X, 47 GB of RAM visible to the VM and an NVIDIA GeForce RTX 4090 24 GB on PCIe 4.0 x16. The driver was 580.95.05 (CUDA 13.0), with Docker 28.1.1 and the NVIDIA Container Toolkit 1.20.1. The software was kwh-host 0.2.0.dev3, kwh-bench 1.0.0rc6, and the engine image `vllm/vllm-openai:v0.30.0` (`sha256:8a69ffad015f…`). The VM ran 55 minutes at $0.513/hr, $0.46 in all.

It took three attempts. The files at the top level are from the third, 09:31–09:38 UTC, which ran every step. The first two are under `attempts/`, because what stopped them is part of the result (below).

| File | What |
| --- | --- |
| `steps.log`, `summary.json` | the script's log, step by step with times, and everything condensed into one file |
| `install.log` | the one-line installer in this attempt. The toolkit was already in place by then; `attempts/2-two-daemons/install.log` shows the installer repairing it |
| `gpu.txt`, `nvidia-smi.txt`, `init.json` | the card and driver, and `kwh-host init` choosing the CUDA 13 engine build for this driver |
| `doctor-before-fetch.txt`, `doctor.txt`, `fetch.json`, `fetch.log` | readiness checks around `kwh-host fetch`; every file was checked against the lock's SHA-256 |
| `bench.txt` | `kwh-host bench` in the sandbox. The certified report is the benchmark repo's [`results/rtx-4090-vast-sandbox.json`](https://github.com/Tim-cryptow/kwh-benchmark/blob/main/results/rtx-4090-vast-sandbox.json) (sha256 `b61ccf0f…`) |
| `register.json` | registration with the mock platform, which accepted Docker hosts with certified reports only |
| `engine-inspect.json`, `sandbox-probes.txt` | the running engine container from outside (`docker inspect`) and from inside (`docker exec` probes) |
| `jobs-*.json` | buyer-side job records: outputs with token ids, usage, units, attempts |
| `run.log`, `run-awq.log`, `hosts-*.json` | the daemon serving the reference model, then the 4-bit substitute, and the platform's view after each |
| `service.txt`, `service-status.txt` | `kwh-host service install`, the systemd user unit while it ran, and the uninstall |
| `mock-platform.log` | the mock platform's startup lines |

## What it showed

- **The certified benchmark ran inside the sandbox.** It scored **101.508 units/hour**, median job 35.465 s, stability 0.0016, 321.6 W mean and 315.6 units per electric kWh. All eight canaries matched the reference exactly (mean delta 0.0000). Attempt 2 on the same VM scored 101.586. The RTX 4090 certified bare metal on RunPod on 2026-09-29 scored 100.56 units/hour at 319.4 W. The sandbox has no measurable cost: no network, read-only root, no capabilities, an ordinary uid, and bounded memory and processes.
- **The lockdown held when checked from both sides.** `docker inspect` showed a read-only root, network `none`, `cap_drop ALL`, `no-new-privileges`, a pids limit of 4096, 35 GiB of memory, user 1002:1002, GPU 0 only, and `/hf` mounted read-only. From inside the container, an outbound connection got *Network is unreachable* and DNS failed. Writes to `/usr` and `/hf` were refused. The process ran as uid 1002 with `CapEff 0000000000000000`, and it saw the RTX 4090.
- **It went live 42 s after `kwh-host run`.** The first challenge passed at mean delta 0.0 over four continuations. The micro-benchmark measured 100.87 units/hour.
- **Jobs.**
  - A chat job of two greedy requests, 256 tokens in 3.97 s.
  - A sampled chat request: a 17-token haiku about rain in Lagos.
  - A 6,065-token prompt, which needs the 8,192-token context of D8.
  - An 8,292-token job was refused with "no live host with capacity for this job".
  - Stopping the daemon removed the engine container within 16 s.
- **The 4-bit substitute was caught.** `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` ran in the same sandbox. When the engine restarted, the platform held the host as `degraded`. Its challenge failed at mean delta **0.1285**: 0.055, 0.219, 0.099 and 0.142 over four continuations, against a tolerance of 0.05. In attempt 2 it failed at 0.0765. The honest engine scored 0.0 in every challenge.
- **The background service went live 58 s after `kwh-host service install`.** The service was a systemd user unit, with lingering enabled so it runs without a login session. `kwh-host service uninstall` removed it cleanly.

## Attempts 1 and 2

**Attempt 1** (08:49, `attempts/1-no-toolkit/`) failed at the benchmark: the engine exited at once with `docker: could not select device driver "" with capabilities: [[gpu]]`. `vm-diag.txt` shows why. The NVIDIA Container Toolkit was not installed: no package and none of its programs. Docker's `daemon.json` still listed an `nvidia` runtime pointing at a program that did not exist. The template's readme says the toolkit is preinstalled. The installer and `doctor` had both taken the runtime entry as proof (`doctor.txt`: "GPU in containers: nvidia runtime"). The fix is in commit 7e67397:
- The installer now runs NVIDIA's sample workload, `docker run --rm --gpus all ubuntu nvidia-smi`, and installs and configures the toolkit when that fails.
- `doctor` requires the toolkit's hook and runs `nvidia-smi` inside the engine image.
- An engine that dies at launch now reports the last lines of its log instead of just "exited early with code 125".

**Attempt 2** (09:13, `attempts/2-two-daemons/`) is where the installer installed toolkit 1.20.1 and the test container saw the GPU (`install.log`). The benchmark certified at 101.586 (`report.json`, sha256 `b773aba3…`) and the host went live. Two faults in the test, not in the host client, spoiled the rest:
- **Chat jobs failed with a 500.** The mock platform's chat template needs jinja2, which transformers does not install. On RunPod it came with vLLM. See `mock-platform.log`.
- **The first daemon never stopped.** The script stopped the daemon by signalling `$!`, which was the `sg`/function wrapper, not the daemon. When the substitute's daemon started, the two shared one host identity and fought over one engine container (`run.log`). The substitute still failed its challenge.

Both are fixed in c969efb. The script stops the daemon by name and waits for it. The mock platform checks its chat template at start, and refuses chat jobs with a 400 that says why.

## Notes on renting

- **Vast.ai VMs log in as root.** `vm-m3.sh` now creates a normal user with passwordless sudo and runs as that user, the way a host would. The test runs detached, and rerunning the same command follows it after a dropped SSH connection.
- **Listed offers can be unavailable.** Two VM offers that the search listed as rentable, in Israel and India, refused the rental with `no_such_ask … is not available`. Both had a `min_bid` above their on-demand price, a sign that another job was on the machine. An offer whose `min_bid` was under its price (Chile) rented at the first try.
- **The template's port was not the problem.** The stock template carries a UDP port mapping `741641`, which is out of range. A private copy without it was used for the rental. Removing it did not by itself make the busy offers rentable, so it was not the cause of the refusals.
- **Fetch output was mixed up.** `kwh-host fetch` printed `docker pull`'s progress on stdout ahead of its JSON (`fetch.json`). It now goes to stderr.
