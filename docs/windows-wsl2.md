# Hosting on Windows (WSL2)

The host client runs inside WSL2, the Linux that ships with Windows. The engine runs in Docker inside that Linux and uses your NVIDIA card through the normal Windows driver.

**Status:** tested on a Windows 11 laptop without an NVIDIA card on 2026-10-06 ([results](../results/wsl-windows11-2026-10-06/)): Ubuntu, the installer, the engine sandbox, the background service, and keeping it running (step 5, which that test rewrote). The GPU itself (steps 2 and 4 with a real card) has not been run on Windows yet. If you run it, tell us what happened.

## What you need

- Windows 11, or Windows 10 version 21H2 or later
- An NVIDIA card with **16 GB or more** (RTX 3090, 4080, 4090, 5080, 5090 and similar)
- The current NVIDIA driver for Windows (Game Ready or Studio). This is the only driver to install: do not install any NVIDIA driver inside Linux.

## 1. Install Ubuntu in WSL2

In PowerShell, run as administrator (right-click Start, **Terminal (Admin)**):

```powershell
wsl --install -d Ubuntu-24.04
```

If WSL itself is missing, this installs it first. Restart if Windows asks you to. Ubuntu then asks for a Linux user name and password; the password is the one `sudo` asks for later.

From here on, commands go in **Ubuntu**, not PowerShell: open **Ubuntu 24.04** from the Start menu, or type `wsl ~` in PowerShell. Its prompt looks like `you@PC:~$`; PowerShell's looks like `PS C:\...>`.

## 2. Check the GPU and systemd

In the Ubuntu window:

```bash
nvidia-smi                       # must show your card, and "CUDA Version" 12.8 or newer; it comes from the Windows driver
systemctl is-system-running      # "running" or "degraded" is fine
```

If `systemctl` says systemd is not running, turn it on and restart WSL:

```bash
printf '[boot]\nsystemd=true\n' | sudo tee -a /etc/wsl.conf
```

Then in PowerShell run `wsl --shutdown`, and open Ubuntu again.

## 3. Install the host client

In Ubuntu:

```bash
curl -fsSL https://raw.githubusercontent.com/Tim-cryptow/kwh-host/main/install.sh | bash
```

The installer checks the card, then asks before each system change: Docker Engine (inside Ubuntu, not Docker Desktop), the NVIDIA Container Toolkit, and adding you to the `docker` group. Answer `y`. Two things look alarming and are not:

- Docker's script recommends Docker Desktop and waits 20 seconds. Let it carry on.
- A red line about `systemd-binfmt.service` failing. WSL handles that part itself.

Afterwards, **run `wsl --shutdown` in PowerShell and open Ubuntu again**, so the `docker` group takes effect everywhere, including for the background service.

## 4. Set up and certify

```bash
kwh-host init --platform <platform URL>
kwh-host doctor            # every line should say ok
kwh-host fetch             # the model (about 9 GB, checked against the published hashes) and the engine (about 9 GB)
kwh-host bench             # the certified benchmark, about 10 minutes
kwh-host register
kwh-host service install   # hosting starts, and restarts on its own if it crashes
```

## 5. Keep Ubuntu running

WSL stops Ubuntu about 15 seconds after its last window closes, background services or not, and hosting stops with it. Two settings fix that: one keeps Ubuntu running with no window open, the other starts it when you log on to Windows. Both are done in PowerShell, as yourself (not as administrator).

**Keep Ubuntu running** (WSL 2.5.4 or later; `wsl --version` shows yours). Open WSL's settings file:

```powershell
notepad "$env:USERPROFILE\.wslconfig"
```

Add these two lines (if the file already has a `[general]` line, add only the second line, under it), save, and run `wsl --shutdown`:

```ini
[general]
instanceIdleTimeout=-1
```

**Start Ubuntu, and the host with it, at logon:**

```powershell
$a = New-ScheduledTaskAction -Execute "wsl.exe" -Argument "-d Ubuntu-24.04 --exec true"
$t = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit 0
Register-ScheduledTask -TaskName "kWh host (WSL)" -Action $a -Trigger $t -Settings $s
```

A window opens for a moment at logon and closes by itself; Ubuntu and the host keep running. After a `wsl --shutdown`, start them again by opening Ubuntu once, or with `Start-ScheduledTask "kWh host (WSL)"`.

**On WSL older than 2.5.4**, skip the settings file and let the task hold Ubuntu open instead: put `--exec sleep infinity` in place of `--exec true`. Its window then stays open: minimize it, don't close it.

Each part of the command matters:

- **`-User`:** without it, Windows lets only an administrator create the task ("Access is denied").
- **`-Settings`:** a task's defaults keep it from starting on battery, stop it when a laptop unplugs, and stop it after 72 hours.
- **`instanceIdleTimeout=-1`:** it keeps every Linux distribution in WSL running, not just Ubuntu, until `wsl --shutdown` or a restart.

The tray app planned after the real platform launches will replace this step.

## If you already use Docker Desktop

Docker Desktop runs containers in its own virtual machine. A Unix socket can't cross from there into Ubuntu, so the engine can't be fully cut off from the network. Either:

- uninstall Docker Desktop, or turn off its WSL integration for Ubuntu, and let the installer put Docker Engine inside Ubuntu (recommended), or
- keep Docker Desktop and run `kwh-host init --platform <URL> --engine-transport tcp`. The engine can then be reached only from your PC, but it can still reach the internet.

`kwh-host doctor` tells you which setup it found.

## Known limits on WSL2

- `nvidia-smi` lists no processes under WSL2, so the host client can't see another program (a game, a miner) using the card. You'll see it as slower micro-benchmarks, and the platform will too.
- Sleep and hibernate stop hosting. Set Windows power settings to keep the PC awake while you host.
- On the laptop we tested, Ubuntu's clock under WSL2 ran about 5% slow and was pulled back to Windows' time in jumps ([results](../results/wsl-windows11-2026-10-06/)). The benchmark times itself on that clock, so since rc.7 it checks the clock against the wall clock and does not certify a run where they disagree (`kwh-host bench` shows the reason, `clock: ...`). Whether this happens on PCs with an NVIDIA card is not known yet.
