# Hosting on Windows (WSL2)

The host client runs inside WSL2, the Linux that ships with Windows. The engine runs in Docker inside that Linux and uses your NVIDIA card through the normal Windows driver.

**Status:** written from NVIDIA's and Microsoft's documentation and tested on Linux. It has not yet been run on a Windows PC with an NVIDIA card. If you run it, tell us what happened.

## What you need

- Windows 11, or Windows 10 version 21H2 or later
- An NVIDIA card with **16 GB or more** (RTX 3090, 4080, 4090, 5080, 5090 and similar)
- The current NVIDIA driver for Windows (Game Ready or Studio). This is the only driver to install: do not install any NVIDIA driver inside Linux.

## 1. Install Ubuntu in WSL2

In PowerShell, run as administrator:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart if Windows asks you to, then open **Ubuntu** from the Start menu and create your Linux user name and password.

## 2. Check the GPU and systemd

In the Ubuntu window:

```bash
nvidia-smi                       # must show your card; it comes from the Windows driver
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

The installer checks the card, then asks before each system change: Docker Engine (inside Ubuntu, not Docker Desktop), the NVIDIA Container Toolkit, and adding you to the `docker` group. Answer `y`. **Close the Ubuntu window and open a new one** afterwards so the group takes effect.

## 4. Set up and certify

```bash
kwh-host init --platform <platform URL>
kwh-host doctor            # every line should say ok
kwh-host fetch             # the model (about 9 GB, checked against the published hashes) and the engine (about 10 GB)
kwh-host bench             # the certified benchmark, about 10 minutes
kwh-host register
kwh-host service install   # hosting starts, and restarts on its own if it crashes
```

## 5. Keep Ubuntu running

WSL2 stops Ubuntu about a minute after its last window closes, and hosting stops with it. Either leave an Ubuntu window open (minimized is fine), or have Windows open one at logon. In PowerShell:

```powershell
$a = New-ScheduledTaskAction -Execute "wsl.exe" -Argument "-d Ubuntu-24.04 --exec sleep infinity"
$t = New-ScheduledTaskTrigger -AtLogOn
Register-ScheduledTask -TaskName "kWh host (WSL)" -Action $a -Trigger $t
```

A console window opens at logon; minimize it, don't close it. The tray app planned after the real platform launches will replace this step.

## If you already use Docker Desktop

Docker Desktop runs containers in its own virtual machine. A Unix socket can't cross from there into Ubuntu, so the engine can't be fully cut off from the network. Either:

- uninstall Docker Desktop, or turn off its WSL integration for Ubuntu, and let the installer put Docker Engine inside Ubuntu (recommended), or
- keep Docker Desktop and run `kwh-host init --platform <URL> --engine-transport tcp`. The engine can then be reached only from your PC, but it can still reach the internet.

`kwh-host doctor` tells you which setup it found.

## Known limits on WSL2

- `nvidia-smi` lists no processes under WSL2, so the host client can't see another program (a game, a miner) using the card. You'll see it as slower micro-benchmarks, and the platform will too.
- Sleep and hibernate stop hosting. Set Windows power settings to keep the PC awake while you host.
