"""`kwh-host service`: run the daemon as a systemd user service, so it starts with the machine
and comes back after a crash. A user service, not a system one: the daemon never needs root.

Without `loginctl enable-linger`, a user's services stop when their last session ends; the
installer and `service install` say so. On WSL2, systemd must be on (`[boot] systemd=true`
in /etc/wsl.conf, the default for new Ubuntu installs) and the distribution has to be running.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

UNIT_NAME = "kwh-host.service"


def unit_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")).expanduser()
    return base / "systemd" / "user" / UNIT_NAME


def executable() -> str:
    """The kwh-host to run: the one running now, as installed."""
    found = shutil.which("kwh-host")
    argv0 = Path(sys.argv[0])
    if argv0.name == "kwh-host" and argv0.exists():
        return str(argv0.resolve())
    if found:
        return str(Path(found).resolve())
    return f"{sys.executable} -m kwh_host.cli"


def unit_text(exe: str, kwh_home: Optional[str] = None) -> str:
    env = f"Environment=KWH_HOST_HOME={kwh_home}\n" if kwh_home else ""
    return f"""[Unit]
Description=kWh Exchange host client (sells this GPU's work; see kwh-host status)
Documentation=https://github.com/Tim-cryptow/kwh-host
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
ExecStart={exe} run
Restart=on-failure
RestartSec=30
TimeoutStopSec=90
{env}
[Install]
WantedBy=default.target
"""


def systemctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True, timeout=60)


def install(start: bool = True) -> List[str]:
    """Write the unit, enable it, start it. Returns what was done, line by line."""
    path = unit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(unit_text(executable(), os.environ.get("KWH_HOST_HOME")), encoding="utf-8")
    done = [f"wrote {path}"]
    for args in (("daemon-reload",), ("enable", "--now", UNIT_NAME) if start else ("enable", UNIT_NAME)):
        r = systemctl(*args)
        if r.returncode != 0:
            raise RuntimeError(f"systemctl --user {' '.join(args)} failed: {(r.stderr or r.stdout).strip()}")
        done.append(f"systemctl --user {' '.join(args)}")
    return done


def uninstall() -> List[str]:
    done = []
    r = systemctl("disable", "--now", UNIT_NAME)
    if r.returncode == 0:
        done.append(f"systemctl --user disable --now {UNIT_NAME}")
    path = unit_path()
    if path.exists():
        path.unlink()
        done.append(f"removed {path}")
    systemctl("daemon-reload")
    return done


def lingering() -> Optional[bool]:
    """Whether this user's services keep running with nobody logged in (None if unknown)."""
    user = os.environ.get("USER") or ""
    if not user or not shutil.which("loginctl"):
        return None
    r = subprocess.run(["loginctl", "show-user", user, "--property=Linger"], capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        return None
    return r.stdout.strip().endswith("yes")
