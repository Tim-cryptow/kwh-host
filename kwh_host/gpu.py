"""GPU sample for the heartbeat (HOST-CLIENT.md §4): utilization, power, VRAM, and whether
any compute process other than ours is on the card. Uses nvidia-smi like the benchmark does."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Iterable, Optional, Set


def _run(argv: list[str], timeout: float = 5.0) -> Optional[str]:
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _f(s: str) -> Optional[float]:
    try:
        return float(s)
    except ValueError:
        return None


def sample(gpu_index: int = 0, own_pids: Optional[Iterable[int]] = None) -> dict:
    """One nvidia-smi sample. `own_pids` are the engine's processes; anything else holding the
    GPU counts as foreign. With `own_pids=None` the foreign count is unknown (None), not zero."""
    out = {"available": False, "util_pct": None, "power_w": None, "mem_used_mib": None, "mem_total_mib": None,
           "temp_c": None, "compute_processes": None, "foreign_processes": None, "foreign_pids": [],
           "unattributed_processes": 0}
    if not shutil.which("nvidia-smi"):
        return out
    q = _run(["nvidia-smi", "-i", str(gpu_index), "--query-gpu=utilization.gpu,power.draw,memory.used,memory.total,temperature.gpu",
              "--format=csv,noheader,nounits"])
    if not q:
        return out
    vals = [v.strip() for v in q.splitlines()[0].split(",")]
    if len(vals) != 5:
        return out
    out.update({"available": True, "util_pct": _f(vals[0]), "power_w": _f(vals[1]), "mem_used_mib": _f(vals[2]),
                "mem_total_mib": _f(vals[3]), "temp_c": _f(vals[4])})
    apps = _run(["nvidia-smi", "-i", str(gpu_index), "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
    pids: Set[int] = set()
    if apps:
        for line in apps.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if parts and parts[0].isdigit():
                pids.add(int(parts[0]))
    out["compute_processes"] = len(pids)
    if own_pids is not None:
        own = set(own_pids)
        # Inside a container nvidia-smi reports host-namespace PIDs, which match nothing here.
        # A PID we cannot see in /proc is unattributable, not foreign; only a visible stranger counts.
        visible = {p for p in pids if p in own or os.path.exists(f"/proc/{p}")}
        foreign = sorted(p for p in visible if p not in own)
        out["foreign_processes"] = len(foreign)
        out["foreign_pids"] = foreign
        out["unattributed_processes"] = len(pids - visible)
    return out


def process_tree(root_pid: int) -> Set[int]:
    """The engine process and all descendants (vLLM forks workers)."""
    try:
        import psutil
        p = psutil.Process(root_pid)
        return {root_pid, *(c.pid for c in p.children(recursive=True))}
    except Exception:  # noqa: BLE001
        return {root_pid}
