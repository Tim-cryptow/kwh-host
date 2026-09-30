"""The engine sandbox (HOST-CLIENT.md §3): one long-lived vLLM serving the Grade I reference
model with the pinned flags, launched and owned by the daemon. Reuses the benchmark's
`VLLMEngine`, so the flags, version check and inference calls are the certified ones."""

from __future__ import annotations

import subprocess
from typing import Optional, Set

import httpx
from kwh_bench.engines import VLLMEngine
from kwh_bench.lockfile import Lock, load_lock

from .config import HostConfig
from .gpu import process_tree


def make_engine(cfg: HostConfig, log_path: Optional[str] = None, lock: Optional[Lock] = None) -> VLLMEngine:
    lock = lock or load_lock()
    docker = cfg.docker_image if cfg.engine_mode == "docker" else None
    return VLLMEngine(revision=lock.model_revision, docker_image=docker, port=cfg.engine_port,
                      log_path=log_path, hf_cache=cfg.hf_cache)


def launch_mode(cfg: HostConfig) -> str:
    return "docker" if cfg.engine_mode == "docker" else "subprocess"


async def health(engine: VLLMEngine) -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{engine.base_url}/health")
            return r.status_code == 200
    except httpx.HTTPError:
        return False


async def describe(engine: VLLMEngine, cfg: HostConfig) -> dict:
    """The `engine` block of a heartbeat."""
    ok = await health(engine)
    version = served = None
    if ok:
        try:
            info = await engine.info()
            version, served = info.version, info.extra.get("served_model")
        except Exception:  # noqa: BLE001
            ok = False
    return {"healthy": ok, "version": version, "served_model": served, "launch_mode": launch_mode(cfg),
            "port": cfg.engine_port}


def own_pids(engine: VLLMEngine, cfg: HostConfig) -> Optional[Set[int]]:
    """PIDs that legitimately hold the GPU: the engine process tree (bare metal) or the
    container's process tree (docker). None when unknown."""
    proc = getattr(engine, "_proc", None)
    popen = getattr(proc, "proc", None)
    if popen is None or popen.poll() is not None:
        return None
    if cfg.engine_mode != "docker":
        return process_tree(popen.pid)
    try:
        cid = subprocess.run(["docker", "ps", "-q", "--filter", f"publish={cfg.engine_port}"],
                             capture_output=True, text=True, timeout=5).stdout.split()
        if not cid:
            return None
        pid = subprocess.run(["docker", "inspect", "-f", "{{.State.Pid}}", cid[0]],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return process_tree(int(pid)) | process_tree(popen.pid) if pid.isdigit() else None
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def check_version(engine_version: Optional[str], lock: Optional[Lock] = None) -> Optional[str]:
    """Reason the engine may not serve, or None when it matches the lock."""
    lock = lock or load_lock()
    if not lock.is_locked:
        return "kwh-bench lock is incomplete"
    if engine_version != lock.vllm_version:
        return f"engine version {engine_version!r} != locked {lock.vllm_version!r}"
    return None
