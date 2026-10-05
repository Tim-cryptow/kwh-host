"""The daemon's engine (HOST-CLIENT.md §3): one long-lived vLLM serving the Grade I reference
model with the pinned flags, launched and owned by the daemon. Docker mode runs it in the
sandbox (`sandbox.py`); bare-metal mode, for pods that cannot run Docker, runs `vllm serve`
directly. Both reuse the benchmark's `VLLMEngine`, so the flags, version check and inference
calls are the certified ones."""

from __future__ import annotations

from typing import Optional, Set

import httpx
from kwh_bench import reference as ref
from kwh_bench.engines import VLLMEngine
from kwh_bench.lockfile import Lock, load_lock

from .config import HostConfig
from .gpu import process_tree
from .sandbox import SandboxedVLLMEngine, SandboxSpec, docker_is_rootless


def sandbox_spec(cfg: HostConfig, model: str, revision: Optional[str]) -> SandboxSpec:
    rootless = docker_is_rootless()
    return SandboxSpec(image=cfg.docker_image, name=f"kwh-engine-gpu{cfg.gpu_index}", socket_dir=cfg.dir / "run",
                       hf_home=cfg.hf_home, cache_dir=cfg.dir / "engine-cache", max_model_len=cfg.max_model_len,
                       gpu_index=None if cfg.extra.get("no_gpu") else cfg.gpu_index, model=model,
                       revision=revision, transport=cfg.engine_transport,
                       port=cfg.engine_port, memory=cfg.engine_memory or "auto",
                       **({"uid": 0, "gid": 0} if rootless else {}))


def make_engine(cfg: HostConfig, log_path: Optional[str] = None, lock: Optional[Lock] = None,
                model: Optional[str] = None) -> VLLMEngine:
    """The certified engine, at the context length in the config (the benchmark certifies the
    same value, so the rate is measured on the engine that serves). `model` overrides the
    reference model for wrong-model tests only: such an engine fails the platform's challenges
    and never goes live."""
    lock = lock or load_lock()
    revision = None if model else lock.model_revision
    if cfg.engine_mode == "docker":
        return SandboxedVLLMEngine(sandbox_spec(cfg, model or ref.MODEL_ID, revision), log_path=log_path)
    return VLLMEngine(model=model or ref.MODEL_ID, revision=revision, port=cfg.engine_port, log_path=log_path,
                      max_model_len=cfg.max_model_len)


def launch_mode(cfg: HostConfig) -> str:
    return "docker" if cfg.engine_mode == "docker" else "subprocess"


async def health(engine: VLLMEngine) -> bool:
    try:
        async with engine.http_client(timeout=5.0) as c:
            r = await c.get("/health")
            return r.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def running_image(cfg: HostConfig) -> Optional[str]:
    """The engine image the daemon launches (pinned by digest); None outside Docker."""
    return cfg.docker_image if cfg.engine_mode == "docker" else None


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
            "image": running_image(cfg), "port": cfg.engine_port}


def own_pids(engine: VLLMEngine, cfg: HostConfig) -> Optional[Set[int]]:
    """PIDs that legitimately hold the GPU: the engine process tree (bare metal) or the
    container's process tree (docker). None when unknown."""
    if isinstance(engine, SandboxedVLLMEngine):
        return engine.own_pids()
    proc = getattr(engine, "_proc", None)
    popen = getattr(proc, "proc", None)
    if popen is None or popen.poll() is not None:
        return None
    return process_tree(popen.pid)


def check_version(engine_version: Optional[str], lock: Optional[Lock] = None) -> Optional[str]:
    """Reason the engine may not serve, or None when it matches the lock."""
    lock = lock or load_lock()
    if not lock.is_locked:
        return "kwh-bench lock is incomplete"
    if engine_version != lock.vllm_version:
        return f"engine version {engine_version!r} != locked {lock.vllm_version!r}"
    return None
