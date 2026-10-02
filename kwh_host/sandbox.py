"""The engine sandbox (HOST-CLIENT.md §3, D4).

One container per GPU runs the certified vLLM image with the pinned flags, serving the
reference model and nothing else. It isolates the host from buyer inputs: whatever a prompt
makes the engine do, it does with no network, a read-only root filesystem, no Linux
capabilities and no way to gain any, as an ordinary user, in bounded memory and processes.
What the container can reach:

    /hf        the Hugging Face cache, read-only: the checkpoint `kwh-host fetch` downloaded
               and checked against the lock's hashes; the engine loads it offline
    /cache     compile caches (torch, Triton, CUDA) and $HOME, so a restart skips recompiling
    /run/kwh   the directory holding the engine's Unix socket: the only way in or out
    /tmp       a size-limited tmpfs

With Docker Desktop (Windows or macOS) a Unix socket cannot cross from the container's VM to
the host, so `engine_transport = "tcp"` publishes the port on 127.0.0.1 instead. The engine
then still cannot be reached from the network, but it can reach out; the native Docker
Engine path (Linux, or inside WSL2) keeps it fully offline.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

from kwh_bench import reference as ref
from kwh_bench.engines import VLLMEngine

SOCKET_DIR_IN_CONTAINER = "/run/kwh"
SOCKET_NAME = "engine.sock"
HF_IN_CONTAINER = "/hf"
CACHE_IN_CONTAINER = "/cache"


NOBODY = 65534


def default_uid() -> int:
    """The engine runs as the daemon's user; if the daemon runs as root, as nobody instead."""
    return NOBODY if os.getuid() == 0 else os.getuid()


def default_gid() -> int:
    return NOBODY if os.getuid() == 0 else os.getgid()


@dataclass
class SandboxSpec:
    image: str
    name: str
    socket_dir: Path
    hf_home: Path
    cache_dir: Path
    max_model_len: int = 8192
    gpu_index: Optional[int] = 0          # None: no GPU (tests only)
    model: str = ref.MODEL_ID
    revision: Optional[str] = None
    transport: str = "uds"                # "uds": no network at all; "tcp": loopback port (Docker Desktop)
    port: int = 8000
    uid: int = field(default_factory=default_uid)
    gid: int = field(default_factory=default_gid)
    memory: str = "auto"                  # docker --memory; "auto" = 3/4 of RAM, capped at 64 GiB
    pids_limit: int = 4096
    shm_size: str = "2g"
    tmp_size: str = "4g"
    extra_docker_args: List[str] = field(default_factory=list)   # tests only

    @property
    def socket_path(self) -> Path:
        return self.socket_dir / SOCKET_NAME


def auto_memory() -> str:
    """Three quarters of this machine's RAM, at least 8 GiB, at most 64 GiB."""
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return "16g"
    return f"{max(8, min(64, int(total * 0.75 / 2**30)))}g"


def container_env() -> dict:
    cache = CACHE_IN_CONTAINER
    return {
        # No passwd entry exists for the host uid inside the image; torch reads USER instead.
        "HOME": f"{cache}/home", "USER": "kwh", "LOGNAME": "kwh",
        "HF_HOME": HF_IN_CONTAINER, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
        "XDG_CACHE_HOME": f"{cache}/xdg", "VLLM_CACHE_ROOT": f"{cache}/vllm", "TRITON_CACHE_DIR": f"{cache}/triton",
        "TORCHINDUCTOR_CACHE_DIR": f"{cache}/inductor", "CUDA_CACHE_PATH": f"{cache}/nv",
    }


def docker_run_argv(spec: SandboxSpec) -> List[str]:
    """`docker run` for the engine. The image's entrypoint is `vllm serve`; the model and the
    benchmark's pinned flags follow the image name."""
    mem = auto_memory() if spec.memory == "auto" else spec.memory
    argv = ["docker", "run", "--rm", "--init", "--name", spec.name, "--label", "kwh.engine=1",
            "--user", f"{spec.uid}:{spec.gid}",
            "--read-only", "--tmpfs", f"/tmp:rw,exec,nosuid,nodev,size={spec.tmp_size}",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
            "--pids-limit", str(spec.pids_limit), "--memory", mem, "--memory-swap", mem,
            "--shm-size", spec.shm_size,
            "-v", f"{spec.hf_home}:{HF_IN_CONTAINER}:ro", "-v", f"{spec.cache_dir}:{CACHE_IN_CONTAINER}:rw"]
    if spec.gpu_index is not None:
        argv += ["--gpus", f"device={spec.gpu_index}"]
    for k, v in container_env().items():
        argv += ["-e", f"{k}={v}"]
    if spec.transport == "uds":
        argv += ["--network", "none", "-v", f"{spec.socket_dir}:{SOCKET_DIR_IN_CONTAINER}:rw"]
        listen = ["--uds", f"{SOCKET_DIR_IN_CONTAINER}/{SOCKET_NAME}"]
    elif spec.transport == "tcp":
        argv += ["-p", f"127.0.0.1:{spec.port}:{spec.port}"]
        listen = ["--host", "0.0.0.0", "--port", str(spec.port)]
    else:
        raise ValueError(f"engine transport {spec.transport!r} is neither 'uds' nor 'tcp'")
    argv += list(spec.extra_docker_args)
    argv += [spec.image, spec.model] + ref.vllm_args(spec.max_model_len)
    if spec.revision:
        argv += ["--revision", spec.revision]
    return argv + listen


def prepare_dirs(spec: SandboxSpec) -> None:
    for d in (spec.socket_dir, spec.cache_dir, spec.cache_dir / "home", spec.hf_home):
        d.mkdir(parents=True, exist_ok=True)
    if os.getuid() == 0 and spec.uid != 0:
        # A root daemon hands the engine's two writable directories to the engine's user.
        for d in (spec.socket_dir, spec.cache_dir, spec.cache_dir / "home"):
            os.chown(d, spec.uid, spec.gid)
    os.chmod(spec.socket_dir, 0o700)
    if spec.socket_path.exists() or spec.socket_path.is_symlink():
        spec.socket_path.unlink()


def docker(*args: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def remove_container(name: str) -> None:
    """Remove the named engine container if it exists (a stale one from a crashed daemon)."""
    try:
        docker("rm", "-f", name)
    except (OSError, subprocess.TimeoutExpired):
        pass


def container_pid(name: str) -> Optional[int]:
    try:
        out = docker("inspect", "-f", "{{.State.Pid}}", name, timeout=5).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return int(out) if out.isdigit() and int(out) > 0 else None


def docker_is_rootless() -> bool:
    """Rootless Docker maps container root to the invoking user, so the engine runs as 0:0 there
    (still unprivileged on the host) and files it writes come out owned by the user."""
    try:
        out = docker("info", "--format", "{{json .SecurityOptions}}", timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "rootless" in out


def redact_home(arg: str) -> str:
    """Reports are public: never publish the host's home directory (it carries the user name)."""
    home = str(Path.home())
    return arg.replace(home, "~") if home and home != "/" else arg


class SandboxedVLLMEngine(VLLMEngine):
    """The benchmark's VLLMEngine with the sandbox as its launcher. Everything after launch
    (the calls, the version and model checks, what the benchmark measures) is the benchmark's."""

    def __init__(self, spec: SandboxSpec, log_path: Optional[str] = None):
        self.spec = spec
        super().__init__(model=spec.model, revision=spec.revision, docker_image=spec.image, port=spec.port,
                         log_path=log_path, max_model_len=spec.max_model_len,
                         uds=str(spec.socket_path) if spec.transport == "uds" else None)

    def build_argv(self) -> List[str]:
        return docker_run_argv(self.spec)

    async def start(self) -> None:
        prepare_dirs(self.spec)
        remove_container(self.spec.name)
        try:
            await super().start()
        except BaseException:
            remove_container(self.spec.name)
            raise

    async def stop(self) -> None:
        try:
            await super().stop()
        finally:
            remove_container(self.spec.name)

    async def info(self):
        info = await super().info()
        info.launch_args = [redact_home(a) for a in info.launch_args]
        return info

    def own_pids(self) -> Optional[Set[int]]:
        from .gpu import process_tree
        pid = container_pid(self.spec.name)
        return process_tree(pid) if pid else None
