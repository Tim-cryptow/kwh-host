"""Host-side state: ~/.kwh-host/ (override with KWH_HOST_HOME).

    config.json    platform URL, engine mode, registration (host_id, token)
    identity.key   ed25519 private key (0600)
    report.json    the latest certified kwh-bench report, signed
    state.json     last known daemon state, for `kwh-host status` while the daemon is down
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# The certified engine (the version the lock pins) is published in two builds. The default one is
# built on CUDA 13 and will not start under an older driver ("driver too old"); the cu129 build of
# the same version runs on CUDA 12.x drivers. `kwh-host init` picks by the driver (engine_image_for).
DEFAULT_DOCKER_IMAGE = "vllm/vllm-openai:v0.30.0"            # CUDA 13: driver 580 or newer
CUDA12_DOCKER_IMAGE = "vllm/vllm-openai:v0.30.0-cu129"       # CUDA 12.9: drivers that support CUDA 12.x
IMAGE_CUDA = {DEFAULT_DOCKER_IMAGE: (13, 0), CUDA12_DOCKER_IMAGE: (12, 0)}   # oldest driver CUDA each build runs on
TESTED_CUDA12 = (12, 8)       # oldest 12.x driver the cu129 build has run on (RTX 3090, driver 570, bare metal, 2026-09-30)


def cuda_tuple(version: Optional[str]) -> Optional[tuple]:
    """'12.8' -> (12, 8); None or unparseable -> None."""
    try:
        major, _, minor = (version or "").strip().partition(".")
        return int(major), int(minor or 0)
    except ValueError:
        return None


def engine_image_for(driver_cuda: Optional[str]) -> str:
    """The engine build for a driver that supports `driver_cuda` (nvidia-smi's "CUDA Version")."""
    cuda = cuda_tuple(driver_cuda)
    return CUDA12_DOCKER_IMAGE if cuda is not None and cuda[0] == 12 else DEFAULT_DOCKER_IMAGE


def home() -> Path:
    return Path(os.environ.get("KWH_HOST_HOME", Path.home() / ".kwh-host")).expanduser()


@dataclass
class HostConfig:
    platform_url: str = "http://127.0.0.1:9000"
    engine_mode: str = "docker"                 # "docker" (D4) | "bare-metal" (testing only; refused for registration)
    docker_image: str = DEFAULT_DOCKER_IMAGE
    engine_port: int = 8000
    # Context length the engine serves and is certified at (D8; the benchmark certifies 1024-8192).
    # It is the longest buyer request this host can take; the rate does not depend on it.
    max_model_len: int = 8192
    gpu_index: int = 0
    hf_cache: Optional[str] = None              # HF_HOME for the engine, mounted read-only (default ~/.kwh-host/hf)
    engine_transport: str = "uds"               # "uds": engine has no network; "tcp": loopback port (Docker Desktop)
    engine_memory: Optional[str] = None         # docker --memory for the engine; None = 3/4 of RAM, max 64 GiB
    host_id: Optional[str] = None
    token: Optional[str] = None
    bare_metal_ok: bool = False                 # set by `init --allow-bare-metal` for pod testing
    extra: dict = field(default_factory=dict)

    # -- paths ---------------------------------------------------------
    @property
    def dir(self) -> Path:
        return home()

    @property
    def path(self) -> Path:
        return self.dir / "config.json"

    @property
    def identity_path(self) -> Path:
        return self.dir / "identity.key"

    @property
    def report_path(self) -> Path:
        return self.dir / "report.json"

    @property
    def state_path(self) -> Path:
        return self.dir / "state.json"

    @property
    def hf_home(self) -> Path:
        return Path(self.hf_cache).expanduser() if self.hf_cache else self.dir / "hf"

    @property
    def registered(self) -> bool:
        return bool(self.host_id and self.token)

    # -- io ------------------------------------------------------------
    @classmethod
    def load(cls) -> "HostConfig":
        p = home() / "config.json"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; run `kwh-host init` first")
        d = json.loads(p.read_text(encoding="utf-8"))
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        return self.path


def write_state(cfg: HostConfig, state: dict) -> None:
    cfg.dir.mkdir(parents=True, exist_ok=True)
    cfg.state_path.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")


def read_state(cfg: HostConfig) -> Optional[dict]:
    if not cfg.state_path.exists():
        return None
    return json.loads(cfg.state_path.read_text(encoding="utf-8"))
