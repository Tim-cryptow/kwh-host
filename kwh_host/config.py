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

DEFAULT_DOCKER_IMAGE = "vllm/vllm-openai:v0.30.0"   # same version the lock pins


def home() -> Path:
    return Path(os.environ.get("KWH_HOST_HOME", Path.home() / ".kwh-host")).expanduser()


@dataclass
class HostConfig:
    platform_url: str = "http://127.0.0.1:9000"
    engine_mode: str = "docker"                 # "docker" (D4) | "bare-metal" (testing only; refused for registration)
    docker_image: str = DEFAULT_DOCKER_IMAGE
    engine_port: int = 8000
    gpu_index: int = 0
    hf_cache: Optional[str] = None              # host dir mounted read-only into the container
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
