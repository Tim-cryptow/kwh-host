import json
import time
from pathlib import Path

import pytest
from kwh_bench.engines import MockEngine
from kwh_bench.runner import run_benchmark

from kwh_host.config import HostConfig
from kwh_host.identity import Identity


class FakeClock:
    def __init__(self, t: float | None = None):
        self.t = time.time() if t is None else t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("KWH_HOST_HOME", str(tmp_path / "kwh-host"))
    return tmp_path / "kwh-host"


@pytest.fixture
def cfg(home):
    c = HostConfig(platform_url="http://mock", engine_mode="bare-metal", bare_metal_ok=True)
    c.save()
    return c


@pytest.fixture
def identity(cfg):
    return Identity.load_or_create(cfg.identity_path)


@pytest.fixture
def mock_engine():
    return MockEngine(step_ms=0.02, prefill_ms_per_1k=0.05)


@pytest.fixture
def report(identity, mock_engine, cfg):
    """A signed benchmark report from the mock engine (uncertified; the mock platform accepts it in test mode)."""
    import asyncio
    r = asyncio.run(run_benchmark(mock_engine, measured_jobs=3, log=lambda s: None))
    identity.sign_report(r)
    cfg.report_path.write_text(json.dumps(r, indent=2))
    return r
