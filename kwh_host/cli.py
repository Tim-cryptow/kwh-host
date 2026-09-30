"""kwh-host command line: init | bench | register | run | status | mock-platform."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import click
from kwh_bench import reference as ref

from . import __version__
from .config import DEFAULT_DOCKER_IMAGE, HostConfig, read_state
from .identity import Identity


def _log(s: str) -> None:
    click.echo(s, err=True)


def _load() -> tuple[HostConfig, Identity]:
    cfg = HostConfig.load()
    return cfg, Identity.load(cfg.identity_path)


def _engine(cfg: HostConfig, log_path: Optional[str], mock: bool):
    """The daemon-owned engine; `mock` (hidden) exercises the whole flow without a GPU."""
    if mock:
        from kwh_bench.engines import MockEngine
        return MockEngine(step_ms=0.05, prefill_ms_per_1k=0.1)
    from .engine import make_engine
    return make_engine(cfg, log_path=log_path)


@click.group()
@click.version_option(__version__, prog_name="kwh-host")
def main():
    """kWh Exchange host client. See HOST-CLIENT.md."""


@main.command()
@click.option("--platform", "platform_url", default="http://127.0.0.1:9000", show_default=True, help="Platform base URL.")
@click.option("--engine", "engine_mode", type=click.Choice(["docker", "bare-metal"]), default="docker", show_default=True)
@click.option("--docker-image", default=DEFAULT_DOCKER_IMAGE, show_default=True)
@click.option("--port", type=int, default=8000, show_default=True, help="Engine port.")
@click.option("--gpu", "gpu_index", type=int, default=0, show_default=True)
@click.option("--hf-cache", default=None, help="Host HF cache dir to mount into the container.")
@click.option("--allow-bare-metal", is_flag=True, help="Testing on pods without Docker; the platform must also allow it.")
def init(platform_url, engine_mode, docker_image, port, gpu_index, hf_cache, allow_bare_metal):
    """Create ~/.kwh-host: config + identity keypair."""
    if engine_mode == "bare-metal" and not allow_bare_metal:
        raise click.UsageError("bare-metal is for testing only; pass --allow-bare-metal to confirm (D4)")
    cfg = HostConfig(platform_url=platform_url, engine_mode=engine_mode, docker_image=docker_image, engine_port=port,
                     gpu_index=gpu_index, hf_cache=hf_cache, bare_metal_ok=allow_bare_metal)
    if cfg.path.exists():
        old = HostConfig.load()
        cfg.host_id, cfg.token = old.host_id, old.token
    cfg.save()
    ident = Identity.load_or_create(cfg.identity_path)
    click.echo(json.dumps({"dir": str(cfg.dir), "platform": cfg.platform_url, "engine": cfg.engine_mode,
                           "public_key": ident.public_key_hex, "registered": cfg.registered}, indent=2))


@main.command()
@click.option("--runs", type=int, default=ref.DEFAULT_MEASURED_JOBS, show_default=True)
@click.option("--engine-log", type=click.Path(path_type=Path), default=None)
@click.option("--ignore-preflight", is_flag=True, help="Run on a busy GPU (report will be uncertified).")
@click.option("--mock-engine", is_flag=True, hidden=True)
def bench(runs, engine_log, ignore_preflight, mock_engine):
    """Run the Grade I benchmark with a daemon-launched engine; sign and verify the report."""
    from .bench import full_benchmark
    cfg, ident = _load()
    engine = _engine(cfg, str(engine_log) if engine_log else str(cfg.dir / "engine-bench.log"), mock_engine)
    try:
        report = asyncio.run(full_benchmark(engine, ident, cfg.report_path, runs=runs, log=_log, ignore_preflight=ignore_preflight))
    except Exception as e:  # noqa: BLE001
        _log(f"error: {type(e).__name__}: {e}")
        sys.exit(2)
    from kwh_bench.report import summarize
    click.echo(summarize(report))
    click.echo(f"report: {cfg.report_path}", err=True)


@main.command()
def register():
    """Send the signed report to the platform; store host_id and token."""
    from .bench import load_report
    from .platform.client import PlatformClient, PlatformError
    cfg, ident = _load()
    if not cfg.report_path.exists():
        raise click.UsageError("no report yet; run `kwh-host bench` first")
    report = load_report(cfg.report_path)

    async def go():
        async with PlatformClient(cfg.platform_url, ident) as c:
            return await c.register(report)

    try:
        out = asyncio.run(go())
    except PlatformError as e:
        _log(f"registration refused: {e.detail}")
        sys.exit(1)
    cfg.host_id, cfg.token = out["host_id"], out["token"]
    cfg.save()
    click.echo(json.dumps({k: out[k] for k in ("host_id", "rate_units_per_hour", "bucket")}, indent=2))


@main.command()
@click.option("--engine-log", type=click.Path(path_type=Path), default=None)
@click.option("--beats", type=int, default=None, hidden=True, help="Stop after N heartbeats (tests).")
@click.option("--no-version-check", is_flag=True, hidden=True)
@click.option("--mock-engine", is_flag=True, hidden=True)
def run(engine_log, beats, no_version_check, mock_engine):
    """Start the engine and the heartbeat loop; stay live until stopped."""
    from .daemon import Daemon
    from .platform.client import PlatformClient
    cfg, ident = _load()
    if not cfg.registered:
        raise click.UsageError("not registered; run `kwh-host bench` then `kwh-host register`")
    engine = _engine(cfg, str(engine_log) if engine_log else str(cfg.dir / "engine.log"), mock_engine)
    no_version_check = no_version_check or mock_engine

    async def go():
        async with PlatformClient(cfg.platform_url, ident, token=cfg.token, host_id=cfg.host_id) as c:
            d = Daemon(cfg, c, engine, log=_log, max_beats=beats, require_lock_version=not no_version_check)
            loop = asyncio.get_running_loop()
            try:
                import signal
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(sig, d.stop)
            except (NotImplementedError, ImportError):
                pass
            return await d.run()

    try:
        final = asyncio.run(go())
    except Exception as e:  # noqa: BLE001
        _log(f"error: {type(e).__name__}: {e}")
        sys.exit(2)
    click.echo(json.dumps({k: final[k] for k in ("state", "beats", "accepted_beats", "minted_total", "balance")}, indent=2))


@main.command()
@click.option("--remote/--local", default=True, help="Ask the platform (default) or show the last local state only.")
def status(remote):
    """Show host state: platform verdict, rate, accrual, last checks."""
    cfg, ident = _load()
    local = read_state(cfg)
    out = {"host_id": cfg.host_id, "platform": cfg.platform_url, "engine": cfg.engine_mode, "local": local}
    if remote and cfg.registered:
        from .platform.client import PlatformClient, PlatformError

        async def go():
            async with PlatformClient(cfg.platform_url, ident, token=cfg.token, host_id=cfg.host_id) as c:
                return await c.status()

        try:
            out["platform_view"] = asyncio.run(go())
        except (PlatformError, Exception) as e:  # noqa: BLE001
            out["platform_view"] = {"error": str(e)}
    click.echo(json.dumps(out, indent=2, default=str))


@main.command("mock-platform")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=9000, show_default=True)
@click.option("--heartbeat", type=float, default=30.0, show_default=True, help="Seconds between heartbeats.")
@click.option("--challenge-every", type=float, default=300.0, show_default=True)
@click.option("--microbench-every", type=float, default=1800.0, show_default=True)
@click.option("--accept-uncertified", is_flag=True, help="Tests only: accept uncertified reports and any engine version.")
@click.option("--allow-bare-metal", is_flag=True, help="Pod testing: accept non-Docker engines (D4).")
def mock_platform(host, port, heartbeat, challenge_every, microbench_every, accept_uncertified, allow_bare_metal):
    """Run the in-memory mock platform (HOST-CLIENT.md §8) for local development."""
    import uvicorn
    from .platform.mock import ChallengePool, MockPlatform, Settings, create_app
    s = Settings(heartbeat_seconds=heartbeat, challenge_every_seconds=challenge_every, microbench_every_seconds=microbench_every,
                 accept_uncertified=accept_uncertified, allow_bare_metal=allow_bare_metal)
    app = create_app(MockPlatform(ChallengePool.from_lock(), s))
    _log(f"mock platform on http://{host}:{port}  (challenges from the public lock; not a guard)")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
