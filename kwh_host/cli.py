"""kwh-host command line: init | fetch | doctor | bench | register | run | status | events | service | burst | mock-platform | submit."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import click
from kwh_bench import reference as ref

from . import __version__
from .config import CUDA12_DOCKER_IMAGE, DEFAULT_DOCKER_IMAGE, IMAGE_TAGS, HostConfig, engine_image_for, read_state
from .identity import Identity


def _log(s: str) -> None:
    click.echo(s, err=True)


def _load() -> tuple[HostConfig, Identity]:
    cfg = HostConfig.load()
    return cfg, Identity.load(cfg.identity_path)


def _engine(cfg: HostConfig, log_path: Optional[str], mock: bool, model: Optional[str] = None):
    """The daemon-owned engine; `mock` (hidden) exercises the whole flow without a GPU, `model`
    (hidden) serves something other than the reference model to prove the platform catches it."""
    if mock:
        from kwh_bench.engines import MockEngine
        return MockEngine(step_ms=0.05, prefill_ms_per_1k=0.1)
    from .engine import make_engine
    return make_engine(cfg, log_path=log_path, model=model)


@click.group()
@click.version_option(__version__, prog_name="kwh-host")
def main():
    """kWh Exchange host client. See HOST-CLIENT.md."""


@main.command()
@click.option("--platform", "platform_url", default="http://127.0.0.1:9000", show_default=True, help="Platform base URL.")
@click.option("--engine", "engine_mode", type=click.Choice(["docker", "bare-metal"]), default="docker", show_default=True)
@click.option("--docker-image", default=None,
              help=f"Engine image. Default: {IMAGE_TAGS[DEFAULT_DOCKER_IMAGE]} for CUDA 13 drivers, "
                   f"{IMAGE_TAGS[CUDA12_DOCKER_IMAGE]} for CUDA 12.x, both pinned by digest.")
@click.option("--port", type=int, default=8000, show_default=True, help="Engine port.")
@click.option("--gpu", "gpu_index", type=int, default=0, show_default=True)
@click.option("--hf-cache", default=None, help="Host HF cache dir to mount into the container.")
@click.option("--allow-bare-metal", is_flag=True, help="Testing on pods without Docker; the platform must also allow it.")
@click.option("--max-model-len", type=click.IntRange(ref.MAX_MODEL_LEN_MIN, ref.MAX_MODEL_LEN_MAX), default=8192,
              show_default=True, help="Context length to certify and serve: the longest buyer request this host takes.")
@click.option("--engine-transport", type=click.Choice(["uds", "tcp"]), default="uds", show_default=True,
              help="uds: the engine has no network, reached through a Unix socket. tcp: a loopback port, for Docker Desktop.")
@click.option("--engine-memory", default=None, help="Memory limit for the engine container, e.g. 24g (default: 3/4 of RAM).")
@click.option("--no-gpu", is_flag=True, hidden=True, help="Tests only: run the engine container without a GPU.")
def init(platform_url, engine_mode, docker_image, port, gpu_index, hf_cache, allow_bare_metal, max_model_len,
         engine_transport, engine_memory, no_gpu):
    """Create ~/.kwh-host: config + identity keypair."""
    if engine_mode == "bare-metal" and not allow_bare_metal:
        raise click.UsageError("bare-metal is for testing only; pass --allow-bare-metal to confirm (D4)")
    driver_cuda = None
    if docker_image is None:
        from kwh_bench.hardware import probe_cuda_version
        driver_cuda = probe_cuda_version()
        docker_image = engine_image_for(driver_cuda)
    cfg = HostConfig(platform_url=platform_url, engine_mode=engine_mode, docker_image=docker_image, engine_port=port,
                     gpu_index=gpu_index, hf_cache=hf_cache, bare_metal_ok=allow_bare_metal, max_model_len=max_model_len,
                     engine_transport=engine_transport, engine_memory=engine_memory,
                     extra={"no_gpu": True} if no_gpu else {})
    if cfg.path.exists():
        old = HostConfig.load()
        cfg.host_id, cfg.token = old.host_id, old.token
    cfg.save()
    ident = Identity.load_or_create(cfg.identity_path)
    out = {"dir": str(cfg.dir), "platform": cfg.platform_url, "engine": cfg.engine_mode,
           "max_model_len": cfg.max_model_len, "public_key": ident.public_key_hex, "registered": cfg.registered}
    if cfg.engine_mode == "docker":
        out["engine_image"] = cfg.docker_image
        out["engine_build"] = IMAGE_TAGS.get(cfg.docker_image, cfg.docker_image)
        if driver_cuda:
            out["driver_cuda"] = driver_cuda
    click.echo(json.dumps(out, indent=2))


@main.command()
@click.option("--model", default=None, hidden=True, help="Fetch another model (wrong-model tests); not hash-checked.")
@click.option("--no-image", is_flag=True, help="Skip pulling the engine image (bare metal, or already pulled).")
def fetch(model, no_image):
    """Download the reference checkpoint at the locked revision, check its hashes, pull the engine image."""
    from kwh_bench.lockfile import load_lock
    from .fetch import fetch as do_fetch
    cfg = HostConfig.load()
    image = None if (no_image or cfg.engine_mode != "docker") else cfg.docker_image
    try:
        out = do_fetch(cfg.hf_home, load_lock(), image, log=_log, model=model)
    except Exception as e:  # noqa: BLE001
        _log(f"error: {type(e).__name__}: {e}")
        sys.exit(2)
    click.echo(json.dumps(out, indent=2))


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output.")
def doctor(as_json):
    """Check this machine is ready to host: driver, GPU, Docker, GPU in containers, image, checkpoint."""
    from kwh_bench.lockfile import load_lock
    from .doctor import run_checks
    try:
        cfg = HostConfig.load()
    except FileNotFoundError:
        cfg = HostConfig()
    checks = run_checks(cfg, load_lock().model_revision)
    if as_json:
        click.echo(json.dumps([c.__dict__ for c in checks], indent=2))
    else:
        for c in checks:
            click.echo(c.line())
    if any(c.status == "fail" for c in checks):
        sys.exit(1)


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
@click.option("--model", default=None, hidden=True, help="Serve this model instead of the reference (wrong-model tests).")
@click.option("--no-jobs", is_flag=True, help="Heartbeat and challenges only; do not open the job channel.")
@click.option("--no-rebench", is_flag=True, hidden=True, help="Tests only: never re-benchmark (the platform holds the host degraded).")
def run(engine_log, beats, no_version_check, mock_engine, model, no_jobs, no_rebench):
    """Start the engine, heartbeat, serve jobs, and re-benchmark when due; stay live until stopped."""
    from .daemon import Daemon
    from .platform.client import PlatformClient
    cfg, ident = _load()
    if not cfg.registered:
        raise click.UsageError("not registered; run `kwh-host bench` then `kwh-host register`")
    if cfg.report_path.exists() and not mock_engine:
        from .bench import context_mismatch, load_report
        problem = context_mismatch(load_report(cfg.report_path), cfg.max_model_len)
        if problem:
            raise click.UsageError(problem)
    engine = _engine(cfg, str(engine_log) if engine_log else str(cfg.dir / "engine.log"), mock_engine, model)
    no_version_check = no_version_check or mock_engine

    def bench_engine():
        """A re-benchmark (§5) runs on a fresh engine with its own log, like `kwh-host bench`."""
        return _engine(cfg, str(cfg.dir / "engine-bench.log"), mock_engine, model)

    async def go():
        from .events import EventLog
        async with PlatformClient(cfg.platform_url, ident, token=cfg.token, host_id=cfg.host_id) as c:
            d = Daemon(cfg, c, engine, log=_log, max_beats=beats, require_lock_version=not no_version_check,
                       jobs=not no_jobs, bench_engine_factory=bench_engine, events=EventLog(cfg.events_path),
                       rebench=not no_rebench)
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
    click.echo(json.dumps({k: final[k] for k in ("state", "beats", "accepted_beats", "minted_total", "balance", "jobs")},
                          indent=2))


def _platform_call(cfg: HostConfig, ident: Identity, call):
    """Run `call(client)` against the platform; returns (result, None) or (None, error text)."""
    from .platform.client import PlatformClient, PlatformError

    async def go():
        async with PlatformClient(cfg.platform_url, ident, token=cfg.token, host_id=cfg.host_id, timeout=10.0) as c:
            return await call(c)

    try:
        return asyncio.run(go()), None
    except PlatformError as e:
        return None, f"{e.status}: {e.detail}"
    except Exception as e:  # noqa: BLE001 - unreachable, DNS, TLS: say so and show what we have
        return None, f"{type(e).__name__}: {e}"[:300]


@main.command()
@click.option("--remote/--local", default=True, help="Ask the platform (default) or show the daemon's last state only.")
@click.option("--json", "as_json", is_flag=True, help="Everything, machine-readable.")
def status(remote, as_json):
    """Host state: the platform's verdict, rate, earnings, engine, GPU, last checks, reliability."""
    from .config import image_label
    from .status import render_status
    cfg, ident = _load()
    local = read_state(cfg)
    platform_view = None
    if remote and cfg.registered:
        platform_view, err = _platform_call(cfg, ident, lambda c: c.status())
        if err:
            platform_view = {"error": err}
    if as_json:
        click.echo(json.dumps({"host_id": cfg.host_id, "platform": cfg.platform_url, "engine": cfg.engine_mode,
                               "local": local, "platform_view": platform_view}, indent=2, default=str))
        return
    host = {"version": __version__, "host_id": cfg.host_id, "platform": cfg.platform_url,
            "engine": cfg.engine_mode, "image": image_label(cfg.docker_image) if cfg.engine_mode == "docker" else None}
    click.echo(render_status(host, local, platform_view))


@main.command()
@click.option("--remote", is_flag=True, help="The platform's log of this host instead of the daemon's own.")
@click.option("--since", default=None, help="Only events in the last e.g. 30m, 6h or 2d.")
@click.option("--limit", type=int, default=50, show_default=True, help="The last N events.")
@click.option("--kind", "kinds", multiple=True, help="Only this kind (repeat for more): heartbeat, challenge, job, rebench, state, ...")
@click.option("--all", "show_all", is_flag=True, help="Include routine events (accepted heartbeats, issued challenges).")
@click.option("--json", "as_json", is_flag=True, help="One JSON object per line.")
def events(remote, since, limit, kinds, show_all, as_json):
    """What happened: heartbeats, challenges, micro-benchmarks, jobs, re-benchmarks, state changes."""
    import time as _time
    from .events import EventLog
    from .status import is_routine, parse_since, render_events
    cfg, ident = _load()
    try:
        t_since = parse_since(since, _time.time())
    except ValueError as e:
        raise click.UsageError(str(e))
    if remote:
        if not cfg.registered:
            raise click.UsageError("not registered; the platform has no log of this host")
        out, err = _platform_call(cfg, ident, lambda c: c.events(since=t_since, limit=max(limit * 4, 200),
                                                                 kinds=list(kinds) or None))
        if err:
            _log(f"platform: {err}")
            sys.exit(1)
        evs = out.get("events") or []
    else:
        evs = EventLog(cfg.events_path).read(since=t_since, kinds=kinds or None)
    if not show_all and not kinds:
        evs = [e for e in evs if not is_routine(e)]
    evs = evs[-limit:]
    if as_json:
        for e in evs:
            click.echo(json.dumps(e, default=str))
    else:
        click.echo(render_events(evs, show_all=True))


@main.group()
def service():
    """Run the daemon as a systemd user service (starts with the machine, restarts on failure)."""


@service.command("install")
@click.option("--no-start", is_flag=True, help="Enable it for the next boot without starting it now.")
def service_install(no_start):
    from .service import install, lingering
    cfg = HostConfig.load()
    if not cfg.registered:
        raise click.UsageError("not registered; run `kwh-host bench` then `kwh-host register` first")
    try:
        for line in install(start=not no_start):
            click.echo(line)
    except (RuntimeError, OSError) as e:
        _log(f"error: {e}")
        sys.exit(2)
    if lingering() is False:
        click.echo("note: the service stops when you log out; to keep it running: sudo loginctl enable-linger $USER")


@service.command("uninstall")
def service_uninstall():
    from .service import uninstall
    for line in uninstall() or ["nothing to remove"]:
        click.echo(line)


@service.command("status")
def service_status():
    import subprocess
    subprocess.run(["systemctl", "--user", "status", "--no-pager", "kwh-host.service"])


def _burst_gpu(engine, pause_service: bool):
    """The burst's engine takes this host's GPU and its engine container's name, so nothing else
    may hold them: the service is paused with --pause-service, and anything else is refused."""
    from contextlib import asynccontextmanager
    from . import service as svc
    from .sandbox import container_pid
    name = getattr(getattr(engine, "spec", None), "name", None)
    if svc.is_active():
        if not pause_service:
            raise click.ClickException("the kwh-host service is running and its engine holds the GPU; pass "
                                       "--pause-service to stop it for the burst and start it again after")
    elif name and container_pid(name):
        raise click.ClickException(f"the engine container {name} is running; is `kwh-host run` open in a terminal? "
                                   "Stop it first")

    @asynccontextmanager
    async def gpu():
        paused = False
        if svc.is_active():
            _log("stopping the kwh-host service for the burst")
            svc.stop()
            paused = True
        try:
            yield
        finally:
            if paused:
                try:
                    svc.start()
                    _log("started the kwh-host service again")
                except RuntimeError as e:
                    _log(f"error: the service did not start again ({e}); start it: systemctl --user start kwh-host")
    return gpu


@main.command()
@click.option("--token", envvar="KWH_BURST_TOKEN", required=True,
              help="The platform's burst token, from its operator (or set KWH_BURST_TOKEN).")
@click.option("--platform", "platform_url", default=None, help="Platform base URL (default: the configured platform).")
@click.option("--pause-service", is_flag=True,
              help="Stop the kwh-host service for the burst and start it again after: one engine fits on the GPU.")
@click.option("--max-continuations", type=int, default=3000, show_default=True, help="New challenge continuations, at most.")
@click.option("--max-verify", type=int, default=2000, show_default=True, help="Queued outputs to judge, at most.")
@click.option("--concurrency", type=int, default=32, show_default=True)
@click.option("--engine-url", default=None,
              help="A vLLM already serving the reference model at the locked version, instead of launching the engine.")
@click.option("--mock-engine", is_flag=True, hidden=True)
def burst(token, platform_url, pause_service, max_continuations, max_verify, concurrency, engine_url, mock_engine):
    """For the platform's operator: run the reference model on this GPU to make fresh challenges and
    judge the buyer outputs queued for verification. Only does anything if the platform needs it."""
    from .burst import run_burst
    try:
        cfg = HostConfig.load()
    except FileNotFoundError:
        cfg = None
    if cfg is None and not (engine_url or mock_engine):
        raise click.UsageError("no config yet: run `kwh-host init` and `kwh-host fetch` first; the burst launches "
                               "the same engine as `kwh-host bench`")
    platform_url = platform_url or (cfg.platform_url if cfg else None)
    if not platform_url:
        raise click.UsageError("give --platform")
    engine = scorer = gpu = None
    if mock_engine:
        from .mockmodel import ToyLM
        from .platform.verifier import ToyScorer
        engine, scorer = _engine(cfg, None, True), ToyScorer(ToyLM())
    elif not engine_url:
        engine = _engine(cfg, str(cfg.dir / "engine-burst.log"), False)
        gpu = _burst_gpu(engine, pause_service)
    try:
        out = asyncio.run(run_burst(platform_url, token, engine=engine, scorer=scorer, engine_url=engine_url,
                                    max_continuations=max_continuations, max_verify=max_verify,
                                    concurrency=concurrency, log=_log, gpu=gpu))
    except click.ClickException:
        raise
    except Exception as e:  # noqa: BLE001
        import httpx
        if isinstance(e, httpx.HTTPStatusError):
            try:
                body = e.response.json()
                said = body.get("detail", body) if isinstance(body, dict) else body
            except ValueError:
                said = e.response.text[:200]
            _log(f"error: the platform answered {e.response.status_code} to {e.request.method} {e.request.url.path}: {said}")
        else:
            _log(f"error: {type(e).__name__}: {e}")
        sys.exit(2)
    click.echo(json.dumps(out, indent=2))
    if not out.get("ok"):
        sys.exit(1)


@main.command("mock-platform")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=9000, show_default=True)
@click.option("--heartbeat", type=float, default=30.0, show_default=True, help="Seconds between heartbeats.")
@click.option("--challenge-every", type=float, default=300.0, show_default=True)
@click.option("--microbench-every", type=float, default=1800.0, show_default=True)
@click.option("--rebench-every", type=float, default=7 * 86400.0, show_default=True,
              help="Seconds before a report must be replaced (re-benchmark tests: e.g. 600).")
@click.option("--accept-uncertified", is_flag=True, help="Tests only: accept uncertified reports and any engine version.")
@click.option("--allow-bare-metal", is_flag=True, help="Pod testing: accept non-Docker engines (D4).")
@click.option("--tokenizer/--no-tokenizer", "use_tokenizer", default=None,
              help="Tokenize text/chat prompts on /v1/mock/jobs with the reference tokenizer (default: if transformers is installed).")
@click.option("--verify-url", default=None, help="A vLLM serving the reference model: verify delivered greedy outputs against it.")
@click.option("--verify-fraction", type=float, default=1.0, show_default=True, help="Share of completed jobs verified (with --verify-url).")
@click.option("--verify-tau", type=float, default=None, help="Gap in nats above which a position is a confident disagreement.")
@click.option("--mock-challenges", is_flag=True, hidden=True, help="Challenges from the benchmark's mock engine (GPU-free demos).")
@click.option("--challenges-from", default=None, hidden=True,
              help="Challenges scored by the vLLM-compatible server at this URL (tests with a fake engine).")
def mock_platform(host, port, heartbeat, challenge_every, microbench_every, rebench_every, accept_uncertified,
                  allow_bare_metal, use_tokenizer, verify_url, verify_fraction, verify_tau, mock_challenges, challenges_from):
    """Run the in-memory mock platform (HOST-CLIENT.md §8) for local development."""
    import uvicorn
    from .platform.mock import ChallengePool, HFTokenizer, MockPlatform, Router, Settings, create_app
    from .platform.verifier import DEFAULT_TAU, VLLMScorer
    s = Settings(heartbeat_seconds=heartbeat, challenge_every_seconds=challenge_every, microbench_every_seconds=microbench_every,
                 rebench_every_seconds=rebench_every, accept_uncertified=accept_uncertified, allow_bare_metal=allow_bare_metal,
                 verify_fraction=verify_fraction if verify_url else 0.0, verify_tau=verify_tau or DEFAULT_TAU)
    if challenges_from:
        from kwh_bench.engines import VLLMEngine
        pool = asyncio.run(ChallengePool.from_engine(VLLMEngine(server_url=challenges_from)))
    elif mock_challenges:
        from kwh_bench.engines import MockEngine
        pool = asyncio.run(ChallengePool.from_engine(MockEngine(step_ms=0.05, prefill_ms_per_1k=0.1)))
    else:
        pool = ChallengePool.from_lock()
    platform = MockPlatform(pool, s)
    tokenizer = None
    if use_tokenizer is not False:
        try:
            from kwh_bench.lockfile import load_lock
            tokenizer = HFTokenizer(revision=load_lock().model_revision)
            if tokenizer.chat_error:
                _log(f"tokenizer loaded, but no chat template ({tokenizer.chat_error}); "
                     "raw text and token ids work, chat messages are refused")
        except Exception as e:  # noqa: BLE001
            if use_tokenizer:
                raise click.ClickException(f"tokenizer unavailable: {type(e).__name__}: {e}")
            _log(f"no tokenizer ({type(e).__name__}); /v1/mock/jobs accepts prompt_token_ids only")
    router = Router(platform, scorer=VLLMScorer(verify_url) if verify_url else None)
    app = create_app(platform, router=router, tokenizer=tokenizer)
    source = challenges_from or ("the mock engine" if mock_challenges else "the public lock")
    _log(f"mock platform on http://{host}:{port}  (challenges from {source}; not a guard)"
         + (f"; verifying greedy outputs against {verify_url}" if verify_url else ""))
    uvicorn.run(app, host=host, port=port, log_level="warning", ws="websockets-sansio")


@main.command()
@click.option("--platform", "platform_url", default=None, help="Mock platform URL (default: the configured platform).")
@click.option("--prompt", "prompts", multiple=True, help="A user message; the platform applies the chat template. Repeat for more requests.")
@click.option("--raw", is_flag=True, help="Send each --prompt as raw text instead of a chat message.")
@click.option("--file", "jobs_file", type=click.Path(path_type=Path, exists=True), default=None,
              help="JSON: a job {\"requests\": [...]} or a list of jobs.")
@click.option("--max-tokens", type=int, default=128, show_default=True)
@click.option("--temperature", type=float, default=0.0, show_default=True)
@click.option("--seed", type=int, default=None)
@click.option("--timeout", "timeout_s", type=float, default=120.0, show_default=True)
@click.option("--concurrent", type=int, default=1, show_default=True, help="Jobs from --file in flight at once.")
@click.option("--out", type=click.Path(path_type=Path), default=None, help="Write the full responses here.")
def submit(platform_url, prompts, raw, jobs_file, max_tokens, temperature, seed, timeout_s, concurrent, out):
    """Send jobs to the mock platform's router as a buyer would (dev; not part of the host contract)."""
    import httpx
    if not platform_url:
        platform_url = HostConfig.load().platform_url
    jobs = []
    if jobs_file:
        loaded = json.loads(Path(jobs_file).read_text())
        jobs = loaded if isinstance(loaded, list) else [loaded]
    if prompts:
        reqs = [{("prompt" if raw else "messages"): (p if raw else [{"role": "user", "content": p}]),
                 "max_tokens": max_tokens, "temperature": temperature, "seed": seed} for p in prompts]
        jobs.append({"requests": reqs})
    if not jobs:
        raise click.UsageError("nothing to submit: give --prompt or --file")

    async def go():
        sem = asyncio.Semaphore(max(1, concurrent))
        async with httpx.AsyncClient(base_url=platform_url.rstrip("/"), timeout=timeout_s + 30) as c:
            async def one(job):
                async with sem:
                    r = await c.post("/v1/mock/jobs", json={"timeout_s": timeout_s, **job})
                    try:
                        return r.json()
                    except ValueError:
                        return {"status": "failed", "reason": f"HTTP {r.status_code}: {r.text[:200]}"}
            return await asyncio.gather(*(one(j) for j in jobs))

    results = asyncio.run(go())
    for res in results:
        v = res.get("verification") or {}
        click.echo(f"{res.get('job_id')}: {res.get('status')} on {res.get('host_id')}  units {res.get('units')}  "
                   f"{res.get('latency_ms')} ms  attempts {[a.get('outcome') for a in res.get('attempts', [])]}"
                   + (f"  verified={v.get('pass')}" if v else "") + (f"  ({res.get('reason') or res.get('detail')})"
                                                                     if res.get('status') != 'completed' else ""))
        for o in res.get("outputs") or []:
            text = (o.get("text") or "").replace("\n", " ")
            click.echo(f"  [{o['index']}] {o.get('finish_reason')} {len(o.get('token_ids') or [])} tok: {text[:160]}")
    if out:
        Path(out).write_text(json.dumps(results, indent=2) + "\n")
        click.echo(f"responses: {out}", err=True)
    if any(r.get("status") != "completed" for r in results):
        sys.exit(1)


@main.group(hidden=True)
def experiment():
    """Verifier calibration on real cards (see kwh_host/experiments.py)."""


@experiment.command("generate")
@click.option("--url", required=True, help="vLLM serving the model under test.")
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--canonical", "n_canonical", type=int, default=32, show_default=True)
@click.option("--chat/--no-chat", default=True, show_default=True)
@click.option("--max-tokens", type=int, default=128, show_default=True)
@click.option("--prompts-from", type=click.Path(path_type=Path, exists=True), default=None,
              help="Reuse the exact prompt ids of an earlier generate run.")
@click.option("--label", default="")
def experiment_generate(url, out, n_canonical, chat, max_tokens, prompts_from, label):
    from .experiments import generate
    asyncio.run(generate(url, out, n_canonical, chat, max_tokens, prompts_from, label, log=_log))


@experiment.command("score")
@click.option("--url", required=True, help="vLLM serving the reference model.")
@click.option("--in", "in_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def experiment_score(url, in_path, out):
    from .experiments import score
    asyncio.run(score(url, in_path, out, log=_log))


if __name__ == "__main__":
    main()
