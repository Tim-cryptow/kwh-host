"""kwh-host command line: init | bench | register | run | status | mock-platform | submit | experiment."""

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
@click.option("--model", default=None, hidden=True, help="Serve this model instead of the reference (wrong-model tests).")
@click.option("--no-jobs", is_flag=True, help="Heartbeat and challenges only; do not open the job channel.")
def run(engine_log, beats, no_version_check, mock_engine, model, no_jobs):
    """Start the engine, heartbeat, and serve jobs; stay live until stopped."""
    from .daemon import Daemon
    from .platform.client import PlatformClient
    cfg, ident = _load()
    if not cfg.registered:
        raise click.UsageError("not registered; run `kwh-host bench` then `kwh-host register`")
    engine = _engine(cfg, str(engine_log) if engine_log else str(cfg.dir / "engine.log"), mock_engine, model)
    no_version_check = no_version_check or mock_engine

    async def go():
        async with PlatformClient(cfg.platform_url, ident, token=cfg.token, host_id=cfg.host_id) as c:
            d = Daemon(cfg, c, engine, log=_log, max_beats=beats, require_lock_version=not no_version_check,
                       jobs=not no_jobs)
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
@click.option("--tokenizer/--no-tokenizer", "use_tokenizer", default=None,
              help="Tokenize text/chat prompts on /v1/mock/jobs with the reference tokenizer (default: if transformers is installed).")
@click.option("--verify-url", default=None, help="A vLLM serving the reference model: verify delivered greedy outputs against it.")
@click.option("--verify-fraction", type=float, default=1.0, show_default=True, help="Share of completed jobs verified (with --verify-url).")
@click.option("--verify-tau", type=float, default=None, help="Gap in nats above which a position is a confident disagreement.")
@click.option("--mock-challenges", is_flag=True, hidden=True, help="Challenges from the benchmark's mock engine (GPU-free demos).")
def mock_platform(host, port, heartbeat, challenge_every, microbench_every, accept_uncertified, allow_bare_metal,
                  use_tokenizer, verify_url, verify_fraction, verify_tau, mock_challenges):
    """Run the in-memory mock platform (HOST-CLIENT.md §8) for local development."""
    import uvicorn
    from .platform.mock import ChallengePool, HFTokenizer, MockPlatform, Router, Settings, create_app
    from .platform.verifier import DEFAULT_TAU, VLLMScorer
    s = Settings(heartbeat_seconds=heartbeat, challenge_every_seconds=challenge_every, microbench_every_seconds=microbench_every,
                 accept_uncertified=accept_uncertified, allow_bare_metal=allow_bare_metal,
                 verify_fraction=verify_fraction if verify_url else 0.0, verify_tau=verify_tau or DEFAULT_TAU)
    if mock_challenges:
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
        except Exception as e:  # noqa: BLE001
            if use_tokenizer:
                raise click.ClickException(f"tokenizer unavailable: {type(e).__name__}: {e}")
            _log(f"no tokenizer ({type(e).__name__}); /v1/mock/jobs accepts prompt_token_ids only")
    router = Router(platform, scorer=VLLMScorer(verify_url) if verify_url else None)
    app = create_app(platform, router=router, tokenizer=tokenizer)
    _log(f"mock platform on http://{host}:{port}  (challenges from the {'mock engine' if mock_challenges else 'public lock'}; not a guard)"
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
