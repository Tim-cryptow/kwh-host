"""The engine sandbox's `docker run` (HOST-CLIENT.md §3, D4), checked without Docker."""

from pathlib import Path

from kwh_bench import reference as ref
from kwh_bench.engines.base import EngineInfo
from kwh_bench.load import JobResult, score_runs
from kwh_bench.lockfile import load_lock
from kwh_bench.report import certification_reasons, evaluate_canary

from kwh_host.config import HostConfig
from kwh_host.engine import make_engine, sandbox_spec
from kwh_host.sandbox import SandboxedVLLMEngine, SandboxSpec, docker_run_argv, redact_home


def spec(tmp_path, **kw):
    base = dict(image="vllm/vllm-openai:v0.30.0", name="kwh-engine-gpu0", socket_dir=tmp_path / "run",
                hf_home=tmp_path / "hf", cache_dir=tmp_path / "cache", revision="024e24c", uid=1000, gid=1000,
                memory="24g")
    base.update(kw)
    return SandboxSpec(**base)


def pairs(argv, flag):
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag]


def test_sandbox_locks_the_container_down(tmp_path):
    argv = docker_run_argv(spec(tmp_path))
    image_at = argv.index("vllm/vllm-openai:v0.30.0")
    docker, engine = argv[:image_at], argv[image_at:]
    for flag in ("--rm", "--init", "--read-only"):
        assert flag in docker
    assert pairs(docker, "--network") == ["none"]                          # no network at all
    assert pairs(docker, "--cap-drop") == ["ALL"] and pairs(docker, "--security-opt") == ["no-new-privileges:true"]
    assert pairs(docker, "--user") == ["1000:1000"]                        # never root
    assert pairs(docker, "--pids-limit") == ["4096"] and pairs(docker, "--memory") == ["24g"]
    assert pairs(docker, "--memory-swap") == ["24g"] and pairs(docker, "--gpus") == ["device=0"]
    assert "-p" not in docker                                              # nothing published
    vols = pairs(docker, "-v")
    assert f"{tmp_path / 'hf'}:/hf:ro" in vols and f"{tmp_path / 'cache'}:/cache:rw" in vols
    assert f"{tmp_path / 'run'}:/run/kwh:rw" in vols and len(vols) == 3
    env = dict(e.split("=", 1) for e in pairs(docker, "-e"))
    assert env["HF_HUB_OFFLINE"] == "1" and env["HF_HOME"] == "/hf" and env["VLLM_NO_USAGE_STATS"] == "1"
    assert all(v.startswith("/cache") for k, v in env.items() if k.endswith(("_DIR", "_ROOT", "_PATH", "HOME")) and k != "HF_HOME")
    # the engine: the reference model, the pinned flags at the host's context, listening on the socket only
    assert engine[1] == ref.MODEL_ID and engine[2:2 + len(ref.vllm_args(8192))] == ref.vllm_args(8192)
    assert pairs(engine, "--revision") == ["024e24c"] and pairs(engine, "--uds") == ["/run/kwh/engine.sock"]
    assert "--port" not in engine and "--host" not in engine


def test_docker_desktop_transport_publishes_on_loopback_only(tmp_path):
    argv = docker_run_argv(spec(tmp_path, transport="tcp", port=8123))
    assert pairs(argv, "-p") == ["127.0.0.1:8123:8123"] and "--network" not in argv and "--uds" not in argv
    assert pairs(argv, "--port") == ["8123"]


def test_sandbox_argv_certifies(tmp_path):
    """The docker flags around the engine must not trip the benchmark's certification checks."""
    argv = [redact_home(a) for a in docker_run_argv(spec(tmp_path))]
    info = EngineInfo(name="vllm", version=load_lock().vllm_version, model_id=ref.MODEL_ID,
                      model_revision=load_lock().model_revision, launch_mode="docker", launch_args=argv)
    runs = [JobResult(job_seconds=40.0, records=[], generated_tokens=ref.GENERATED_TOKENS_PER_JOB, failures=0)] * 3
    canary = evaluate_canary({c.prompt_id: c.reference_mean_logprob for c in load_lock().canaries}, load_lock())
    assert certification_reasons(info, runs, score_runs([40.0] * 3), canary, load_lock()) == []


def test_reports_never_carry_the_home_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", "/home/alice")
    assert redact_home("/home/alice/.kwh-host/hf:/hf:ro") == "~/.kwh-host/hf:/hf:ro"
    assert redact_home("/srv/models:/hf:ro") == "/srv/models:/hf:ro"


def test_make_engine_sandboxes_docker_mode(home, monkeypatch):
    monkeypatch.setattr("kwh_host.engine.docker_is_rootless", lambda: False)
    cfg = HostConfig(engine_mode="docker", gpu_index=1, max_model_len=4096)
    e = make_engine(cfg)
    assert isinstance(e, SandboxedVLLMEngine) and e.uds == str(cfg.dir / "run" / "engine.sock")
    assert e.base_url == "http://engine" and e.spec.name == "kwh-engine-gpu1" and e.spec.gpu_index == 1
    assert e.spec.hf_home == cfg.dir / "hf" and e.spec.max_model_len == 4096
    tcp = make_engine(HostConfig(engine_mode="docker", engine_transport="tcp", engine_port=8123))
    assert tcp.uds is None and tcp.base_url == "http://127.0.0.1:8123"
    # rootless Docker: container root is the invoking user
    monkeypatch.setattr("kwh_host.engine.docker_is_rootless", lambda: True)
    s = sandbox_spec(cfg, ref.MODEL_ID, None)
    assert (s.uid, s.gid) == (0, 0)
    bare = make_engine(HostConfig(engine_mode="bare-metal"))
    assert not isinstance(bare, SandboxedVLLMEngine) and bare.docker_image is None


def test_hf_home_defaults_inside_the_host_dir(home):
    cfg = HostConfig()
    assert cfg.hf_home == cfg.dir / "hf"
    assert HostConfig(hf_cache="~/models").hf_home == Path("~/models").expanduser()


def test_engine_user_is_never_root(monkeypatch):
    from kwh_host import sandbox
    monkeypatch.setattr(sandbox.os, "getuid", lambda: 0)
    monkeypatch.setattr(sandbox.os, "getgid", lambda: 0)
    assert (sandbox.default_uid(), sandbox.default_gid()) == (65534, 65534)      # a root daemon: the engine runs as nobody
    monkeypatch.setattr(sandbox.os, "getuid", lambda: 1000)
    monkeypatch.setattr(sandbox.os, "getgid", lambda: 1000)
    assert (sandbox.default_uid(), sandbox.default_gid()) == (1000, 1000)


def test_an_engine_that_dies_early_says_why(tmp_path, monkeypatch):
    """`exited early with code 125` told a host nothing; the docker error in the log does."""
    import asyncio

    import pytest
    from kwh_bench.engines.vllm import VLLMEngine

    from kwh_host import sandbox

    log = tmp_path / "engine.log"
    log.write_text('docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]\n\n'
                   "Run 'docker run --help' for more information\n")
    monkeypatch.setattr(sandbox, "prepare_dirs", lambda s: None)
    removed = []
    monkeypatch.setattr(sandbox, "remove_container", removed.append)

    async def dies(self):
        raise RuntimeError("engine process exited early with code 125")
    monkeypatch.setattr(VLLMEngine, "start", dies)
    engine = SandboxedVLLMEngine(spec(tmp_path), log_path=str(log))
    with pytest.raises(RuntimeError) as e:
        asyncio.run(engine.start())
    assert "code 125" in str(e.value) and "could not select device driver" in str(e.value)
    assert "docker run --help" not in str(e.value) and removed == ["kwh-engine-gpu0", "kwh-engine-gpu0"]
    assert sandbox.log_tail(None) == "" and sandbox.log_tail(str(tmp_path / "missing.log")) == ""
