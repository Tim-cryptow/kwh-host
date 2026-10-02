"""The sandbox for real: the production `docker run` (minus the GPU) launching a fake vLLM
(tests/fake_engine), then checked from outside the way an attacker inside would meet it.

Skipped unless Docker works and KWH_TEST_ENGINE_IMAGE names the fake engine image:

    docker build -t kwh-fake-engine:test tests/fake_engine
    KWH_TEST_ENGINE_IMAGE=kwh-fake-engine:test pytest tests/test_sandbox_docker.py
"""

import json
import os
import shutil
import subprocess

import pytest
from kwh_bench import reference as ref

from kwh_host.jobs import VLLMExecutor
from kwh_host.jobspec import parse_job
from kwh_host.sandbox import SandboxedVLLMEngine, SandboxSpec

IMAGE = os.environ.get("KWH_TEST_ENGINE_IMAGE")


def docker_ok() -> bool:
    if not IMAGE or not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(not docker_ok(), reason="needs Docker and KWH_TEST_ENGINE_IMAGE (tests/fake_engine)")


def inside(name: str, code: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", "exec", name, "/opt/venv/bin/python", "-c", code],
                          capture_output=True, text=True, timeout=60)


@pytest.fixture
def spec(tmp_path):
    # A short socket path: Unix socket paths are limited to ~107 bytes.
    d = tmp_path / "k"
    return SandboxSpec(image=IMAGE, name=f"kwh-engine-test-{os.getpid()}", socket_dir=d / "run", hf_home=d / "hf",
                       cache_dir=d / "cache", gpu_index=None, max_model_len=8192, revision="024e24c", memory="2g")


async def test_engine_serves_only_through_its_socket_and_cannot_get_out(spec, tmp_path):
    engine = SandboxedVLLMEngine(spec, log_path=str(tmp_path / "engine.log"))
    async with engine:
        info = await engine.info()
        assert info.launch_mode == "docker" and info.version == "0.30.0" and info.model_id == ref.MODEL_ID
        assert "--uds" in info.launch_args and "--max-model-len" in info.launch_args

        # how Docker sees the container
        insp = json.loads(subprocess.run(["docker", "inspect", spec.name], capture_output=True, text=True).stdout)[0]
        hc = insp["HostConfig"]
        assert hc["ReadonlyRootfs"] is True and hc["NetworkMode"] == "none" and hc["CapDrop"] == ["ALL"]
        assert "no-new-privileges:true" in hc["SecurityOpt"] and hc["PidsLimit"] == 4096
        assert hc["Memory"] == 2 * 2**30 and hc["Init"] is True and not hc["PortBindings"]
        assert insp["Config"]["User"] == f"{spec.uid}:{spec.gid}" and spec.uid != 0      # never root, even under a root daemon
        mounts = {m["Destination"]: m for m in insp["Mounts"]}
        assert mounts["/hf"]["RW"] is False and mounts["/cache"]["RW"] is True and mounts["/run/kwh"]["RW"] is True

        # from the inside: no route out, nothing writable but /tmp and /cache, no root
        r = inside(spec.name, "import socket; socket.create_connection(('1.1.1.1', 53), 3)")
        assert r.returncode != 0 and "OSError" in r.stderr
        r = inside(spec.name, "open('/usr/evil', 'w')")
        assert r.returncode != 0 and "Read-only file system" in r.stderr
        r = inside(spec.name, "open('/hf/evil', 'w')")
        assert r.returncode != 0 and "Read-only file system" in r.stderr
        r = inside(spec.name, "import os; open('/tmp/ok', 'w').write('x'); open('/cache/ok', 'w').write('x'); "
                              "print(os.getuid(), open('/proc/self/status').read().split('CapEff:')[1].split()[0])")
        assert r.returncode == 0, r.stderr
        uid, cap_eff = r.stdout.split()
        assert int(uid) == spec.uid and int(cap_eff, 16) == 0             # no effective capabilities

        # and it serves, over the socket only: the benchmark's calls and a buyer job
        ids = await engine.tokenize("the quick brown fox jumps")
        done = await engine.complete(ids, 8, want_token_ids=True)
        assert len(done.token_ids) == 8 and done.completion_tokens == 8
        lps = await engine.score_continuation(ids, done.token_ids)
        assert len(lps) == 8 and max(lps) <= 0
        ex = VLLMExecutor(engine.base_url, ref.MODEL_ID, transport=engine.make_transport())
        job = {"job_id": "j_sandbox.1", "timeout_s": 30, "units_reserved": 0.001,
               "requests": [{"prompt_token_ids": [128000, 9906, 1917], "max_tokens": 12, "temperature": 0.0, "top_p": 1.0,
                             "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0, "seed": None, "stop": [],
                             "stop_token_ids": [], "ignore_eos": True, "logprobs": None}]}
        out = await ex.generate(parse_job(job, 8192)[0])
        await ex.aclose()
        assert len(out["token_ids"]) == 12 and out["finish_reason"] == "length"
        assert engine.own_pids()                                          # the GPU sampler can tell it apart

    gone = subprocess.run(["docker", "inspect", spec.name], capture_output=True)
    assert gone.returncode != 0                                           # stopped means removed


async def test_a_stale_engine_container_is_replaced(spec, tmp_path):
    """A daemon that crashed leaves its container behind; the next start removes it first."""
    subprocess.run(["docker", "run", "-d", "--name", spec.name, "--entrypoint", "sleep", IMAGE, "300"],
                   capture_output=True, check=True)
    engine = SandboxedVLLMEngine(spec, log_path=str(tmp_path / "engine.log"))
    async with engine:
        assert (await engine.info()).version == "0.30.0"
    assert subprocess.run(["docker", "inspect", spec.name], capture_output=True).returncode != 0
