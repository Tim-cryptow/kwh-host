"""`kwh-host burst` (kwh_host/burst.py) against a stand-in for the platform's burst API: the pool is
filled with continuations made the way the lock made its canaries, queued outputs are judged by
teacher forcing, the GPU is taken only when there is work and always given back, and the host's own
service is paused around the burst only when asked."""

import contextlib
import json
import subprocess

import httpx
import pytest
from click.testing import CliRunner
from kwh_bench.engines import MockEngine

from kwh_host import burst as burst_mod
from kwh_host import cli, service
from kwh_host.burst import run_burst, undecided_to_post
from kwh_host.mockmodel import ToyLM
from kwh_host.platform.verifier import ToyScorer

TOKEN = "burst-" + "b" * 30


class BurstAPI:
    """The platform's /burst endpoints, recording what a burst posts."""

    def __init__(self, wanted=0, queue=(), fail_continuations=False):
        self.wanted, self.queue, self.fail_continuations = wanted, list(queue), fail_continuations
        self.continuations, self.verdicts, self.posts = [], [], []

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.headers.get("authorization") != f"Bearer {TOKEN}":
            return httpx.Response(401, json={"detail": "bad burst token"})
        if request.method == "GET" and request.url.path == "/burst/work":
            limit = int(request.url.params["verify_limit"])
            return httpx.Response(200, json={"burst_id": "burst_t", "continuations_wanted": self.wanted,
                                             "pool_unissued": 0, "verify": self.queue[:limit], "tau": 1.0})
        body = json.loads(request.content)
        if request.url.path == "/burst/continuations":
            if self.fail_continuations:
                return httpx.Response(500, json={"detail": "database down"})
            self.continuations += body["items"]
            self.posts.append(("continuations", len(body["items"])))
            return httpx.Response(200, json={"added": len(body["items"])})
        if request.url.path == "/burst/verdicts":
            self.verdicts += body["items"]
            self.posts.append(("verdicts", len(body["items"])))
            return httpx.Response(200, json={"judged": len(body["items"])})
        return httpx.Response(404)

    def transport(self):
        return httpx.MockTransport(self.handle)


def queued(honest, substitute):
    """Delivered greedy outputs waiting for the reference: some from the reference model itself,
    some from another model."""
    items = []
    for i in range(honest + substitute):
        prompt = [(7 * i + k) % 500 + 3 for k in range(12)]
        lm = ToyLM() if i < honest else ToyLM(drift=1.0)
        out, _ = lm.generate(prompt, 16, ignore_eos=True)
        items.append({"id": i + 1, "host_id": "h_x", "job_id": f"j_{i}", "prompt_token_ids": prompt,
                      "output_token_ids": out})
    return items


class Recording(MockEngine):
    def __init__(self, events):
        super().__init__(step_ms=0.01, prefill_ms_per_1k=0.02)
        self.events = events

    async def start(self):
        self.events.append("engine up")
        await super().start()

    async def stop(self):
        await super().stop()
        self.events.append("engine down")


def gpu_hook(events):
    @contextlib.asynccontextmanager
    async def gpu():
        events.append("gpu taken")
        try:
            yield
        finally:
            events.append("gpu given back")
    return gpu


async def burst(api, events, engine=None, scorer=None, **kw):
    return await run_burst("http://platform", TOKEN, engine=engine or Recording(events),
                           scorer=scorer or ToyScorer(ToyLM()), seed=7, log=lambda s: None,
                           gpu=gpu_hook(events), transport=api.transport(), concurrency=4, **kw)


async def test_a_burst_fills_the_pool_and_judges_the_queue(monkeypatch):
    monkeypatch.setattr(burst_mod, "BATCH", 3)
    api, events = BurstAPI(wanted=7, queue=queued(4, 2)), []
    out = await burst(api, events)
    assert out["ok"] and out["continuations"] == 7 and out["verdicts"] == 6
    assert events == ["gpu taken", "engine up", "engine down", "gpu given back"]
    assert api.posts == [("continuations", 3), ("continuations", 3), ("continuations", 1),
                         ("verdicts", 3), ("verdicts", 3)]
    assert len({c["prompt_text"] for c in api.continuations}) == 7          # fresh prompts, none twice
    async with MockEngine(step_ms=0.01, prefill_ms_per_1k=0.02) as ref:
        for c in api.continuations:
            assert c["prompt_tokens"] == 512 and len(c["continuation_token_ids"]) == 32
            ids = (await ref.tokenize(c["prompt_text"]))[:512]
            assert (await ref.complete(ids, 32, want_token_ids=True)).token_ids == c["continuation_token_ids"]
            lps = await ref.score_continuation(ids, c["continuation_token_ids"])
            assert c["reference_mean_logprob"] == pytest.approx(sum(lps) / len(lps)) and c["reference_mean_logprob"] <= 0
    verdicts = {v["id"]: v["verdict"] for v in api.verdicts}
    assert [verdicts[i]["pass"] for i in range(1, 7)] == [True] * 4 + [False] * 2
    assert all(verdicts[i]["confident_disagreements"] > 0 for i in (5, 6))


async def test_nothing_to_do_leaves_the_gpu_alone():
    api, events = BurstAPI(wanted=0), []
    out = await burst(api, events)
    assert out["ok"] and out["note"] == "nothing to do"
    assert events == [] and api.posts == []


async def test_the_gpu_is_given_back_when_a_burst_fails():
    api, events = BurstAPI(wanted=3, fail_continuations=True), []
    with pytest.raises(httpx.HTTPStatusError):
        await burst(api, events)
    assert events == ["gpu taken", "engine up", "engine down", "gpu given back"]


class Impostor(Recording):
    def __init__(self, events, model_id, version):
        super().__init__(events)
        self.model_id, self.version = model_id, version

    async def info(self):
        info = await super().info()
        info.name, info.model_id, info.version = "vllm", self.model_id, self.version
        return info


async def test_an_engine_that_is_not_the_reference_makes_nothing():
    from kwh_bench import reference as ref
    from kwh_bench.lockfile import load_lock
    locked = load_lock().vllm_version
    for model_id, version, why in (("meta-llama/Llama-3.1-8B-Instruct", locked, "not the reference checkpoint"),
                                   (ref.MODEL_ID, "0.0.1", "the reference is the locked"),
                                   (ref.MODEL_ID, None, "the reference is the locked")):
        api, events = BurstAPI(wanted=3, queue=queued(1, 0)), []
        with pytest.raises(RuntimeError, match=why):
            await burst(api, events, engine=Impostor(events, model_id, version))
        assert api.posts == [] and events[-1] == "gpu given back"


class Flaky:
    """A reference that cannot judge some outputs."""

    def __init__(self, failing):
        self.failing, self.inner = failing, ToyScorer(ToyLM())

    async def positions(self, prompt_ids, output_ids):
        if self.failing(prompt_ids):
            raise RuntimeError("reference engine 500")
        return await self.inner.positions(prompt_ids, output_ids)


async def test_outputs_the_reference_cannot_judge():
    items = queued(10, 0)
    # one output it cannot judge: something about that output, so it is closed as undecided
    one = items[3]["prompt_token_ids"]
    api, events = BurstAPI(queue=items), []
    out = await burst(api, events, scorer=Flaky(lambda p: p == one))
    assert out["ok"] and out["verdicts"] == 10
    assert [v["verdict"]["pass"] for v in api.verdicts].count(None) == 1
    # none judged: the engine failed, so they stay queued for the next burst
    api, events = BurstAPI(queue=items), []
    out = await burst(api, events, scorer=Flaky(lambda p: True))
    assert not out["ok"] and out["verdicts"] == 0 and api.verdicts == []
    assert undecided_to_post([{"verdict": {"pass": None}}]) == [{"verdict": {"pass": None}}]


def test_the_service_state_is_read_from_systemd(monkeypatch):
    monkeypatch.setattr(service.shutil, "which", lambda name: "/usr/bin/" + name)
    for said, active in (("active\n", True), ("activating\n", True), ("inactive\n", False), ("failed\n", False),
                         ("", None)):
        monkeypatch.setattr(service, "systemctl", lambda *a, s=said: subprocess.CompletedProcess(a, 0, s, ""))
        assert service.is_active() is active, said
    monkeypatch.setattr(service.shutil, "which", lambda name: None)
    assert service.is_active() is None                     # no systemd: nothing to ask


def test_the_cli_pauses_the_service_only_when_asked(cfg, monkeypatch):
    state, calls = {"active": True}, []
    monkeypatch.setattr(service, "is_active", lambda: state["active"])

    def stop():
        calls.append("stop")
        state["active"] = False

    def start():
        calls.append("start")
        state["active"] = True
    monkeypatch.setattr(service, "stop", stop)
    monkeypatch.setattr(service, "start", start)

    async def fake_burst(platform_url, token, *, gpu, **kw):
        calls.append(f"work from {platform_url}")
        async with gpu():
            calls.append("burst")
        return {"ok": True, "burst_id": "burst_t"}
    monkeypatch.setattr(burst_mod, "run_burst", fake_burst)

    r = CliRunner().invoke(cli.main, ["burst", "--token", TOKEN])
    assert r.exit_code == 1 and "--pause-service" in r.output and calls == []
    r = CliRunner().invoke(cli.main, ["burst", "--token", TOKEN, "--pause-service"])
    assert r.exit_code == 0, r.output
    assert calls == ["work from http://mock", "stop", "burst", "start"] and state["active"]

    async def failing(platform_url, token, *, gpu, **kw):
        async with gpu():
            raise RuntimeError("the engine died")
    monkeypatch.setattr(burst_mod, "run_burst", failing)
    calls.clear()
    r = CliRunner().invoke(cli.main, ["burst", "--token", TOKEN, "--pause-service"])
    assert r.exit_code == 2 and "the engine died" in r.output
    assert calls == ["stop", "start"] and state["active"]  # the host serves again either way


def test_the_token_can_live_in_a_file(cfg, monkeypatch):
    """The daily burst from cron reads ~/.kwh-host/burst-token, so the token is not on a command line."""
    seen = []

    async def fake_burst(platform_url, token, **kw):
        seen.append(token)
        return {"ok": True}
    monkeypatch.setattr(burst_mod, "run_burst", fake_burst)
    monkeypatch.setattr(service, "is_active", lambda: False)
    monkeypatch.delenv("KWH_BURST_TOKEN", raising=False)
    r = CliRunner().invoke(cli.main, ["burst"])
    assert r.exit_code == 2 and "burst-token" in r.output and seen == []
    (cfg.dir / "burst-token").write_text(TOKEN + "\n")
    r = CliRunner().invoke(cli.main, ["burst"])
    assert r.exit_code == 0, r.output
    assert seen == [TOKEN]


def test_register_waits_for_an_invitation(cfg, report, monkeypatch):
    """A closed platform answers 403 to a key it has not invited; `register --wait` keeps asking."""
    import time
    from kwh_host.platform.client import PlatformClient, PlatformError
    answers = [PlatformError(403, "closed beta: ask the operator to invite this host's key"),
               httpx.ConnectError("the platform is redeploying"),
               PlatformError(403, "closed beta: ask the operator to invite this host's key"),
               {"host_id": "h_1", "token": "t", "rate_units_per_hour": 100.0, "bucket": "b"}]
    tries, sleeps = [], []

    async def register(self, report):
        tries.append(1)
        a = answers[len(tries) - 1]
        if isinstance(a, Exception):
            raise a
        return a
    monkeypatch.setattr(PlatformClient, "register", register)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    r = CliRunner().invoke(cli.main, ["register"])                     # without --wait: refused at once
    assert r.exit_code == 1 and "registration refused" in r.output and len(tries) == 1
    tries.clear()
    r = CliRunner().invoke(cli.main, ["register", "--wait", "60"])
    assert r.exit_code == 0, r.output
    assert len(tries) == 4 and sleeps == [30, 30, 30]
    assert r.output.count("this host's key: ") == 2 and "waiting: the platform is unreachable" in r.output
    assert json.loads(r.stdout)["host_id"] == "h_1"
    from kwh_host.config import HostConfig
    assert HostConfig.load().host_id == "h_1"
    answers.insert(0, PlatformError(400, "report rejected: uncertified"))
    tries.clear()
    r = CliRunner().invoke(cli.main, ["register", "--wait", "60"])     # anything else is final
    assert r.exit_code == 1 and "report rejected" in r.output and len(tries) == 1
