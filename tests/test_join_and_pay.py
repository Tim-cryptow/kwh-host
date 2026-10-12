"""Adding a GPU in one command (`kwh-host join`) and reading what it earns (`kwh-host payout`,
`kwh-host status`), against the mock platform as a real server; and the service taking the docker
group with sg when the user's systemd started before the installer added it."""

import asyncio
import threading
import time

import pytest
import uvicorn
from click.testing import CliRunner
from kwh_bench.engines import MockEngine

from conftest import challenge_from
from kwh_host import cli, service
from kwh_host.config import HostConfig
from kwh_host.platform.client import PlatformClient, PlatformError
from kwh_host.platform.mock import ChallengePool, MockPlatform, Router, Settings, create_app
from kwh_host.status import render_status

PAY = {"host_id": "h_abc", "share": 0.8, "earned_usdc": "1.234567", "paid_usdc": "0.000000",
       "adjusted_usdc": "0.000000", "owed_usdc": "1.234567", "payout_address": None, "last_payout_at": None,
       "payouts": {"cadence": "weekly", "min_usdc": "5.000000", "chain": "base", "token": "USDC", "from": "0x" + "d" * 40},
       "recent": [{"kind": "earning", "amount_usdc": "0.000004", "ref": "j_1", "note": None, "at": 1_791_000_000.0}]}


@pytest.fixture
def platform_url(home):
    """The mock platform on a real port, in a thread of its own, so CLI commands can reach it."""
    ch = asyncio.run(challenge_from(MockEngine(step_ms=0.01, prefill_ms_per_1k=0.02)))
    platform = MockPlatform(ChallengePool([ch]), Settings(accept_uncertified=True, allow_bare_metal=True,
                                                          require_engine_version=None))
    server = uvicorn.Server(uvicorn.Config(create_app(platform, Router(platform)), host="127.0.0.1", port=0,
                                           log_level="warning", lifespan="off", ws="websockets-sansio"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(500):
        if server.started:
            break
        time.sleep(0.01)
    yield f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    server.should_exit = True
    thread.join(10)


def test_join_adds_a_gpu_in_one_command_and_again_does_only_what_is_left(platform_url):
    payout = "0x" + "5" * 40
    r = CliRunner().invoke(cli.main, ["join", platform_url, "--mock-engine", "--payout", payout])
    assert r.exit_code == 0, r.output
    cfg = HostConfig.load()
    assert cfg.registered and cfg.platform_url == platform_url and cfg.report_path.exists()
    for said in (f"setting up for {platform_url}", "the certified benchmark", "registering",
                 "not set (404",                       # the mock platform does not pay hosts
                 "start serving with  kwh-host run"):
        assert said in r.output, said
    host_id = cfg.host_id
    r = CliRunner().invoke(cli.main, ["join", platform_url, "--mock-engine"])
    assert r.exit_code == 0, r.output
    assert f"set up for {platform_url} already" in r.output and "registered already" in r.output
    assert "the certified benchmark" not in r.output and HostConfig.load().host_id == host_id
    r = CliRunner().invoke(cli.main, ["payout"])
    assert r.exit_code == 1 and "this platform does not pay hosts" in r.output


def test_a_new_platform_means_a_new_registration(home):
    CliRunner().invoke(cli.main, ["init", "--platform", "https://a.example", "--engine", "bare-metal",
                                  "--allow-bare-metal"])
    cfg = HostConfig.load()
    cfg.host_id, cfg.token = "h_a", "t_a"
    cfg.save()
    CliRunner().invoke(cli.main, ["init", "--platform", "https://a.example/", "--engine", "bare-metal",
                                  "--allow-bare-metal"])
    assert HostConfig.load().host_id == "h_a"                              # the same platform keeps it
    CliRunner().invoke(cli.main, ["init", "--platform", "https://b.example", "--engine", "bare-metal",
                                  "--allow-bare-metal"])
    assert HostConfig.load().host_id is None and HostConfig.load().token is None


def test_payout_shows_earnings_and_sets_the_address(cfg, identity, monkeypatch):
    cfg.host_id, cfg.token = "h_abc", "tok"
    cfg.save()
    calls = []

    async def earnings(self):
        calls.append("earnings")
        return PAY

    async def set_payout_address(self, address):
        calls.append(address)
        return {**PAY, "payout_address": address}
    monkeypatch.setattr(PlatformClient, "earnings", earnings)
    monkeypatch.setattr(PlatformClient, "set_payout_address", set_payout_address)
    r = CliRunner().invoke(cli.main, ["payout"])
    assert r.exit_code == 0, r.output
    for line in ("Earned   1.234567 USDC (80% of each job this GPU delivers)", "Paid     0.000000 USDC",
                 "Owed     1.234567 USDC; paid weekly once it reaches 5.000000 USDC",
                 "Payout   not set: kwh-host payout 0x...", "earning       +0.000004  j_1"):
        assert line in r.output, line
    r = CliRunner().invoke(cli.main, ["payout", "0x1234"])
    assert r.exit_code == 2 and "not an address" in r.output
    r = CliRunner().invoke(cli.main, ["payout", "0x" + "A" * 40])
    assert r.exit_code == 0 and f"Payout   0x{'A' * 40} (USDC on base)" in r.output
    assert calls == ["earnings", "0x" + "A" * 40]

    async def refused(self):
        raise PlatformError(503, "database down")
    monkeypatch.setattr(PlatformClient, "earnings", refused)
    r = CliRunner().invoke(cli.main, ["payout"])
    assert r.exit_code == 1 and "platform: 503: database down" in r.output


def test_status_shows_what_the_gpu_earned():
    host = {"version": "0.3.0", "host_id": "h_abc", "platform": "https://platform.example"}
    view = {"state": "live", "state_since": 1_791_000_000.0, "now": 1_791_000_100.0, "pay": PAY}
    text = render_status(host, None, view, now=1_791_000_100.0)
    assert "Earnings     1.234567 USDC earned, 0.000000 paid, 1.234567 owed" in text
    assert "no payout address yet: kwh-host payout 0x..." in text
    text = render_status(host, None, {**view, "pay": {**PAY, "payout_address": "0x" + "5" * 40}}, now=1_791_000_100.0)
    assert f"paid to 0x{'5' * 40}, weekly from 5.000000 USDC" in text


def test_the_service_takes_the_docker_group_when_its_systemd_lacks_it(monkeypatch):
    import grp
    import getpass
    from types import SimpleNamespace
    monkeypatch.setattr(grp, "getgrnam", lambda name: SimpleNamespace(gr_gid=999, gr_mem=["kwh"]))
    monkeypatch.setattr(getpass, "getuser", lambda: "kwh")
    monkeypatch.setattr(service.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(service, "manager_groups", lambda: ["4", "27"])    # started before the installer added it
    assert service.docker_via_sg() == "/usr/bin/sg"
    text = service.unit_text("/home/kwh/.local/bin/kwh-host", None, "/usr/bin/sg")
    assert "ExecStart=/usr/bin/sg docker -c 'exec /home/kwh/.local/bin/kwh-host run'" in text
    monkeypatch.setattr(service, "manager_groups", lambda: ["4", "27", "999"])   # after the next login
    assert service.docker_via_sg() is None
    monkeypatch.setattr(service, "manager_groups", lambda: None)                  # no systemd to ask
    assert service.docker_via_sg() is None
    assert "ExecStart=/x/kwh-host run" in service.unit_text("/x/kwh-host")
