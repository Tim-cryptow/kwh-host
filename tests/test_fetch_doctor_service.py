"""kwh-host fetch, doctor and service, without the network, a GPU or systemd."""

import json
import subprocess
from pathlib import Path

import pytest
from kwh_bench import reference as ref
from kwh_bench.lockfile import Lock

from kwh_host import doctor, fetch, service
from kwh_host.config import HostConfig


def fake_snapshot(hf_home: Path, revision: str, files: dict) -> Path:
    snap = hf_home / "hub" / ("models--" + ref.MODEL_ID.replace("/", "--")) / "snapshots" / revision
    snap.mkdir(parents=True)
    for name, data in files.items():
        (snap / name).write_bytes(data)
    return snap


def test_fetch_checks_every_file_against_the_lock(tmp_path, monkeypatch):
    files = {"config.json": b"{}", "model-00001-of-00002.safetensors": b"weights"}
    snap = fake_snapshot(tmp_path / "hf", "024e24c", files)
    lock = Lock(model_revision="024e24c", model_files={k: fetch.sha256_file(snap / k) for k in files})
    assert fetch.snapshot_dir(tmp_path / "hf", ref.MODEL_ID, "024e24c") == snap
    assert fetch.verify_model(snap, lock) == []
    (snap / "config.json").write_bytes(b'{"tampered": true}')
    problems = fetch.verify_model(snap, lock)
    assert len(problems) == 1 and problems[0].startswith("config.json: sha256")
    (snap / "model-00001-of-00002.safetensors").unlink()
    assert any("missing" in p for p in fetch.verify_model(snap, lock))
    # the whole command: download (faked), verify, refuse a mismatch
    monkeypatch.setattr(fetch, "download_model", lambda hf_home, model, revision, log: snap)
    with pytest.raises(RuntimeError, match="does not match the lock"):
        fetch.fetch(tmp_path / "hf", lock, image=None, log=lambda s: None)
    out = fetch.fetch(tmp_path / "hf", lock, image=None, log=lambda s: None, model="hugging-quants/x")
    assert out["verified"] is None and out["revision"] is None          # a wrong-model test is not hash-checked


def runner(responses):
    def run(argv):
        for prefix, (code, out, err) in responses.items():
            if " ".join(argv).startswith(prefix):
                return subprocess.CompletedProcess(argv, code, out, err)
        return subprocess.CompletedProcess(argv, 127, "", "not found")
    return run


def test_doctor_reads_the_gpu_and_names_the_fix(home):
    cfg = HostConfig()
    ok = doctor.check_gpu(cfg, runner({"nvidia-smi": (0, "0, NVIDIA GeForce RTX 4090, 24564, 580.95.05\n", "")}))
    assert ok.status == "ok" and "RTX 4090" in ok.detail and "24 GB" in ok.detail
    small = doctor.check_gpu(cfg, runner({"nvidia-smi": (0, "0, NVIDIA GeForce RTX 4060, 8188, 580.95.05\n", "")}))
    assert small.status == "fail" and "16 GB" in small.detail
    none = doctor.check_gpu(cfg, runner({}))
    assert none.status == "fail" and none.fix


def test_doctor_on_docker_desktop_flags_the_socket_and_offers_tcp(home):
    desktop = {"OperatingSystem": "Docker Desktop", "ServerVersion": "28.0", "Runtimes": {"runc": {}}}
    c = doctor.check_transport(HostConfig(), desktop)
    assert c.status == "fail" and "--engine-transport tcp" in c.fix
    assert doctor.check_transport(HostConfig(engine_transport="tcp"), desktop).status == "warn"
    engine = {"OperatingSystem": "Ubuntu 24.04 LTS", "ServerVersion": "28.0", "Runtimes": {"runc": {}, "nvidia": {}}}
    assert doctor.check_transport(HostConfig(), engine).status == "ok"
    assert doctor.check_gpu_runtime(engine).status == "ok"
    assert doctor.check_gpu_runtime({"Runtimes": {"runc": {}}, "OperatingSystem": "Ubuntu"}).status in ("fail", "ok")


def test_doctor_finds_the_fetched_checkpoint(home):
    cfg = HostConfig()
    assert doctor.check_model(cfg, "024e24c").status == "fail"
    fake_snapshot(cfg.hf_home, "024e24c", {"config.json": b"{}"})
    assert doctor.check_model(cfg, "024e24c").status == "ok"


def test_service_unit_runs_the_daemon_as_the_user(tmp_path, monkeypatch):
    text = service.unit_text("/home/alice/.kwh-host/venv/bin/kwh-host", kwh_home="/data/kwh")
    assert "ExecStart=/home/alice/.kwh-host/venv/bin/kwh-host run" in text and "Restart=on-failure" in text
    assert "Environment=KWH_HOST_HOME=/data/kwh" in text and "WantedBy=default.target" in text
    assert "User=" not in text                                         # a user service: never root
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    calls = []
    monkeypatch.setattr(service, "systemctl", lambda *a: calls.append(a) or subprocess.CompletedProcess(a, 0, "", ""))
    done = service.install()
    assert service.unit_path().exists() and calls == [("daemon-reload",), ("enable", "--now", "kwh-host.service")]
    assert done[0].startswith("wrote ")
    service.uninstall()
    assert not service.unit_path().exists()
