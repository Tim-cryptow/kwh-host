"""`kwh-host doctor`: is this machine ready to host? One line per check, a fix for each failure.

The checks are what the sandboxed engine needs: an NVIDIA driver with a big enough card and new
enough for the engine image's CUDA build, Docker reachable without sudo, Docker able to hand the
GPU to a container (the NVIDIA Container Toolkit, tried with a real container once the image is
there), Docker Engine rather than Docker Desktop when the engine talks over a Unix socket, and the
image and the checkpoint already fetched.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from kwh_bench import reference as ref

from .config import IMAGE_CUDA, TESTED_CUDA12, HostConfig, cuda_tuple, engine_image_for, image_label
from .fetch import snapshot_dir

MIN_VRAM_MIB = 16 * 1024 - 512           # "16 GB" cards report a little under 16384 MiB

Run = Callable[[List[str]], subprocess.CompletedProcess]


def _run(argv: List[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return subprocess.CompletedProcess(argv, 127, "", f"{argv[0]}: not found")
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(argv, 124, "", f"{argv[0]}: timed out")


@dataclass
class Check:
    name: str
    status: str            # ok | warn | fail
    detail: str
    fix: Optional[str] = None

    def line(self) -> str:
        mark = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL"}[self.status]
        return f"[{mark}] {self.name}: {self.detail}" + (f"\n         fix: {self.fix}" if self.fix else "")


def is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def check_platform() -> Check:
    machine, system = platform.machine(), platform.system()
    if system != "Linux" or machine not in ("x86_64", "AMD64"):
        return Check("platform", "fail", f"{system} {machine}",
                     "the host client runs on x86-64 Linux, or on Windows inside WSL2 (HOST-CLIENT.md §12)")
    return Check("platform", "ok", "WSL2 (Windows)" if is_wsl() else f"Linux {platform.release()}")


def check_gpu(cfg: HostConfig, run: Run = _run) -> Check:
    r = run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    if r.returncode != 0:
        fix = ("install the NVIDIA driver for Windows (it brings the GPU into WSL2); nothing to install inside Linux"
               if is_wsl() else "install the NVIDIA driver: sudo ubuntu-drivers install, then reboot")
        return Check("NVIDIA driver", "fail", "nvidia-smi did not run", fix)
    gpus = {}
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4 and parts[0].isdigit():
            gpus[int(parts[0])] = (parts[1], int(float(parts[2])), parts[3])
    if cfg.gpu_index not in gpus:
        return Check("NVIDIA driver", "fail", f"no GPU {cfg.gpu_index} (found {sorted(gpus)})",
                     "kwh-host init --gpu <index>")
    name, mib, driver = gpus[cfg.gpu_index]
    detail = f"GPU {cfg.gpu_index}: {name}, {mib / 1024:.0f} GB, driver {driver}"
    if mib < MIN_VRAM_MIB:
        return Check("NVIDIA driver", "fail", detail + "; the reference model needs a 16 GB card or larger", None)
    return Check("NVIDIA driver", "ok", detail)


def driver_cuda(run: Run = _run) -> Optional[str]:
    """The newest CUDA the driver supports, from nvidia-smi's banner ("CUDA Version: 13.0")."""
    r = run(["nvidia-smi"])
    if r.returncode == 0:
        for line in r.stdout.splitlines():
            if "CUDA Version:" in line:
                return line.split("CUDA Version:")[1].split("|")[0].strip()
    return None


def check_engine_build(cfg: HostConfig, run: Run = _run) -> Check:
    """The engine image is built for a CUDA version; the driver has to support it. The default
    build needs CUDA 13 (driver 580 or newer); the cu129 build runs on CUDA 12.x drivers."""
    name, image = "engine build", cfg.docker_image
    cuda_s = driver_cuda(run)
    cuda = cuda_tuple(cuda_s)
    driver_fix = "update the NVIDIA driver" + (" for Windows" if is_wsl() else "")
    if cuda is None:
        return Check(name, "warn", f"{image_label(image)}; could not read the driver's CUDA version from nvidia-smi")
    if cuda[0] < 12:
        return Check(name, "fail", f"the driver supports CUDA {cuda_s}; the engine needs CUDA 12.8 or newer",
                     driver_fix + " to version 570 or newer")
    need = IMAGE_CUDA.get(image)
    if need is None:
        return Check(name, "warn", f"{image} is not a published build of the engine; cannot tell which CUDA it needs")
    if cuda < need:
        return Check(name, "fail", f"{image_label(image)} needs a CUDA {need[0]} driver (580 or newer); this one supports CUDA {cuda_s}",
                     f"run kwh-host init again with the same options (it picks {image_label(engine_image_for(cuda_s))}), "
                     f"or {driver_fix} to 580 or newer")
    if cuda[0] == 12 and cuda < TESTED_CUDA12:
        return Check(name, "warn", f"{image_label(image)} on a CUDA {cuda_s} driver; it has run on 12.8 and newer",
                     f"if the engine does not start, {driver_fix} to 570 or newer")
    return Check(name, "ok", f"{image_label(image)}, on a driver that supports CUDA {cuda_s}")


def docker_info(run: Run = _run) -> Optional[dict]:
    r = run(["docker", "info", "--format", "{{json .}}"])
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except ValueError:
        return None


def check_docker(info: Optional[dict], run: Run = _run) -> Check:
    if not shutil.which("docker"):
        return Check("Docker", "fail", "not installed", "rerun the installer, or: curl -fsSL https://get.docker.com | sh")
    if info is None:
        r = run(["docker", "version"])
        if "permission denied" in (r.stderr or "").lower():
            return Check("Docker", "fail", "this user may not use Docker",
                         "sudo usermod -aG docker $USER, then log out and back in")
        return Check("Docker", "fail", "the Docker daemon is not reachable", "sudo systemctl start docker")
    return Check("Docker", "ok", f"{info.get('OperatingSystem', '?')}, server {info.get('ServerVersion', '?')}")


TOOLKIT_FIX = ("rerun the installer (it tries a GPU container and installs the toolkit if that fails), or install "
               "nvidia-container-toolkit, then: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker")


def check_gpu_runtime(cfg: HostConfig, info: Optional[dict], run: Run = _run,
                      which: Callable[[str], Optional[str]] = shutil.which) -> Check:
    """Can Docker hand the GPU to a container? An "nvidia" runtime in Docker's settings is not
    enough: on a machine whose toolkit was removed the runtime stays registered while its programs
    are gone (a rented VM image, 2026-10-05), and `docker run --gpus` fails. So the toolkit's hook
    has to exist and, once the engine image is pulled, a throwaway container from it has to see
    the GPU."""
    name = "GPU in containers"
    if info is None:
        return Check(name, "fail", "Docker not reachable", None)
    if "Docker Desktop" in str(info.get("OperatingSystem", "")):
        return Check(name, "ok", "Docker Desktop GPU support")
    if not which("nvidia-container-runtime-hook"):
        detail = ("Docker lists an nvidia runtime, but the NVIDIA Container Toolkit's programs are missing"
                  if "nvidia" in (info.get("Runtimes") or {}) else "the NVIDIA Container Toolkit is not installed")
        return Check(name, "fail", detail, TOOLKIT_FIX)
    if run(["docker", "image", "inspect", "--format", "{{.Id}}", cfg.docker_image]).returncode != 0:
        return Check(name, "warn", "toolkit installed; tried with a container once the engine image is fetched",
                     "kwh-host fetch")
    r = run(["docker", "run", "--rm", "--gpus", f"device={cfg.gpu_index}", "--entrypoint", "nvidia-smi",
             cfg.docker_image, "-L"])
    if r.returncode == 0 and "GPU" in r.stdout:
        return Check(name, "ok", "a container sees " + r.stdout.strip().splitlines()[0].split(" (UUID")[0])
    said = ((r.stderr or r.stdout or "").strip().splitlines() or ["no output"])[0][:200]
    return Check(name, "fail", f"a test container could not use GPU {cfg.gpu_index}: {said}", TOOLKIT_FIX)


def check_transport(cfg: HostConfig, info: Optional[dict]) -> Check:
    desktop = info is not None and "Docker Desktop" in str(info.get("OperatingSystem", ""))
    if cfg.engine_transport == "uds" and desktop:
        return Check("engine network", "fail", "Docker Desktop cannot share the engine's Unix socket with this machine",
                     "use Docker Engine inside WSL2 (HOST-CLIENT.md §12), or: kwh-host init --engine-transport tcp")
    if cfg.engine_transport == "tcp":
        return Check("engine network", "warn", "tcp: reachable only from this machine, but the engine can reach out",
                     "with Docker Engine (not Desktop), kwh-host init --engine-transport uds cuts it off entirely")
    return Check("engine network", "ok", "none; the engine is reached through a Unix socket only")


def check_image(cfg: HostConfig, run: Run = _run) -> Check:
    r = run(["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", cfg.docker_image])
    if r.returncode != 0:
        return Check("engine image", "fail", f"{image_label(cfg.docker_image)} is not pulled", "kwh-host fetch")
    return Check("engine image", "ok", image_label(cfg.docker_image))


def check_model(cfg: HostConfig, revision: Optional[str]) -> Check:
    snap = snapshot_dir(cfg.hf_home, ref.MODEL_ID, revision)
    if snap is None:
        return Check("checkpoint", "fail", f"{ref.MODEL_ID} @ {(revision or '?')[:7]} not in {cfg.hf_home}", "kwh-host fetch")
    return Check("checkpoint", "ok", f"{ref.MODEL_ID} @ {(revision or '?')[:7]} (kwh-host fetch checked its hashes)")


def check_mode(cfg: HostConfig) -> Check:
    if cfg.engine_mode != "docker":
        return Check("engine mode", "warn", "bare-metal: for test pods only; the platform registers Docker hosts (D4)",
                     "kwh-host init --engine docker")
    return Check("engine mode", "ok", f"docker, context {cfg.max_model_len} tokens")


def run_checks(cfg: HostConfig, revision: Optional[str], run: Run = _run) -> List[Check]:
    gpu = check_gpu(cfg, run)
    checks = [check_platform(), gpu, check_mode(cfg)]
    if cfg.engine_mode == "docker":
        if gpu.status != "fail":
            checks.append(check_engine_build(cfg, run))
        info = docker_info(run) if shutil.which("docker") else None
        checks += [check_docker(info, run), check_gpu_runtime(cfg, info, run), check_transport(cfg, info)]
        if info is not None:
            checks.append(check_image(cfg, run))
    checks.append(check_model(cfg, revision))
    return checks


def systemd_user_available() -> bool:
    return shutil.which("systemctl") is not None and os.path.isdir("/run/systemd/system")
