"""`kwh-host fetch`: put the reference checkpoint and the engine image on this machine.

The engine runs offline from a read-only cache (sandbox.py), so everything it needs arrives
here first: the checkpoint at the revision the lock pins, every file checked against the
lock's SHA-256 (the same hashes the reference node recorded), and the certified engine image.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional

from kwh_bench import reference as ref
from kwh_bench.lockfile import Lock

Log = Callable[[str], None]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_dir(hf_home: Path, model: str, revision: Optional[str]) -> Optional[Path]:
    """Where a revision of `model` lives in the cache under `hf_home`, if it is there."""
    repo = hf_home / "hub" / ("models--" + model.replace("/", "--"))
    if revision:
        d = repo / "snapshots" / revision
        return d if d.is_dir() else None
    ref_main = repo / "refs" / "main"
    if ref_main.exists():
        d = repo / "snapshots" / ref_main.read_text().strip()
        return d if d.is_dir() else None
    return None


def download_model(hf_home: Path, model: str, revision: Optional[str], log: Log) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:   # pragma: no cover - a declared dependency
        raise RuntimeError("huggingface_hub is required (pip install huggingface_hub)") from e
    log(f"downloading {model}" + (f" @ {revision}" if revision else "") + f" into {hf_home} (about 9 GB the first time)")
    return Path(snapshot_download(repo_id=model, revision=revision, cache_dir=str(hf_home / "hub")))


def verify_model(snapshot: Path, lock: Lock, log: Log = lambda s: None) -> List[str]:
    """Every file the lock hashed, compared with what is on disk. Empty when it all matches."""
    problems: List[str] = []
    if not lock.model_files:
        return ["the kwh-bench lock records no model file hashes"]
    for name, want in sorted(lock.model_files.items()):
        f = snapshot / name
        if not f.exists():
            problems.append(f"{name}: missing")
            continue
        got = sha256_file(f)
        log(f"  {name}: {'ok' if got == want else 'MISMATCH'}")
        if got != want:
            problems.append(f"{name}: sha256 {got[:16]}… is not the locked {want[:16]}…")
    return problems


def pull_image(image: str, log: Log) -> Optional[str]:
    """`docker pull`, progress on the terminal; returns the image's repo digest."""
    log(f"pulling {image} (about {14 if 'cu129' in image else 9} GB the first time)")
    # docker prints its progress on stdout; send it to stderr so `kwh-host fetch` prints only its JSON
    subprocess.run(["docker", "pull", image], check=True, stdout=sys.stderr)
    out = subprocess.run(["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", image],
                         capture_output=True, text=True, timeout=30)
    digests = [d for d in (out.stdout.strip().strip("[]").replace('"', "").split(",")) if d]
    return digests[0] if digests else None


def fetch(hf_home: Path, lock: Lock, image: Optional[str], log: Log, model: Optional[str] = None,
          verify: bool = True) -> dict:
    """Download (or find) the checkpoint, check it, pull the image. `model` overrides the
    reference model for wrong-model tests; such a model is not hash-checked."""
    hf_home.mkdir(parents=True, exist_ok=True)
    reference = model is None
    model = model or ref.MODEL_ID
    revision = lock.model_revision if reference else None
    snap = download_model(hf_home, model, revision, log)
    out = {"model": model, "revision": revision, "snapshot": str(snap), "verified": None, "image": image,
           "image_digest": None}
    if reference and verify:
        log("checking every file against the lock")
        problems = verify_model(snap, lock, log)
        if problems:
            raise RuntimeError("checkpoint does not match the lock: " + "; ".join(problems))
        out["verified"] = True
    if image:
        out["image_digest"] = pull_image(image, log)
    return out
