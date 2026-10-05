"""When a benchmark report stops describing the host (HOST-CLIENT.md §5).

A report is a measurement of one GPU, on one driver, with one engine build, at one point in
time. It stops counting when any of those changes, and after seven days in any case. The
platform and the daemon apply the same rules: the platform to hold the host degraded (no
minting, no jobs) until a new report is accepted, the daemon to notice first and produce one.
What the platform knows of the GPU, driver and image comes from the host's heartbeats; a host
that lies about them skips a re-benchmark, and the challenges and micro-benchmarks are what
catch a rig that no longer performs as its report says.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from .config import image_label, pinned_image

REBENCH_EVERY_SECONDS = 7 * 86400.0
MICROBENCH_REASON = "micro-benchmark outside 10% of the rate twice in a row"


def iso_ts(s: Optional[str]) -> Optional[float]:
    """'2026-10-05T09:35:39Z' -> unix seconds; None when absent or unreadable."""
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def span(seconds: float) -> str:
    """7 * 86400 -> '7 days'; 3600 -> '1 hour'; 90 -> '90 seconds'."""
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
        if seconds >= size and seconds % size == 0:
            n = int(seconds // size)
            return f"{n} {unit}" + ("s" if n != 1 else "")
    return f"{seconds:g} seconds"


def report_gpus(report: dict) -> Dict[str, Optional[str]]:
    """GPU UUID -> driver version, for every GPU the report recorded (none for in-process engines)."""
    gpus = (report.get("hardware") or {}).get("gpus") or []
    return {g["uuid"]: g.get("driver_version") for g in gpus if isinstance(g, dict) and g.get("uuid")}


def report_image(report: dict) -> Optional[str]:
    """The engine image a Docker report was measured with, pinned (older reports name the tag)."""
    eng = report.get("engine") or {}
    image = (eng.get("extra") or {}).get("docker_image") if eng.get("launch_mode") == "docker" else None
    return pinned_image(image) if image else None


def change_reasons(gpus: Dict[str, Optional[str]], image: Optional[str], uuid: Optional[str],
                   driver: Optional[str], running_image: Optional[str]) -> List[str]:
    """What changed between the report (`gpus`, `image`) and the running host. Each reason is a
    stable string, so the same change is one reason however many heartbeats repeat it."""
    out: List[str] = []
    if gpus and uuid:
        if uuid not in gpus:
            out.append(f"GPU changed: {uuid} is not the GPU benchmarked")
        elif driver and gpus[uuid] and driver != gpus[uuid]:
            out.append(f"driver changed: {gpus[uuid]} benchmarked, {driver} now")
    if image and running_image and pinned_image(running_image) != image:
        out.append(f"engine image changed: {image_label(image)} benchmarked, "
                   f"{image_label(pinned_image(running_image))} now")
    return out


def age_reason(report_at: Optional[float], now: float, every: float = REBENCH_EVERY_SECONDS) -> Optional[str]:
    if report_at is not None and now - report_at >= every:
        return f"report older than {span(every)}"
    return None


def local_reasons(report: dict, sample: dict, running_image: Optional[str], now: float,
                  every: float = REBENCH_EVERY_SECONDS) -> List[str]:
    """The daemon's own check: its report against its GPU sample, its engine image and the clock."""
    out = change_reasons(report_gpus(report), report_image(report), sample.get("uuid"), sample.get("driver_version"),
                         running_image)
    old = age_reason(iso_ts(report.get("finished_at")), now, every)
    return out + ([old] if old else [])
