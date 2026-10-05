"""The daemon's own event log (HOST-CLIENT.md §5): ~/.kwh-host/events.jsonl, one JSON object per line.

These are raw events as the host saw them: heartbeats sent and the platform's verdict on each,
heartbeats that could not be sent, challenges with their deltas, micro-benchmarks, jobs with
their latency, engine starts and stops, re-benchmarks. The platform keeps its own record of
the same things and scores from that one, since a host can edit this file; this one is for the
host to read (`kwh-host events`) and for telling the two views apart when they disagree.

The file rotates at 5 MB to events.jsonl.1, so the log keeps two to four weeks of a busy host.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional

MAX_BYTES = 5 * 1024 * 1024


class EventLog:
    def __init__(self, path: Path, max_bytes: int = MAX_BYTES, clock: Callable[[], float] = time.time):
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.clock = clock

    @property
    def rotated(self) -> Path:
        return self.path.with_name(self.path.name + ".1")

    def add(self, kind: str, **data) -> dict:
        """Append one event. Never raises: a full disk must not stop the host from hosting."""
        ev = {"t": round(self.clock(), 3), "kind": kind, **data}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_size >= self.max_bytes:
                os.replace(self.path, self.rotated)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, default=str, separators=(",", ":")) + "\n")
        except OSError:
            pass
        return ev

    def read(self, since: Optional[float] = None, limit: Optional[int] = None,
             kinds: Optional[Iterable[str]] = None) -> List[dict]:
        """Events after `since`, oldest first, the last `limit` of them."""
        want = set(kinds) if kinds else None
        out: List[dict] = []
        for p in (self.rotated, self.path):
            try:
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        try:
                            ev = json.loads(line)
                        except ValueError:
                            continue          # a line cut short by a crash
                        if not isinstance(ev, dict):
                            continue
                        if since is not None and (ev.get("t") or 0) <= since:
                            continue
                        if want is not None and ev.get("kind") not in want:
                            continue
                        out.append(ev)
            except OSError:
                continue
        return out[-limit:] if limit else out


class NullEventLog(EventLog):
    """For callers that keep no log."""

    def __init__(self):
        super().__init__(Path(os.devnull))

    def add(self, kind: str, **data) -> dict:
        return {"t": self.clock(), "kind": kind, **data}

    def read(self, since=None, limit=None, kinds=None) -> List[dict]:
        return []
