"""Client side of the platform API contract (HOST-CLIENT.md §8).

Every request is signed with the host identity over (timestamp, method, path, body);
after registration the bearer token is sent as well. The job channel is a WebSocket
opened by the host (D3): its handshake is signed the same way, as a GET with an empty
body. The mock platform in `mock.py` implements the other side; the real platform
(step 4) replaces the base URL.
"""

from __future__ import annotations

import time
from typing import Callable, Optional
from urllib.parse import urlsplit, urlunsplit

import httpx

from .. import __version__
from ..identity import Identity, canonical_bytes

# Job envelopes carry prompt token ids; 256 requests x 1,024 ids is ~2 MB of JSON.
WS_MAX_MESSAGE_BYTES = 32 * 1024 * 1024


class PlatformError(RuntimeError):
    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(f"platform {status}: {detail}")


class PlatformClient:
    def __init__(self, base_url: str, identity: Identity, token: Optional[str] = None,
                 host_id: Optional[str] = None, transport: Optional[httpx.AsyncBaseTransport] = None,
                 timeout: float = 30.0, clock: Callable[[], float] = time.time):
        self.base_url = base_url.rstrip("/")
        self._base_path = urlsplit(self.base_url).path.rstrip("/")
        self.identity = identity
        self.token = token
        self.host_id = host_id
        self.clock = clock
        self._http = httpx.AsyncClient(base_url=self.base_url, transport=transport, timeout=timeout)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "PlatformClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    # -- transport -----------------------------------------------------
    def _headers(self, method: str, path: str, raw: bytes) -> dict:
        headers = {"User-Agent": f"kwh-host/{__version__}"}
        # Sign the path as the platform will see it (base URL path + endpoint path).
        headers.update(self.identity.sign_request(raw, method=method, path=self._base_path + path,
                                                  timestamp=int(self.clock())))
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def _call(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        # Sign exactly the bytes that go on the wire: a GET has an empty body, so it signs b"".
        raw = b"" if method == "GET" else canonical_bytes(body if body is not None else {})
        headers = self._headers(method, path, raw)
        if method != "GET":
            headers["Content-Type"] = "application/json"
        r = await self._http.request(method, path, content=raw if method != "GET" else None, headers=headers)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise PlatformError(r.status_code, str(detail))
        return r.json()

    def _hp(self, suffix: str = "") -> str:
        if not self.host_id:
            raise PlatformError(0, "not registered")
        return f"/v1/hosts/{self.host_id}{suffix}"

    # -- contract ------------------------------------------------------
    async def register(self, report: dict) -> dict:
        out = await self._call("POST", "/v1/hosts", {
            "report": report,
            "public_key": self.identity.public_key_hex,
            "client_version": __version__,
        })
        self.host_id = out["host_id"]
        self.token = out["token"]
        return out

    async def heartbeat(self, engine: dict, gpu_sample: dict, in_flight: int = 0, wants_mint: bool = True) -> dict:
        return await self._call("POST", self._hp("/heartbeat"), {
            "engine": engine, "gpu_sample": gpu_sample, "in_flight": in_flight,
            "wants_mint": wants_mint, "client_version": __version__,
        })

    async def liveness(self, challenge_id: str, mean_logprob: Optional[float], elapsed_ms: int) -> dict:
        return await self._call("POST", self._hp("/liveness"), {
            "challenge_id": challenge_id, "mean_logprob": mean_logprob, "elapsed_ms": elapsed_ms,
        })

    async def microbench(self, units_per_hour: float, job_seconds: float, gpu_sample: dict) -> dict:
        return await self._call("POST", self._hp("/microbench"), {
            "units_per_hour": units_per_hour, "job_seconds": job_seconds, "gpu_sample": gpu_sample,
        })

    async def upload_report(self, report: dict) -> dict:
        return await self._call("POST", self._hp("/reports"), {"report": report})

    async def status(self) -> dict:
        return await self._call("GET", self._hp())

    # -- job channel (§7) ----------------------------------------------
    def jobs_url(self) -> str:
        parts = urlsplit(self.base_url)
        scheme = {"http": "ws", "https": "wss"}.get(parts.scheme, parts.scheme)
        return urlunsplit((scheme, parts.netloc, self._base_path + self._hp("/jobs"), "", ""))

    def connect_jobs(self, open_timeout: float = 20.0):
        """An async context manager yielding the job WebSocket (websockets' client connection)."""
        from websockets.asyncio.client import connect
        return connect(
            self.jobs_url(),
            additional_headers=self._headers("GET", self._hp("/jobs"), b""),
            open_timeout=open_timeout,
            max_size=WS_MAX_MESSAGE_BYTES,
            ping_interval=20,
            ping_timeout=20,
        )
