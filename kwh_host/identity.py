"""Host identity: an ed25519 keypair generated once at `kwh-host init` (HOST-CLIENT.md §8).

The public key is the host's durable identity across reinstalls and token rotations.
Three things are signed with it, each over a message that names what it is, so a
signature made for one purpose can never be replayed as another:

    request   "kwh-req-v1\\n{timestamp}\\n{METHOD}\\n{path}\\n{sha256(body)}"   every platform call,
              including the WebSocket handshake (GET, empty body)
    report    the report's own content hash, in the report's `signature` field
              (the schema reserved the field; kept as-is so published reports still verify)
    result    "kwh-result-v1\\n{result_sha256}"   every job result, so a delivered output is
              attributable to the host that produced it (what step 3 slashes against)
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

MAX_CLOCK_SKEW_S = 300
REQUEST_PURPOSE = "kwh-req-v1"
RESULT_PURPOSE = "kwh-result-v1"


def canonical_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(obj: dict, exclude: Iterable[str]) -> str:
    """sha256 of the canonical JSON of `obj` with the `exclude` fields nulled (the same
    construction as kwh-bench's report_sha256)."""
    clone = dict(obj)
    for k in exclude:
        clone[k] = None
    return sha256_hex(canonical_bytes(clone))


def request_message(timestamp: int, method: str, path: str, body: bytes) -> bytes:
    return f"{REQUEST_PURPOSE}\n{timestamp}\n{method.upper()}\n{path}\n{sha256_hex(body)}".encode("utf-8")


class Identity:
    def __init__(self, private_key: Ed25519PrivateKey):
        self._key = private_key

    # -- lifecycle -----------------------------------------------------
    @classmethod
    def generate(cls) -> "Identity":
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def load(cls, path: Path) -> "Identity":
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError(f"{path} is not an ed25519 key")
        return cls(key)

    @classmethod
    def load_or_create(cls, path: Path) -> "Identity":
        if path.exists():
            return cls.load(path)
        ident = cls.generate()
        ident.save(path)
        return ident

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pem = self._key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(pem)

    # -- keys ----------------------------------------------------------
    @property
    def public_key_hex(self) -> str:
        return self._key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw).hex()

    # -- signing -------------------------------------------------------
    def sign(self, message: bytes) -> str:
        return base64.b64encode(self._key.sign(message)).decode("ascii")

    def sign_request(self, body: bytes, method: str, path: str, timestamp: Optional[int] = None) -> dict:
        ts = int(timestamp if timestamp is not None else time.time())
        return {
            "X-Kwh-Timestamp": str(ts),
            "X-Kwh-Public-Key": self.public_key_hex,
            "X-Kwh-Signature": self.sign(request_message(ts, method, path, body)),
        }

    def sign_report(self, report: dict) -> dict:
        """Fill the report's `signature` field. The content hash excludes the field, so signing
        does not change `report_sha256`."""
        report["signature"] = {
            "alg": "ed25519",
            "public_key": self.public_key_hex,
            "signed": "report_sha256",
            "sig": self.sign(report["report_sha256"].encode("ascii")),
        }
        return report

    def sign_result(self, result: dict) -> dict:
        """Hash the result (excluding its hash and signature) and sign the hash."""
        result["result_sha256"] = content_hash(result, ("result_sha256", "signature"))
        result["signature"] = {
            "alg": "ed25519",
            "public_key": self.public_key_hex,
            "signed": f"{RESULT_PURPOSE}:result_sha256",
            "sig": self.sign(f"{RESULT_PURPOSE}\n{result['result_sha256']}".encode("ascii")),
        }
        return result


def verify(public_key_hex: str, message: bytes, signature_b64: str) -> bool:
    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        pub.verify(base64.b64decode(signature_b64), message)
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def verify_request(headers: dict, body: bytes, method: str, path: str, expected_public_key: Optional[str] = None,
                   now: Optional[float] = None) -> Optional[str]:
    """Return the verified public key, or None. Headers are matched case-insensitively."""
    h = {k.lower(): v for k, v in headers.items()}
    try:
        ts = int(h.get("x-kwh-timestamp", ""))
    except ValueError:
        return None
    pub = h.get("x-kwh-public-key", "")
    sig = h.get("x-kwh-signature", "")
    if expected_public_key and pub != expected_public_key:
        return None
    if abs((now if now is not None else time.time()) - ts) > MAX_CLOCK_SKEW_S:
        return None
    if not verify(pub, request_message(ts, method, path, body), sig):
        return None
    return pub


def verify_report_signature(report: dict) -> Optional[str]:
    """Return the signing public key if the report's signature is valid, else None."""
    sig = report.get("signature") or {}
    if sig.get("alg") != "ed25519" or sig.get("signed") != "report_sha256":
        return None
    pub = sig.get("public_key", "")
    if verify(pub, report["report_sha256"].encode("ascii"), sig.get("sig", "")):
        return pub
    return None


def verify_result_signature(result: dict) -> Optional[str]:
    """Return the signing public key if the result's hash matches its content and the
    signature over it is valid, else None."""
    sig = result.get("signature") or {}
    if sig.get("alg") != "ed25519" or sig.get("signed") != f"{RESULT_PURPOSE}:result_sha256":
        return None
    digest = result.get("result_sha256") or ""
    if content_hash(result, ("result_sha256", "signature")) != digest:
        return None
    pub = sig.get("public_key", "")
    if verify(pub, f"{RESULT_PURPOSE}\n{digest}".encode("ascii"), sig.get("sig", "")):
        return pub
    return None
