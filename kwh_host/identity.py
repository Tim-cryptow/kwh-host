"""Host identity: an ed25519 keypair generated once at `kwh-host init` (HOST-CLIENT.md §8).

The public key is the host's durable identity across reinstalls and token rotations.
Every platform request is signed over (timestamp, sha256(body)); the benchmark report
is signed over its own content hash and carried in the report's `signature` field.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

MAX_CLOCK_SKEW_S = 300


def canonical_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def request_message(timestamp: int, body: bytes) -> bytes:
    return f"{timestamp}\n{hashlib.sha256(body).hexdigest()}".encode("ascii")


class Identity:
    def __init__(self, private_key: Ed25519PrivateKey):
        self._key = private_key

    # -- lifecycle -----------------------------------------------------
    @classmethod
    def generate(cls) -> "Identity":
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def load(cls, path: Path) -> "Identity":
        raw = path.read_bytes()
        key = serialization.load_pem_private_key(raw, password=None)
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

    def sign_request(self, body: bytes, timestamp: Optional[int] = None) -> dict:
        ts = int(timestamp if timestamp is not None else time.time())
        return {
            "X-Kwh-Timestamp": str(ts),
            "X-Kwh-Public-Key": self.public_key_hex,
            "X-Kwh-Signature": self.sign(request_message(ts, body)),
        }

    def sign_report(self, report: dict) -> dict:
        """Fill the report's `signature` field. The content hash excludes the field, so signing
        does not change `report_sha256`."""
        digest = report["report_sha256"]
        report["signature"] = {
            "alg": "ed25519",
            "public_key": self.public_key_hex,
            "signed": "report_sha256",
            "sig": self.sign(digest.encode("ascii")),
        }
        return report


def verify(public_key_hex: str, message: bytes, signature_b64: str) -> bool:
    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        pub.verify(base64.b64decode(signature_b64), message)
        return True
    except (InvalidSignature, ValueError):
        return False


def verify_request(headers: dict, body: bytes, expected_public_key: Optional[str] = None,
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
    if not verify(pub, request_message(ts, body), sig):
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
