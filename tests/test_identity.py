import os
import stat

from kwh_host.identity import (Identity, canonical_bytes, request_message, verify, verify_report_signature,
                               verify_request, verify_result_signature)


def test_keypair_persists_with_0600(tmp_path):
    p = tmp_path / "id.key"
    a = Identity.load_or_create(p)
    b = Identity.load_or_create(p)
    assert a.public_key_hex == b.public_key_hex and len(a.public_key_hex) == 64
    if os.name == "posix":
        assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_request_signature_binds_body_time_method_and_path():
    ident = Identity.generate()
    body = canonical_bytes({"b": 1, "a": [1, 2]})
    assert body == b'{"a":[1,2],"b":1}'
    path = "/v1/hosts/h_1/heartbeat"
    headers = ident.sign_request(body, method="POST", path=path, timestamp=1000)
    assert verify_request(headers, body, "POST", path, now=1000) == ident.public_key_hex
    assert verify_request(headers, body + b" ", "POST", path, now=1000) is None                  # body tampered
    assert verify_request(headers, body, "POST", path, now=1000 + 301) is None                    # stale
    assert verify_request(headers, body, "POST", "/v1/hosts/h_1/liveness", now=1000) is None      # replayed elsewhere
    assert verify_request(headers, body, "GET", path, now=1000) is None                           # other method
    assert verify_request(headers, body, "POST", path, expected_public_key="00" * 32, now=1000) is None
    other = Identity.generate()
    assert verify(other.public_key_hex, request_message(1000, "POST", path, body), headers["X-Kwh-Signature"]) is False


def test_report_signature_does_not_change_hash():
    from kwh_bench.report import report_hash
    ident = Identity.generate()
    report = {"spec": {"series": "I-1"}, "score": {"units_per_hour": 1.0}, "signature": None}
    report["report_sha256"] = report_hash(report)
    ident.sign_report(report)
    assert report_hash(report) == report["report_sha256"]
    assert verify_report_signature(report) == ident.public_key_hex
    report["signature"]["sig"] = report["signature"]["sig"][:-4] + "AAAA"
    assert verify_report_signature(report) is None


def test_result_signature_covers_content_and_is_not_a_report_signature():
    ident = Identity.generate()
    result = ident.sign_result({"job_id": "j_1.1", "status": "completed", "outputs": [{"index": 0, "token_ids": [1, 2]}],
                                "result_sha256": None, "signature": None})
    assert verify_result_signature(result) == ident.public_key_hex
    tampered = {**result, "outputs": [{"index": 0, "token_ids": [1, 3]}]}
    assert verify_result_signature(tampered) is None
    # A report-style signature over the same digest does not pass as a result signature.
    forged = dict(result)
    forged["signature"] = {**result["signature"], "sig": ident.sign(result["result_sha256"].encode("ascii"))}
    assert verify_result_signature(forged) is None
