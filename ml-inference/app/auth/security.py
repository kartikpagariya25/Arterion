import os
import hashlib
import hmac
import time
import json
import base64

JWT_SECRET = os.environ.get("CARDIOSENSE_JWT_SECRET", "cardiosense-dev-secret-change-me")
JWT_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return base64.b64encode(salt + digest).decode()


def verify_password(password: str, stored_hash: str) -> bool:
    raw = base64.b64decode(stored_hash.encode())
    salt, digest = raw[:16], raw[16:]
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return hmac.compare_digest(digest, check)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_token(user_id: int, role: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {"sub": user_id, "role": role, "exp": int(time.time()) + JWT_TTL_SECONDS}
    segments = [
        _b64url_encode(json.dumps(header).encode()),
        _b64url_encode(json.dumps(payload).encode()),
    ]
    signing_input = ".".join(segments).encode()
    signature = hmac.new(JWT_SECRET.encode(), signing_input, hashlib.sha256).digest()
    segments.append(_b64url_encode(signature))
    return ".".join(segments)


def decode_token(token: str) -> dict:
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
    except ValueError:
        raise ValueError("Malformed token")

    signing_input = f"{header_b64}.{payload_b64}".encode()
    expected_sig = hmac.new(JWT_SECRET.encode(), signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(expected_sig, _b64url_decode(sig_b64)):
        raise ValueError("Invalid token signature")

    payload = json.loads(_b64url_decode(payload_b64))
    if payload.get("exp", 0) < time.time():
        raise ValueError("Token expired")
    return payload