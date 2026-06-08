"""core/crypto.py — AES-256-GCM encryption for API keys stored in SQLite.

Keys are encrypted before INSERT/UPDATE and decrypted after SELECT.
The encryption key is derived from SECRET_KEY via PBKDF2-HMAC-SHA256
with a fixed application salt so the same plaintext always produces a
deterministic *key* (the per-value nonce still makes ciphertexts unique).

Ciphertext format (base64-encoded, safe to store as TEXT in SQLite):
    <16-byte salt> + <12-byte nonce> + <ciphertext> + <16-byte tag>
    → base64url-encoded → stored as "enc:<b64>"

Plaintext values that are empty strings are stored as empty strings
(no encryption overhead for empty slots).
"""
from __future__ import annotations

import base64
import os
import hashlib

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_PREFIX = "enc:"


def _derive_key(secret: str) -> bytes:
    """Derive a 32-byte AES key from the application SECRET_KEY."""
    return hashlib.pbkdf2_hmac(
        "sha256",
        secret.encode("utf-8"),
        b"signalmind-apikey-salt-v1",
        iterations=200_000,
        dklen=32,
    )


def _get_secret() -> str:
    secret = os.getenv("SECRET_KEY", "")
    if not secret:
        raise RuntimeError(
            "SECRET_KEY is not set — cannot encrypt/decrypt API keys. "
            "Add SECRET_KEY to your .env file."
        )
    return secret


def encrypt_key(plaintext: str) -> str:
    """Encrypt an API key string. Returns an opaque 'enc:...' string."""
    if not plaintext:
        return plaintext
    aes_key = _derive_key(_get_secret())
    nonce = os.urandom(12)
    aesgcm = AESGCM(aes_key)
    ct = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    blob = nonce + ct  # nonce (12) + ciphertext + tag (16)
    return _PREFIX + base64.b64encode(blob).decode("ascii")


def decrypt_key(stored: str) -> str:
    """Decrypt a stored API key. Returns the original plaintext."""
    if not stored or not stored.startswith(_PREFIX):
        # Already plaintext (legacy row or empty) — return as-is
        return stored
    aes_key = _derive_key(_get_secret())
    blob = base64.b64decode(stored[len(_PREFIX):])
    nonce, ct = blob[:12], blob[12:]
    aesgcm = AESGCM(aes_key)
    try:
        return aesgcm.decrypt(nonce, ct, None).decode("utf-8")
    except Exception:
        # Wrong key or corrupted — return empty so the user re-enters
        return ""


def encrypt_if_needed(value: str) -> str:
    """Idempotent encrypt: skip if already encrypted or empty."""
    if not value or value.startswith(_PREFIX):
        return value
    return encrypt_key(value)


def decrypt_if_needed(value: str) -> str:
    """Idempotent decrypt: skip if not encrypted or empty."""
    if not value or not value.startswith(_PREFIX):
        return value
    return decrypt_key(value)
