"""
Symmetric encryption for secrets stored in the database (broker logins/tokens).

Key source, in order:
  1. env `BROKER_ENCRYPTION_KEY` — a urlsafe-base64 Fernet key, and
  2. a key file `broker.key` created on first use NEXT TO THE SQLITE DB
     (so on HAOS it lands in /data and survives add-on updates, and a DB
     backup taken by copying the data directory keeps the two together).

The key is deliberately NOT derived from SECRET_KEY: that setting ships with a
weak public default, and rotating the JWT secret must not silently destroy
every stored broker login.

If the key changes (file deleted, env var changed), stored values can no
longer be decrypted. That raises SecretsUnavailable with a message telling the
user to reconnect — never a raw InvalidToken traceback.
"""
from __future__ import annotations

import os
import threading

from cryptography.fernet import Fernet, InvalidToken

KEY_FILENAME = "broker.key"

_lock = threading.Lock()
_fernet: Fernet | None = None
_fernet_source: str | None = None


class SecretsUnavailable(Exception):
    """Stored secrets can't be decrypted with the current key."""


def _db_directory() -> str:
    """Directory holding the SQLite file named by settings.database_url."""
    from config import get_settings
    url = get_settings().database_url or ""
    path = None
    if url.startswith("sqlite:///"):
        path = url[len("sqlite:///"):]
    if not path:
        return os.path.abspath("data")
    # sqlite:////data/x.db -> "/data/x.db"; sqlite:///./data/x.db -> "./data/x.db"
    return os.path.dirname(os.path.abspath(path)) or os.path.abspath(".")


def key_path() -> str:
    return os.path.join(_db_directory(), KEY_FILENAME)


def _load_or_create_key() -> tuple[bytes, str]:
    env = os.environ.get("BROKER_ENCRYPTION_KEY", "").strip()
    if env:
        return env.encode(), "env"
    path = key_path()
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            return fh.read().strip(), path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    key = Fernet.generate_key()
    # O_EXCL: two workers racing on first use must not each write a different key.
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        with open(path, "rb") as fh:
            return fh.read().strip(), path
    with os.fdopen(fd, "wb") as fh:
        fh.write(key)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass  # Windows / odd filesystems — best effort
    return key, path


def _get() -> Fernet:
    global _fernet, _fernet_source
    with _lock:
        if _fernet is None:
            key, source = _load_or_create_key()
            try:
                _fernet = Fernet(key)
            except (ValueError, TypeError) as exc:
                raise SecretsUnavailable(
                    "The broker encryption key is malformed (" + source + "). "
                    "Fix or remove it, then reconnect your broker."
                ) from exc
            _fernet_source = source
        return _fernet


def reset_cache() -> None:
    """Forget the loaded key (tests, or after the key file is replaced)."""
    global _fernet, _fernet_source
    with _lock:
        _fernet = None
        _fernet_source = None


def encrypt(plaintext: str) -> str:
    if plaintext is None:
        raise ValueError("cannot encrypt None")
    return _get().encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt(ciphertext: str) -> str:
    if not ciphertext:
        raise SecretsUnavailable("No stored value.")
    try:
        return _get().decrypt(ciphertext.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise SecretsUnavailable(
            "Stored broker credentials can't be decrypted (the encryption key "
            "changed); reconnect your broker account."
        ) from exc


def encrypt_opt(plaintext: str | None) -> str | None:
    return encrypt(plaintext) if plaintext else None


def decrypt_opt(ciphertext: str | None) -> str | None:
    return decrypt(ciphertext) if ciphertext else None
