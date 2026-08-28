import os
from typing import Set

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER_NAME = "X-API-Key"

# Why an env var instead of hardcoding keys?
# Keeps secrets out of source control while staying trivial to configure for
# local development. `auto_error=False` lets us raise our own 401 with a
# clear message instead of FastAPI/Starlette's generic "not authenticated".
_api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


def _load_valid_api_keys() -> Set[str]:
    """
    Reads the comma-separated ROUTER_API_KEYS env var on every call (rather
    than caching it once at import time) so tests can flip valid keys via
    monkeypatch without needing to reload this module.
    """
    raw = os.environ.get("ROUTER_API_KEYS", "")
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    return keys or {"dev-local-key"}  # sane default so local runs work out of the box


def verify_api_key(api_key: str = Security(_api_key_header)) -> str:
    """
    FastAPI dependency that authenticates a request via the X-API-Key header.
    Returns the validated key so downstream dependencies (e.g. the rate
    limiter) can use it as the per-client bucket identifier.
    """
    if not api_key or api_key not in _load_valid_api_keys():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key
