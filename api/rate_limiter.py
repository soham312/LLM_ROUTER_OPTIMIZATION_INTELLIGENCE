import os
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import Depends, HTTPException, status

from api.auth import verify_api_key


class RateLimiter:
    """
    STAGE 10a: In-memory sliding-window rate limiter, keyed per API key.

    Why sliding window over a fixed window? A fixed window (e.g. reset on
    the clock every 60s) lets a client burst up to 2x the limit right at the
    window boundary. Looking back exactly `window_seconds` from "now" closes
    that gap, so the limit holds no matter when a burst happens.

    Why in-memory instead of Redis? This is a single-process deployment (no
    horizontal scaling), so a deque per key is simpler, has zero extra
    infrastructure, and is still correct within one process. A multi-worker
    deployment would need a shared store (Redis) instead - noted as the
    natural next step, not implemented here.
    """

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        """
        Records a request for `key` and raises 429 if it exceeds the limit.
        Uses time.monotonic() so the limiter is immune to system clock
        adjustments (NTP sync, DST, manual changes).
        """
        now = time.monotonic()
        with self._lock:
            timestamps = self._requests[key]

            # Drop timestamps that have aged out of the window.
            while timestamps and now - timestamps[0] > self.window_seconds:
                timestamps.popleft()

            if len(timestamps) >= self.max_requests:
                retry_after = self.window_seconds - (now - timestamps[0])
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": f"{max(retry_after, 0.0):.1f}"},
                )

            timestamps.append(now)


def _load_rate_limit_config() -> RateLimiter:
    max_requests = int(os.environ.get("ROUTER_RATE_LIMIT_MAX_REQUESTS", "60"))
    window_seconds = float(os.environ.get("ROUTER_RATE_LIMIT_WINDOW_SECONDS", "60"))
    return RateLimiter(max_requests=max_requests, window_seconds=window_seconds)


# Module-level singleton shared by every request in this process. Tests
# monkeypatch this attribute (not the RateLimiter class) to install a
# limiter with a tiny window without touching global env state.
rate_limiter = _load_rate_limit_config()


def enforce_rate_limit(api_key: str = Depends(verify_api_key)) -> str:
    """
    Composed dependency: authenticate first, then rate-limit by the
    authenticated key. Ordering matters - an invalid key is rejected with
    401 before it can consume any quota, so unauthenticated traffic can't be
    used to exhaust a legitimate client's rate limit bucket.
    """
    rate_limiter.check(api_key)
    return api_key
