"""API-key auth and per-client rate limiting for the expensive endpoints.

API_KEYS empty -> auth disabled (local development; a warning is logged at startup).
The rate limiter is in-memory and per process: with several workers each has its own budget.
"""

from __future__ import annotations

import hmac
import math
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0, clock=time.monotonic):
        self.limit = limit
        self.window = window_seconds
        self.clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, identity: str) -> float | None:
        """Record a hit; return None if allowed, else seconds until the next slot frees up."""
        if self.limit <= 0:
            return None
        now = self.clock()
        hits = self._hits[identity]
        while hits and hits[0] <= now - self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return max(0.0, hits[0] + self.window - now)
        hits.append(now)
        return None


def parse_keys(raw: str) -> list[str]:
    return [k.strip() for k in raw.split(",") if k.strip()]


def authenticate(request: Request) -> str:
    """Returns the caller identity (key fingerprint, or client IP when auth is disabled)."""
    keys: list[str] = request.app.state.api_keys
    if not keys:
        return f"ip:{request.client.host if request.client else 'unknown'}"
    given = request.headers.get("x-api-key", "")
    for k in keys:
        if given and hmac.compare_digest(given.encode(), k.encode()):
            return f"key:{k[:4]}…{len(k)}"
    raise HTTPException(401, "Missing or invalid X-API-Key header.", headers={"WWW-Authenticate": "API-Key"})


def guard_check(request: Request) -> str:
    """Dependency for the check endpoints: auth, then rate limit."""
    identity = authenticate(request)
    wait = request.app.state.rate_limiter.check(identity)
    if wait is not None:
        raise HTTPException(429, "Rate limit exceeded; try again later.", headers={"Retry-After": str(math.ceil(wait))})
    return identity
