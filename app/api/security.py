"""API-key auth and per-client rate limiting for the expensive endpoints.

API_KEYS empty -> auth disabled (local development; a warning is logged at startup).
With DATABASE_URL set the rate limiter lives in Postgres (PgRateLimiter), so all API workers share one budget
per client; without a database it is in-memory and per process (SlidingWindowLimiter).
"""

from __future__ import annotations

import hmac
import inspect
import math
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncEngine


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


class PgRateLimiter:
    """Sliding-window limit shared by every process using the same database.

    One row per accepted request in rate_limit_hits; a per-identity transaction-level advisory lock
    serialises concurrent checks for the same client, and the database clock is the only clock used.
    """

    def __init__(self, engine: AsyncEngine, table, limit: int, window_seconds: float = 60.0):
        self.engine = engine
        self.t = table
        self.limit = limit
        self.window = window_seconds

    async def check(self, identity: str) -> float | None:
        if self.limit <= 0:
            return None
        t = self.t
        window = text(f"interval '{float(self.window)} seconds'")
        async with self.engine.begin() as conn:
            await conn.execute(select(func.pg_advisory_xact_lock(func.hashtextextended(identity, 0))))
            expired = t.c.at <= func.now() - window
            await conn.execute(t.delete().where(t.c.identity == identity, expired))
            # Also expire other clients' old hits (rows another check holds are skipped: no lock waits or
            # deadlocks), so the table stays about one window of requests.
            stale = select(t.c.id).where(expired).limit(500).with_for_update(skip_locked=True)
            await conn.execute(t.delete().where(t.c.id.in_(stale.scalar_subquery())))
            row = (await conn.execute(
                select(func.count(), func.extract("epoch", func.min(t.c.at) + window - func.now()))
                .where(t.c.identity == identity)
            )).one()
            if row[0] >= self.limit:
                return max(0.0, float(row[1] or 0.0))
            await conn.execute(t.insert().values(identity=identity))
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


async def guard_check(request: Request) -> str:
    """Dependency for the check endpoints: auth, then rate limit."""
    identity = authenticate(request)
    wait = request.app.state.rate_limiter.check(identity)
    if inspect.isawaitable(wait):  # PgRateLimiter
        wait = await wait
    if wait is not None:
        raise HTTPException(429, "Rate limit exceeded; try again later.", headers={"Retry-After": str(math.ceil(wait))})
    return identity
