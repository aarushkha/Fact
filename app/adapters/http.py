"""Shared HTTP plumbing for API adapters: JSON POST/GET with bounded retries on transient errors."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

log = logging.getLogger("fact.adapters")
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class ApiError(RuntimeError):
    def __init__(self, provider: str, status: int, message: str):
        super().__init__(f"{provider} HTTP {status}: {message}")
        self.provider = provider
        self.status = status
        self.transient = status in RETRY_STATUSES or status == 0


def _error_message(r: httpx.Response) -> str:
    try:
        body = r.json()
    except ValueError:
        return r.text[:300]
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or err)[:300]
    return str(err or body)[:300]


async def request_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    provider: str,
    retries: int = 2,
    backoff: float = 1.5,
    **kwargs: Any,
) -> Any:
    """Send a request and return parsed JSON. Retries network errors, 429 and 5xx. Never logs headers."""
    for attempt in range(retries + 1):
        try:
            r = await client.request(method, url, **kwargs)
        except httpx.TransportError as exc:
            err = ApiError(provider, 0, f"{type(exc).__name__}: {exc}")
        else:
            if r.status_code < 400:
                return r.json()
            err = ApiError(provider, r.status_code, _error_message(r))
        if not err.transient or attempt == retries:
            raise err
        delay = backoff * (2**attempt)
        log.warning("%s; retrying in %.1fs", err, delay)
        await asyncio.sleep(delay)
    raise AssertionError("unreachable")
