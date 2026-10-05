"""Embedder / NLIVerifier adapters that call the model server (app/model_server.py)."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

import httpx

from app.adapters.http import request_json
from app.models.schemas import NLIScore

BATCH = 64  # the server's per-request limit


def internal_host(host: str) -> bool:
    """Single-label names (Compose service "models", "localhost") and private/loopback IPs."""
    if "." not in host:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback


def check_transport(base_url: str, token: str) -> None:
    """Refuse to send MODEL_SERVER_TOKEN (and claim text) in cleartext to a public host. Plain HTTP stays
    allowed to internal hosts, which keeps the documented http://models:8001 Compose setup working."""
    parts = urlparse(base_url)
    if token and parts.scheme == "http" and not internal_host(parts.hostname or ""):
        raise ValueError(f"MODEL_SERVER_URL {base_url} is plain HTTP to a public host: use https:// "
                         "so MODEL_SERVER_TOKEN is not sent in cleartext")


class _Remote:
    def __init__(self, base_url: str, token: str = "", timeout: float = 120, client: httpx.AsyncClient | None = None,
                 configured: str = ""):
        check_transport(base_url, token)
        self.base_url = base_url.rstrip("/")
        self._headers = {"X-Model-Token": token} if token else {}
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self.model_version = f"remote:{configured}"  # replaced by the server-reported version on first call

    async def _post(self, path: str, body: dict) -> dict:
        data = await request_json(self._client, "POST", f"{self.base_url}{path}", provider="model-server",
                                  headers=self._headers, json=body)
        self.model_version = data.get("model_version") or self.model_version
        return data


class RemoteEmbedder(_Remote):
    def __init__(self, *args, dim: int = 1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), BATCH):
            out += (await self._post("/embed", {"texts": texts[i : i + BATCH]}))["vectors"]
        if out and len(out[0]) != self.dim:
            raise ValueError(f"model server returned {len(out[0])}-dim vectors but EMBEDDING_DIM={self.dim}")
        return out


class RemoteNLI(_Remote):
    async def score(self, pairs: list[tuple[str, str]]) -> list[NLIScore]:
        out: list[NLIScore] = []
        for i in range(0, len(pairs), BATCH):
            data = await self._post("/nli", {"pairs": [list(p) for p in pairs[i : i + BATCH]]})
            out += [NLIScore(**x) for x in data["scores"]]
        return out
