"""Embedder / NLIVerifier adapters that call the model server (app/model_server.py)."""

from __future__ import annotations

import httpx

from app.adapters.http import request_json
from app.models.schemas import NLIScore

BATCH = 64  # the server's per-request limit


class _Remote:
    def __init__(self, base_url: str, token: str = "", timeout: float = 120, client: httpx.AsyncClient | None = None,
                 configured: str = ""):
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
