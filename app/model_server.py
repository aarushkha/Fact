"""Self-hosted model server: BGE-M3 embeddings and mDeBERTa NLI behind HTTP.

    uvicorn app.model_server:app --port 8001          (compose: docker compose --profile models up)

Run it once and point the API and worker at it with MODEL_SERVER_URL, so app processes start fast and
stay small (no torch), and several workers share one copy of the weights.
Optional shared secret: MODEL_SERVER_TOKEN (sent as X-Model-Token).
"""

from __future__ import annotations

import hmac

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app.adapters.local_models import BGEM3Embedder, MDebertaNLI
from app.config import get_settings

MAX_ITEMS = 64
MAX_CHARS = 8000


class EmbedRequest(BaseModel):
    texts: list[str] = Field(max_length=MAX_ITEMS)


class NLIRequest(BaseModel):
    pairs: list[tuple[str, str]] = Field(max_length=MAX_ITEMS)  # (premise, hypothesis)


def create_model_app() -> FastAPI:
    s = get_settings()
    embedder = BGEM3Embedder(s.embedder_model, s.embedder_revision or None,
                             max_seq_length=s.embedder_max_seq_length, dim=s.embedding_dim)
    nli = MDebertaNLI(s.nli_model, s.nli_revision or None)
    app = FastAPI(title="Fact model server", version="0.1.0")

    def check_token(request: Request) -> None:
        expected = s.model_server_token
        given = request.headers.get("x-model-token", "")
        if expected and not hmac.compare_digest(given.encode(), expected.encode()):
            raise HTTPException(401, "Invalid X-Model-Token")

    def clip(texts: list[str]) -> list[str]:
        return [t[:MAX_CHARS] for t in texts]

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True, "embedder": embedder.model_version, "nli": nli.model_version, "dim": s.embedding_dim}

    @app.post("/embed", dependencies=[Depends(check_token)])
    async def embed(req: EmbedRequest) -> dict:
        return {"model_version": embedder.model_version, "vectors": await embedder.embed(clip(req.texts))}

    @app.post("/nli", dependencies=[Depends(check_token)])
    async def score(req: NLIRequest) -> dict:
        pairs = [(p[:MAX_CHARS], h[:MAX_CHARS]) for p, h in req.pairs]
        scores = await nli.score(pairs)
        return {"model_version": nli.model_version, "scores": [x.model_dump() for x in scores]}

    return app


app = create_model_app()
