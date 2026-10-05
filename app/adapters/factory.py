"""Builds the adapter bundle from settings. MOCK_MODE=true forces every model adapter to its mock."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine

from app.adapters import mock
from app.adapters.base import Adapters, Embedder, Search
from app.config import Settings
from app.db.tables import Tables
from app.sources import load_whitelist


def build_embedder(settings: Settings) -> Embedder:
    if settings.mock_mode:
        return mock.MockEmbedder(settings.embedding_dim)
    # Real (self-hosted BGE-M3) embedder lands in build step 3.
    raise NotImplementedError("Real embedder is not implemented yet; set MOCK_MODE=true.")


def build_search(
    settings: Settings, embedder: Embedder, engine: AsyncEngine | None, tables: Tables | None
) -> Search:
    backend = settings.effective_search_backend
    if backend == "postgres":
        if engine is None or tables is None:
            raise ValueError("SEARCH_BACKEND=postgres needs DATABASE_URL.")
        from app.db.pg_search import PgSearch

        return PgSearch(
            engine, tables, embedder,
            rrf_k=settings.search_rrf_k, min_vector_similarity=settings.search_min_vector_similarity,
        )
    if backend == "mock":
        if not settings.mock_mode:
            raise ValueError("The mock search corpus is only available with MOCK_MODE=true.")
        return mock.MockSearch(load_whitelist(str(settings.effective_sources_file)))
    raise ValueError(f"Unknown SEARCH_BACKEND={backend!r}")


def build_adapters(settings: Settings, engine: AsyncEngine | None = None, tables: Tables | None = None) -> Adapters:
    if settings.mock_mode:
        embedder = build_embedder(settings)
        return Adapters(
            vision=mock.MockVisionReader(),
            translator=mock.MockTranslator(),
            llm=mock.MockLLM(),
            classifier=mock.MockClassifier(),
            embedder=embedder,
            nli=mock.MockNLIVerifier(),
            factcheck=mock.MockFactCheckSearch(),
            search=build_search(settings, embedder, engine, tables),
        )
    # Real adapters are added in build step 3, one at a time, each written from the provider's docs.
    raise NotImplementedError("Real adapters are not implemented yet; set MOCK_MODE=true.")
