"""FastAPI app factory. `uvicorn app.main:app`"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator, Callable

from fastapi import FastAPI

from app.adapters.base import Adapters
from app.adapters.factory import build_adapters
from app.api.routes import router
from app.config import Settings, get_settings
from app.db.store import InMemoryStore, Store
from app.pipeline.orchestrator import Pipeline
from app.sources import load_whitelist_file

log = logging.getLogger("fact")


def create_app(
    settings: Settings | None = None,
    adapters: Adapters | None = None,
    store: Store | None = None,
    clock: Callable[[], datetime] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level)
    whitelist = load_whitelist_file(settings.effective_sources_file)
    if not len(whitelist):
        log.warning(
            "Source whitelist %s has no usable entries; every claim will be UNVERIFIED.",
            settings.effective_sources_file,
        )

    engine = tables = None
    if settings.database_url:
        from app.db.pg_store import PgStore
        from app.db.tables import build_tables, make_engine

        engine, tables = make_engine(settings.database_url), build_tables(settings.embedding_dim)
        store = store or PgStore(engine, tables)
    adapters = adapters or build_adapters(settings, engine, tables)
    pipeline = Pipeline(settings, adapters, store or InMemoryStore(), whitelist, clock=clock)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if engine is not None and tables is not None:
            from app.db.index import sync_sources
            from app.db.seed import seed_mock_corpus
            from app.db.tables import init_db

            await init_db(engine, tables)
            await sync_sources(engine, tables, whitelist)
            if settings.mock_mode and settings.effective_search_backend == "postgres":
                await seed_mock_corpus(engine, tables, whitelist, adapters.embedder)
        yield
        if engine is not None:
            await engine.dispose()

    app = FastAPI(title="Fact", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.pipeline = pipeline
    app.include_router(router)
    return app


app = create_app()
