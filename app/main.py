"""FastAPI app factory. `uvicorn app.main:app`"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator, Callable

from fastapi import FastAPI

from app.adapters.base import Adapters
from app.api.routes import router
from app.api.settings_routes import router as settings_router
from app.config import Settings, get_settings
from app.db.store import InMemoryStore, Store
from app.runtime import ModelReuse, Runtime, build, prepare_backend
from app.runtime_settings import DbOverrides, FileOverrides, MemoryOverrides, OverrideStore

log = logging.getLogger("fact")


def create_app(
    settings: Settings | None = None,
    adapters: Adapters | None = None,
    store: Store | None = None,
    clock: Callable[[], datetime] | None = None,
    overrides: OverrideStore | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level)

    engine = tables = None
    if settings.database_url:
        from app.db.pg_store import PgStore
        from app.db.tables import build_tables, make_engine

        engine, tables = make_engine(settings.database_url), build_tables(settings.embedding_dim)
        store = store or PgStore(engine, tables)
    store = store or InMemoryStore()
    models = ModelReuse()
    built = build(settings, store, models, engine, tables, clock, adapters)
    if overrides is None:
        if engine is not None and tables is not None:
            overrides = DbOverrides(engine, tables)
        elif settings.settings_overrides_file:
            overrides = FileOverrides(settings.settings_overrides_file)
        else:
            overrides = MemoryOverrides()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if engine is not None and tables is not None:
            from app.db.tables import init_db

            await init_db(engine, tables)
            await prepare_backend(settings, built, engine, tables)
        await runtime.load()  # settings saved from the web page win over the environment
        yield
        if engine is not None:
            await engine.dispose()

    app = FastAPI(title="Fact", version="0.1.0", lifespan=lifespan)
    runtime = Runtime(app, settings, overrides, store, models, engine, tables, clock)
    runtime.install(built)
    app.state.runtime = runtime
    app.state.overrides_kind = overrides.kind
    if not app.state.api_keys:
        log.warning("API_KEYS is empty: the API is open to anyone who can reach it (fine for local testing only).")
    app.include_router(router)
    app.include_router(settings_router)
    return app


app = create_app()
