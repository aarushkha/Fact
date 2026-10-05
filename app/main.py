"""FastAPI app factory. `uvicorn app.main:app`"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable

from fastapi import FastAPI

from app.adapters.base import Adapters
from app.adapters.factory import build_adapters
from app.api.routes import router
from app.config import Settings, get_settings
from app.db.store import InMemoryStore, Store
from app.pipeline.orchestrator import Pipeline
from app.sources import load_whitelist_file


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
        logging.getLogger("fact").warning(
            "Source whitelist %s has no usable entries; every claim will be UNVERIFIED.",
            settings.effective_sources_file,
        )
    pipeline = Pipeline(
        settings,
        adapters or build_adapters(settings),
        store or InMemoryStore(),
        whitelist,
        clock=clock,
    )
    app = FastAPI(title="Fact", version="0.1.0")
    app.state.settings = settings
    app.state.pipeline = pipeline
    app.include_router(router)
    return app


app = create_app()
