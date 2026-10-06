"""The live configuration of the API process: build the pipeline from settings, and swap it when they change.

`Runtime.apply` builds the new adapters / pipeline first and touches nothing if that fails, so a bad value
from the settings page never takes the running service down. Loaded models are reused when the model
settings did not change (loading BGE-M3 takes minutes).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from fastapi import FastAPI

from app.adapters.base import Adapters
from app.adapters.factory import build_adapters
from app.api.security import PgRateLimiter, SlidingWindowLimiter, parse_keys
from app.config import Settings
from app.db.store import Store
from app.pipeline.orchestrator import Pipeline
from app.runtime_settings import (
    OverrideStore,
    SettingsInvalid,
    check_readiness,
    clean_overrides,
    merge_settings,
)
from app.sources import Whitelist, load_whitelist_file

log = logging.getLogger("fact.runtime")


def _embedder_key(s: Settings) -> tuple:
    return (s.mock_mode, s.model_server_url, s.model_server_token, s.embedder_model, s.embedder_revision,
            s.embedding_dim, s.embedder_max_seq_length)


def _nli_key(s: Settings) -> tuple:
    return (s.mock_mode, s.model_server_url, s.model_server_token, s.nli_model, s.nli_revision)


class ModelReuse:
    """Remembers the embedder / NLI built for a settings snapshot, to hand them to the next build."""

    def __init__(self) -> None:
        self._embedder: tuple[tuple, Any] | None = None
        self._nli: tuple[tuple, Any] | None = None

    def build(self, s: Settings, engine=None, tables=None) -> Adapters:
        embedder = self._embedder[1] if self._embedder and self._embedder[0] == _embedder_key(s) else None
        nli = self._nli[1] if self._nli and self._nli[0] == _nli_key(s) and not s.mock_mode else None
        adapters = build_adapters(s, engine, tables, embedder=embedder, nli=nli)
        self._embedder = (_embedder_key(s), adapters.embedder)
        if not s.mock_mode:
            self._nli = (_nli_key(s), adapters.nli)
        return adapters


@dataclass
class Built:
    settings: Settings
    adapters: Adapters
    pipeline: Pipeline
    whitelist: Whitelist


def build(
    s: Settings, store: Store, models: ModelReuse, engine=None, tables=None,
    clock: Callable[[], datetime] | None = None, adapters: Adapters | None = None,
) -> Built:
    whitelist = load_whitelist_file(s.effective_sources_file)
    if not len(whitelist):
        log.warning("Source whitelist %s has no usable entries; every claim will be UNVERIFIED.", s.effective_sources_file)
    adapters = adapters or models.build(s, engine, tables)
    return Built(s, adapters, Pipeline(s, adapters, store, whitelist, clock=clock), whitelist)


async def prepare_backend(s: Settings, built: Built, engine, tables) -> None:
    """Database side of a (re)configuration: sources table, and the demo corpus in mock mode."""
    if engine is None or tables is None:
        return
    from app.db.index import sync_sources
    from app.db.seed import seed_mock_corpus

    await sync_sources(engine, tables, built.whitelist)
    if s.mock_mode and s.effective_search_backend == "postgres":
        await seed_mock_corpus(engine, tables, built.whitelist, built.adapters.embedder)


def make_limiter(s: Settings, engine, tables):
    if engine is not None and tables is not None:
        return PgRateLimiter(engine, tables.rate_limit_hits, s.rate_limit_per_minute)
    return SlidingWindowLimiter(s.rate_limit_per_minute)


class Runtime:
    def __init__(
        self, app: FastAPI, base: Settings, overrides_store: OverrideStore, store: Store, models: ModelReuse,
        engine=None, tables=None, clock: Callable[[], datetime] | None = None,
    ):
        self.app, self.base, self.overrides_store = app, base, overrides_store
        self.store, self.models, self.engine, self.tables, self.clock = store, models, engine, tables, clock
        self.overrides: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    @property
    def current(self) -> Settings:
        return self.app.state.settings

    def install(self, built: Built) -> None:
        """Swap the live pieces in. Synchronous, so a request sees either the old set or the new one."""
        s, app = built.settings, self.app
        app.state.settings = s
        app.state.pipeline = built.pipeline
        app.state.api_keys = parse_keys(s.api_keys)
        app.state.admin_token = s.admin_token
        app.state.rate_limiter = make_limiter(s, self.engine, self.tables)
        logging.getLogger().setLevel(s.log_level)

    async def load(self) -> None:
        """Startup: apply the overrides saved earlier. A bad saved value is logged, and the environment is used."""
        try:
            saved = await self.overrides_store.load()
            if saved:
                await self._apply(saved)
        except Exception:
            log.exception("could not apply the saved settings; running with the environment settings")

    async def update(self, values: dict[str, Any], clear: list[str]) -> None:
        """Apply changes from the settings page. Raises SettingsInvalid; nothing changes in that case."""
        if self.base.admin_token and "admin_token" in values:
            raise SettingsInvalid({"admin_token": "set in the environment (ADMIN_TOKEN); change it there"})
        async with self._lock:
            new = dict(self.overrides)
            for name in clear:
                new.pop(name, None)
            new.update(clean_overrides(values))
            await self._apply(new, persist=True)

    async def _apply(self, overrides: dict[str, Any], *, persist: bool = False) -> None:
        s = merge_settings(self.base, overrides)
        problems = check_readiness(s).errors
        if problems:
            raise SettingsInvalid({"": " ".join(problems)})
        try:
            built = build(s, self.store, self.models, self.engine, self.tables, self.clock)
            await prepare_backend(s, built, self.engine, self.tables)
        except SettingsInvalid:
            raise
        except Exception as exc:
            log.exception("settings rejected")
            raise SettingsInvalid({"": f"{type(exc).__name__}: {exc}"}) from exc
        # A failed save must leave the running pipeline and authentication unchanged.
        if persist:
            await self.overrides_store.save(overrides)
        self.overrides = dict(overrides)
        self.install(built)
