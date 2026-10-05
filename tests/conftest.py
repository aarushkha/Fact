from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.adapters.factory import build_adapters
from app.config import Settings
from app.db.store import InMemoryStore
from app.main import create_app
from app.pipeline.orchestrator import Pipeline
from app.sources import load_whitelist_file

NOW = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, mock_mode=True)


@pytest.fixture
def whitelist(settings):
    return load_whitelist_file(settings.effective_sources_file)


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def adapters(settings):
    return build_adapters(settings)


@pytest.fixture
def pipeline(settings, adapters, store, whitelist) -> Pipeline:
    return Pipeline(settings, adapters, store, whitelist, clock=lambda: NOW)


@pytest.fixture
def client(settings, store) -> TestClient:
    return TestClient(create_app(settings, store=store, clock=lambda: NOW))
