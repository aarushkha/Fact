"""The settings page API: validation, hot reload, secrets, auth, persistence."""

import logging
import os

import pytest
from fastapi.testclient import TestClient

from app.adapters.mock import MockEmbedder
from app.config import Settings
from app.main import create_app
from app.runtime_settings import (
    FIELDS, GROUPS, DbOverrides, FileOverrides, SettingsInvalid, check_readiness, clean_overrides, merge_settings,
)

from .conftest import NOW

URL = os.environ.get("TEST_DATABASE_URL")
NO_KEYS = dict(gemini_api_key="", google_factcheck_api_key="", sarvam_api_key="", openrouter_api_key="",
               api_keys="", admin_token="", model_server_url="", model_server_token="")


def base(**kw) -> Settings:
    return Settings(_env_file=None, mock_mode=True, **{**NO_KEYS, **kw})


def make(settings=None, **kw) -> TestClient:
    return TestClient(create_app(settings or base(), clock=lambda: NOW, **kw))


def field(state, name):
    return next(f for g in state["groups"] for f in g["fields"] if f["name"] == name)


def test_every_field_exists_on_settings_and_is_never_infrastructure():
    for name in FIELDS:
        assert name in Settings.model_fields
    for forbidden in ("database_url", "sources_file", "gemini_base_url", "embedding_dim", "embedder_model",
                      "nli_model", "web_search_enabled", "search_backend", "settings_overrides_file"):
        assert forbidden not in FIELDS
    assert len({f.name for g in GROUPS for f in g.fields}) == sum(len(g.fields) for g in GROUPS)


def test_get_describes_fields_and_hides_secrets():
    c = make(base(gemini_api_key="SECRET-VALUE-123"))
    r = c.get("/api/settings")
    assert r.status_code == 200
    assert "SECRET-VALUE-123" not in r.text
    st = r.json()
    assert st["mock_mode"] is True and st["auth"] == "open" and st["persistence"] == "memory"
    key = field(st, "gemini_api_key")
    assert key["is_set"] is True and "value" not in key and key["source"] == "env"
    assert field(st, "sarvam_api_key")["is_set"] is False
    assert field(st, "confidence_threshold")["value"] == 0.6
    assert field(st, "confidence_threshold")["source"] == "default"
    assert c.get("/settings").status_code == 200


def test_change_threshold_applies_to_the_running_pipeline():
    c = make()
    p0 = c.app.state.pipeline
    r = c.put("/api/settings", json={"values": {"confidence_threshold": 0.8, "rate_limit_per_minute": 3}})
    assert r.status_code == 200, r.text
    f = field(r.json(), "confidence_threshold")
    assert f["value"] == 0.8 and f["source"] == "ui"
    assert c.app.state.pipeline is not p0
    assert c.app.state.pipeline.thresholds.confidence == 0.8
    assert c.app.state.rate_limiter.limit == 3
    # the models were not rebuilt for a threshold change
    assert c.app.state.pipeline.a.embedder is p0.a.embedder
    # and checks still work afterwards
    assert c.post("/api/check", data={"text": "Mumbai airport is closed for a week."}).status_code == 200


def test_clear_reverts_to_environment_value():
    c = make(base(confidence_threshold=0.7))
    c.put("/api/settings", json={"values": {"confidence_threshold": 0.9}})
    r = c.put("/api/settings", json={"clear": ["confidence_threshold"]})
    f = field(r.json(), "confidence_threshold")
    assert f["value"] == 0.7 and f["source"] == "env"


@pytest.mark.parametrize("values, name", [
    ({"confidence_threshold": 1.5}, "confidence_threshold"),
    ({"confidence_threshold": "high"}, "confidence_threshold"),
    ({"retrieval_top_k": 2.5}, "retrieval_top_k"),
    ({"mock_mode": "yes"}, "mock_mode"),
    ({"summary_language": "klingon"}, "summary_language"),
    ({"database_url": "postgresql://evil"}, "database_url"),
    ({"gemini_api_key": "a\nb"}, "gemini_api_key"),
])
def test_invalid_values_are_rejected_and_nothing_changes(values, name):
    c = make()
    p0 = c.app.state.pipeline
    r = c.put("/api/settings", json={"values": values})
    assert r.status_code == 422
    assert name in r.json()["detail"]["errors"]
    assert c.app.state.pipeline is p0


def test_real_mode_needs_keys_and_models():
    c = make()
    r = c.put("/api/settings", json={"values": {"mock_mode": False}})
    assert r.status_code == 422
    assert "Real mode needs" in r.json()["detail"]["errors"][""]
    assert c.get("/api/health").json()["mock_mode"] is True


def test_real_mode_needs_a_database():
    c = make()
    r = c.put("/api/settings", json={"values": {"mock_mode": False}})
    assert "DATABASE_URL" in r.json()["detail"]["errors"][""]


@pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")
async def test_switch_to_real_and_back_with_a_database():
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    from app.db.tables import build_tables

    engine = create_async_engine(URL, poolclass=NullPool)
    t = build_tables(1024)
    async with engine.begin() as conn:
        await conn.run_sync(t.metadata.drop_all)
        await conn.exec_driver_sql("DROP TABLE IF EXISTS alembic_version")
    await engine.dispose()

    keys = {"gemini_api_key": "g", "google_factcheck_api_key": "f", "sarvam_api_key": "s", "openrouter_api_key": "o",
            "model_server_url": "http://models:8001", "mock_mode": False}
    with make(base(database_url=URL)) as c:
        assert c.get("/api/settings").json()["persistence"] == "database"
        r = c.put("/api/settings", json={"values": keys})
        assert r.status_code == 200, r.text
        assert r.json()["mock_mode"] is False and r.json()["readiness"]["errors"] == []
        assert c.get("/api/health").json()["mock_mode"] is False
        assert all("value" not in f for g in r.json()["groups"] for f in g["fields"] if f["kind"] == "secret")
    # a new process (new app) starts in real mode: the saved settings come from the database
    with make(base(database_url=URL)) as c2:
        assert c2.get("/api/health").json()["mock_mode"] is False
        assert c2.put("/api/settings", json={"values": {"mock_mode": True}}).status_code == 200
        assert c2.get("/api/health").json()["mock_mode"] is True
        assert c2.post("/api/check", data={"text": "Mumbai airport is closed for a week."}).status_code == 200


def test_secret_is_write_only_and_applied():
    c = make()
    r = c.put("/api/settings", json={"values": {"api_keys": "k-one-1234, k-two-5678"}})
    assert r.status_code == 200
    assert "k-one-1234" not in r.text and field(r.json(), "api_keys")["is_set"] is True
    assert c.app.state.api_keys == ["k-one-1234", "k-two-5678"]
    data = {"text": "Mumbai airport is closed for a week."}
    assert c.post("/api/check", data=data).status_code == 401
    assert c.post("/api/check", data=data, headers={"X-API-Key": "k-two-5678"}).status_code == 200


def test_auth_modes():
    c = make(base(api_keys="k1"))
    assert c.get("/api/settings").status_code == 401
    assert c.get("/api/settings", headers={"X-API-Key": "k1"}).json()["auth"] == "api_key"
    # an admin token takes over: an API key no longer opens the settings
    c = make(base(api_keys="k1", admin_token="adm1n"))
    assert c.get("/api/settings", headers={"X-API-Key": "k1"}).status_code == 401
    assert c.get("/api/settings", headers={"X-Admin-Token": "nope"}).status_code == 401
    ok = c.get("/api/settings", headers={"X-Admin-Token": "adm1n"})
    assert ok.status_code == 200 and ok.json()["auth"] == "admin_token"
    assert field(ok.json(), "admin_token")["locked"] is True
    r = c.put("/api/settings", json={"values": {"admin_token": "other"}}, headers={"X-Admin-Token": "adm1n"})
    assert r.status_code == 422 and "admin_token" in r.json()["detail"]["errors"]
    assert c.put("/api/settings", json={"values": {"log_level": "DEBUG"}}).status_code == 401


def test_setting_an_admin_token_from_an_open_server_locks_it():
    c = make()
    saved = c.put("/api/settings", json={"values": {"admin_token": "first-token"}})
    assert saved.status_code == 200
    assert saved.json()["auth"] == "admin_token"
    assert c.get("/api/settings").status_code == 401
    assert c.get("/api/settings", headers={"X-Admin-Token": "first-token"}).json()["auth"] == "admin_token"


def test_overrides_persist_in_a_file_across_apps(tmp_path):
    path = tmp_path / "overrides.json"
    s = base(settings_overrides_file=str(path))
    with make(s) as c:
        assert c.get("/api/settings").json()["persistence"] == "file"
        assert c.put("/api/settings", json={"values": {"confidence_threshold": 0.75}}).status_code == 200
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    with make(s) as c2:  # lifespan loads the saved overrides
        assert c2.app.state.pipeline.thresholds.confidence == 0.75
        assert field(c2.get("/api/settings").json(), "confidence_threshold")["source"] == "ui"


def test_bad_saved_settings_do_not_stop_startup(tmp_path):
    path = tmp_path / "overrides.json"
    path.write_text('{"confidence_threshold": "oops", "mock_mode": false}')
    with make(base(settings_overrides_file=str(path))) as c:
        assert c.get("/api/health").json()["mock_mode"] is True


def test_models_survive_a_reload():
    c = make()
    emb = c.app.state.pipeline.a.embedder
    assert isinstance(emb, MockEmbedder)
    c.put("/api/settings", json={"values": {"retrieval_top_k": 5}})
    assert c.app.state.pipeline.a.embedder is emb


def test_clean_and_merge_units():
    assert clean_overrides({"retrieval_top_k": 4.0}) == {"retrieval_top_k": 4}
    with pytest.raises(SettingsInvalid):
        clean_overrides({"nope": 1})
    s = base(admin_token="env-token")
    assert merge_settings(s, {"admin_token": "ui"}).admin_token == "env-token"  # environment wins
    assert check_readiness(base()).errors == []


# ----------------------------------------------------------------------------------------------- Postgres

@pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")
async def test_db_overrides_roundtrip():
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    from app.db.tables import build_tables, init_db

    engine = create_async_engine(URL, poolclass=NullPool)
    t = build_tables(1024)
    try:
        await init_db(engine, t)
        store = DbOverrides(engine, t)
        await store.save({"confidence_threshold": 0.7, "mock_mode": False, "log_level": "DEBUG"})
        assert await store.load() == {"confidence_threshold": 0.7, "mock_mode": False, "log_level": "DEBUG"}
        await store.save({"confidence_threshold": 0.8})  # replaces: the other keys are gone
        assert await store.load() == {"confidence_threshold": 0.8}
        await store.save({})
        assert await store.load() == {}
    finally:
        await engine.dispose()


async def test_file_overrides_missing_file(tmp_path):
    assert await FileOverrides(tmp_path / "none.json").load() == {}


def test_failed_persistence_keeps_running_settings_and_auth(monkeypatch):
    c = make()
    rt = c.app.state.runtime
    pipeline = c.app.state.pipeline

    async def fail_save(values):
        raise OSError("disk is full")

    monkeypatch.setattr(rt.overrides_store, "save", fail_save)
    with pytest.raises(OSError, match="disk is full"):
        c.put("/api/settings", json={"values": {"confidence_threshold": 0.9, "admin_token": "unsaved-token"}})

    assert c.app.state.pipeline is pipeline
    assert rt.current.confidence_threshold == 0.6
    assert rt.overrides == {}
    assert c.get("/api/settings").json()["auth"] == "open"
    assert c.post("/api/check", data={"text": "Mumbai airport is closed for a week."}).status_code == 200


def test_invalid_effective_log_level_does_not_save_or_install(tmp_path):
    path = tmp_path / "overrides.json"
    with make(base(settings_overrides_file=str(path))) as c:
        rt = c.app.state.runtime
        assert c.put("/api/settings", json={"values": {"log_level": "INFO"}}).status_code == 200
        # Exercise reverting an override to an invalid base value, bypassing submitted-field validation.
        rt.base = rt.base.model_copy(update={"log_level": "INVALID"})
        old_settings = c.app.state.settings
        old_pipeline = c.app.state.pipeline
        old_limiter = c.app.state.rate_limiter
        old_overrides = dict(rt.overrides)
        old_saved = path.read_bytes()
        old_level = logging.getLogger().level
        response = c.put("/api/settings", json={
            "clear": ["log_level"],
            "values": {"confidence_threshold": 0.9, "api_keys": "new-key", "admin_token": "new-admin",
                       "rate_limit_per_minute": 2},
        })
        assert response.status_code == 422, response.text
        assert "log level" in response.json()["detail"]["errors"][""].lower()
        assert path.read_bytes() == old_saved
        assert rt.overrides == old_overrides
        assert c.app.state.settings is old_settings
        assert c.app.state.pipeline is old_pipeline
        assert c.app.state.rate_limiter is old_limiter
        assert c.app.state.api_keys == []
        assert c.app.state.admin_token == ""
        assert logging.getLogger().level == old_level
        assert c.post("/api/check", data={"text": "Mumbai airport is closed for a week."}).status_code == 200
        # A valid override can still mask the invalid base and be saved and installed.
        try:
            valid = c.put("/api/settings", json={"values": {"log_level": "WARNING"}})
            assert valid.status_code == 200
            assert field(valid.json(), "log_level")["value"] == "WARNING"
            assert logging.getLogger().level == logging.WARNING
        finally:
            logging.getLogger().setLevel(old_level)


def test_invalid_saved_log_level_does_not_partially_apply_on_startup(tmp_path):
    path = tmp_path / "overrides.json"
    path.write_text('{"log_level": "INVALID", "confidence_threshold": 0.9, "admin_token": "bad-saved-token"}')
    with make(base(settings_overrides_file=str(path))) as c:
        state = c.get("/api/settings")
        assert state.status_code == 200
        assert field(state.json(), "confidence_threshold")["value"] == 0.6
        assert c.app.state.runtime.overrides == {}
        assert c.app.state.admin_token == ""


@pytest.mark.parametrize("level", ["INFO", "WARNING", "WARN", "CRITICAL", "FATAL", "NOTSET"])
def test_readiness_accepts_valid_logging_levels(level):
    assert check_readiness(base(log_level=level)).errors == []
