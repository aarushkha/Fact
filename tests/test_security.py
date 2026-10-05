from fastapi.testclient import TestClient

from app.api.security import SlidingWindowLimiter
from app.main import create_app

from .conftest import NOW


def make_client(settings, **kw):
    return TestClient(create_app(settings.model_copy(update=kw), clock=lambda: NOW))


def test_sliding_window():
    t = [0.0]
    lim = SlidingWindowLimiter(2, 60, clock=lambda: t[0])
    assert lim.check("a") is None and lim.check("a") is None
    assert lim.check("a") == 60.0 and lim.check("b") is None
    t[0] = 60.5
    assert lim.check("a") is None
    assert SlidingWindowLimiter(0).check("x") is None  # disabled


def test_auth_required_when_keys_set(settings):
    c = make_client(settings, api_keys="k1, k2", rate_limit_per_minute=0)
    assert c.get("/api/health").json()["auth_required"] is True
    assert c.get("/").status_code == 200
    data = {"text": "Mumbai airport is closed for a week."}
    assert c.post("/api/check", data=data).status_code == 401
    assert c.post("/api/check", data=data, headers={"X-API-Key": "nope"}).status_code == 401
    r = c.post("/api/check", data=data, headers={"X-API-Key": "k2"})
    assert r.status_code == 200
    assert c.post("/api/check/stream", data=data).status_code == 401
    cid = r.json()["check_id"]
    assert c.get(f"/api/checks/{cid}").status_code == 401
    assert c.get(f"/api/checks/{cid}", headers={"X-API-Key": "k1"}).status_code == 200


def test_rate_limit_per_key(settings):
    c = make_client(settings, api_keys="a,b", rate_limit_per_minute=2)
    data = {"text": "Nashik is the most beautiful city in India."}
    for _ in range(2):
        assert c.post("/api/check", data=data, headers={"X-API-Key": "a"}).status_code == 200
    r = c.post("/api/check", data=data, headers={"X-API-Key": "a"})
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0
    assert c.post("/api/check", data=data, headers={"X-API-Key": "b"}).status_code == 200


def test_open_mode_limits_by_ip(settings):
    c = make_client(settings, api_keys="", rate_limit_per_minute=1)
    data = {"text": "Nashik is the most beautiful city in India."}
    assert c.post("/api/check", data=data).status_code == 200
    assert c.post("/api/check", data=data).status_code == 429
