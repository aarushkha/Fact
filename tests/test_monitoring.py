from app.monitoring import summarize_activity


def test_monitoring_endpoint(client):
    for text in ["Mumbai airport is closed for a week.", "Mumbai airport is closed for a week.",
                 "A fire broke out at a chemical factory in Thane."]:
        assert client.post("/api/check", data={"text": text}).status_code == 200
    d = client.get("/api/monitoring?hours=24").json()
    assert d["checks"]["total"] == 3 and d["claims"]["total"] == 3
    assert d["claims"]["by_status"]["CONTRADICTED"] == 2 and d["claims"]["abstain_rate"] == 0.333
    assert d["stages"]["judge"]["runs"] == 2 and d["stages"]["judge"]["latency_ms_p95"] is not None
    assert d["nli"]["sentences_deleted"] >= 1  # the mock writer's unsupported sentence
    assert d["top_cascades"][0]["claim"] == "Mumbai airport is closed for a week." and d["top_cascades"][0]["submissions_24h"] == 2
    assert client.get("/api/monitoring?hours=0").status_code == 422
    assert client.get("/monitor").status_code == 200


def test_monitoring_requires_key(settings):
    from fastapi.testclient import TestClient
    from app.main import create_app

    c = TestClient(create_app(settings.model_copy(update={"api_keys": "k"})))
    assert c.get("/api/monitoring").status_code == 401
    assert c.get("/api/monitoring", headers={"X-API-Key": "k"}).status_code == 200


def test_summary_counts_fallbacks_and_errors():
    data = {
        "stage_runs": [
            {"stage": "judge", "latency_ms": 10, "model_version": "jev (fallback used: llm)", "error": None, "outputs": None},
            {"stage": "judge", "latency_ms": 30, "model_version": "jev", "error": "boom", "outputs": None},
        ],
        "checks": [{"id": "a", "status": "error"}],
        "claims": [],
    }
    s = summarize_activity(data)
    assert s["fallback_runs"] == 1 and s["stages"]["judge"]["errors"] == 1 and s["claims"]["abstain_rate"] is None
