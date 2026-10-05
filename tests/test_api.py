import json

from app.adapters.mock import make_mock_png
from app.models.schemas import CheckResponse, Status


def parse_sse(body: str) -> list[tuple[str, dict]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Fact test console" in r.text


def test_health(client):
    r = client.get("/api/health").json()
    assert r["mock_mode"] is True and r["model_versions"]["llm"] == "mock-1"


def test_check_json_contract(client):
    r = client.post("/api/check", data={"text": "Mumbai airport is closed for a week."})
    assert r.status_code == 200
    body = CheckResponse.model_validate(r.json())
    assert body.claims[0].status == Status.CONTRADICTED
    assert set(r.json()) == {"check_id", "input_type", "languages", "claims", "sources", "model_versions",
                             "checked_at", "recheck_at"}
    assert set(r.json()["claims"][0]) == {"text_original", "text_en", "type", "status", "confidence", "summary",
                                          "expected_evidence", "would_change_if"}
    assert client.get(f"/api/checks/{body.check_id}").json() == r.json()


def test_stream(client):
    r = client.post("/api/check/stream", data={"text": "Mumbai airport is closed for a week."})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert [n for n, _ in events] == ["claims_extracted", "evidence", "verdict", "done"]


def test_screenshot_upload_with_post_date(client):
    png = make_mock_png({"post_text": "A fire broke out at a chemical factory in Thane.", "post_date": "2h"})
    r = client.post("/api/check", files={"image": ("s.png", png, "image/png")},
                    data={"post_date": "2026-09-01T10:00:00Z"})
    assert r.status_code == 200
    body = r.json()
    assert body["input_type"] == "screenshot"
    # user date (34 days old) overrides the screenshot's "2h"
    assert body["claims"][0]["status"] == "UNVERIFIED_EVIDENCE_MISSING"


def test_validation_errors(client):
    assert client.post("/api/check", data={"text": "  "}).status_code == 422
    assert client.post("/api/check", data={"text": "x", "post_date": "soon"}).status_code == 422
    r = client.post("/api/check", files={"image": ("a.txt", b"hi", "text/plain")})
    assert r.status_code == 415
    assert client.get("/api/checks/nope").status_code == 404
