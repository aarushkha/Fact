"""Real adapters against recorded response shapes (httpx.MockTransport): no network, no keys."""

import json

import httpx
import pytest

from app.adapters.gemini import GeminiClient, GeminiLLM, GeminiVisionReader
from app.adapters.google_factcheck import GoogleFactCheckSearch
from app.adapters.http import ApiError
from app.adapters.jev import JevClassifier
from app.adapters.llm_judge import FallbackClassifier
from app.adapters.mock import MockClassifier
from app.adapters.sarvam import SarvamTranslator, from_sarvam, split_for_limit
from app.models.schemas import ClaimType, Stance, Status

from .helpers import claim, passage


def transport(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def gemini_ok(payload: dict, model: str = "gemini-3.8-flash") -> httpx.Response:
    return httpx.Response(200, json={
        "status": "completed", "model": model, "object": "interaction",
        "steps": [{"type": "thought", "signature": "x"},
                  {"type": "model_output", "content": [{"type": "text", "text": json.dumps(payload)}]}],
    })


# ----------------------------------------------------------------------------- Gemini

async def test_gemini_request_format_and_parse():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"], seen["key"], seen["body"] = str(req.url), req.headers["x-goog-api-key"], json.loads(req.content)
        return gemini_ok({"claims": [{"text_original": "a", "text_en": "A is closed.",
                                      "entities": [{"text": "A", "kind": "place"}, {"text": "x", "kind": "weird"}]}]})

    llm = GeminiLLM(GeminiClient("k", "gemini-3.8-flash", client=transport(handler)))
    out = await llm.extract_claims("a", "A is closed.", ["en"])
    assert seen["url"].endswith("/v1beta/interactions") and seen["key"] == "k"
    b = seen["body"]
    assert b["model"] == "gemini-3.8-flash" and b["store"] is False
    assert b["response_format"]["mime_type"] == "application/json" and "schema" in b["response_format"]
    assert out[0].text_en == "A is closed." and [e.kind for e in out[0].entities] == ["place", "other"]
    assert llm.model_version == "gemini/gemini-3.8-flash"


async def test_gemini_falls_back_on_overload_and_cools_down(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr("app.adapters.http.asyncio.sleep", no_sleep)
    calls = []

    def handler(req: httpx.Request):
        model = json.loads(req.content)["model"]
        calls.append(model)
        if model == "primary":
            return httpx.Response(503, json={"error": {"message": "high demand", "code": "service_unavailable"}})
        return gemini_ok({"translation": "hello"}, model=model)

    llm = GeminiLLM(GeminiClient("k", "primary", ["backup"], client=transport(handler)))
    assert await llm.translate("namaste", "hi") == "hello"
    assert llm.model_version == "gemini/backup"
    n = len(calls)
    await llm.translate("namaste", "hi")
    assert calls[n:] == ["backup"]  # primary is cooling down, not retried


async def test_gemini_non_transient_error_raises():
    def handler(req):
        return httpx.Response(400, json={"error": {"message": "bad schema"}})

    with pytest.raises(ApiError, match="bad schema"):
        await GeminiLLM(GeminiClient("k", "m", ["n"], client=transport(handler))).translate("x", "hi")


async def test_gemini_write_restricts_citations_to_given_ids():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return gemini_ok({"sentences": [{"sentence": "S says so [p1].", "passage_ids": ["p1"]},
                                        {"sentence": "T (p1, p2) too.", "passage_ids": ["p1", "p2"]}]})

    llm = GeminiLLM(GeminiClient("k", "m", client=transport(handler)))
    out = await llm.write_summary(claim(), [passage("p1"), passage("p2")], "CONFIRMED")
    enum = seen["body"]["response_format"]["schema"]["properties"]["sentences"]["items"]["properties"]["passage_ids"]["items"]["enum"]
    assert enum == ["p1", "p2"] and out[0].passage_ids == ["p1"]
    assert [d.sentence for d in out] == ["S says so.", "T too."]
    assert await llm.write_summary(claim(), [], "CONFIRMED") == []


async def test_gemini_vision_sends_inline_image():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return gemini_ok({"post_text": "Bridge fell", "account_handle": "@a", "post_date": "", "image_description": ""})

    out = await GeminiVisionReader(GeminiClient("k", "m", client=transport(handler))).read(b"\x89PNGdata", "image/png")
    img = seen["body"]["input"][1]
    assert img["type"] == "image" and img["mime_type"] == "image/png" and img["data"]
    assert out.post_text == "Bridge fell" and out.post_date_raw is None


def test_gemini_requires_key():
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiClient("", "m")


# ----------------------------------------------------------------------------- Sarvam

def test_sarvam_language_mapping():
    assert from_sarvam("hi-IN", "Latn") == "hi-Latn"
    assert from_sarvam("hi-IN", "Deva") == "hi"
    assert from_sarvam("mr-IN", "Deva") == "mr"
    assert from_sarvam("en-IN", "Latn") == "en"


def test_split_for_limit():
    text = ". ".join(["word " * 30] * 10)
    pieces = split_for_limit(text, 400)
    assert all(len(p) <= 400 for p in pieces) and len(pieces) > 1
    assert split_for_limit("x" * 50, 20) == ["x" * 20, "x" * 20, "x" * 10]


async def test_sarvam_detect_and_translate():
    seen = []

    def handler(req: httpx.Request):
        body = json.loads(req.content)
        seen.append((req.url.path, req.headers["api-subscription-key"], body))
        if req.url.path == "/text-lid":
            return httpx.Response(200, json={"request_id": "r", "language_code": "hi-IN", "script_code": "Latn"})
        return httpx.Response(200, json={"request_id": "r", "translated_text": "Airport closed.", "source_language_code": "hi-IN"})

    t = SarvamTranslator("sk", client=transport(handler))
    assert await t.detect("Mumbai airport band hai") == ["hi-Latn"]
    assert await t.translate("Mumbai airport band hai", "hi-Latn") == "Airport closed."
    path, key, body = seen[-1]
    assert path == "/translate" and key == "sk"
    assert body == {"input": "Mumbai airport band hai", "source_language_code": "hi-IN",
                    "target_language_code": "en-IN", "model": "mayura:v1"}
    assert await t.translate("same", "en", "en") == "same"


# ----------------------------------------------------------------------------- Google Fact Check

async def test_factcheck_parse_and_auth_header():
    seen = {}

    def handler(req: httpx.Request):
        seen["key"], seen["params"] = req.headers["x-goog-api-key"], dict(req.url.params)
        return httpx.Response(200, json={"claims": [{
            "text": "Claim", "claimant": "Users", "claimDate": "2026-10-02T00:00:00Z",
            "claimReview": [
                {"publisher": {"name": "AajTak", "site": "aajtak.in"}, "url": "https://aajtak.in/x",
                 "title": "T", "reviewDate": "2026-10-03T00:00:00Z", "textualRating": "Half true", "languageCode": "hi"},
                {"publisher": {"name": "NoUrl"}, "textualRating": "False"},
            ]}]})

    hits = await GoogleFactCheckSearch("gk", client=transport(handler)).search("मुंबई", "hi-Latn")
    assert seen["key"] == "gk" and "key" not in seen["params"]  # key in header, never in the URL
    assert seen["params"]["languageCode"] == "hi"
    assert len(hits) == 1 and hits[0].publisher_site == "aajtak.in" and hits[0].textual_rating == "Half true"


async def test_factcheck_empty_result():
    hits = await GoogleFactCheckSearch("gk", client=transport(lambda r: httpx.Response(200, json={}))).search("q")
    assert hits == []


# ----------------------------------------------------------------------------- Jev

def jev_answer(answers: dict) -> httpx.Response:
    return httpx.Response(200, json={"model": "typesafe/jev-1.13-20260917", "answers": answers,
                                     "usage": {"input_tokens": 1, "output_tokens": 1}, "provider": "TypeSafe"})


async def test_jev_request_format_and_stance():
    seen = {}

    def handler(req: httpx.Request):
        seen["auth"], seen["body"] = req.headers["authorization"], json.loads(req.content)
        return jev_answer({"stance": {"type": "choice", "choice": "contradicts", "confidence": 0.9,
                                      "probabilities": {"supports": 0.05, "contradicts": 0.9, "irrelevant": 0.05}}})

    jev = JevClassifier("or-key", client=transport(handler))
    j = await jev.judge_passage(claim(), passage("p1", text="Airport operating normally."))
    assert seen["auth"] == "Bearer or-key" and seen["body"]["model"] == "typesafe/jev-1.13"
    q = seen["body"]["questions"]["stance"]
    assert q["type"] == "choice" and set(q["criteria"]) == {"supports", "contradicts", "irrelevant"}
    assert j.stance == Stance.CONTRADICTS and j.probability == pytest.approx(0.9)
    assert jev.model_version == "openrouter/typesafe/jev-1.13-20260917"


async def test_jev_verdict_maps_unverified_and_claim_type():
    answers = iter([
        {"verdict": {"type": "choice", "choice": "UNVERIFIED", "confidence": 0.7,
                     "probabilities": {"CONFIRMED": 0.2, "CONTRADICTED": 0.05, "MISLEADING_CONTEXT": 0.05, "UNVERIFIED": 0.7}}},
        {"type": {"type": "choice", "choice": "opinion", "confidence": 1, "probabilities": {"opinion": 1}}},
    ])
    jev = JevClassifier("k", client=transport(lambda r: jev_answer(next(answers))))
    v = await jev.judge_claim(claim(), [], [], [], 5.0)
    assert v.probabilities[Status.UNVERIFIED_EVIDENCE_MISSING] == pytest.approx(0.7)
    assert await jev.claim_type(claim()) == (ClaimType.OPINION, 1.0)


async def test_jev_expected_evidence_uses_catalog():
    def handler(req):
        qs = json.loads(req.content)["questions"]
        assert all(q["type"] == "noul" for q in qs.values())
        return jev_answer({k: {"type": "noul", "noul": 0.9 if k in ("expect_police", "expect_news") else 0.1} for k in qs})

    out = await JevClassifier("k", client=transport(handler)).expected_evidence(claim(), [], [])
    assert [e.item for e in out] == ["Statement or FIR from police / the responsible authority",
                                     "Report by a credible news outlet or wire service"]


async def test_jev_rejects_latest_alias_and_missing_answers():
    with pytest.raises(ValueError):
        JevClassifier("k", "~typesafe/jev-latest")
    jev = JevClassifier("k", client=transport(lambda r: jev_answer({})))
    with pytest.raises(ApiError, match="missing answers"):
        await jev.claim_type(claim())


async def test_fallback_classifier_switches_on_error():
    jev = JevClassifier("k", client=transport(lambda r: httpx.Response(401, json={"error": {"message": "bad key"}})))
    fb = FallbackClassifier(jev, MockClassifier())
    ctype, _ = await fb.claim_type(claim())
    assert ctype == ClaimType.CHECKABLE and "fallback used: mock-1" in fb.model_version
