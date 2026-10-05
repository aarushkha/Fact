"""Opt-in live checks against the real APIs and models. Skipped unless enabled:

    RUN_LIVE_TESTS=1 pytest tests/test_live.py      # uses keys from .env / environment
    RUN_MODEL_TESTS=1 pytest tests/test_live.py     # loads BGE-M3 + mDeBERTa (~3 GB download)
"""

import os

import pytest

from app.config import Settings
from app.models.schemas import ClaimType, RawClaim

live = pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1")
models = pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1")


@pytest.fixture
def s() -> Settings:
    return Settings(mock_mode=False)


@live
async def test_sarvam_live(s):
    from app.adapters.sarvam import SarvamTranslator

    t = SarvamTranslator(s.sarvam_api_key, s.sarvam_translate_model)
    assert await t.detect("Mumbai airport ek hafte ke liye band hai.") == ["hi-Latn"]
    assert "airport" in (await t.translate("मुंबई एयरपोर्ट एक हफ्ते के लिए बंद है।", "hi")).lower()


@live
async def test_factcheck_live(s):
    from app.adapters.google_factcheck import GoogleFactCheckSearch

    hits = await GoogleFactCheckSearch(s.google_factcheck_api_key).search("vaccine", "en")
    assert hits and all(h.review_url and h.textual_rating for h in hits)


@live
async def test_jev_live(s):
    from app.adapters.jev import JevClassifier

    jev = JevClassifier(s.openrouter_api_key, s.jev_model_version)
    ctype, p = await jev.claim_type(RawClaim(text_original="x", text_en="Pune is the best city in the world."))
    assert ctype == ClaimType.OPINION and p > 0.5


@live
async def test_gemini_live(s):
    from app.adapters.gemini import GeminiClient, GeminiLLM

    llm = GeminiLLM(GeminiClient(s.gemini_api_key, s.llm_model, s.llm_fallback_models.split(",")))
    claims = await llm.extract_claims("Mumbai airport band hai aur 3 log mare.", "Mumbai airport is closed and 3 people died.", ["hi-Latn"])
    assert len(claims) >= 2


@models
async def test_local_models(s):
    from app.adapters.local_models import BGEM3Embedder, MDebertaNLI
    from app.db.store import cosine

    e = BGEM3Embedder(s.embedder_model, s.embedder_revision, dim=s.embedding_dim)
    a, b, c = await e.embed(["Mumbai airport is closed.", "मुंबई एयरपोर्ट बंद है।", "Cricket scores today."])
    assert cosine(a, b) > cosine(a, c)
    n = MDebertaNLI(s.nli_model, s.nli_revision)
    yes, no = await n.score([("The bridge collapsed on Sunday.", "A bridge collapsed."),
                             ("The bridge collapsed on Sunday.", "Dozens died in a flood.")])
    assert yes.entailment > 0.8 and no.entailment < 0.2
