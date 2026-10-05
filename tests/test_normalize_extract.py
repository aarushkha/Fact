import pytest

from app.adapters.mock import MockClassifier, MockLLM, MockTranslator
from app.models.schemas import ClaimType, Normalized
from app.pipeline.extract import extract_claims, resolve_type
from app.pipeline.normalize import normalize


@pytest.mark.parametrize(
    "text,lang",
    [
        ("नाशिकमध्ये गोदावरी नदीवरील पादचारी पूल रविवारी कोसळला.", "mr"),
        ("मुंबई एयरपोर्ट एक हफ्ते के लिए बंद है।", "hi"),
        ("Mumbai airport ek hafte ke liye band hai.", "hi-Latn"),
        ("Mumbai airport is closed for a week.", "en"),
    ],
)
async def test_language_detection_and_translation(text, lang):
    norm, used = await normalize(text, MockTranslator(), MockLLM())
    assert norm.languages[0] == lang
    assert norm.text_original == text
    if lang == "en":
        assert norm.text_en == text and used == "none"
    else:
        assert norm.text_en != text and used == "translator"


async def test_translator_failure_falls_back_to_llm():
    class Broken(MockTranslator):
        async def translate(self, text, source, target="en"):
            raise RuntimeError("down")

    norm, used = await normalize("Mumbai airport ek hafte ke liye band hai.", Broken(), MockLLM())
    assert used == "llm"
    assert norm.text_en == "Mumbai airport is closed for a week."


async def test_extract_splits_atomic_claims_and_tags_type():
    text = "Mumbai airport is closed for a week. Nashik is the most beautiful city in India."
    norm = Normalized(text_original=text, text_en=text, languages=["en"])
    claims = await extract_claims(norm, MockLLM(), MockClassifier(), 0.6)
    assert [c.id for c in claims] == ["c1", "c2"]
    assert claims[0].type == ClaimType.CHECKABLE
    assert claims[1].type == ClaimType.OPINION
    assert {e.text for e in claims[0].entities} >= {"Mumbai", "airport"}


def test_low_confidence_non_checkable_falls_back_to_checkable():
    assert resolve_type(ClaimType.OPINION, 0.55, 0.6) == (ClaimType.CHECKABLE, pytest.approx(0.45))
    assert resolve_type(ClaimType.OPINION, 0.8, 0.6) == (ClaimType.OPINION, 0.8)
    assert resolve_type(ClaimType.CHECKABLE, 0.3, 0.6) == (ClaimType.CHECKABLE, 0.3)
