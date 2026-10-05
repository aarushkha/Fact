"""Stage 3: split text into atomic claims and tag each claim's type."""

from __future__ import annotations

import asyncio

from app.adapters.base import LLM, Classifier
from app.models.schemas import Claim, ClaimType, Normalized, RawClaim


def resolve_type(predicted: ClaimType, probability: float, threshold: float) -> tuple[ClaimType, float]:
    """Only stop checking a claim when the classifier is confident it is not checkable.

    Mislabelling a checkable claim as opinion would silently skip it; the reverse only costs an
    UNVERIFIED verdict. So a low-confidence non-checkable label falls back to checkable.
    """
    if predicted != ClaimType.CHECKABLE and probability < threshold:
        return ClaimType.CHECKABLE, 1 - probability
    return predicted, probability


async def extract_claims(
    norm: Normalized, llm: LLM, classifier: Classifier, type_threshold: float
) -> list[Claim]:
    raw: list[RawClaim] = await llm.extract_claims(norm.text_original, norm.text_en, norm.languages)
    raw = [r for r in raw if r.text_en.strip()]
    typed = await asyncio.gather(*(classifier.claim_type(r) for r in raw))
    claims = []
    for i, (r, (ctype, prob)) in enumerate(zip(raw, typed), start=1):
        ctype, prob = resolve_type(ctype, prob, type_threshold)
        claims.append(Claim(id=f"c{i}", type=ctype, type_confidence=prob, **r.model_dump()))
    return claims
