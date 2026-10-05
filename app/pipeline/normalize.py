"""Stage 2: detect language, keep the original text and an English translation."""

from __future__ import annotations

import logging

from app.adapters.base import LLM, Translator
from app.models.schemas import Normalized

log = logging.getLogger("fact.pipeline")


async def normalize(text: str, translator: Translator, llm: LLM) -> tuple[Normalized, str]:
    """Returns (normalized, translator_used) where translator_used is "none", "translator" or "llm"."""
    languages = await translator.detect(text) or ["en"]
    primary = languages[0]
    if primary == "en":
        return Normalized(text_original=text, text_en=text, languages=languages), "none"
    try:
        text_en = await translator.translate(text, source=primary, target="en")
        used = "translator"
    except Exception as exc:
        log.warning("translator failed (%s); falling back to LLM translation", exc)
        text_en = await llm.translate(text, source=primary, target="en")
        used = "llm"
    return Normalized(text_original=text, text_en=text_en, languages=languages), used
