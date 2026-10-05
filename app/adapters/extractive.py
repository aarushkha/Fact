"""Gemini-free claim extraction and summary writing (CLAIM_EXTRACTOR=sentences, SUMMARY_WRITER=extractive).

- SentenceExtractor: one claim per sentence; English via the translator (Sarvam) when sentence counts
  differ. Cruder than LLM extraction (no splitting of compound sentences, heuristic entities).
- ExtractiveWriter: quotes, verbatim, the passage sentence most similar to the claim (multilingual
  embeddings), at most one per passage, up to 3. Cannot hallucinate; NLI still verifies every sentence.
"""

from __future__ import annotations

import asyncio

from app.adapters.base import LLM, Embedder, Translator
from app.db.store import cosine
from app.models.schemas import Claim, DraftSentence, Passage, RawClaim
from app.text import guess_entities, sentences

MAX_CLAIMS = 8


class SentenceExtractor:
    def __init__(self, translator: Translator):
        self.translator = translator

    async def extract_claims(self, text_original: str, text_en: str, languages: list[str]) -> list[RawClaim]:
        orig = sentences(text_original)[:MAX_CLAIMS] or [text_original]
        en = sentences(text_en)
        if len(en) != len(orig):
            primary = languages[0] if languages else "en"
            en = list(await asyncio.gather(*(self.translator.translate(s, primary) for s in orig))) if primary != "en" else orig
        return [
            RawClaim(text_original=o, text_en=e, entities=guess_entities(e))
            for o, e in zip(orig, en) if e.strip()
        ]


class ExtractiveWriter:
    def __init__(self, embedder: Embedder, max_sentences: int = 3):
        self.embedder = embedder
        self.max_sentences = max_sentences

    async def write_summary(self, claim: Claim, passages: list[Passage], status: str) -> list[DraftSentence]:
        candidates = [(p, s) for p in passages[: self.max_sentences] for s in sentences(p.text) if len(s.split()) >= 4]
        if not candidates:
            return []
        vecs = await self.embedder.embed([claim.text_en] + [s for _, s in candidates])
        best: dict[str, tuple[float, str]] = {}
        for (p, s), v in zip(candidates, vecs[1:]):
            score = cosine(vecs[0], v)
            if p.id not in best or score > best[p.id][0]:
                best[p.id] = (score, s)
        out, seen = [], set()
        for p in passages[: self.max_sentences]:
            if p.id in best and best[p.id][1] not in seen:
                seen.add(best[p.id][1])
                out.append(DraftSentence(sentence=best[p.id][1], passage_ids=[p.id]))
        return out


class HybridLLM:
    """Routes the LLM roles: extraction and writing to cheap implementations, translation to the LLM."""

    def __init__(self, llm: LLM, extractor=None, writer=None):
        self.llm = llm
        self.extractor = extractor or llm
        self.writer = writer or llm

    @property
    def model_version(self) -> str:
        parts = []
        parts.append("extract=sentences" if isinstance(self.extractor, SentenceExtractor) else f"extract={self.llm.model_version}")
        parts.append("write=extractive" if isinstance(self.writer, ExtractiveWriter) else f"write={self.llm.model_version}")
        return ";".join(parts)

    async def extract_claims(self, text_original: str, text_en: str, languages: list[str]) -> list[RawClaim]:
        return await self.extractor.extract_claims(text_original, text_en, languages)

    async def write_summary(self, claim: Claim, passages: list[Passage], status: str) -> list[DraftSentence]:
        return await self.writer.write_summary(claim, passages, status)

    async def translate(self, text: str, source: str, target: str = "en") -> str:
        return await self.llm.translate(text, source, target)
