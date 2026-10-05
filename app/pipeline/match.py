"""Stage 4: past-verdict cache and Google Fact Check match. A hit short-circuits the claim."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime

from app.adapters.base import Embedder, FactCheckSearch
from app.db.store import CachedVerdict, Store, cosine
from app.models.schemas import Claim, FactCheckHit, Passage, Status
from app.sources import Whitelist
from app.text import source_id_for, verdict_sentence

# Conservative exact-match table (after lowercasing and stripping punctuation).
# Ratings not listed here never short-circuit; whitelisted reviews then go to the judge as evidence.
RATING_MAP: dict[str, Status] = {
    "false": Status.CONTRADICTED,
    "fake": Status.CONTRADICTED,
    "incorrect": Status.CONTRADICTED,
    "pants on fire": Status.CONTRADICTED,
    "fabricated": Status.CONTRADICTED,
    "गलत": Status.CONTRADICTED,
    "फर्जी": Status.CONTRADICTED,
    "खोटे": Status.CONTRADICTED,
    "चूक": Status.CONTRADICTED,
    "true": Status.CONFIRMED,
    "correct": Status.CONFIRMED,
    "accurate": Status.CONFIRMED,
    "misleading": Status.MISLEADING_CONTEXT,
    "missing context": Status.MISLEADING_CONTEXT,
    "out of context": Status.MISLEADING_CONTEXT,
    "misattributed": Status.MISLEADING_CONTEXT,
    "भ्रामक": Status.MISLEADING_CONTEXT,
    "दिशाभूल करणारे": Status.MISLEADING_CONTEXT,
    "दिशाभूल": Status.MISLEADING_CONTEXT,
}


def map_rating(textual_rating: str) -> Status | None:
    key = re.sub(r"[^\w\sऀ-ॿ]", "", textual_rating.lower()).strip()
    key = re.sub(r"\s+", " ", key)
    return RATING_MAP.get(key)


@dataclass
class FactCheckMatch:
    hit: FactCheckHit
    passage: Passage  # the review, as a citable passage
    status: Status | None  # None -> usable as evidence but does not short-circuit
    similarity: float


def review_passage(hit: FactCheckHit, whitelist: Whitelist) -> Passage | None:
    entry = whitelist.lookup(hit.review_url) or whitelist.lookup(hit.publisher_site)
    if entry is None:
        return None
    title = (hit.review_title or "").strip().rstrip(".")
    text = (f"{title}. " if title else "") + verdict_sentence(hit.textual_rating, hit.claim_text)
    return Passage(
        id="fc_" + source_id_for(hit.review_url)[4:],
        source_id=source_id_for(hit.review_url),
        url=hit.review_url,
        publisher=entry.name,
        tier=entry.tier,
        kind=entry.kind,
        language=hit.language,
        published_at=hit.review_date,
        text=text,
        relevance=1.0,
        rating=hit.textual_rating,
    )


async def match_factchecks(
    claim: Claim,
    claim_embedding: list[float],
    languages: list[str],
    factcheck: FactCheckSearch,
    embedder: Embedder,
    whitelist: Whitelist,
    similarity_threshold: float,
) -> list[FactCheckMatch]:
    """Query the fact-check API in English and the original language; keep whitelisted, similar reviews."""
    queries = [(claim.text_en, "en")]
    if claim.text_original.strip() != claim.text_en.strip() and languages:
        queries.append((claim.text_original, languages[0].split("-")[0]))
    results = await asyncio.gather(*(factcheck.search(q, lang) for q, lang in queries))
    hits = {h.review_url: h for batch in results for h in batch}
    if not hits:
        return []
    hit_list = list(hits.values())
    vectors = await embedder.embed([h.claim_text for h in hit_list])
    matches = []
    for hit, vec in zip(hit_list, vectors):
        sim = cosine(claim_embedding, vec)
        if sim < similarity_threshold:
            continue
        passage = review_passage(hit, whitelist)
        if passage is None:
            continue  # rule 4: only whitelisted fact-checkers count
        matches.append(FactCheckMatch(hit=hit, passage=passage, status=map_rating(hit.textual_rating), similarity=sim))
    matches.sort(key=lambda m: (m.passage.tier, -m.similarity))
    return matches


def decisive_factcheck(matches: list[FactCheckMatch]) -> FactCheckMatch | None:
    """A fact-check short-circuits only when all mapped whitelisted reviews agree."""
    mapped = [m for m in matches if m.status is not None]
    if not mapped or len({m.status for m in mapped}) > 1:
        return None
    return mapped[0]


async def lookup_cache(
    claim: Claim, embedding: list[float], embedding_model: str, store: Store, threshold: float, now: datetime
) -> CachedVerdict | None:
    return await store.find_cached(embedding, embedding_model, claim.entities, threshold, now)
