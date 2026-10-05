"""Stage 5: retrieve evidence in the original language AND English, whitelist it, rank it."""

from __future__ import annotations

from datetime import datetime, timezone

from app.adapters.base import Search
from app.models.schemas import Claim, Passage
from app.sources import Whitelist
from app.text import document_key

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def apply_whitelist(passages: list[Passage], whitelist: Whitelist) -> list[Passage]:
    """Drop passages from non-whitelisted sources. The whitelist is the source of truth for tier/publisher."""
    out = []
    for p in passages:
        entry = whitelist.lookup(p.url)
        if entry is None:
            continue
        out.append(p.model_copy(update={"publisher": entry.name, "tier": entry.tier, "kind": entry.kind}))
    return out


def rank_passages(passages: list[Passage]) -> list[Passage]:
    """Source tier first (1 = primary), then relevance, then recency."""
    return sorted(passages, key=lambda p: (p.tier, -p.relevance, -(p.published_at or _EPOCH).timestamp()))


def dedupe(passages: list[Passage]) -> list[Passage]:
    best: dict[str, Passage] = {}
    for p in passages:
        if p.id not in best or p.relevance > best[p.id].relevance:
            best[p.id] = p
    return list(best.values())


async def retrieve(
    claim: Claim,
    languages: list[str],
    embedding: list[float] | None,
    search: Search,
    whitelist: Whitelist,
    k: int,
    web_search: Search | None = None,
    exclude: set[str] | frozenset[str] = frozenset(),
) -> list[Passage]:
    """Top-k whitelisted passages. `exclude` (document_key of URLs, eval-only) is applied before the
    top-k cut, so an excluded review never takes the slot of an eligible passage."""
    queries: list[tuple[str, str | None]] = [(claim.text_en, "en")]
    if claim.text_original.strip() and claim.text_original.strip() != claim.text_en.strip():
        queries.insert(0, (claim.text_original, languages[0] if languages else None))
    found = await search.search(queries, embedding, k=k * 2)
    passages = [p for p in apply_whitelist(found, whitelist) if document_key(p.url) not in exclude]
    if not passages and web_search is not None:
        passages = [p for p in apply_whitelist(await web_search.search(queries, embedding, k=k * 2), whitelist)
                    if document_key(p.url) not in exclude]
    return rank_passages(dedupe(passages))[:k]
