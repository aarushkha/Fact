"""Hybrid evidence search over Postgres: full-text (keyword) + pgvector (cosine), fused with RRF."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.adapters.base import Embedder
from app.db.tables import Tables
from app.models.schemas import Passage
from app.text import source_id_for, tokens


def keyword_query_text(text: str) -> str:
    """OR together the content words. websearch_to_tsquery never raises on user text."""
    return " or ".join(dict.fromkeys(tokens(text)))


def rrf_fuse(ranked_lists: list[list[str]], k: int) -> dict[str, float]:
    """Reciprocal rank fusion, normalised to 0-1 by the best possible score."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, pid in enumerate(ranked, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    if not ranked_lists:
        return {}
    best = len(ranked_lists) / (k + 1)
    return {pid: s / best for pid, s in scores.items()}


class PgSearch:
    def __init__(
        self,
        engine: AsyncEngine,
        tables: Tables,
        embedder: Embedder,
        rrf_k: int = 60,
        candidates: int = 30,
        min_vector_similarity: float = 0.4,
    ):
        self.engine = engine
        self.min_vector_similarity = min_vector_similarity
        self.t = tables
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.candidates = candidates
        self.model_version = f"pg-hybrid-rrf{rrf_k}+{embedder.model_version}"

    async def _query_vectors(
        self, queries: list[tuple[str, str | None]], embedding: list[float] | None
    ) -> list[list[float]]:
        """Reuse the claim's English embedding; embed the other-language queries."""
        missing = [q for q, lang in queries if not (lang == "en" and embedding is not None)]
        embedded = iter(await self.embedder.embed(missing)) if missing else iter(())
        return [embedding if (lang == "en" and embedding is not None) else next(embedded) for _, lang in queries]

    async def search(
        self,
        queries: list[tuple[str, str | None]],
        embedding: list[float] | None,
        k: int,
        published_after: datetime | None = None,
    ) -> list[Passage]:
        p, d, s = self.t.passages, self.t.documents, self.t.sources
        base_from = p.join(d, d.c.id == p.c.document_id)
        date_filter = [d.c.published_at >= published_after] if published_after else []
        vectors = await self._query_vectors(queries, embedding)
        ranked_lists: list[list[str]] = []

        async with self.engine.connect() as conn:
            for (text, _lang), vec in zip(queries, vectors):
                q = keyword_query_text(text)
                if q:
                    tsq = func.websearch_to_tsquery("simple", q).op("||")(func.websearch_to_tsquery("english", q))
                    rank = func.ts_rank_cd(p.c.tsv, tsq).label("rank")
                    rows = await conn.execute(
                        select(p.c.id, rank).select_from(base_from)
                        .where(p.c.tsv.op("@@")(tsq), *date_filter)
                        .order_by(desc("rank")).limit(self.candidates)
                    )
                    ranked_lists.append([r.id for r in rows])
                if vec is not None:
                    dist = p.c.embedding.cosine_distance(vec)
                    rows = await conn.execute(
                        select(p.c.id).select_from(base_from)
                        .where(
                            p.c.embedding_model == self.embedder.model_version,
                            dist <= 1 - self.min_vector_similarity,  # nearest neighbours are not always relevant
                            *date_filter,
                        )
                        .order_by(dist).limit(self.candidates)
                    )
                    ranked_lists.append([r.id for r in rows])

            fused = rrf_fuse([lst for lst in ranked_lists if lst], self.rrf_k)
            top = sorted(fused, key=lambda pid: -fused[pid])[:k]
            if not top:
                return []
            rows = await conn.execute(
                select(
                    p.c.id, p.c.text, p.c.language, d.c.url, d.c.published_at, d.c.title,
                    s.c.name.label("publisher"), s.c.tier, s.c.kind,
                    d.c.claim_review["rating"].astext.label("rating"),
                )
                .select_from(base_from.join(s, s.c.id == d.c.source_id))
                .where(p.c.id.in_(top))
            )
            by_id = {r.id: r for r in rows}

        return [
            Passage(
                id=pid,
                source_id=source_id_for(by_id[pid].url),
                url=by_id[pid].url,
                publisher=by_id[pid].publisher,
                tier=by_id[pid].tier,
                kind=by_id[pid].kind,
                language=by_id[pid].language,
                published_at=by_id[pid].published_at,
                text=by_id[pid].text,
                title=by_id[pid].title,
                relevance=round(fused[pid], 4),
                rating=by_id[pid].rating,
            )
            for pid in top
            if pid in by_id
        ]
