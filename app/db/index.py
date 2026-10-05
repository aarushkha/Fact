"""Writes to the evidence index: whitelist sync, documents and their passages."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.tables import Tables
from app.sources import SourceEntry, Whitelist, host_of


@dataclass
class ChunkIn:
    id: str
    text: str
    embedding: list[float]


def passage_id(url: str, chunk_index: int) -> str:
    return "psg_" + hashlib.sha1(f"{url}#{chunk_index}".encode("utf-8")).hexdigest()[:16]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def sync_sources(engine: AsyncEngine, t: Tables, whitelist: Whitelist) -> dict[str, int]:
    """Upsert whitelist entries into `sources`; returns {domain: source_id}."""
    async with engine.begin() as conn:
        for e in whitelist.entries:
            values = {
                "domain": host_of(e.domain), "name": e.name, "tier": e.tier, "kind": e.kind,
                "language": e.language, "rss_url": " ".join(e.feeds) or None, "sitemap_url": e.sitemap_url,
            }
            stmt = insert(t.sources).values(**values)
            await conn.execute(stmt.on_conflict_do_update(index_elements=["domain"], set_=values))
        rows = (await conn.execute(select(t.sources.c.domain, t.sources.c.id))).all()
    return {r.domain: r.id for r in rows}


async def known_urls(engine: AsyncEngine, t: Tables, urls: list[str]) -> set[str]:
    if not urls:
        return set()
    async with engine.connect() as conn:
        rows = await conn.execute(select(t.documents.c.url).where(t.documents.c.url.in_(urls)))
        return {r.url for r in rows}


async def upsert_document(
    engine: AsyncEngine,
    t: Tables,
    *,
    source_id: int,
    entry: SourceEntry,
    url: str,
    title: str | None,
    language: str | None,
    published_at: datetime | None,
    full_text: str,
    chunks: list[ChunkIn],
    embedding_model: str,
    claim_review: dict | None = None,
) -> int:
    """Insert or replace a document and all of its passages in one transaction."""
    async with engine.begin() as conn:
        values = {
            "source_id": source_id, "url": url, "title": title, "language": language or entry.language,
            "published_at": published_at, "content_hash": content_hash(full_text), "claim_review": claim_review,
        }
        stmt = insert(t.documents).values(**values)
        doc_id = (
            await conn.execute(
                stmt.on_conflict_do_update(index_elements=["url"], set_=values).returning(t.documents.c.id)
            )
        ).scalar_one()
        await conn.execute(delete(t.passages).where(t.passages.c.document_id == doc_id))
        if chunks:
            await conn.execute(
                insert(t.passages),
                [
                    {
                        "id": c.id, "document_id": doc_id, "chunk_index": i, "text": c.text,
                        "language": language or entry.language, "embedding": c.embedding,
                        "embedding_model": embedding_model,
                    }
                    for i, c in enumerate(chunks)
                ],
            )
    return doc_id


async def passage_count(engine: AsyncEngine, t: Tables, embedding_model: str | None = None) -> int:
    stmt = select(func.count()).select_from(t.passages)
    if embedding_model:
        stmt = stmt.where(t.passages.c.embedding_model == embedding_model)
    async with engine.connect() as conn:
        return (await conn.execute(stmt)).scalar_one()
