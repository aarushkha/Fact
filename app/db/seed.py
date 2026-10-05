"""Load the MOCK corpus into Postgres so MOCK_MODE can exercise real hybrid search."""

from __future__ import annotations

import logging

import yaml
from sqlalchemy.ext.asyncio import AsyncEngine

from app.adapters.base import Embedder
from app.adapters.mock import DATA_DIR
from app.db.index import ChunkIn, passage_count, sync_sources, upsert_document
from app.db.tables import Tables
from app.sources import Whitelist, host_of

log = logging.getLogger("fact.db")


async def seed_mock_corpus(engine: AsyncEngine, t: Tables, whitelist: Whitelist, embedder: Embedder) -> int:
    """Idempotent. Non-whitelisted mock rows are skipped, exactly as the crawler would skip them."""
    if await passage_count(engine, t, embedder.model_version):
        return 0
    source_ids = await sync_sources(engine, t, whitelist)
    rows = yaml.safe_load((DATA_DIR / "corpus.yaml").read_text(encoding="utf-8"))["passages"]
    seeded = 0
    for row in rows:
        entry = whitelist.lookup(row["url"])
        if entry is None:
            continue
        (vec,) = await embedder.embed([row["text"]])
        await upsert_document(
            engine, t,
            source_id=source_ids[host_of(entry.domain)], entry=entry, url=row["url"], title=None,
            language=row.get("language"), published_at=row.get("published_at"), full_text=row["text"],
            chunks=[ChunkIn(id=row["id"], text=row["text"], embedding=vec)],
            embedding_model=embedder.model_version,
        )
        seeded += 1
    log.info("seeded %d mock passages", seeded)
    return seeded
