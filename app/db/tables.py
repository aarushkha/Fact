"""Postgres schema (SQLAlchemy Core). Everything lives in Postgres + pgvector.

Schema changes go through Alembic (app/db/migrations); `init_db` upgrades to head at app startup and
in the crawler. After editing these tables: alembic revision --autogenerate -m "..."
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Column,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    Table,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

# Keyword half of hybrid search: 'simple' keeps every language's tokens (Marathi/Hindi/Hinglish),
# 'english' adds stemming for English text.
TSV_EXPR = "to_tsvector('simple', text) || to_tsvector('english', text)"


@dataclass(frozen=True)
class Tables:
    metadata: MetaData
    sources: Table
    documents: Table
    passages: Table
    checks: Table
    stage_runs: Table
    claims: Table
    verdicts: Table
    dim: int


@lru_cache
def build_tables(dim: int) -> Tables:
    md = MetaData()
    tz = DateTime(timezone=True)

    sources = Table(
        "sources", md,
        Column("id", Integer, primary_key=True),
        Column("domain", Text, nullable=False, unique=True),
        Column("name", Text, nullable=False),
        Column("tier", SmallInteger, nullable=False),
        Column("kind", Text),
        Column("language", Text),
        Column("rss_url", Text),
        Column("sitemap_url", Text),
        Column("updated_at", tz, server_default=func.now(), nullable=False),
    )
    documents = Table(
        "documents", md,
        Column("id", BigInteger, primary_key=True),
        Column("source_id", Integer, ForeignKey("sources.id", ondelete="CASCADE"), nullable=False),
        Column("url", Text, nullable=False, unique=True),
        Column("title", Text),
        Column("language", Text),
        Column("published_at", tz),
        Column("fetched_at", tz, server_default=func.now(), nullable=False),
        Column("content_hash", Text),
        Column("claim_review", JSONB(none_as_null=True)),  # schema.org ClaimReview found on the page (fact-check articles)
    )
    passages = Table(
        "passages", md,
        Column("id", Text, primary_key=True),
        Column("document_id", BigInteger, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        Column("chunk_index", Integer, nullable=False),
        Column("text", Text, nullable=False),
        Column("language", Text),
        Column("embedding", Vector(dim)),
        Column("embedding_model", Text, nullable=False),
        Column("tsv", TSVECTOR, Computed(TSV_EXPR, persisted=True)),
        Index("ix_passages_tsv", "tsv", postgresql_using="gin"),
        Index(
            "ix_passages_embedding", "embedding",
            postgresql_using="hnsw", postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_passages_embedding_model", "embedding_model"),
    )
    checks = Table(
        "checks", md,
        Column("id", Text, primary_key=True),
        Column("input_type", Text, nullable=False),
        Column("meta", JSONB),
        Column("status", Text, nullable=False, server_default="running"),
        Column("created_at", tz, server_default=func.now(), nullable=False),
        Column("finished_at", tz),
        Column("response", JSONB),
    )
    stage_runs = Table(
        "stage_runs", md,
        Column("id", BigInteger, primary_key=True),
        Column("check_id", Text, ForeignKey("checks.id", ondelete="CASCADE"), nullable=False, index=True),
        Column("claim_id", Text),
        Column("stage", Text, nullable=False),
        Column("inputs", JSONB),
        Column("outputs", JSONB),
        Column("latency_ms", Float, nullable=False),
        Column("model_version", Text),
        Column("error", Text),
        Column("started_at", tz, nullable=False),
    )
    claims = Table(
        "claims", md,
        Column("id", BigInteger, primary_key=True),
        Column("check_id", Text, ForeignKey("checks.id", ondelete="CASCADE"), nullable=False, index=True),
        Column("claim_key", Text, nullable=False),
        Column("text_original", Text, nullable=False),
        Column("text_en", Text, nullable=False),
        Column("type", Text, nullable=False),
        Column("entities", JSONB),
        Column("entity_keys", ARRAY(Text), nullable=False),
        Column("embedding", Vector(dim)),
        Column("embedding_model", Text, nullable=False),
        Column("signals", JSONB(none_as_null=True)),  # rumour-cascade signals at check time
        Column("post_date", tz),  # resolved post date; rechecks reuse it
        Column("created_at", tz, server_default=func.now(), nullable=False),
        Index("ix_claims_entity_keys", "entity_keys", postgresql_using="gin"),
        Index(
            "ix_claims_embedding", "embedding",
            postgresql_using="hnsw", postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
    verdicts = Table(
        "verdicts", md,
        Column("id", BigInteger, primary_key=True),
        Column("claim_id", BigInteger, ForeignKey("claims.id", ondelete="CASCADE"), nullable=False, index=True),
        Column("status", Text, nullable=False),
        Column("confidence", Float, nullable=False),
        Column("result", JSONB, nullable=False),
        Column("sources", JSONB, nullable=False),
        Column("model_versions", JSONB, nullable=False),
        Column("recheck_at", tz),
        Column("rechecked_from", BigInteger, ForeignKey("verdicts.id", ondelete="SET NULL")),
        Column("superseded_at", tz),  # set when a recheck produced a newer verdict
        Column("created_at", tz, server_default=func.now(), nullable=False),
        Index("ix_verdicts_due", "recheck_at", postgresql_where=text("superseded_at IS NULL")),
    )
    return Tables(md, sources, documents, passages, checks, stage_runs, claims, verdicts, dim)


def make_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(database_url, pool_pre_ping=True)


async def init_db(engine: AsyncEngine, t: Tables) -> None:
    """Bring the database to the latest Alembic revision (creates everything on a fresh database)."""
    from app.db.migrate import run_migrations

    await run_migrations(engine, t.dim)
