"""Alembic environment. Uses the connection handed over by app/db/migrate.py when present,
otherwise connects with DATABASE_URL (CLI use)."""

from __future__ import annotations

import asyncio

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings
from app.db.tables import build_tables

config = context.config
dim = int(config.attributes.get("embedding_dim") or get_settings().embedding_dim)
config.attributes["embedding_dim"] = dim  # migrations read it from here (they must not import app code)
target_metadata = build_tables(dim).metadata


def _run(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:
    context.configure(url=get_settings().database_url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


async def _run_async() -> None:
    engine = create_async_engine(get_settings().database_url)
    async with engine.begin() as conn:
        await conn.run_sync(_run)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
elif config.attributes.get("connection") is not None:
    _run(config.attributes["connection"])
else:
    asyncio.run(_run_async())
