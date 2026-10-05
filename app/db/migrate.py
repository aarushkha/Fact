"""Run Alembic migrations from async code (app startup, crawler, tests).

    python -m app.db.migrate            # upgrade to head using DATABASE_URL
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncEngine

MIGRATIONS = Path(__file__).parent / "migrations"
BASELINE = "0001"


def _config(connection, dim: int) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.attributes["connection"] = connection
    cfg.attributes["embedding_dim"] = dim
    return cfg


def _upgrade(connection, dim: int) -> None:
    cfg = _config(connection, dim)
    tables = set(inspect(connection).get_table_names())
    if "checks" in tables and "alembic_version" not in tables:
        command.stamp(cfg, BASELINE)  # database created by create_all before migrations existed
    command.upgrade(cfg, "head")


async def run_migrations(engine: AsyncEngine, dim: int) -> None:
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(_upgrade, dim)


if __name__ == "__main__":
    import asyncio

    from app.config import get_settings
    from app.db.tables import make_engine

    async def _main() -> None:
        s = get_settings()
        engine = make_engine(s.database_url)
        try:
            await run_migrations(engine, s.embedding_dim)
            print("database is at head")
        finally:
            await engine.dispose()

    asyncio.run(_main())
