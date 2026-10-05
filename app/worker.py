"""Background worker: periodic rechecks of UNVERIFIED claims and periodic crawling.

    python -m app.worker                 # loop forever (docker compose service "worker")
    python -m app.worker --once          # one recheck pass + one crawl pass, then exit (cron-friendly)
    python -m app.worker --once --no-crawl
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time


from app.adapters.factory import build_adapters
from app.config import get_settings
from app.db.pg_store import PgStore
from app.db.tables import build_tables, init_db, make_engine
from app.pipeline.orchestrator import Pipeline
from app.pipeline.recheck import run_rechecks
from app.sources import load_whitelist_file

log = logging.getLogger("fact.worker")


async def main_async(once: bool, crawl: bool) -> None:
    from crawler.run import crawl as run_crawl
    from crawler.run import crawl_client

    s = get_settings()
    logging.basicConfig(level=s.log_level)
    if not s.database_url:
        raise SystemExit("The worker needs DATABASE_URL.")
    engine, tables = make_engine(s.database_url), build_tables(s.embedding_dim)
    await init_db(engine, tables)
    adapters = build_adapters(s, engine, tables)
    whitelist = load_whitelist_file(s.effective_sources_file)
    pipeline = Pipeline(s, adapters, PgStore(engine, tables), whitelist)
    next_crawl = next_recheck = 0.0
    try:
        while True:
            now = time.monotonic()
            # A failed pass is logged and waits for its next slot: one bad pass must not stop the worker.
            # Crawl first, so a recheck that is due at the same time sees the newly indexed evidence.
            if crawl and not s.mock_mode and now >= next_crawl:
                next_crawl = now + s.crawl_interval_minutes * 60
                try:
                    async with crawl_client(s) as client:
                        stats = await run_crawl(s, engine=engine, tables=tables, whitelist=whitelist,
                                                embedder=adapters.embedder, client=client)
                    log.info("crawl pass: stored=%d passages=%d failed=%d", stats.stored, stats.passages, stats.failed)
                except Exception:
                    log.exception("crawl pass failed")
            if now >= next_recheck:
                next_recheck = now + s.recheck_interval_minutes * 60
                try:
                    results = await run_rechecks(pipeline, s.recheck_batch_size)
                    log.info("recheck pass: %d claims", len(results))
                except Exception:
                    log.exception("recheck pass failed")
            if once:
                return
            await asyncio.sleep(30)
    finally:
        await engine.dispose()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-crawl", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args.once, not args.no_crawl))


if __name__ == "__main__":
    main()
