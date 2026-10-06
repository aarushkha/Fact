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


from app.config import get_settings
from app.db.pg_store import PgStore
from app.db.tables import build_tables, init_db, make_engine
from app.pipeline.recheck import run_rechecks
from app.runtime import ModelReuse, build
from app.runtime_settings import DbOverrides, check_readiness, merge_settings

log = logging.getLogger("fact.worker")


async def main_async(once: bool, crawl: bool) -> None:
    from crawler.run import crawl as run_crawl
    from crawler.run import crawl_client

    base = get_settings()
    logging.basicConfig(level=base.log_level)
    if not base.database_url:
        raise SystemExit("The worker needs DATABASE_URL.")
    engine, tables = make_engine(base.database_url), build_tables(base.embedding_dim)
    await init_db(engine, tables)
    store, models, overrides_store = PgStore(engine, tables), ModelReuse(), DbOverrides(engine, tables)
    s = base
    built = build(s, store, models, engine, tables)
    adapters, whitelist, pipeline = built.adapters, built.whitelist, built.pipeline
    seen: dict = {}  # the saved web-page settings already looked at
    next_crawl = next_recheck = 0.0
    try:
        while True:
            # Settings saved on the web page apply without a restart. A bad set is logged and the previous
            # settings stay; it is remembered so it is not retried every 30 s.
            try:
                saved = await overrides_store.load()
                if saved != seen:
                    seen = saved
                    new = merge_settings(base, saved)
                    if problems := check_readiness(new).errors:
                        raise ValueError(" ".join(problems))
                    built = build(new, store, models, engine, tables)
                    s, adapters, whitelist, pipeline = new, built.adapters, built.whitelist, built.pipeline
                    logging.getLogger().setLevel(s.log_level)
                    log.info("settings reloaded (mock_mode=%s)", s.mock_mode)
            except Exception:
                log.exception("could not apply the saved settings; keeping the previous ones")
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
