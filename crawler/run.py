"""Crawl whitelisted sources into the evidence index.

    python -m crawler.run                       # every source in sources.yaml with an rss_url/sitemap_url
    python -m crawler.run --source example.org  # one source
    python -m crawler.run --dry-run             # discover only, print candidate URLs
    python -m crawler.run --reindex             # also re-extract/re-embed articles already indexed

Polls each source's RSS/Atom feed and/or sitemap, keeps only URLs on that source's domain, respects
robots.txt, extracts article text with trafilatura, chunks it, embeds it and stores it with url,
publisher, tier and published_at.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from sqlalchemy.ext.asyncio import AsyncEngine

from app.adapters.base import Embedder
from app.adapters.factory import build_embedder
from app.config import Settings, get_settings
from app.db.index import ChunkIn, known_urls, passage_id, sync_sources, upsert_document
from app.db.tables import Tables, build_tables, init_db, make_engine
from app.sources import SourceEntry, Whitelist, host_of, load_whitelist_file
from crawler.chunk import chunk_text
from crawler.discover import Candidate, parse_feed, parse_sitemap
from crawler.extract import claim_review_text, extract_article, extract_claim_review

log = logging.getLogger("fact.crawler")
MAX_CHILD_SITEMAPS = 5


@dataclass
class CrawlStats:
    discovered: int = 0
    skipped_known: int = 0
    skipped_robots: int = 0
    skipped_offsite: int = 0
    failed: int = 0
    stored: int = 0
    passages: int = 0
    errors: list[str] = field(default_factory=list)


class Robots:
    """Per-host robots.txt cache. An unreachable robots.txt is treated as allow-all (standard practice)."""

    def __init__(self, client: httpx.AsyncClient, user_agent: str):
        self.client = client
        self.user_agent = user_agent
        self._cache: dict[str, RobotFileParser] = {}

    async def allowed(self, url: str) -> bool:
        parts = urlparse(url)
        base = f"{parts.scheme}://{parts.netloc}"
        if base not in self._cache:
            rp = RobotFileParser()
            try:
                r = await self.client.get(urljoin(base, "/robots.txt"))
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except httpx.HTTPError:
                rp.parse([])
            self._cache[base] = rp
        return self._cache[base].can_fetch(self.user_agent, url)


async def discover(client: httpx.AsyncClient, entry: SourceEntry) -> list[Candidate]:
    """Candidates from every feed and sitemap of a source. One failing URL is logged and skipped;
    only a source whose discovery URLs all fail raises."""
    found: list[Candidate] = []
    tried, failures = 0, []

    async def fetch(url: str) -> bytes | None:
        nonlocal tried
        tried += 1
        try:
            r = await client.get(url)
            r.raise_for_status()
            return r.content
        except httpx.HTTPError as exc:
            failures.append(f"{url}: {type(exc).__name__}: {exc}")
            log.warning("discovery failed %s: %s", url, exc)
            return None

    for feed in entry.feed_specs:
        content = await fetch(feed.url)
        if content is not None:
            for c in parse_feed(content):
                c.language = feed.language
                found.append(c)
    if entry.sitemap_url:
        queue, seen = [entry.sitemap_url], 0
        while queue and seen <= MAX_CHILD_SITEMAPS:
            url = queue.pop(0)
            seen += 1
            content = await fetch(url)
            if content is None:
                continue
            items, children = parse_sitemap(content)
            found += items
            queue += children
    if tried and len(failures) == tried:
        raise RuntimeError("all discovery URLs failed: " + "; ".join(failures))
    # Dedupe; prefer the entry that carries a date.
    by_url: dict[str, Candidate] = {}
    for c in found:
        if c.url not in by_url or (by_url[c.url].published_at is None and c.published_at):
            by_url[c.url] = c
    return list(by_url.values())


def _newest_first(c: Candidate) -> float:
    return -(c.published_at.timestamp() if c.published_at else 0)


async def crawl_source(
    entry: SourceEntry,
    source_id: int,
    *,
    client: httpx.AsyncClient,
    robots: Robots,
    whitelist: Whitelist,
    engine: AsyncEngine,
    tables: Tables,
    embedder: Embedder,
    settings: Settings,
    limit: int,
    since: datetime | None,
    stats: CrawlStats,
    reindex: bool = False,
) -> None:
    candidates = await discover(client, entry)
    stats.discovered += len(candidates)
    on_site = [c for c in candidates if whitelist.lookup(c.url) is entry]
    stats.skipped_offsite += len(candidates) - len(on_site)
    if since:
        on_site = [c for c in on_site if c.published_at is None or c.published_at >= since]
    on_site.sort(key=_newest_first)
    on_site = on_site[:limit]
    # reindex: re-extract and re-embed articles already stored (after a change to chunking, embedding or
    # ClaimReview handling); upsert_document replaces their passages.
    known = set() if reindex else await known_urls(engine, tables, [c.url for c in on_site], source_id)
    stats.skipped_known += len(known)
    todo = [c for c in on_site if c.url not in known]
    sem = asyncio.Semaphore(settings.crawler_concurrency)

    async def one(c: Candidate) -> None:
        async with sem:
            try:
                if not await robots.allowed(c.url):
                    stats.skipped_robots += 1
                    return
                r = await client.get(c.url)
                r.raise_for_status()
                final_url = str(r.url)
                if whitelist.lookup(final_url) is not entry:
                    stats.skipped_offsite += 1  # redirected off the whitelisted domain
                    return
                article = extract_article(r.text, final_url)
                if article is None:
                    stats.failed += 1
                    return
                chunks = chunk_text(article.text, settings.chunk_max_words)
                review = extract_claim_review(r.text)
                if review:  # a fact-check's structured verdict becomes its own, searchable passage
                    chunks.append(claim_review_text(review))
                title = article.title or c.title
                # Embed with the title so generic chunks ("Hence the claim is false") keep their subject.
                vectors = await embedder.embed([f"{title}\n{ch}" if title else ch for ch in chunks])
                await upsert_document(
                    engine, tables,
                    source_id=source_id, entry=entry, url=final_url, title=article.title or c.title,
                    language=c.language or entry.language, published_at=c.published_at or article.published_at,
                    full_text=article.text,
                    chunks=[ChunkIn(passage_id(final_url, i), t, v) for i, (t, v) in enumerate(zip(chunks, vectors))],
                    embedding_model=embedder.model_version, claim_review=review,
                )
                stats.stored += 1
                stats.passages += len(chunks)
            except Exception as exc:  # one bad article must not stop the crawl
                stats.failed += 1
                stats.errors.append(f"{c.url}: {type(exc).__name__}: {exc}")
                log.warning("failed %s: %s", c.url, exc)

    await asyncio.gather(*(one(c) for c in todo))


async def crawl(
    settings: Settings,
    *,
    engine: AsyncEngine,
    tables: Tables,
    whitelist: Whitelist,
    embedder: Embedder,
    client: httpx.AsyncClient,
    only_domain: str | None = None,
    limit: int | None = None,
    since_days: float | None = None,
    reindex: bool = False,
) -> CrawlStats:
    await init_db(engine, tables)
    source_ids = await sync_sources(engine, tables, whitelist)
    robots = Robots(client, settings.crawler_user_agent)
    since = datetime.now(timezone.utc) - timedelta(days=since_days) if since_days else None
    stats = CrawlStats()
    for entry in whitelist.entries:
        if only_domain and host_of(entry.domain) != host_of(only_domain):
            continue
        if not (entry.feeds or entry.sitemap_url):
            log.info("skip %s: no rss_url or sitemap_url", entry.name)
            continue
        try:
            await crawl_source(
                entry, source_ids[host_of(entry.domain)], client=client, robots=robots, whitelist=whitelist,
                engine=engine, tables=tables, embedder=embedder, settings=settings,
                limit=limit or settings.crawler_max_articles_per_source, since=since, stats=stats,
                reindex=reindex,
            )
        except Exception as exc:
            stats.errors.append(f"{entry.name}: {type(exc).__name__}: {exc}")
            log.warning("source %s failed: %s", entry.name, exc)
    return stats


async def _main(args: argparse.Namespace) -> int:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    whitelist = load_whitelist_file(settings.effective_sources_file)
    if not len(whitelist):
        print(f"No usable sources in {settings.effective_sources_file} (fill in the TODO entries).")
        return 1
    headers = {"User-Agent": settings.crawler_user_agent}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=20) as client:
        if args.dry_run:
            for entry in whitelist.entries:
                if args.source and host_of(entry.domain) != host_of(args.source):
                    continue
                if entry.feeds or entry.sitemap_url:
                    for c in (await discover(client, entry))[: args.limit or 20]:
                        print(f"{entry.name}\t{c.published_at}\t{c.url}")
            return 0
        if not settings.database_url:
            print("DATABASE_URL is not set.")
            return 1
        engine = make_engine(settings.database_url)
        try:
            stats = await crawl(
                settings, engine=engine, tables=build_tables(settings.embedding_dim), whitelist=whitelist,
                embedder=build_embedder(settings), client=client, only_domain=args.source,
                limit=args.limit, since_days=args.since_days, reindex=args.reindex,
            )
        finally:
            await engine.dispose()
    print(
        f"discovered={stats.discovered} stored={stats.stored} passages={stats.passages} "
        f"known={stats.skipped_known} robots={stats.skipped_robots} offsite={stats.skipped_offsite} "
        f"failed={stats.failed}"
    )
    for e in stats.errors[:20]:
        print("  error:", e)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="only crawl this domain")
    ap.add_argument("--limit", type=int, help="max new articles per source")
    ap.add_argument("--since-days", type=float, help="skip articles older than this")
    ap.add_argument("--dry-run", action="store_true", help="discover and print URLs; store nothing")
    ap.add_argument("--reindex", action="store_true",
                    help="re-extract and re-embed articles already in the index (after an indexing change)")
    raise SystemExit(asyncio.run(_main(ap.parse_args())))


if __name__ == "__main__":
    main()
