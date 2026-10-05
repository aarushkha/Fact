"""Postgres + pgvector integration tests. Set TEST_DATABASE_URL to run them, e.g.
TEST_DATABASE_URL=postgresql+asyncpg://fact:fact@localhost:5432/fact_test pytest tests/test_pg.py
"""

import asyncio
import os
from datetime import timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy import text as sql
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.adapters.factory import build_adapters
from app.adapters.mock import MockEmbedder
from app.db.index import ChunkIn, known_urls, passage_count, sync_sources, upsert_document
from app.db.pg_search import PgSearch
from app.db.pg_store import PgStore
from app.db.seed import seed_mock_corpus
from app.db.store import StageRun
from app.db.tables import build_tables, init_db
from app.models.schemas import CheckInput, ClaimResult, ClaimType, Entity, Status
from app.pipeline.orchestrator import Pipeline

from .conftest import NOW
from .test_pipeline import CASES

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL not set")
DIM = 1024


@pytest.fixture
async def db():
    engine = create_async_engine(URL, poolclass=NullPool)
    t = build_tables(DIM)
    async with engine.begin() as conn:
        await conn.run_sync(t.metadata.drop_all)
        await conn.exec_driver_sql("DROP TABLE IF EXISTS alembic_version")
    await init_db(engine, t)  # runs the Alembic migrations
    yield engine, t
    await engine.dispose()


@pytest.fixture
async def seeded(db, whitelist):
    engine, t = db
    emb = MockEmbedder(DIM)
    assert await seed_mock_corpus(engine, t, whitelist, emb) == 6  # 7 rows, 1 not whitelisted
    assert await seed_mock_corpus(engine, t, whitelist, emb) == 0  # idempotent
    return engine, t, emb


def _result(status: Status) -> ClaimResult:
    return ClaimResult(text_original="x", text_en="x", type=ClaimType.CHECKABLE, status=status, confidence=0.9)


async def test_store_roundtrip_and_cache(db):
    engine, t = db
    store = PgStore(engine, t)
    emb = MockEmbedder(DIM)
    (v,) = await emb.embed(["Mumbai airport is closed for a week"])
    (other,) = await emb.embed(["completely different words about cricket scores"])
    mumbai = [Entity(text="Mumbai", kind="place")]

    await store.create_check("chk1", "text", {"created_at": NOW})
    await store.log_stage(StageRun("chk1", "c1", "judge", {"a": 1}, {"b": "\x00ok"}, 1.5, "mock-1", None, NOW))
    await store.save_claim("chk1", "c1", _result(Status.CONTRADICTED), v, "mock-1", mumbai, [], {"llm": "m"}, None)
    await store.save_claim("chk1", "c2", _result(Status.UNVERIFIED_TOO_EARLY), v, "mock-1", mumbai, [], {}, NOW + timedelta(hours=6))

    hit = await store.find_cached(v, "mock-1", [Entity(text="mumbai")], 0.92, NOW)
    assert hit and hit.check_id == "chk1" and hit.claim.status == Status.CONTRADICTED and hit.similarity > 0.99
    assert await store.find_cached(v, "mock-1", [Entity(text="Pune")], 0.92, NOW) is None
    assert await store.find_cached(v, "other-model", mumbai, 0.92, NOW) is None
    assert await store.find_cached(other, "mock-1", mumbai, 0.92, NOW) is None
    assert await store.find_cached(v, "mock-1", [], 0.92, NOW) is None

    async with engine.connect() as conn:
        assert (await conn.execute(select(func.count()).select_from(t.stage_runs))).scalar_one() == 1
        assert (await conn.execute(select(t.stage_runs.c.outputs))).scalar_one() == {"b": "ok"}

    await store.fail_check("chk1", "boom")
    async with engine.connect() as conn:
        row = (await conn.execute(select(t.checks.c.status, t.checks.c.meta))).one()
    assert row.status == "error" and row.meta["error"] == "boom"
    assert await store.get_check("chk1") is None


async def test_upsert_document_replaces_passages(db, whitelist):
    engine, t = db
    ids = await sync_sources(engine, t, whitelist)
    entry = whitelist.lookup("https://wire.mock.example/")
    emb = MockEmbedder(DIM)
    kw = dict(source_id=ids["wire.mock.example"], entry=entry, url="https://wire.mock.example/a", title="A",
              language="en", published_at=NOW, embedding_model="mock-1")
    (v,) = await emb.embed(["x"])
    await upsert_document(engine, t, full_text="a b", chunks=[ChunkIn("p1", "a", v), ChunkIn("p2", "b", v)], **kw)
    await upsert_document(engine, t, full_text="c", chunks=[ChunkIn("p3", "c", v)], **kw)
    assert await passage_count(engine, t) == 1


async def test_known_urls_matches_feed_links_to_redirected_urls(db, whitelist):
    # Stored under the URL after redirects; the feed links it with a trailing slash and www.
    engine, t = db
    ids = await sync_sources(engine, t, whitelist)
    entry = whitelist.lookup("https://wire.mock.example/")
    (v,) = await MockEmbedder(DIM).embed(["x"])
    await upsert_document(engine, t, source_id=ids["wire.mock.example"], entry=entry, url="https://wire.mock.example/a",
                          title="A", language="en", published_at=NOW, full_text="a", chunks=[ChunkIn("p1", "a", v)],
                          embedding_model="mock-1")
    feed = ["https://www.wire.mock.example/a/", "https://wire.mock.example/b/"]
    assert await known_urls(engine, t, feed, ids["wire.mock.example"]) == {"https://www.wire.mock.example/a/"}
    assert await known_urls(engine, t, ["https://wire.mock.example/a"]) == {"https://wire.mock.example/a"}


async def test_hybrid_search_both_languages(seeded):
    engine, t, emb = seeded
    search = PgSearch(engine, t, emb)
    claim_en = "A footbridge over the Godavari river in Nashik collapsed on Sunday."
    (vec,) = await emb.embed([claim_en])
    out = await search.search([("नाशिकमध्ये गोदावरी नदीवरील पादचारी पूल रविवारी कोसळला.", "mr"), (claim_en, "en")], vec, k=8)
    ids = [p.id for p in out]
    assert {"psg_bridge_police", "psg_bridge_daily", "psg_bridge_police_mr"} <= set(ids)
    assert "psg_metro_blog" not in ids  # never indexed: not whitelisted
    police = next(p for p in out if p.id == "psg_bridge_police")
    assert police.tier == 1 and police.publisher == "Nashik City Police (MOCK)" and 0 < police.relevance <= 1


async def test_hybrid_search_english_stemming_and_date_filter(seeded):
    engine, t, emb = seeded
    search = PgSearch(engine, t, emb, min_vector_similarity=0.99)  # keyword half only
    out = await search.search([("bridge collapses", "en")], None, k=8)
    assert "psg_bridge_daily" in [p.id for p in out]  # 'collapses' ~ 'collapse(d)' via english stemming
    later = await search.search([("bridge collapses", "en")], None, k=8, published_after=NOW)
    assert later == []


async def test_search_with_no_match_returns_nothing(seeded):
    engine, t, emb = seeded
    (vec,) = await emb.embed(["A fire broke out at a chemical factory in Thane."])
    assert await PgSearch(engine, t, emb).search([("A fire broke out at a chemical factory in Thane.", "en")], vec, k=8) == []


@pytest.mark.parametrize("text,post_date,expected", CASES)
async def test_pipeline_end_to_end_on_postgres(seeded, settings, whitelist, text, post_date, expected):
    engine, t, _ = seeded
    s = settings.model_copy(update={"database_url": URL})
    pipeline = Pipeline(s, build_adapters(s, engine, t), PgStore(engine, t), whitelist, clock=lambda: NOW)
    assert pipeline.a.search.model_version.startswith("pg-hybrid")
    res = await pipeline.run(CheckInput(text=text, post_date=post_date))
    assert [c.status for c in res.claims] == [expected]
    assert await PgStore(engine, t).get_check(res.check_id) == res
    async with engine.connect() as conn:
        stages = (await conn.execute(sql("select count(*) from stage_runs where check_id = :c"), {"c": res.check_id})).scalar_one()
    assert stages >= 3


async def test_cache_hit_on_postgres(seeded, settings, whitelist):
    engine, t, _ = seeded
    s = settings.model_copy(update={"database_url": URL})
    pipeline = Pipeline(s, build_adapters(s, engine, t), PgStore(engine, t), whitelist, clock=lambda: NOW)
    await pipeline.run(CheckInput(text="Mumbai airport is closed for a week."))
    names = [e.name async for e in pipeline.stream(CheckInput(text="Mumbai airport is closed for a week."))]
    assert names == ["claims_extracted", "cache_hit", "verdict", "done"]


async def test_crawl_end_to_end(db, settings):
    import httpx

    from app.sources import SourceEntry, Whitelist
    from crawler.run import crawl
    from .test_crawler import article_html

    engine, t = db
    wl = Whitelist([
        SourceEntry(name="Wire (MOCK)", domain="wire.mock.example", tier=1, kind="wire",
                    rss_url="https://wire.mock.example/feed.xml", language="en"),
        SourceEntry(name="Other (MOCK)", domain="other.mock.example", tier=2),
    ])
    body = "Officials of Thane Municipal Corporation confirmed a fire at a chemical factory in Thane. " * 4
    feed = f"""<?xml version="1.0"?><rss version="2.0"><channel><title>W</title>
      <item><title>Fire</title><link>https://wire.mock.example/fire</link><pubDate>Sun, 04 Oct 2026 10:00:00 GMT</pubDate></item>
      <item><title>Blocked</title><link>https://wire.mock.example/private/x</link></item>
      <item><title>Offsite</title><link>https://evil.example/fake</link></item>
      <item><title>Redirect</title><link>https://wire.mock.example/redirect</link></item>
      <item><title>Short</title><link>https://wire.mock.example/short</link></item>
    </channel></rss>"""
    pages = {
        "/feed.xml": httpx.Response(200, text=feed),
        "/robots.txt": httpx.Response(200, text="User-agent: *\nDisallow: /private/\n"),
        "/fire": httpx.Response(200, text=article_html(body, "Fire in Thane")),
        "/short": httpx.Response(200, text=article_html("Too short.")),
        "/redirect": httpx.Response(302, headers={"Location": "https://evil.example/landing"}),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host != "wire.mock.example":
            return httpx.Response(200, text=article_html(body))
        return pages.get(request.url.path, httpx.Response(404))

    emb = MockEmbedder(DIM)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        kw = dict(engine=engine, tables=t, whitelist=wl, embedder=emb, client=client)
        stats = await crawl(settings, **kw)
        assert (stats.stored, stats.skipped_robots, stats.skipped_offsite, stats.failed) == (1, 1, 2, 1)
        again = await crawl(settings, **kw)
        assert again.stored == 0 and again.skipped_known == 1

    async with engine.connect() as conn:
        doc = (await conn.execute(select(t.documents))).one()
    assert doc.url == "https://wire.mock.example/fire" and doc.title == "Fire in Thane"
    assert doc.published_at.isoformat().startswith("2026-10-04T10:00")

    (vec,) = await emb.embed(["A fire broke out at a chemical factory in Thane."])
    out = await PgSearch(engine, t, emb).search([("A fire broke out at a chemical factory in Thane.", "en")], vec, k=5)
    assert out and out[0].publisher == "Wire (MOCK)" and out[0].tier == 1


async def test_recheck_on_postgres(seeded, settings, whitelist):
    from app.pipeline.recheck import run_rechecks

    engine, t, _ = seeded
    s = settings.model_copy(update={"database_url": URL})
    store = PgStore(engine, t)
    pipeline = Pipeline(s, build_adapters(s, engine, t), store, whitelist, clock=lambda: NOW)
    await pipeline.run(CheckInput(text="A fire broke out at a chemical factory in Thane.", post_date=NOW - timedelta(hours=2)))
    pipeline.clock = lambda: NOW + timedelta(days=4)
    results = await run_rechecks(pipeline)
    assert [(r["old"], r["new"]) for r in results] == [("UNVERIFIED_TOO_EARLY", "UNVERIFIED_EVIDENCE_MISSING")]
    async with engine.connect() as conn:
        rows = (await conn.execute(select(t.verdicts.c.id, t.verdicts.c.superseded_at, t.verdicts.c.rechecked_from)
                                   .order_by(t.verdicts.c.id))).all()
        post_dates = (await conn.execute(select(t.claims.c.post_date))).scalars().all()
    assert rows[0].superseded_at is not None and rows[1].rechecked_from == rows[0].id
    assert post_dates[0] == post_dates[1] == NOW - timedelta(hours=2)
    assert await run_rechecks(pipeline) == []


async def test_known_urls_keeps_identifying_query_parameters(db, whitelist):
    # RBI press releases differ only by ?prid=; tracking parameters must not make a stored URL look new.
    engine, t = db
    ids = await sync_sources(engine, t, whitelist)
    entry = whitelist.lookup("https://wire.mock.example/")
    (v,) = await MockEmbedder(DIM).embed(["x"])
    await upsert_document(engine, t, source_id=ids["wire.mock.example"], entry=entry,
                          url="https://wire.mock.example/show.aspx?prid=1", title="A", language="en", published_at=NOW,
                          full_text="a", chunks=[ChunkIn("p1", "a", v)], embedding_model="mock-1")
    feed = ["https://wire.mock.example/show.aspx?prid=1&utm_source=rss", "https://wire.mock.example/show.aspx?prid=2"]
    assert await known_urls(engine, t, feed, ids["wire.mock.example"]) == {feed[0]}


async def test_rate_limit_is_shared_across_workers(db):
    from app.api.security import PgRateLimiter

    engine, t = db
    # Two limiters on one database = two API worker processes.
    a, b = PgRateLimiter(engine, t.rate_limit_hits, 3), PgRateLimiter(engine, t.rate_limit_hits, 3)
    results = await asyncio.gather(*(lim.check("key:abc") for lim in (a, b, a, b, a)))
    assert sum(r is None for r in results) == 3  # concurrent checks never over-admit
    wait = await b.check("key:abc")
    assert wait is not None and 0 < wait <= 60
    assert await a.check("key:other") is None
    assert await PgRateLimiter(engine, t.rate_limit_hits, 0).check("key:abc") is None  # disabled


async def test_rate_limit_expires_old_hits(db):
    from app.api.security import PgRateLimiter

    engine, t = db
    hits = t.rate_limit_hits
    async with engine.begin() as conn:
        await conn.execute(hits.insert().values(identity="ip:gone", at=func.now() - sql("interval '2 minutes'")))
        await conn.execute(hits.insert().values(identity="key:abc", at=func.now() - sql("interval '2 minutes'")))
    assert await PgRateLimiter(engine, hits, 1).check("key:abc") is None  # the old hit no longer counts
    async with engine.connect() as conn:
        assert (await conn.execute(select(hits.c.identity))).scalars().all() == ["key:abc"]
