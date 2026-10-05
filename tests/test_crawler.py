import pytest
from datetime import datetime, timezone

from crawler.chunk import chunk_text
from crawler.discover import parse_feed, parse_sitemap
from crawler.extract import extract_article

RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>T</title>
<item><title>One</title><link>https://wire.mock.example/one</link><pubDate>Sun, 27 Sep 2026 14:00:00 GMT</pubDate></item>
<item><title>No date</title><link>https://wire.mock.example/two</link></item>
<item><title>No link</title></item>
</channel></rss>"""

SITEMAP = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">
<url><loc>https://wire.mock.example/a</loc><lastmod>2026-09-01</lastmod></url>
<url><loc>https://wire.mock.example/b</loc><news:news><news:publication_date>2026-09-02T10:00:00+05:30</news:publication_date><news:title>B</news:title></news:news></url>
</urlset>"""

INDEX = b"""<?xml version="1.0"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<sitemap><loc>https://wire.mock.example/sitemap-1.xml</loc></sitemap></sitemapindex>"""

ENTITY_BOMB = b"""<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;&a;">]>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://x.example/&b;</loc></url></urlset>"""


def test_parse_feed():
    out = parse_feed(RSS)
    assert [c.url for c in out] == ["https://wire.mock.example/one", "https://wire.mock.example/two"]
    assert out[0].published_at == datetime(2026, 9, 27, 14, tzinfo=timezone.utc)
    assert out[1].published_at is None


def test_parse_sitemap_and_index():
    items, children = parse_sitemap(SITEMAP)
    assert children == [] and [c.url for c in items] == ["https://wire.mock.example/a", "https://wire.mock.example/b"]
    assert items[0].published_at == datetime(2026, 9, 1, tzinfo=timezone.utc)
    assert items[1].published_at.isoformat() == "2026-09-02T10:00:00+05:30" and items[1].title == "B"
    assert parse_sitemap(INDEX) == ([], ["https://wire.mock.example/sitemap-1.xml"])
    assert parse_sitemap(b"not xml") == ([], [])


def test_sitemap_entities_not_expanded():
    items, _ = parse_sitemap(ENTITY_BOMB)
    assert all("aaaaa" not in c.url for c in items)


def test_chunk_text_packs_sentences_with_overlap():
    text = " ".join(f"Sentence number {i} has five words." for i in range(10))  # 6 words each
    chunks = chunk_text(text, max_words=20)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 20 for c in chunks)
    assert chunks[0].split(". ")[-1] in chunks[1]  # last sentence carried over


def test_chunk_text_devanagari_and_long_sentence():
    assert chunk_text("पहिले वाक्य आहे। दुसरे वाक्य आहे।", max_words=100) == ["पहिले वाक्य आहे। दुसरे वाक्य आहे।"]
    long = " ".join(["word"] * 45)
    assert [len(c.split()) for c in chunk_text(long, max_words=20)] == [20, 20, 5]
    assert chunk_text("", 20) == []


def article_html(body: str, title: str = "Headline") -> str:
    return (f"<html><head><title>{title}</title></head><body><nav>Home | About</nav>"
            f"<article><h1>{title}</h1><p>{body}</p></article><footer>Copyright</footer></body></html>")


def test_extract_article():
    body = "Nashik City Police confirmed that a section of the old footbridge collapsed on Sunday. " * 4
    a = extract_article(article_html(body), "https://police-nashik.mock.example/a")
    assert a and "footbridge collapsed" in a.text and "Copyright" not in a.text
    assert extract_article(article_html("Too short."), "https://x.example/") is None


def test_feed_urls_lose_fragments():
    rss = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>T</title>
    <item><title>A</title><link>https://x.example/a#publisher=newsstand</link></item></channel></rss>"""
    assert parse_feed(rss)[0].url == "https://x.example/a"


def test_extract_claim_review_from_json_ld():
    from crawler.extract import claim_review_text, extract_claim_review

    html = """<html><head><script type="application/ld+json">{"@context":"https://schema.org","@graph":[
      {"@type":"WebPage"},
      {"@type":"ClaimReview","claimReviewed":"Video shows floods in Kolhapur this week","datePublished":"2026-10-01",
       "reviewRating":{"@type":"Rating","alternateName":"Misleading"},
       "itemReviewed":{"@type":"Claim","author":{"@type":"Organization","name":"Social media users"}}}]}
    </script></head><body></body></html>"""
    r = extract_claim_review(html)
    assert r == {"claim_reviewed": "Video shows floods in Kolhapur this week", "rating": "Misleading",
                 "date_published": "2026-10-01", "claimant": "Social media users"}
    assert claim_review_text(r).startswith('Fact-check. Claim reviewed: "Video shows')
    assert extract_claim_review("<script type='application/ld+json'>not json</script>") is None


def test_source_entry_accepts_several_feeds():
    from app.sources import SourceEntry

    assert SourceEntry(name="x", domain="x.in", tier=2, rss_url=["a", "b"]).feeds == ["a", "b"]
    assert SourceEntry(name="x", domain="x.in", tier=2, rss_url="a").feeds == ["a"]
    assert SourceEntry(name="x", domain="x.in", tier=2).feeds == []


def test_extract_claim_review_skips_malformed_nodes():
    from crawler.extract import extract_claim_review

    bad = '{"@type": "ClaimReview", "claimReviewed": "x", "reviewRating": "False", "itemReviewed": "y"}'
    good = '{"@type": "ClaimReview", "claimReviewed": "Video shows floods", "reviewRating": {"alternateName": "False"}}'
    html = f'<script type="application/ld+json">[{bad}, {good}]</script>'
    assert extract_claim_review(html)["claim_reviewed"] == "Video shows floods"


def test_feed_language_overrides_entry_language():
    from app.sources import SourceEntry

    e = SourceEntry(name="x", domain="x.in", tier=2, language="hi",
                    rss_url=["https://x.in/feed", {"url": "https://x.in/mr/feed", "language": "mr"}])
    assert [(f.url, f.language) for f in e.feed_specs] == [("https://x.in/feed", "hi"), ("https://x.in/mr/feed", "mr")]
    assert e.feeds == ["https://x.in/feed", "https://x.in/mr/feed"]


async def test_discover_survives_one_failing_feed_and_tags_languages():
    import httpx
    import pytest

    from app.sources import SourceEntry
    from crawler.run import discover

    mr_rss = RSS.replace(b"wire.mock.example/one", b"wire.mock.example/mr-one")

    def handler(request: httpx.Request) -> httpx.Response:
        return {"/down": httpx.Response(503), "/mr": httpx.Response(200, content=mr_rss)}.get(
            request.url.path, httpx.Response(200, content=RSS))

    entry = SourceEntry(name="w", domain="wire.mock.example", tier=1, language="en",
                        rss_url=["https://wire.mock.example/down", "https://wire.mock.example/feed",
                                 {"url": "https://wire.mock.example/mr", "language": "mr"}])
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        found = {c.url: c.language for c in await discover(client, entry)}
        assert found["https://wire.mock.example/one"] == "en"
        assert found["https://wire.mock.example/mr-one"] == "mr"
        dead = SourceEntry(name="d", domain="wire.mock.example", tier=1, rss_url="https://wire.mock.example/down")
        with pytest.raises(RuntimeError, match="all discovery URLs failed"):
            await discover(client, dead)


async def test_crawler_never_follows_redirects_or_child_sitemaps_off_domain():
    import httpx

    from app.sources import SourceEntry
    from crawler.run import OffsiteRedirect, discover, safe_get

    index = b"""<?xml version="1.0"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<sitemap><loc>http://169.254.169.254/latest/meta-data/</loc></sitemap>
<sitemap><loc>https://wire.mock.example/sitemap-1.xml</loc></sitemap></sitemapindex>"""
    requested = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if request.url.path == "/sitemap.xml":
            return httpx.Response(200, content=index)
        if request.url.path == "/sitemap-1.xml":
            return httpx.Response(200, content=SITEMAP)
        if request.url.path == "/moved":
            return httpx.Response(302, headers={"location": "http://10.0.0.5/admin"})
        if request.url.path == "/hop":
            return httpx.Response(301, headers={"location": "/sitemap-1.xml"})
        return httpx.Response(404)

    entry = SourceEntry(name="w", domain="wire.mock.example", tier=1, sitemap_url="https://wire.mock.example/sitemap.xml")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True) as client:
        found = await discover(client, entry)
        assert {c.url for c in found} == {"https://wire.mock.example/a", "https://wire.mock.example/b"}
        assert not any("169.254" in u for u in requested)
        with pytest.raises(OffsiteRedirect):
            await safe_get(client, "https://wire.mock.example/moved", {"wire.mock.example"})
        assert not any("10.0.0.5" in u for u in requested)
        r = await safe_get(client, "https://wire.mock.example/hop", {"wire.mock.example"})  # on-domain hop is fine
        assert r.status_code == 200 and str(r.url).endswith("/sitemap-1.xml")
