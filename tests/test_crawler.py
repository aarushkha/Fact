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
