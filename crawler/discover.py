"""Find article URLs (with dates when available) from RSS/Atom feeds and XML sitemaps."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import datetime, timezone

import feedparser
from lxml import etree

SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"
NEWS_NS = "http://www.google.com/schemas/sitemap-news/0.9"


@dataclass
class Candidate:
    url: str
    published_at: datetime | None = None
    title: str | None = None


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parse_feed(content: bytes) -> list[Candidate]:
    feed = feedparser.parse(content)
    out = []
    for e in feed.entries:
        url = e.get("link")
        if not url:
            continue
        parsed = e.get("published_parsed") or e.get("updated_parsed")
        published = datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc) if parsed else None
        out.append(Candidate(url=url, published_at=published, title=e.get("title")))
    return out


def parse_sitemap(content: bytes) -> tuple[list[Candidate], list[str]]:
    """Returns (article candidates, child sitemap URLs). Entities and network access are disabled."""
    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)
    try:
        root = etree.fromstring(content, parser=parser)
    except etree.XMLSyntaxError:
        return [], []
    ns = {"s": SITEMAP_NS, "n": NEWS_NS}
    if etree.QName(root).localname == "sitemapindex":
        return [], [loc.strip() for loc in root.xpath("//s:sitemap/s:loc/text()", namespaces=ns)]
    out = []
    for node in root.xpath("//s:url", namespaces=ns):
        loc = node.xpath("s:loc/text()", namespaces=ns)
        if not loc:
            continue
        date = node.xpath("n:news/n:publication_date/text()", namespaces=ns) or node.xpath("s:lastmod/text()", namespaces=ns)
        title = node.xpath("n:news/n:title/text()", namespaces=ns)
        out.append(Candidate(url=loc[0].strip(), published_at=parse_datetime(date[0] if date else None),
                             title=title[0] if title else None))
    return out, []
