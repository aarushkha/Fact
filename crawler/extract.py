"""Article text extraction with trafilatura."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import trafilatura

from crawler.discover import parse_datetime

MIN_CHARS = 200  # shorter extractions are usually index pages or paywalls


@dataclass
class Article:
    text: str
    title: str | None
    published_at: datetime | None


def extract_article(html: str | bytes, url: str) -> Article | None:
    doc = trafilatura.bare_extraction(html, url=url, with_metadata=True, include_comments=False, include_tables=False)
    if doc is None or not doc.text or len(doc.text) < MIN_CHARS:
        return None
    return Article(text=doc.text, title=doc.title, published_at=parse_datetime(doc.date))
