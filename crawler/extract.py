"""Article text extraction with trafilatura."""

from __future__ import annotations

import json
import re
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


_LD_JSON = re.compile(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.S | re.I)


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def extract_claim_review(html: str) -> dict | None:
    """First schema.org ClaimReview in the page's JSON-LD, reduced to the fields we use."""
    for block in _LD_JSON.findall(html or ""):
        try:
            data = json.loads(block.strip())
        except ValueError:
            continue
        for node in _walk(data):
            types = node.get("@type")
            if "ClaimReview" not in (types if isinstance(types, list) else [types]):
                continue
            rating = node.get("reviewRating") or {}
            item = node.get("itemReviewed") or {}
            author = item.get("author") or {}
            out = {
                "claim_reviewed": str(node.get("claimReviewed") or "").strip(),
                "rating": str(rating.get("alternateName") or rating.get("ratingValue") or "").strip(),
                "date_published": node.get("datePublished"),
                "claimant": author.get("name") if isinstance(author, dict) else None,
            }
            if out["claim_reviewed"] and out["rating"]:
                return out
    return None


def claim_review_text(review: dict) -> str:
    return f'Fact-check. Claim reviewed: "{review["claim_reviewed"]}". Rating: {review["rating"]}.'
