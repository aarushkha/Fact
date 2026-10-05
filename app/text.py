"""Small, dependency-free text helpers shared by stages and mocks."""

from __future__ import annotations

import hashlib
import re

# \w alone splits Devanagari words at vowel signs (combining marks), so include the block explicitly.
_WORD_RE = re.compile(r"[\w\u0900-\u0963\u0966-\u097F]+", re.UNICODE)  # excludes the danda punctuation marks
_SENT_RE = re.compile(r"(?<=[.!?।])\s+")

STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with from as is are was were be been being this that these
    those it its into over under than then there their they them he she his her we our you your i me my not no
    so such can could would should will shall may might must do does did done has have had having said says say
    all any some more most other also just only very about after before during while up down out off""".split()
)


def tokens(text: str) -> list[str]:
    return [t for t in (w.lower() for w in _WORD_RE.findall(text)) if len(t) > 2 and t not in STOPWORDS]


def content_words(text: str) -> set[str]:
    return set(tokens(text))


def overlap_ratio(query: str, doc: str) -> float:
    """Share of the query's content words found in doc (0-1)."""
    q = content_words(query)
    if not q:
        return 0.0
    return len(q & content_words(doc)) / len(q)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]


def source_id_for(url: str) -> str:
    return "src_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]


def has_devanagari(text: str) -> bool:
    return any("ऀ" <= ch <= "ॿ" for ch in text)


_STOP_CAPS = {"The", "A", "An", "This", "That", "It", "In", "On", "At", "He", "She", "They", "We", "I", "But", "And"}


def guess_entities(text_en: str):
    """Heuristic entities for the Gemini-free path: capitalized words/phrases and 4-digit years."""
    from app.models.schemas import Entity

    found: dict[str, Entity] = {}
    for m in re.finditer(r"\b(?:[A-Z][a-zA-Z]+)(?:\s+[A-Z][a-zA-Z]+)*\b|\b(?:19|20)\d{2}\b", text_en):
        phrase = m.group(0)
        words = [w for w in phrase.split() if w not in _STOP_CAPS]
        if not words:
            continue
        phrase = " ".join(words)
        kind = "date" if phrase.isdigit() else "other"
        found.setdefault(phrase.lower(), Entity(text=phrase, kind=kind))
    return list(found.values())[:10]


def url_key(url: str) -> str:
    """Comparable form of a URL: no scheme, www, fragment, query, trailing slash or AMP segment."""
    from urllib.parse import urlsplit

    parts = urlsplit(url.strip().lower())
    host = parts.netloc[4:] if parts.netloc.startswith("www.") else parts.netloc
    path = "/".join(seg for seg in parts.path.split("/") if seg and seg != "amp")
    return f"{host}/{path}"
