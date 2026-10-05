"""The source whitelist (sources.yaml). Only whitelisted sources count as evidence."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field


class Feed(BaseModel):
    url: str
    language: str | None = None  # defaults to the entry's language


class SourceEntry(BaseModel):
    name: str
    domain: str
    # One feed or several; a multilingual source lists {url, language} per feed so each article keeps its language.
    rss_url: str | list[str | Feed] | None = None
    sitemap_url: str | None = None
    language: str | None = None
    tier: int = Field(ge=1, le=3)  # 1 = primary, 2 = original reporting, 3 = aggregator
    kind: str | None = None  # police, court, government, institution, wire, outlet, factchecker, aggregator
    todo: bool = False  # placeholder entries are never used as evidence

    @property
    def feed_specs(self) -> list[Feed]:
        if not self.rss_url:
            return []
        items = [self.rss_url] if isinstance(self.rss_url, str) else self.rss_url
        out = [Feed(url=f) if isinstance(f, str) else f for f in items]
        return [Feed(url=f.url, language=f.language or self.language) for f in out]

    @property
    def feeds(self) -> list[str]:
        return [f.url for f in self.feed_specs]


def host_of(url_or_domain: str) -> str:
    host = urlparse(url_or_domain).hostname if "//" in url_or_domain else url_or_domain
    host = (host or "").lower().strip(".")
    return host[4:] if host.startswith("www.") else host


class Whitelist:
    def __init__(self, entries: list[SourceEntry]):
        self.entries = [e for e in entries if not e.todo]
        self._by_domain = {host_of(e.domain): e for e in self.entries}

    def lookup(self, url_or_domain: str | None) -> SourceEntry | None:
        """Match a URL or bare domain to a whitelisted source (exact domain or any subdomain)."""
        if not url_or_domain:
            return None
        host = host_of(url_or_domain)
        while host:
            if host in self._by_domain:
                return self._by_domain[host]
            if "." not in host:
                return None
            host = host.split(".", 1)[1]
        return None

    def __len__(self) -> int:
        return len(self.entries)


def load_whitelist_file(path: Path) -> Whitelist:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Whitelist([SourceEntry(**row) for row in data.get("sources", [])])


@lru_cache
def load_whitelist(path: str) -> Whitelist:
    return load_whitelist_file(Path(path))
