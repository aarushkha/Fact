"""The source whitelist (sources.yaml). Only whitelisted sources count as evidence."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, Field


class SourceEntry(BaseModel):
    name: str
    domain: str
    rss_url: str | None = None
    sitemap_url: str | None = None
    language: str | None = None
    tier: int = Field(ge=1, le=3)  # 1 = primary, 2 = original reporting, 3 = aggregator
    kind: str | None = None  # police, court, government, institution, wire, outlet, factchecker, aggregator
    todo: bool = False  # placeholder entries are never used as evidence


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
