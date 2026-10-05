from pathlib import Path

from app.config import ROOT_DIR
from app.sources import SourceEntry, Whitelist, load_whitelist_file


def wl() -> Whitelist:
    return Whitelist([
        SourceEntry(name="Police", domain="police.example.org", tier=1, kind="police"),
        SourceEntry(name="Placeholder", domain="todo.invalid", tier=1, todo=True),
    ])


def test_lookup_matches_domain_and_subdomains():
    w = wl()
    assert w.lookup("https://police.example.org/a").name == "Police"
    assert w.lookup("https://www.police.example.org/a").name == "Police"
    assert w.lookup("https://press.police.example.org/a").name == "Police"
    assert w.lookup("police.example.org").name == "Police"


def test_lookup_rejects_lookalikes_and_unknown():
    w = wl()
    assert w.lookup("https://evilpolice.example.org/a") is None
    assert w.lookup("https://police.example.org.evil.com/a") is None
    assert w.lookup("https://example.org/") is None
    assert w.lookup(None) is None


def test_todo_entries_are_ignored():
    assert wl().lookup("https://todo.invalid/x") is None
    assert len(wl()) == 1


def test_repo_sources_yaml_is_valid():
    w = load_whitelist_file(ROOT_DIR / "sources.yaml")
    assert len(w) >= 10 and all(e.rss_url or e.sitemap_url for e in w.entries)
