"""Fetch the fact-check article behind every real eval row and chunk it the way the crawler does.

Output: bench/data/reviews.jsonl, one line per row: {id, url, title, language, chunks: [...]}.
Robots.txt is honoured and each host is hit at most once per second.

    python -m bench.fetch_reviews [--files eval/factchecks.jsonl,eval/news_confirmed.jsonl]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from app.config import get_settings
from crawler.chunk import chunk_text
from crawler.extract import extract_article

OUT = Path(__file__).parent / "data" / "reviews.jsonl"


class Polite:
    def __init__(self, agent: str):
        self.agent = agent
        self.robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self.last: dict[str, float] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    async def allowed(self, client: httpx.AsyncClient, url: str) -> bool:
        host = urlsplit(url).netloc
        if host not in self.robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = await client.get(f"https://{host}/robots.txt")
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except httpx.HTTPError:
                rp.parse([])
            self.robots[host] = rp
        rp = self.robots[host]
        return rp is None or rp.can_fetch(self.agent, url)

    async def wait(self, url: str) -> None:
        host = urlsplit(url).netloc
        lock = self.locks.setdefault(host, asyncio.Lock())
        async with lock:
            gap = time.monotonic() - self.last.get(host, 0)
            if gap < 1.0:
                await asyncio.sleep(1.0 - gap)
            self.last[host] = time.monotonic()


async def fetch_one(client, polite: Polite, row: dict, sem: asyncio.Semaphore) -> dict | None:
    url = row["review_url"]
    async with sem:
        try:
            if not await polite.allowed(client, url):
                return {"id": row["id"], "url": url, "error": "robots"}
            await polite.wait(url)
            r = await client.get(url, follow_redirects=True)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            return {"id": row["id"], "url": url, "error": type(exc).__name__}
    art = extract_article(r.content, str(r.url))
    if art is None:
        return {"id": row["id"], "url": url, "error": "extract"}
    return {"id": row["id"], "url": url, "title": art.title, "language": row["language"],
            "chunks": chunk_text(art.text, 180)}


async def main(files: list[str]) -> None:
    rows = [json.loads(line) for f in files for line in open(f) if line.strip()]
    rows = [r for r in rows if r.get("review_url")]
    done = {}
    if OUT.exists():
        done = {d["id"]: d for d in map(json.loads, OUT.open()) if "chunks" in d}
    todo = [r for r in rows if r["id"] not in done]
    settings = get_settings()
    agent = settings.crawler_user_agent
    polite = Polite(agent)
    sem = asyncio.Semaphore(6)
    async with httpx.AsyncClient(headers={"User-Agent": agent}, timeout=30) as client:
        results = await asyncio.gather(*(fetch_one(client, polite, r, sem) for r in todo))
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        for d in list(done.values()) + [r for r in results if r]:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    ok = sum(1 for r in results if r and "chunks" in r)
    errors: dict[str, int] = {}
    for r in results:
        if r and "error" in r:
            errors[r["error"]] = errors.get(r["error"], 0) + 1
    print(f"fetched {ok}/{len(todo)} (+{len(done)} cached); errors {errors}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default="eval/factchecks.jsonl,eval/news_confirmed.jsonl")
    asyncio.run(main(ap.parse_args().files.split(",")))
