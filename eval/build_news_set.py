"""Build eval/news_confirmed.jsonl: CONFIRMED rows from real news reported by two independent outlets.

    python -m eval.build_news_set [--n 40] [--threshold 0.75] [--hours 48] [--dry-run]

Fact-checks are ~95% false claims, so CONFIRMED is nearly absent from eval/factchecks.jsonl. A row here is
a headline from whitelisted outlet A whose event another whitelisted source B (a different domain) also
reported within --hours, judged by multilingual embedding similarity of the two headlines >= --threshold.
Labels are silver (two outlets agree; no fact-checker verdict) and marked "label_source": "news".
Each row excludes A's own article, so the system has to find B (or other reporting) itself; when the index
has neither, abstaining is the right answer, exactly as for the fact-check rows.

Appends to --out, never changing an existing row. Run --dry-run first and read the pairs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
import re
from datetime import timedelta
from pathlib import Path

from app.adapters.factory import build_embedder
from app.config import ROOT_DIR, get_settings
from app.db.store import cosine
from app.sources import host_of, load_whitelist_file
from crawler.run import crawl_client, discover
from eval.build_factcheck_set import load_existing, norm_text

# Headlines that are not a statement of fact (live blogs, galleries, explainers, any question, opinion).
NOT_A_CLAIM = re.compile(
    r"^(watch|live|video|photos?|in pics|explained|explainer|opinion|quiz|horoscope|today'?s|top \d+|"
    r"\d+ (things|ways|reasons))\b|\?|\b(live updates|highlights|recap)\b",
    re.IGNORECASE,
)


def is_claim_like(title: str) -> bool:
    return len(title.split()) >= 6 and not NOT_A_CLAIM.search(title.strip())


def pair_headlines(items: list[dict], vectors: list[list[float]], threshold: float, hours: float) -> list[dict]:
    """For each headline, the most similar headline from another domain published within `hours`."""
    pairs = []
    for i, a in enumerate(items):
        best, best_sim = None, threshold
        for j, b in enumerate(items):
            if a["domain"] == b["domain"] or abs(a["published_at"] - b["published_at"]) > timedelta(hours=hours):
                continue
            sim = cosine(vectors[i], vectors[j])
            if sim >= best_sim:
                best, best_sim = b, sim
        if best is not None:
            pairs.append({"a": a, "b": best, "similarity": round(best_sim, 4)})
    return pairs


def to_row(pair: dict) -> dict:
    a, b = pair["a"], pair["b"]
    rid = "news_" + hashlib.sha1(a["url"].encode()).hexdigest()[:10]
    return {
        "id": rid,
        "split": "hidden" if int(rid[5:], 16) % 5 == 0 else "tune",
        "synthetic": False,
        "label_source": "news",
        "input_text": a["title"],
        "language": a["language"],
        "post_date": a["published_at"].isoformat(),
        "expected_status": "CONFIRMED",
        "notes": f"REAL headline from {a['publisher']}; also reported by {b['publisher']} "
                 f"(headline similarity {pair['similarity']}). Silver label.",
        "publisher": a["publisher"],
        "review_url": a["url"],
        "corroborated_by": b["url"],
        "exclude_urls": [a["url"]],
        "single_claim": True,
    }


async def main_async(n: int, threshold: float, hours: float, out: Path, dry_run: bool) -> None:
    settings = get_settings()
    whitelist = load_whitelist_file(settings.effective_sources_file)
    outlets = [e for e in whitelist.entries if e.tier <= 2 and e.kind != "factchecker" and (e.feeds or e.sitemap_url)]
    async with crawl_client(settings) as client:
        results = await asyncio.gather(*(discover(client, e) for e in outlets), return_exceptions=True)
    items, seen = [], set()
    for entry, cands in zip(outlets, results):
        if isinstance(cands, BaseException):
            print(f"skip {entry.name}: {cands}")
            continue
        for c in cands:
            title = " ".join(html.unescape(c.title or "").split())
            if c.published_at is None or not is_claim_like(title) or norm_text(title) in seen:
                continue
            seen.add(norm_text(title))
            items.append({"title": title, "url": c.url, "published_at": c.published_at, "domain": host_of(entry.domain),
                          "publisher": entry.name, "language": c.language or entry.language})
    print(f"{len(items)} headlines from {len(outlets)} outlets")
    embedder = build_embedder(settings)
    vectors = await embedder.embed([i["title"] for i in items])
    pairs = sorted(pair_headlines(items, vectors, threshold, hours), key=lambda p: -p["similarity"])
    existing = load_existing(out)
    known = {r["id"] for r in existing} | {norm_text(r["input_text"]) for r in existing}
    rows, used = [], set()
    for p in pairs:
        row = to_row(p)
        # One row per event: skip a headline whose corroborating article already labels another row.
        if row["id"] in known or norm_text(row["input_text"]) in known or {p["a"]["url"], p["b"]["url"]} & used:
            continue
        used |= {p["a"]["url"], p["b"]["url"]}
        rows.append(row)
        if len(rows) >= n:
            break
    for r in rows:
        print(f"[{r['language']}] {r['input_text'][:100]}\n    <- {r['corroborated_by']}\n    {r['notes']}")
    if dry_run:
        print(f"dry run: {len(rows)} rows, nothing written")
        return
    all_rows = sorted(existing + rows, key=lambda r: r["id"])
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows), encoding="utf-8")
    print(f"wrote {len(all_rows)} rows to {out} ({len(rows)} new)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40, help="new rows to add")
    # 0.72-0.88 pairs read on 2026-10-05 were all the same event; 0.75 leaves a margin.
    ap.add_argument("--threshold", type=float, default=0.75, help="min headline cosine similarity (BGE-M3)")
    ap.add_argument("--hours", type=float, default=48, help="max gap between the two reports")
    ap.add_argument("--out", default=str(ROOT_DIR / "eval" / "news_confirmed.jsonl"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args.n, args.threshold, args.hours, Path(args.out), args.dry_run))


if __name__ == "__main__":
    main()
