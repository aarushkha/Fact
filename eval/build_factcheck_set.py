"""Build eval/factchecks.jsonl: real claims labelled by published fact-checks (Google Fact Check API).

    python -m eval.build_factcheck_set [--n 200] [--seed 7]

Labels come from each review's textualRating through the same conservative exact-match table the
pipeline uses (app/pipeline/match.py RATING_MAP); unmappable ratings ("Half true", "Altered", ...) are
skipped. Each row carries exclude_urls = [the labelling review], which the eval runner removes from the
evidence so the label is not simply looked up. Other outlets' reports on the same claim still count.

The label is the fact-checker's verdict. When our index has no other evidence the correct system
behaviour is to abstain (UNVERIFIED_*): that lowers accuracy but is never counted as confident-wrong.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import httpx

from app.adapters.google_factcheck import URL, parse_claims
from app.config import ROOT_DIR, get_settings
from app.pipeline.match import map_rating

PUBLISHERS = [
    "altnews.in", "factly.in", "vishvasnews.com", "boomlive.in", "thequint.com", "aajtak.in",
    "factcrescendo.com", "marathi.factcrescendo.com", "lokmat.com", "indiatoday.in",
]
LANGS = {"en": 0.4, "hi": 0.35, "mr": 0.25}  # target language mix


async def fetch(site: str, key: str, pages: int = 3) -> list[dict]:
    rows, token = [], None
    async with httpx.AsyncClient(timeout=30, headers={"x-goog-api-key": key}) as c:
        for _ in range(pages):
            params = {"reviewPublisherSiteFilter": site, "pageSize": 100}
            if token:
                params["pageToken"] = token
            r = await c.get(URL, params=params)
            r.raise_for_status()
            data = r.json()
            for hit in parse_claims(data):
                rows.append({"site": site, "hit": hit})
            token = data.get("nextPageToken")
            if not token:
                break
    return rows


def to_row(site: str, hit) -> dict | None:
    status = map_rating(hit.textual_rating)
    lang = (hit.language or "").split("-")[0]
    if status is None or lang not in LANGS or len(hit.claim_text.strip()) < 15:
        return None
    rid = "fc_" + hashlib.sha1(hit.review_url.encode()).hexdigest()[:10]
    date = hit.claim_date or hit.review_date
    return {
        "id": rid,
        "split": "hidden" if int(rid[3:], 16) % 5 == 0 else "tune",  # deterministic ~80/20
        "synthetic": False,
        "input_text": hit.claim_text.strip(),
        "language": lang,
        "post_date": date.isoformat() if date else None,
        "expected_status": status.value,
        "notes": f"REAL claim. Label from {hit.publisher_name or site} rated '{hit.textual_rating}'.",
        "publisher": hit.publisher_name or site,
        "review_url": hit.review_url,
        "exclude_urls": [hit.review_url],
        "single_claim": True,
    }


async def main_async(n: int, seed: int, out: Path) -> None:
    key = get_settings().google_factcheck_api_key
    if not key:
        raise SystemExit("GOOGLE_FACTCHECK_API_KEY is not set")
    batches = await asyncio.gather(*(fetch(s, key) for s in PUBLISHERS))
    seen, by_lang = set(), defaultdict(list)
    for batch in batches:
        for item in batch:
            row = to_row(item["site"], item["hit"])
            norm = " ".join((row or {}).get("input_text", "").lower().split())
            if row and norm not in seen:
                seen.add(norm)
                by_lang[row["language"]].append(row)
    rng = random.Random(seed)
    picked = []
    for lang, share in LANGS.items():
        quota = round(n * share)
        pool = by_lang[lang]
        rng.shuffle(pool)
        by_status = defaultdict(list)
        for r in pool:
            by_status[r["expected_status"]].append(r)
        # Rare labels are capped so CONTRADICTED (most fact-checks) still makes up about half.
        take = by_status["CONFIRMED"][: round(quota * 0.1)] + by_status["MISLEADING_CONTEXT"][: round(quota * 0.4)]
        take += by_status["CONTRADICTED"][: quota - len(take)]
        if len(take) < quota:  # language has too few of one label: top up with whatever is left
            rest = [r for r in pool if r not in take]
            take += rest[: quota - len(take)]
        picked += take
    picked.sort(key=lambda r: r["id"])
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picked), encoding="utf-8")
    counts = defaultdict(int)
    for r in picked:
        counts[(r["language"], r["expected_status"])] += 1
    print(f"wrote {len(picked)} rows to {out}")
    for k in sorted(counts):
        print(f"  {k[0]:<3} {k[1]:<20} {counts[k]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT_DIR / "eval" / "factchecks.jsonl"))
    args = ap.parse_args()
    asyncio.run(main_async(args.n, args.seed, Path(args.out)))


if __name__ == "__main__":
    main()
