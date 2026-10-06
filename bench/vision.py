"""Screenshot reading benchmark: real eval post texts rendered as social-media screenshots (Chromium), so the
ground truth is exact. Each card has a fictional handle, a printed date, like/share chrome and a reply by
another user, which must all be handled as prompts.VISION_SYSTEM says (chrome and replies excluded).

Every model gets the production vision prompt and schema (as GeminiVisionReader sends them).
Metrics: character error rate of post_text (whitespace-normalised), exact handle, exact date, reply leaked
into post_text, and "not transcribed" (Devanagari post returned without Devanagari, i.e. translated).

    python -m bench.vision --render --n 12          # per language: en, hi, mr
    python -m bench.vision --models google/gemini-3.8-flash,... --max-usd 0.5
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import html
import json
import random
import re
from pathlib import Path

import httpx

from app.adapters import prompts
from app.text import has_devanagari
from bench.classifier import Budget, OpenRouterJSON

HERE = Path(__file__).parent
IMG, OUT = HERE / "data" / "screens", HERE / "results"
DATES = ["3 March", "12/09/2026", "2h", "Yesterday at 9:41 PM", "14 Aug 2026", "5d"]
REPLIES = {"en": "This is fake, please check before sharing.", "hi": "यह खबर झूठी है, कृपया जांच करें।",
           "mr": "ही बातमी खोटी आहे, कृपया तपासा."}

CARD = """<html><head><meta charset="utf-8"><style>
body{{margin:0;background:{bg};font-family:'Noto Sans','Noto Sans Devanagari',sans-serif;color:{fg}}}
.card{{width:520px;margin:16px;padding:16px;border:1px solid {line};border-radius:12px;background:{card}}}
.hd{{display:flex;align-items:center;gap:10px}} .av{{width:40px;height:40px;border-radius:50%;background:#7a9}}
.nm{{font-weight:700}} .meta{{color:{muted};font-size:13px}}
.tx{{margin:12px 0;font-size:16px;line-height:1.45;white-space:pre-wrap}}
.bar{{display:flex;gap:28px;color:{muted};font-size:13px;border-top:1px solid {line};padding-top:8px}}
.rp{{margin-top:10px;font-size:14px;background:{replybg};padding:8px;border-radius:8px}}
</style></head><body><div class="card"><div class="hd"><div class="av"></div><div>
<div class="nm">{name}</div><div class="meta">{handle} · {date}</div></div></div>
<div class="tx">{text}</div>
<div class="bar"><span>♥ {likes}</span><span>💬 {comments}</span><span>↗ Share</span></div>
<div class="rp"><b>{rname}</b> {reply}</div></div></body></html>"""


def render(n: int) -> None:
    from playwright.sync_api import sync_playwright

    rows = [json.loads(line) for line in open(HERE.parent / "eval" / "factchecks.jsonl")]
    rng = random.Random(0)
    picked = []
    for lang in ("en", "hi", "mr"):
        rs = sorted((r for r in rows if r["language"] == lang), key=lambda r: -len(r["input_text"]))[: n * 3]
        picked += rng.sample(rs, n)
    IMG.mkdir(parents=True, exist_ok=True)
    truth = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/opt/pw-browsers/chromium")  # pre-installed build
        page = browser.new_page(device_scale_factor=1.5)
        for i, r in enumerate(picked):
            dark = i % 2 == 1
            handle = f"@{rng.choice(['news', 'india', 'sach', 'khabar', 'mumbai'])}_{rng.randrange(100, 9999)}"
            date = DATES[i % len(DATES)]
            page.set_content(CARD.format(
                bg="#000" if dark else "#f0f2f5", card="#16181c" if dark else "#fff", fg="#e7e9ea" if dark else "#111",
                muted="#8b98a5" if dark else "#65676b", line="#2f3336" if dark else "#e4e6eb",
                replybg="#202327" if dark else "#f0f2f5", name=f"Viral Updates {i}", handle=handle, date=date,
                text=html.escape(r["input_text"]), likes=f"{rng.randrange(1, 99)}.{rng.randrange(10)}K",
                comments=rng.randrange(10, 900), rname="Reader", reply=REPLIES[r["language"]]))
            path = IMG / f"{r['id']}.png"
            page.locator(".card").screenshot(path=str(path))
            truth.append({"id": r["id"], "language": r["language"], "path": str(path), "post_text": r["input_text"],
                          "account_handle": handle, "post_date": date, "reply": REPLIES[r["language"]]})
        browser.close()
    (IMG / "truth.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in truth))
    print(f"rendered {len(truth)} screenshots")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def cer(hyp: str, ref: str) -> float:
    a, b = norm(hyp), norm(ref)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(1, len(b))


async def read_all(model: str, truth: list[dict], budget: Budget) -> list[dict]:
    async with httpx.AsyncClient(timeout=180) as client:
        llm = OpenRouterJSON(model, budget, client)

        async def one(t):
            b64 = base64.b64encode(Path(t["path"]).read_bytes()).decode()
            parts = [{"type": "text", "text": "Read this social-media screenshot."},
                     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]
            try:
                return await llm.generate_json(prompts.VISION_SYSTEM, parts, prompts.VISION_SCHEMA)
            except SystemExit:
                raise
            except Exception as exc:
                return {"error": str(exc)[:200]}

        return await asyncio.gather(*(one(t) for t in truth))


def score(truth: list[dict], outs: list[dict]) -> dict:
    ok = [(t, o) for t, o in zip(truth, outs) if "error" not in o]

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    res = {"n": len(ok), "errors": len(truth) - len(ok),
           "post_text CER (lower=better)": mean([cer(o.get("post_text", ""), t["post_text"]) for t, o in ok]),
           "post_text exact": mean([norm(o.get("post_text")) == norm(t["post_text"]) for t, o in ok]),
           "handle exact": mean([norm(o.get("account_handle")).lstrip("@") == t["account_handle"].lstrip("@") for t, o in ok]),
           "date exact": mean([norm(o.get("post_date")) == t["post_date"] for t, o in ok]),
           "reply leaked into post_text": mean([norm(t["reply"])[:20] in norm(o.get("post_text")) for t, o in ok]),
           "not transcribed (translated)": mean([has_devanagari(t["post_text"]) and not has_devanagari(o.get("post_text", ""))
                                                 for t, o in ok])}
    for lang in ("en", "hi", "mr"):
        res[f"CER {lang}"] = mean([cer(o.get("post_text", ""), t["post_text"]) for t, o in ok if t["language"] == lang])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--models", default="")
    ap.add_argument("--max-usd", type=float, default=0.5)
    args = ap.parse_args()
    if args.render:
        render(args.n)
    truth = [json.loads(line) for line in open(IMG / "truth.jsonl")]
    budget = Budget(args.max_usd)
    OUT.mkdir(exist_ok=True)
    out_file = OUT / "vision.json"
    results = json.loads(out_file.read_text()) if out_file.exists() else {}
    for m in filter(None, args.models.split(",")):
        before = budget.spent
        outs = asyncio.run(read_all(m, truth, budget))
        results[m] = {"score": score(truth, outs), "usd": round(budget.spent - before, 4), "raw": outs}
        print(m, results[m]["score"], f"${budget.spent - before:.4f}", flush=True)
        out_file.write_text(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
