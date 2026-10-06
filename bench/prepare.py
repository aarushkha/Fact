"""Shared inputs for the model benchmarks (run once; everything is cached under bench/data/).

1. text_en for every real row and every not-checkable item, via the production translator
   (Sarvam mayura:v1), as the pipeline would produce it.
2. BGE-M3 (production embedder, pinned) vectors for claims and for `title\\nchunk` of every review chunk.
3. Judge pairs per real row: its own article's chunk most similar to the claim (same event) and the two most
   similar chunks from *other* articles (hard negatives; lesson 1). Another article whose own claim is close
   to this one (cosine >= 0.75) may debunk the same rumour, so it is never used as a negative.

    python -m bench.prepare
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from app.adapters.local_models import BGEM3Embedder
from app.adapters.sarvam import SarvamTranslator
from app.config import get_settings

DATA = Path(__file__).parent / "data"
EVAL = Path(__file__).parent.parent / "eval"
SAME_RUMOUR = 0.75


def load_rows() -> list[dict]:
    rows = []
    for name in ("factchecks.jsonl", "news_confirmed.jsonl"):
        rows += [json.loads(line) for line in open(EVAL / name) if line.strip()]
    return rows


async def translate_all(items: list[tuple[str, str, str]]) -> dict[str, str]:
    """items: (key, text, language). Returns key -> English (cached in translations.json)."""
    path = DATA / "translations.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = [(k, t, lang) for k, t, lang in items if k not in cache and lang != "en"]
    tr = SarvamTranslator("injected-by-proxy", get_settings().sarvam_translate_model)
    sem = asyncio.Semaphore(4)

    async def one(k, t, lang):
        async with sem:
            src = "hi" if lang == "hi-Latn" else lang
            try:
                cache[k] = await tr.translate(t, src, "en")
            except Exception as exc:  # keep going; a missing translation is reported below
                print("translate failed", k, exc)

    await asyncio.gather(*(one(*i) for i in todo))
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=0))
    for k, t, lang in items:
        if lang == "en":
            cache[k] = t
    return cache


def main() -> None:
    s = get_settings()
    rows = load_rows()
    reviews = {d["id"]: d for d in map(json.loads, open(DATA / "reviews.jsonl")) if d.get("chunks")}
    nc = [json.loads(line) for line in open(DATA / "not_checkable.jsonl")]
    items = [(r["id"], r["input_text"], r["language"]) for r in rows if "input_text" in r]
    items += [(n["id"], n["text"], n["language"]) for n in nc]
    text_en = asyncio.run(translate_all(items))
    missing = [k for k, *_ in items if k not in text_en]
    print(f"translations: {len(items) - len(missing)}/{len(items)} (missing {missing[:5]})")

    real = [r for r in rows if r["id"] in reviews and r["id"] in text_en]
    chunks = [(r["id"], i, reviews[r["id"]].get("title") or "", ch)
              for r in real for i, ch in enumerate(reviews[r["id"]]["chunks"])]
    emb_path = DATA / "bge_m3.npz"
    if emb_path.exists():
        z = np.load(emb_path)
        q_en, q_orig, c = z["q_en"], z["q_orig"], z["chunks"]
    else:
        e = BGEM3Embedder(s.embedder_model, s.embedder_revision, max_seq_length=s.embedder_max_seq_length)
        q_en = np.array(asyncio.run(e.embed([text_en[r["id"]] for r in real])))
        q_orig = np.array(asyncio.run(e.embed([r["input_text"] for r in real])))
        c = np.array(asyncio.run(e.embed([f"{t}\n{ch}" if t else ch for _, _, t, ch in chunks])))
        np.savez(emb_path, q_en=q_en, q_orig=q_orig, chunks=c)
    (DATA / "corpus.json").write_text(json.dumps(
        {"rows": [r["id"] for r in real], "chunks": [[rid, i] for rid, i, _, _ in chunks]}))

    owner = np.array([rid for rid, *_ in chunks])
    claim_sim = q_en @ q_en.T
    sims = q_en @ c.T
    pairs = []
    for qi, r in enumerate(real):
        own = np.where(owner == r["id"])[0]
        best_own = own[np.argmax(sims[qi, own])]
        same_rumour = {real[j]["id"] for j in np.where(claim_sim[qi] >= SAME_RUMOUR)[0]}
        neg = [j for j in np.argsort(-sims[qi]) if owner[j] not in same_rumour][:2]

        def passage(j, rid=None):
            rid_, i, title, ch = chunks[j]
            return {"row": rid_, "chunk": i, "title": title, "text": ch, "url": reviews[rid_]["url"],
                    "sim": round(float(sims[qi, j]), 3)}

        pairs.append({
            "id": r["id"], "language": r["language"], "label": r["expected_status"],
            "publisher": r.get("publisher"), "text_original": r["input_text"], "text_en": text_en[r["id"]],
            "own": passage(best_own), "negatives": [passage(j) for j in neg],
        })
    with open(DATA / "pairs.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"pairs: {len(pairs)}; chunks: {len(chunks)}")


if __name__ == "__main__":
    main()
