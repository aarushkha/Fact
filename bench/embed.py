"""Embedder benchmark: does the claim retrieve its own fact-check article?

Corpus: every chunk (`title\\nchunk`, as the crawler embeds it) of every fetched review article.
Queries: the claim's English text (what the pipeline embeds) and, separately, the original post text.
A query hits when a chunk of its own article is in the top k (k = RETRIEVAL_TOP_K = 8). Other articles
about the same rumour count as misses for every model alike, so absolute numbers are a lower bound.
Dense retrieval only (production adds Postgres FTS through RRF).

Baseline: production BGE-M3 vectors from bench.prepare (bge_m3.npz). Others via OpenRouter /embeddings,
each with its model card's query/document format.

    python -m bench.embed --models qwen/qwen3-embedding-8b,google/gemini-embedding-2 --max-usd 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np

from app.adapters.http import request_json
from bench.classifier import Budget

HERE = Path(__file__).parent
DATA, OUT = HERE / "data", HERE / "results"
URL = "https://openrouter.ai/api/v1/embeddings"
TASK = "Given a social media claim, retrieve fact-check or news passages about the same event"

# query/document formats from each model card
FORMATS = {
    "qwen/": (lambda q: f"Instruct: {TASK}\nQuery:{q}", lambda d: d),
    "intfloat/multilingual-e5": (lambda q: f"query: {q}", lambda d: f"passage: {d}"),
}


def fmt(model: str):
    for prefix, f in FORMATS.items():
        if model.startswith(prefix):
            return f
    return (lambda q: q), (lambda d: d)


async def embed_api(model: str, texts: list[str], tag: str, budget: Budget, bs: int = 32) -> np.ndarray:
    """Unit-normalised vectors, cached per (model, tag) so a re-run never pays twice."""
    cache_file = DATA / "emb_cache" / (model.replace("/", "__").replace(":", "_") + f".{tag}.npy")
    if cache_file.exists():
        return np.load(cache_file)
    sem = asyncio.Semaphore(4)
    out: list = [None] * len(texts)
    async with httpx.AsyncClient(timeout=180) as client:
        async def batch(i):
            async with sem:
                data = await request_json(client, "POST", URL, provider=model, retries=4,
                                          headers={"Authorization": "Bearer injected-by-proxy"},
                                          json={"model": model, "input": texts[i: i + bs]})
            budget.add((data.get("usage") or {}).get("cost", 0.0))
            for d in data["data"]:
                out[i + d["index"]] = d["embedding"]
        await asyncio.gather(*(batch(i) for i in range(0, len(texts), bs)))
    v = np.array(out, dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    cache_file.parent.mkdir(exist_ok=True)
    np.save(cache_file, v)
    return v


def evaluate(q: np.ndarray, c: np.ndarray, rows: list[str], owner: np.ndarray, lang: dict, k: int = 8) -> dict:
    sims = q @ c.T
    hits, rr = [], []
    for i, rid in enumerate(rows):
        order = np.argsort(-sims[i])
        own = np.where(owner[order] == rid)[0]
        first = int(own[0]) if len(own) else 10**9
        hits.append(first < k)
        rr.append(1 / (first + 1) if first < 100 else 0.0)
    by = defaultdict(list)
    for h, rid in zip(hits, rows):
        by[lang[rid]].append(h)
    return {f"recall@{k}": round(float(np.mean(hits)), 3), "mrr@100": round(float(np.mean(rr)), 3),
            **{f"recall@{k} {lg}": round(float(np.mean(v)), 3) for lg, v in sorted(by.items())}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="")
    ap.add_argument("--max-usd", type=float, default=1.0)
    args = ap.parse_args()
    corpus = json.loads((DATA / "corpus.json").read_text())
    reviews = {d["id"]: d for d in map(json.loads, open(DATA / "reviews.jsonl")) if d.get("chunks")}
    pairs = {p["id"]: p for p in map(json.loads, open(DATA / "pairs.jsonl"))}
    rows = corpus["rows"]
    owner = np.array([rid for rid, _ in corpus["chunks"]])
    docs = []
    for rid, i in corpus["chunks"]:
        title = reviews[rid].get("title") or ""
        ch = reviews[rid]["chunks"][i]
        docs.append(f"{title}\n{ch}" if title else ch)
    lang = {rid: pairs[rid]["language"] for rid in rows}
    q_en = [pairs[rid]["text_en"] for rid in rows]
    q_orig = [pairs[rid]["text_original"] for rid in rows]

    OUT.mkdir(exist_ok=True)
    out_file = OUT / "embed.json"
    results = json.loads(out_file.read_text()) if out_file.exists() else {}
    z = np.load(DATA / "bge_m3.npz")
    results["BAAI/bge-m3 (production, local)"] = {
        "query=text_en": evaluate(z["q_en"], z["chunks"], rows, owner, lang),
        "query=original": evaluate(z["q_orig"], z["chunks"], rows, owner, lang),
    }
    print("bge-m3 local", results["BAAI/bge-m3 (production, local)"], flush=True)
    budget = Budget(args.max_usd)
    for model in filter(None, args.models.split(",")):
        fq, fd = fmt(model)
        before = budget.spent
        try:
            c = asyncio.run(embed_api(model, [fd(d) for d in docs], "docs", budget))
            qe = asyncio.run(embed_api(model, [fq(t) for t in q_en], "q_en", budget))
            qo = asyncio.run(embed_api(model, [fq(t) for t in q_orig], "q_orig", budget))
        except SystemExit:
            raise
        except Exception as exc:  # report and continue with the next model
            print(model, "failed:", exc, flush=True)
            continue
        results[model] = {"query=text_en": evaluate(qe, c, rows, owner, lang),
                          "query=original": evaluate(qo, c, rows, owner, lang),
                          "usd": round(budget.spent - before, 4)}
        print(model, results[model], flush=True)
        out_file.write_text(json.dumps(results, indent=1))
    out_file.write_text(json.dumps(results, indent=1))



if __name__ == "__main__":
    main()
