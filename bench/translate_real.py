"""Translation on the project's own data: the real Hindi/Marathi eval posts -> English.

Baseline: the production translation (Sarvam mayura:v1, made by bench.prepare, translations.json).
Candidates: OpenRouter LLMs with the production LLM translation prompt.
Metric: MetricX-24 hybrid (google/metricx-24-hybrid-large-v2p6) in reference-free QE mode, as its repo's
predict.py builds the input ("source: ... candidate: ..."; EOS removed). Score 0-25, lower is better.
Needs the official code: git clone https://github.com/google-research/metricx (pass --metricx-repo).

    python -m bench.translate_real --metricx-repo DIR --systems google/gemini-3.8-flash,... [--max-usd 0.5]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

import httpx

from app.adapters import prompts
from app.text import has_devanagari
from bench.classifier import Budget, Cache, OpenRouterJSON

HERE = Path(__file__).parent
DATA, OUT = HERE / "data", HERE / "results"
BASE = "sarvam:mayura:v1 (production)"


async def llm_translate(system: str, items: list[dict], budget: Budget) -> list[str]:
    cache = Cache("translate-" + system)
    async with httpx.AsyncClient(timeout=120) as client:
        llm = OpenRouterJSON(system, budget, client)

        async def one(it):
            src = "hi" if it["language"] == "hi-Latn" else it["language"]
            k = Cache.key(system, it["text"], src)
            if k not in cache.data:
                data = await llm.generate_json(
                    prompts.TRANSLATE_SYSTEM, f"Source language: {src}\nTarget language: en\n\nTEXT:\n{it['text']}",
                    prompts.TRANSLATE_SCHEMA)
                cache.put(k, str(data.get("translation", "")).strip())
            return cache.data[k]

        return await asyncio.gather(*(one(it) for it in items))


class MetricX:
    def __init__(self, repo: Path, model="google/metricx-24-hybrid-large-v2p6", tokenizer="google/mt5-large"):
        import torch
        import transformers

        sys.path.insert(0, str(repo))
        from metricx24.models import MT5ForRegression

        torch.set_num_threads(4)
        self.torch = torch
        self.tok = transformers.AutoTokenizer.from_pretrained(tokenizer)
        self.model = MT5ForRegression.from_pretrained(model, torch_dtype=torch.float32).eval()

    def score(self, source: str, hyp: str) -> float:
        enc = self.tok("source: " + source + " candidate: " + hyp, max_length=1536, truncation=True)
        ids = self.torch.tensor([enc["input_ids"][:-1]])
        mask = self.torch.tensor([enc["attention_mask"][:-1]])
        with self.torch.no_grad():
            out = self.model(input_ids=ids, attention_mask=mask)
        return float(out.predictions.reshape(-1)[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metricx-repo", required=True, type=Path)
    ap.add_argument("--systems", required=True)
    ap.add_argument("--max-usd", type=float, default=0.5)
    args = ap.parse_args()
    translations = json.loads((DATA / "translations.json").read_text())
    rows = [json.loads(line) for line in open(HERE.parent / "eval" / "factchecks.jsonl")]
    items = [{"id": r["id"], "language": r["language"], "text": r["input_text"], BASE: translations[r["id"]]}
             for r in rows if r["language"] != "en" and r["id"] in translations]
    budget = Budget(args.max_usd)
    systems = args.systems.split(",")
    for s in systems:
        for it, tr in zip(items, asyncio.run(llm_translate(s, items, budget))):
            it[s] = tr
        print(s, "translated", flush=True)
    names = [BASE, *systems]
    scores_file = DATA / "metricx_scores.json"
    scores = json.loads(scores_file.read_text()) if scores_file.exists() else {}
    mx = MetricX(args.metricx_repo)
    for it in items:
        for n in names:
            k = Cache.key(it["text"], it[n])
            if k not in scores:
                scores[k] = mx.score(it["text"], it[n])
        scores_file.write_text(json.dumps(scores))
    rng = random.Random(1)
    res = {}
    for n in names:
        per = [scores[Cache.key(it["text"], it[n])] for it in items]
        base = [scores[Cache.key(it["text"], it[BASE])] for it in items]
        row = {"metricx_qe_mean (lower=better)": round(sum(per) / len(per), 3),
               "untranslated (Devanagari left)": sum(has_devanagari(it[n]) for it in items)}
        for lang in ("hi", "mr"):
            xs = [scores[Cache.key(it["text"], it[n])] for it in items if it["language"] == lang]
            row[f"mean {lang}"] = round(sum(xs) / len(xs), 3)
        if n != BASE:
            d = [a - b for a, b in zip(per, base)]
            boots = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(2000))
            row["diff vs production 95%CI (negative = better)"] = (round(boots[50], 3), round(boots[1949], 3))
            row["better than production on"] = f"{sum(x < -0.5 for x in d)}/{len(d)} posts (>0.5 pt)"
            row["worse than production on"] = f"{sum(x > 0.5 for x in d)}/{len(d)} posts (>0.5 pt)"
        res[n] = row
        print(n, row, flush=True)
    res["_n_posts"] = len(items)
    res["_spend_usd"] = round(budget.spent, 4)
    OUT.mkdir(exist_ok=True)
    (OUT / "translate_real.json").write_text(json.dumps(res, indent=1))
    (OUT / "translate_real_outputs.json").write_text(json.dumps(items, ensure_ascii=False, indent=0))


if __name__ == "__main__":
    main()
