"""Translation benchmark, Hindi/Marathi -> English, the direction the pipeline uses (normalize, to_english).

Data: FLORES-200 devtest (haoranxu/FLORES-200 hi-en, mr-en), a fixed random sample per language.
Systems: Sarvam (production adapter, any model) and OpenRouter LLMs with the production LLM translation prompt
(prompts.TRANSLATE_SYSTEM, as GeminiLLM.translate sends it).
Metric: chrF++ (sacrebleu), with a paired bootstrap 95% interval for the difference to the baseline.
COMET (wmt22-comet-da) is added by bench.comet_score from the saved outputs.

    python -m bench.translate --data DIR --systems sarvam:mayura:v1,google/gemini-3.8-flash --n 80
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

import httpx
import pandas as pd
import sacrebleu

from app.adapters import prompts
from app.adapters.sarvam import SarvamTranslator
from bench.classifier import Budget, Cache, OpenRouterJSON

OUT = Path(__file__).parent / "results"


def load(data: Path, n: int) -> dict[str, list[tuple[str, str]]]:
    sets = {}
    for lang in ("hi", "mr"):
        df = pd.read_parquet(data / "flores" / f"{lang}-en.parquet")
        col = df.columns[0]
        pairs = [(r[lang], r["en"]) for r in df[col]]
        sets[lang] = random.Random(0).sample(pairs, n)
    return sets


async def translate_all(system: str, items: list[tuple[str, str]], budget: Budget) -> list[str]:
    cache = Cache("translate-" + system)
    async with httpx.AsyncClient(timeout=120) as client:
        if system.startswith("sarvam:"):
            tr = SarvamTranslator("injected-by-proxy", system.split(":", 1)[1], client=client)
            sem = asyncio.Semaphore(2)

            async def one(text, lang):
                k = Cache.key(system, text, lang)
                if k not in cache.data:
                    async with sem:
                        cache.put(k, await tr.translate(text, lang, "en"))
                return cache.data[k]
        else:
            llm = OpenRouterJSON(system, budget, client)

            async def one(text, lang):
                k = Cache.key(system, text, lang)
                if k not in cache.data:
                    data = await llm.generate_json(
                        prompts.TRANSLATE_SYSTEM, f"Source language: {lang}\nTarget language: en\n\nTEXT:\n{text}",
                        prompts.TRANSLATE_SCHEMA)
                    cache.put(k, str(data.get("translation", "")).strip())
                return cache.data[k]

        return await asyncio.gather(*(one(t, lang) for t, lang in items))


def chrf(hyps: list[str], refs: list[str]) -> float:
    return sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score


def bootstrap_diff(h1, h0, refs, iters=1000) -> tuple[float, float]:
    rng = random.Random(1)
    n = len(refs)
    diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        r = [refs[i] for i in idx]
        diffs.append(chrf([h1[i] for i in idx], r) - chrf([h0[i] for i in idx], r))
    diffs.sort()
    return round(diffs[int(0.025 * iters)], 2), round(diffs[int(0.975 * iters)], 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--systems", required=True)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--max-usd", type=float, default=1.0)
    args = ap.parse_args()
    sets = load(args.data, args.n)
    budget = Budget(args.max_usd)
    systems = args.systems.split(",")
    outputs = {}
    for system in systems:
        outputs[system] = {}
        for lang, pairs in sets.items():
            outputs[system][lang] = asyncio.run(translate_all(system, [(s, lang) for s, _ in pairs], budget))
            print(system, lang, "done", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "translate_outputs.json").write_text(json.dumps(
        {"sets": sets, "outputs": outputs}, ensure_ascii=False, indent=0))
    base = systems[0]
    res = {}
    for system in systems:
        res[system] = {}
        for lang, pairs in sets.items():
            refs = [r for _, r in pairs]
            hyps = outputs[system][lang]
            row = {"chrF++": round(chrf(hyps, refs), 2)}
            if system != base:
                row["diff_vs_" + base + " 95%CI"] = bootstrap_diff(hyps, outputs[base][lang], refs)
            res[system][lang] = row
        print(system, res[system], flush=True)
    res["_spend_usd"] = round(budget.spent, 4)
    (OUT / "translate.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
