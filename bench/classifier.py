"""Classifier benchmark: Jev (production) vs LLM judges, through the production judging code.

Every candidate answers the production prompts (Jev questions / LLMClassifier prompts in prompts.py), and its
answers go through the production rules: resolve_type (claim type) and judge.judge_claim (same-event gate,
relevance filter, evidence guards, decide_status). Inputs come from bench.prepare.

Tests
  claim type   real fact-checked rows must stay checkable; bench/data/not_checkable.jsonl must not.
  present      passages = the row's own fact-check chunk + its 2 most similar chunks from other fact-checks.
               Expected: the row's label (CONTRADICTED / MISLEADING_CONTEXT / CONFIRMED).
  absent       passages = only those 2 look-alike chunks. Expected: UNVERIFIED (rule 1, lesson 1).

    python -m bench.classifier --models typesafe/jev-1.13,google/gemini-3.8-flash --n 70 --max-usd 2
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import httpx

from app.adapters import prompts
from app.adapters.http import ApiError, request_json
from app.adapters.jev import JevClassifier
from app.adapters.llm_judge import LLMClassifier
from app.config import get_settings
from app.models.schemas import Claim, ClaimType, Passage, RawClaim, Stance, Status
from app.pipeline.extract import resolve_type
from app.pipeline.judge import Thresholds, judge_claim
from app.sources import load_whitelist

HERE = Path(__file__).parent
DATA, CACHE, OUT = HERE / "data", HERE / "cache", HERE / "results"
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
FAILS = {Status.CONTRADICTED, Status.MISLEADING_CONTEXT}
UNVERIFIED = {Status.UNVERIFIED_EVIDENCE_MISSING, Status.UNVERIFIED_TOO_EARLY}


class Budget:
    def __init__(self, max_usd: float):
        self.max_usd, self.spent = max_usd, 0.0

    def add(self, cost: float) -> None:
        self.spent += cost or 0.0
        if self.spent > self.max_usd:
            raise SystemExit(f"budget exceeded: ${self.spent:.3f} > ${self.max_usd}")


class Cache:
    """Append-only JSONL cache so a re-run never pays twice."""

    def __init__(self, name: str):
        CACHE.mkdir(exist_ok=True)
        self.path = CACHE / (re.sub(r"[^\w.-]", "_", name) + ".jsonl")
        self.data = {}
        if self.path.exists():
            for line in self.path.open():
                d = json.loads(line)
                self.data[d["k"]] = d["v"]

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def put(self, k: str, v) -> None:
        self.data[k] = v
        with self.path.open("a") as f:
            f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")


class OpenRouterJSON:
    """Stands in for GeminiClient (generate_json) so LLMClassifier runs unchanged on any OpenRouter model."""

    def __init__(self, model: str, budget: Budget, client: httpx.AsyncClient, reasoning: str = "low"):
        self.model, self.budget, self.client, self.reasoning = model, budget, client, reasoning
        self.cache = Cache(model)
        self.sem = asyncio.Semaphore(8)
        self.model_version = model

    async def generate_json(self, system: str, parts, schema: dict, temperature: float = 0.0):
        k = Cache.key(self.model, system, parts, schema)
        if k in self.cache.data:
            return self.cache.data[k]
        body = {
            "model": self.model, "temperature": temperature, "usage": {"include": True},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": parts}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "strict": False, "schema": schema}},
        }
        if self.reasoning:
            body["reasoning"] = {"effort": self.reasoning}
        async with self.sem:
            data = await request_json(self.client, "POST", OPENROUTER, provider=self.model,
                                      headers={"Authorization": "Bearer injected-by-proxy"}, json=body, retries=3)
        self.budget.add((data.get("usage") or {}).get("cost", 0.0))
        text = data["choices"][0]["message"].get("content") or ""
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
        out = json.loads(text)
        self.cache.put(k, out)
        return out


class CachedJev(JevClassifier):
    def __init__(self, model: str, budget: Budget, client: httpx.AsyncClient):
        super().__init__("injected-by-proxy", model, client=client)
        self.budget, self.cache, self.sem = budget, Cache(model), asyncio.Semaphore(8)

    async def _ask(self, state: dict, questions: dict) -> dict:
        k = Cache.key(self.model, state, questions)
        if k in self.cache.data:
            return self.cache.data[k]
        async with self.sem:
            data = await request_json(self._client, "POST", self.url, provider="jev", headers=self._headers,
                                      json={"model": self.model, "state": state, "questions": questions}, retries=3)
        self.budget.add((data.get("usage") or {}).get("cost", 0.0))
        answers = data.get("answers") or {}
        if set(questions) - set(answers):
            raise ApiError("jev", 200, "missing answers")
        self.cache.put(k, answers)
        return answers


def make_classifier(model: str, budget: Budget, client: httpx.AsyncClient):
    if model.startswith("typesafe/jev"):
        return CachedJev(model, budget, client)
    return LLMClassifier(OpenRouterJSON(model, budget, client))


def to_passage(d: dict, pid: str, whitelist) -> Passage:
    src = whitelist.lookup(d["url"])
    return Passage(id=pid, source_id=src.domain if src else "unknown", url=d["url"],
                   publisher=src.name if src else "unknown", tier=src.tier if src else 2,
                   kind=src.kind if src else "factchecker", title=d["title"] or None, text=d["text"])


def sample(pairs: list[dict], n: int) -> list[dict]:
    """Deterministic, stratified by language; news (CONFIRMED) rows kept in proportion."""
    rng = random.Random(0)
    by_lang = defaultdict(list)
    for p in pairs:
        by_lang[p["language"]].append(p)
    out = []
    for lang, ps in sorted(by_lang.items()):
        rng.shuffle(ps)
        out += ps[: max(1, round(n * len(ps) / len(pairs)))]
    return out


async def run_model(model: str, pairs: list[dict], nc: list[dict], text_en: dict, budget: Budget) -> dict:
    s = get_settings()
    whitelist = load_whitelist(str(s.effective_sources_file))
    t = Thresholds(confidence=s.confidence_threshold, passage_relevance=s.passage_relevance_threshold,
                   too_early_window_hours=s.too_early_window_hours, same_event=s.same_event_threshold)
    async with httpx.AsyncClient(timeout=120) as client:
        clf = make_classifier(model, budget, client)

        async def ctype(text_original: str, en: str) -> str:
            pred, prob = await clf.claim_type(RawClaim(text_original=text_original, text_en=en))
            return resolve_type(pred, prob, s.claim_type_threshold)[0].value

        async def one(p: dict) -> dict:
            claim_type = await ctype(p["text_original"], p["text_en"])
            claim = Claim(id="c1", text_original=p["text_original"], text_en=p["text_en"],
                          type=ClaimType.CHECKABLE, type_confidence=1.0)
            own = to_passage(p["own"], "own", whitelist)
            negs = [to_passage(d, f"neg{i}", whitelist) for i, d in enumerate(p["negatives"])]
            present = await judge_claim(claim, [own, *negs], clf, None, t)
            absent = await judge_claim(claim, negs, clf, None, t)
            j = {x.passage_id: x for x in present.judgments}
            return {
                "id": p["id"], "language": p["language"], "label": p["label"], "claim_type": claim_type,
                "own_same_event": j["own"].same_event, "own_stance": j["own"].stance.value,
                "neg_counted": [x.stance != Stance.IRRELEVANT and x.probability >= t.passage_relevance
                                for k, x in j.items() if k != "own"],
                "present": present.status.value, "present_conf": present.confidence,
                "absent": absent.status.value,
            }

        async def safe(coro_fn, *a):
            try:
                return await coro_fn(*a)
            except SystemExit:
                raise
            except Exception as exc:  # a row that errors is reported, not silently dropped
                return {"error": f"{type(exc).__name__}: {exc}"[:200]}

        rows = await asyncio.gather(*(safe(one, p) for p in pairs))
        nc_types = await asyncio.gather(*(safe(ctype, n["text"], text_en[n["id"]]) for n in nc))
    return {"rows": rows, "nc": [{"id": n["id"], "type": n["type"], "language": n["language"], "pred": r}
                                 for n, r in zip(nc, nc_types)]}


def score(res: dict) -> dict:
    rows = [r for r in res["rows"] if "error" not in r]
    fails = [r for r in rows if Status(r["label"]) in FAILS]
    conf = [r for r in rows if r["label"] == Status.CONFIRMED.value]

    def rate(xs, pred):
        return round(sum(1 for x in xs if pred(x)) / len(xs), 3) if xs else None

    def present_ok(r):
        return r["present"] == r["label"]

    def fails_ok(r):  # C vs M is mostly fact-checker disagreement (lessons 8, 16)
        return Status(r["present"]) in FAILS if Status(r["label"]) in FAILS else present_ok(r)

    def direction_wrong(r):
        p, lab = Status(r["present"]), Status(r["label"])
        return (lab in FAILS and p == Status.CONFIRMED) or (lab == Status.CONFIRMED and p in FAILS)

    def own_direction_right(r):
        want = Stance.SUPPORTS.value if r["label"] == Status.CONFIRMED.value else Stance.CONTRADICTS.value
        return r["own_stance"] == want

    nc = [n for n in res["nc"] if isinstance(n["pred"], str)]
    out = {
        "n_rows": len(rows), "errors": len(res["rows"]) - len(rows),
        "claim_type: real rows kept checkable": rate(rows, lambda r: r["claim_type"] == "checkable"),
        "claim_type: not-checkable caught": rate(nc, lambda n: n["pred"] != "checkable"),
        "own passage: same-event >= 0.6": rate(rows, lambda r: (r["own_same_event"] or 0) >= 0.6),
        "own passage: right stance (after gate)": rate(rows, own_direction_right),
        "look-alike passages counted as evidence": rate([c for r in rows for c in r["neg_counted"]], bool),
        "present: exact status": rate(rows, present_ok),
        "present: fails-direction correct": rate(rows, fails_ok),
        "present: abstained": rate(rows, lambda r: Status(r["present"]) in UNVERIFIED),
        "present: direction-wrong (dangerous)": rate(rows, direction_wrong),
        "present: CONFIRMED rows correct": rate(conf, present_ok),
        "absent: abstained (correct)": rate(rows, lambda r: Status(r["absent"]) in UNVERIFIED),
        "absent: definitive verdict from wrong evidence": rate(rows, lambda r: Status(r["absent"]) not in UNVERIFIED),
    }
    per_lang = {}
    for lang in sorted({r["language"] for r in rows}):
        rs = [r for r in rows if r["language"] == lang]
        per_lang[lang] = {"n": len(rs), "fails-direction correct": rate(rs, fails_ok),
                          "absent abstained": rate(rs, lambda r: Status(r["absent"]) in UNVERIFIED),
                          "checkable kept": rate(rs, lambda r: r["claim_type"] == "checkable")}
    out["per_language"] = per_lang
    out["present status counts"] = dict(Counter(r["present"] for r in rows))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--n", type=int, default=70, help="rows per model (0 = all)")
    ap.add_argument("--max-usd", type=float, default=2.0, help="stop if this run's OpenRouter spend exceeds it")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    pairs = [json.loads(line) for line in open(DATA / "pairs.jsonl")]
    pairs = sample(pairs, args.n) if args.n else pairs
    nc = [json.loads(line) for line in open(DATA / "not_checkable.jsonl")]
    text_en = json.loads((DATA / "translations.json").read_text())
    text_en.update({n["id"]: n["text"] for n in nc if n["language"] == "en"})
    budget = Budget(args.max_usd)
    OUT.mkdir(exist_ok=True)
    out_file = OUT / "classifier.json"
    results = json.loads(out_file.read_text()) if out_file.exists() else {}
    for model in args.models.split(","):
        before = budget.spent
        res = asyncio.run(run_model(model, pairs, nc, text_en, budget))
        key = model + args.tag
        results[key] = {"score": score(res), "usd": round(budget.spent - before, 4), "raw": res}
        out_file.write_text(json.dumps(results, indent=1, ensure_ascii=False))
        print(key, json.dumps(results[key]["score"], indent=1), f"${budget.spent - before:.4f}", flush=True)


if __name__ == "__main__":
    main()
