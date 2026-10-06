"""NLI benchmark in the shape production uses it: an English sentence (hypothesis) checked against the
original passage (premise) in English, Hindi or Marathi.

Data: XNLI test (English, facebook/xnli) and IndicXNLI forward test (hi, mr; Divyanshu/indicxnli), which is
XNLI test translated, row-aligned with the English set. Labels 0 entail / 1 neutral / 2 contradict.
"long" settings append 3 unrelated premises (same language) around the real one, as a multi-sentence
passage would (lesson 4: long premises under-score).

Only P(entailment) is used downstream (write.py), so we score it as a binary detector:
precision at the production threshold (0.5) = how often a sentence that passes is really supported (rule 2),
recall = how many supported sentences survive, plus ROC-AUC (threshold-free).

    python -m bench.nli --data DIR --models m1,m2 [--n 600]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "results"


def load(data: Path, n: int, seed: int = 0) -> dict[str, list[tuple[str, str, int]]]:
    x = pd.read_parquet(data / "xnli" / "test.parquet")
    en_p = [r["en"] for r in x["premise"]]
    en_h = [dict(zip(r["language"], r["translation"]))["en"] for r in x["hypothesis"]]
    labels = list(x["label"])
    prem = {"en": en_p}
    for lang in ("hi", "mr"):
        rows = json.load(open(data / "indicxnli" / f"forward_test_{lang}.json"))["test"]
        assert len(rows) == len(en_p), (lang, len(rows), len(en_p))
        assert all(r["label"] == lab for r, lab in zip(rows, labels)), f"{lang} labels not aligned"
        prem[lang] = [r["premise"] for r in rows]
    rng = random.Random(seed)
    # XNLI repeats each premise 3x (one per label); sample distinct premises, keep the label balance
    idx = rng.sample(range(len(labels)), n)
    sets = {}
    for lang in ("en", "hi", "mr"):
        sets[f"{lang}->en"] = [(prem[lang][i], en_h[i], labels[i]) for i in idx]
        long = []
        for i in idx:
            others = rng.sample(range(len(labels)), 3)
            parts = [prem[lang][j] for j in others]
            parts.insert(rng.randrange(4), prem[lang][i])
            long.append((" ".join(parts), en_h[i], labels[i]))
        sets[f"{lang}->en long"] = long
    return sets


class HFNLI:
    def __init__(self, name: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(4)
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.ent = next(k for k, v in id2label.items() if v.startswith("entail"))
        self.torch = torch

    def entail(self, pairs: list[tuple[str, str]], bs: int = 16) -> list[float]:
        out = []
        for i in range(0, len(pairs), bs):
            batch = pairs[i: i + bs]
            enc = self.tok([p for p, _ in batch], [h for _, h in batch], truncation="only_first",
                           max_length=512, padding=True, return_tensors="pt")
            with self.torch.no_grad():
                probs = self.model(**enc).logits.softmax(-1)
            out += probs[:, self.ent].tolist()
        return out


def auc(scores: list[float], gold: list[int]) -> float:
    pos = [s for s, g in zip(scores, gold) if g]
    neg = [s for s, g in zip(scores, gold) if not g]
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def metrics(scores: list[float], labels: list[int], thr: float = 0.5) -> dict:
    gold = [int(lab == 0) for lab in labels]
    pred = [int(s >= thr) for s in scores]
    tp = sum(p and g for p, g in zip(pred, gold))
    fp = sum(p and not g for p, g in zip(pred, gold))
    fn = sum(g and not p for p, g in zip(pred, gold))
    # contradictions that pass are the worst case: a sentence saying the opposite of its source
    contra_pass = sum(p for p, lab in zip(pred, labels) if lab == 2) / max(1, labels.count(2))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(2 * prec * rec / max(1e-9, prec + rec), 3), "auc": round(auc(scores, gold), 3),
            "contradiction_pass_rate": round(contra_pass, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--models", required=True)
    ap.add_argument("--n", type=int, default=600)
    args = ap.parse_args()
    sets = load(args.data, args.n)
    OUT.mkdir(exist_ok=True)
    out_file = OUT / "nli.json"
    results = json.loads(out_file.read_text()) if out_file.exists() else {}
    for name in args.models.split(","):
        model = HFNLI(name)
        res = {}
        for key, rows in sets.items():
            t = time.time()
            scores = model.entail([(p, h) for p, h, _ in rows])
            res[key] = metrics(scores, [lab for *_, lab in rows]) | {"sec_per_pair": round((time.time() - t) / len(rows), 3)}
            print(name, key, res[key], flush=True)
        results[name] = res
        out_file.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
