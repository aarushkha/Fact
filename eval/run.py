"""Evaluation harness.

    python -m eval.run --split tune|hidden|all [--file eval/claims.jsonl] [--out eval/out]
    python -m eval.run --split tune --threshold-sweep --target 0.05
    python -m eval.run --file eval/factchecks.jsonl --split hidden     # real fact-checked claims

Rows may set "exclude_urls" (evidence ignored for that row, e.g. the fact-check that labels it) and
"single_claim" (skip claim extraction). Real fact-check rows use both.

Each example is scored on its FIRST claim (eval inputs are single-claim posts).
- confident-wrong rate: wrong and not abstained (abstained = UNVERIFIED_*), over all examples that ran
  (an example whose run failed is counted under "errors", never as a verdict)
- direction-wrong rate: answered "holds" (CONFIRMED) when the label says "fails" (CONTRADICTED /
  MISLEADING_CONTEXT) or the reverse: the dangerous subset of confident-wrong
- abstain rate, accuracy
- citation precision: share of written sentences that pass the NLI check (kept / (kept + dropped))
- calibration: accuracy per confidence bucket for non-abstained answers, plus ECE
- p50 / p95 latency per example
Writes a per-example CSV and a summary JSON to --out.

--threshold-sweep runs the pipeline once with CONFIDENCE_THRESHOLD=0 and replays each threshold
post hoc (exact: a definitive answer at threshold t is kept iff its confidence >= t), then prints the
lowest threshold whose confident-wrong rate is <= --target.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo

from app.adapters.factory import build_adapters
from app.config import ROOT_DIR, Settings, get_settings
from app.db.store import InMemoryStore
from app.models.schemas import DEFINITIVE_STATUSES, UNVERIFIED_STATUSES, CheckInput, Status
from app.pipeline.orchestrator import Pipeline
from app.sources import load_whitelist_file

DEFINITIVE = {s.value for s in DEFINITIVE_STATUSES}
ABSTAIN = {s.value for s in UNVERIFIED_STATUSES}
# "Direction" of a status: does it say the claim holds or not? Wrong direction = the dangerous error.
DIRECTION = {"CONFIRMED": "holds", "CONTRADICTED": "fails", "MISLEADING_CONTEXT": "fails"}
BUCKETS = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


@dataclass
class Row:
    id: str
    split: str
    language: str
    expected: str
    predicted: str
    confidence: float
    correct: bool
    abstained: bool
    confident_wrong: bool
    n_claims: int
    sentences_kept: int
    sentences_dropped: int
    latency_ms: float
    error: str = ""


def load_examples(path: Path, split: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        Status(r["expected_status"])  # validate
        if not (r.get("input_text") or r.get("image_path")):
            raise ValueError(f"{r['id']}: needs input_text or image_path")
    return rows if split == "all" else [r for r in rows if r.get("split", "tune") == split]


def parse_post_date(value: str | None, now: datetime, tz: tzinfo = timezone.utc) -> datetime | None:
    """ISO 8601, or relative to the run start: "-2h", "-10d", "-30m". Naive values (e.g. date-only) are
    read in tz, the configured TIMEZONE, like post dates in the app."""
    if not value:
        return None
    m = re.fullmatch(r"-(\d+(?:\.\d+)?)([mhd])", value.strip())
    if m:
        unit = {"m": "minutes", "h": "hours", "d": "days"}[m.group(2)]
        return now - timedelta(**{unit: float(m.group(1))})
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=tz)


async def run_example(ex: dict, settings: Settings, adapters, whitelist, now: datetime, trace: list | None = None) -> Row:
    store = InMemoryStore()  # fresh per example: no cache hits between eval examples
    pipeline = Pipeline(settings, adapters, store, whitelist, clock=lambda: now)
    image = (ROOT_DIR / ex["image_path"]).read_bytes() if ex.get("image_path") else None
    inp = CheckInput(text=ex.get("input_text"), image=image, image_mime="image/png" if image else None,
                     post_date=parse_post_date(ex.get("post_date"), now, ZoneInfo(settings.timezone)),
                     exclude_urls=ex.get("exclude_urls", []), single_claim=ex.get("single_claim", False))
    t0 = time.perf_counter()
    predicted, conf, n, err = "ERROR", 0.0, 0, ""
    try:
        res = None
        events = []
        async for ev in pipeline.stream(inp):
            events.append({"event": ev.name, "data": ev.data})
            if ev.name == "done":
                from app.models.schemas import CheckResponse

                res = CheckResponse.model_validate(ev.data)
            elif ev.name == "error" and ev.data.get("fatal"):
                raise RuntimeError(ev.data.get("message"))
        if trace is not None:
            stages = {r.stage: r.outputs for r in store.stage_runs if r.stage in ("factcheck", "retrieve", "judge", "verify")}
            trace.append({"id": ex["id"], "expected": ex["expected_status"], "input": ex.get("input_text"),
                          "events": [e for e in events if e["event"] in ("cache_hit", "evidence", "verdict")],
                          "stages": stages})
        if res and res.claims:
            predicted, conf, n = res.claims[0].status.value, res.claims[0].confidence, len(res.claims)
        else:
            predicted = "NO_CLAIMS"
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
    latency = (time.perf_counter() - t0) * 1000
    kept = dropped = 0
    for run in store.stage_runs:
        if run.stage == "verify" and isinstance(run.outputs, dict):
            kept += len(run.outputs.get("kept", []))
            dropped += len(run.outputs.get("dropped", []))
    expected = ex["expected_status"]
    abstained = predicted in ABSTAIN
    correct = predicted == expected
    confident_wrong = not (correct or abstained or err)  # a failed run is an error, not a wrong verdict
    return Row(ex["id"], ex.get("split", "tune"), ex.get("language", ""), expected, predicted, conf, correct,
               abstained, confident_wrong, n, kept, dropped, round(latency, 1), err)


def at_threshold(r: Row, t: float) -> tuple[bool, bool]:
    """(abstained, confident_wrong) if the run had used confidence threshold t."""
    if r.predicted in DEFINITIVE and r.confidence < t:
        return True, False
    return r.abstained, r.confident_wrong


def summarize(rows: list[Row], threshold: float | None = None) -> dict:
    errors = sum(bool(r.error) for r in rows)
    rows = [r for r in rows if not r.error]  # failed runs issued no verdict: kept out of every rate
    n = len(rows) or 1
    if threshold is not None:
        flags = [at_threshold(r, threshold) for r in rows]
    else:
        flags = [(r.abstained, r.confident_wrong) for r in rows]
    # A definitive answer the replayed threshold turns into an abstention is no longer correct
    # (which UNVERIFIED_* it would become is not recorded, so it never counts as a correct abstention).
    correct = [r.correct and (r.abstained or not ab) for r, (ab, _) in zip(rows, flags)]
    direction_wrong = sum(
        1 for r, (ab, _) in zip(rows, flags)
        if not ab and r.predicted in DIRECTION and r.expected in DIRECTION and DIRECTION[r.predicted] != DIRECTION[r.expected]
    )
    answered = [r for r, (ab, _) in zip(rows, flags) if not ab]
    kept = sum(r.sentences_kept for r in rows)
    written = kept + sum(r.sentences_dropped for r in rows)
    calib = []
    ece = 0.0
    for lo, hi in BUCKETS:
        b = [r for r in answered if lo <= r.confidence < hi]
        if b:
            acc = sum(r.correct for r in b) / len(b)
            avg = sum(r.confidence for r in b) / len(b)
            ece += len(b) / max(len(answered), 1) * abs(acc - avg)
            calib.append({"bucket": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": len(b), "avg_conf": round(avg, 3), "accuracy": round(acc, 3)})
    lat = sorted(r.latency_ms for r in rows)
    return {
        "n": len(rows),
        "accuracy": round(sum(correct) / n, 3),
        "confident_wrong_rate": round(sum(cw for _, cw in flags) / n, 3),
        "direction_wrong_rate": round(direction_wrong / n, 3),
        "abstain_rate": round(sum(ab for ab, _ in flags) / n, 3),
        "citation_precision": round(kept / written, 3) if written else None,
        "calibration": calib,
        "ece": round(ece, 3),
        "latency_ms_p50": round(statistics.median(lat), 1) if lat else None,
        "latency_ms_p95": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1) if lat else None,
        "errors": errors,
    }


def sweep(rows: list[Row], target: float) -> tuple[float | None, list[tuple[float, float, float]]]:
    table = []
    best = None
    for i in range(0, 101):
        t = i / 100
        s = summarize(rows, t)
        table.append((t, s["confident_wrong_rate"], s["abstain_rate"]))
        if best is None and s["confident_wrong_rate"] <= target:
            best = t
    return best, table


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.threshold_sweep:
        settings = settings.model_copy(update={"confidence_threshold": 0.0})
    examples = load_examples(Path(args.file), args.split)
    if args.ids:
        wanted = set(args.ids.split(","))
        examples = [e for e in examples if e["id"] in wanted]
    examples = examples[: args.limit or None]
    trace: list | None = [] if args.trace else None
    if not examples:
        print("No examples for split", args.split)
        return 1
    engine = tables = None
    if settings.database_url:
        from app.db.tables import build_tables, make_engine

        engine, tables = make_engine(settings.database_url), build_tables(settings.embedding_dim)
    adapters = build_adapters(settings, engine, tables)
    whitelist = load_whitelist_file(settings.effective_sources_file)
    now = datetime.now(timezone.utc)
    rows = []
    try:
        for ex in examples:  # sequential: keeps latency numbers honest and respects API rate limits
            rows.append(await run_example(ex, settings, adapters, whitelist, now, trace))
            r = rows[-1]
            print(f"{r.id:<5} {r.expected:<28} -> {r.predicted:<28} {r.confidence:.2f} {r.latency_ms / 1000:5.1f}s {r.error}")
    finally:
        if engine is not None:
            await engine.dispose()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    csv_path = out / f"eval_{args.split}_{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0])))
        w.writeheader()
        w.writerows(asdict(r) for r in rows)

    report = {"split": args.split, "mock_mode": settings.mock_mode, "model_versions": adapters.model_versions()}
    if args.threshold_sweep:
        configured = get_settings().confidence_threshold
        best, table = sweep(rows, args.target)
        report["at_configured_threshold"] = {"threshold": configured, **summarize(rows, configured)}
        report["sweep"] = {"target": args.target, "lowest_threshold": best,
                           "table": [{"t": t, "confident_wrong": cw, "abstain": ab} for t, cw, ab in table]}
    else:
        report["metrics"] = {"threshold": settings.confidence_threshold, **summarize(rows)}
    (out / f"eval_{args.split}_{stamp}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if trace is not None:
        tpath = out / f"trace_{args.split}_{stamp}.jsonl"
        tpath.write_text("".join(json.dumps(t, ensure_ascii=False, default=str) + "\n" for t in trace), encoding="utf-8")
        print(f"trace: {tpath}")

    m = report.get("metrics") or report["at_configured_threshold"]
    print(f"\nthreshold={m['threshold']}  n={m['n']}  accuracy={m['accuracy']}  confident_wrong={m['confident_wrong_rate']}  direction_wrong={m['direction_wrong_rate']}  "
          f"abstain={m['abstain_rate']}  citation_precision={m['citation_precision']}  ece={m['ece']}  "
          f"p50={m['latency_ms_p50']}ms  p95={m['latency_ms_p95']}ms  errors={m['errors']}")
    for b in m["calibration"]:
        print(f"  conf {b['bucket']}: n={b['n']} avg_conf={b['avg_conf']} accuracy={b['accuracy']}")
    if args.threshold_sweep:
        best = report["sweep"]["lowest_threshold"]
        print(f"\nlowest threshold with confident-wrong <= {args.target}: {best if best is not None else 'none'}")
        for t, cw, ab in table:
            if t * 10 == int(t * 10):
                print(f"  t={t:.1f}  confident_wrong={cw}  abstain={ab}")
    print(f"\nwrote {csv_path}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["tune", "hidden", "all"], default="tune")
    ap.add_argument("--file", default=str(ROOT_DIR / "eval" / "claims.jsonl"))
    ap.add_argument("--out", default=str(ROOT_DIR / "eval" / "out"))
    ap.add_argument("--limit", type=int, help="only the first N examples of the split")
    ap.add_argument("--ids", help="comma-separated example ids to run")
    ap.add_argument("--trace", action="store_true", help="write per-example events + judge/verify outputs (jsonl)")
    ap.add_argument("--threshold-sweep", action="store_true")
    ap.add_argument("--target", type=float, default=0.05, help="max confident-wrong rate for the sweep")
    raise SystemExit(asyncio.run(main_async(ap.parse_args())))


if __name__ == "__main__":
    main()
