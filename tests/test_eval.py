from datetime import datetime, timedelta, timezone

from app.models.schemas import Status
from eval.run import Row, load_examples, parse_post_date, run_example, summarize, sweep
from app.config import ROOT_DIR

NOW = datetime(2026, 10, 5, tzinfo=timezone.utc)


def row(pred, exp, conf, **kw):
    ab = pred.startswith("UNVERIFIED")
    return Row("x", "tune", "en", exp, pred, conf, pred == exp, ab, pred != exp and not ab, 1,
               kw.get("kept", 1), kw.get("dropped", 0), kw.get("lat", 10.0))


def test_dataset_covers_all_six_statuses_and_is_synthetic():
    rows = load_examples(ROOT_DIR / "eval" / "claims.jsonl", "all")
    assert len(rows) == 10 and all(r["synthetic"] and "SYNTHETIC" in r["notes"] for r in rows)
    assert {r["expected_status"] for r in rows} == {s.value for s in Status}
    assert {r["split"] for r in rows} == {"tune", "hidden"}


def test_relative_post_dates():
    assert parse_post_date("-2h", NOW) == NOW - timedelta(hours=2)
    assert parse_post_date("-10d", NOW) == NOW - timedelta(days=10)
    assert parse_post_date("2026-10-01", NOW) == datetime(2026, 10, 1, tzinfo=timezone.utc)
    assert parse_post_date(None, NOW) is None


def test_metrics():
    rows = [
        row("CONFIRMED", "CONFIRMED", 0.9, kept=2),
        row("CONTRADICTED", "CONFIRMED", 0.65, kept=1, dropped=1),  # confident-wrong
        row("UNVERIFIED_EVIDENCE_MISSING", "CONFIRMED", 0.7, lat=30.0),  # abstained
        row("NOT_CHECKABLE", "NOT_CHECKABLE", 0.9, kept=0),
    ]
    s = summarize(rows)
    assert s["accuracy"] == 0.5 and s["confident_wrong_rate"] == 0.25 and s["abstain_rate"] == 0.25
    assert s["direction_wrong_rate"] == 0.25  # said CONTRADICTED for a CONFIRMED label
    assert s["citation_precision"] == 0.8  # 4 kept of 5 written
    assert s["latency_ms_p50"] == 10.0 and s["latency_ms_p95"] == 30.0
    assert {b["bucket"] for b in s["calibration"]} == {"0.6-0.7", "0.9-1.0"}


def test_sweep_finds_lowest_safe_threshold():
    rows = [row("CONTRADICTED", "CONFIRMED", 0.65), row("CONFIRMED", "CONFIRMED", 0.9)]
    best, table = sweep(rows, target=0.0)
    assert best == 0.66
    assert summarize(rows, 0.66)["abstain_rate"] == 0.5
    assert table[0] == (0.0, 0.5, 0.0)


async def test_mock_run_over_dataset(settings, adapters, whitelist):
    for ex in load_examples(ROOT_DIR / "eval" / "claims.jsonl", "all"):
        r = await run_example(ex, settings, adapters, whitelist, NOW)
        assert r.correct, (ex["id"], r.predicted, r.error)


def test_factcheck_set_is_real_and_guarded():
    rows = load_examples(ROOT_DIR / "eval" / "factchecks.jsonl", "all")
    assert len(rows) == 200 and not any(r["synthetic"] for r in rows)
    assert all(r["exclude_urls"] == [r["review_url"]] and r["single_claim"] for r in rows)
    assert {r["language"] for r in rows} == {"en", "hi", "mr"}


def test_contradicted_vs_misleading_is_not_direction_wrong():
    rows = [row("CONTRADICTED", "MISLEADING_CONTEXT", 0.9)]
    s = summarize(rows)
    assert s["confident_wrong_rate"] == 1.0 and s["direction_wrong_rate"] == 0.0


def test_date_only_post_date_uses_configured_timezone():
    from zoneinfo import ZoneInfo

    ist = ZoneInfo("Asia/Kolkata")
    assert parse_post_date("2026-10-01", NOW, ist) == datetime(2026, 10, 1, tzinfo=ist)
    assert parse_post_date("2026-10-01T00:00:00+00:00", NOW, ist) == datetime(2026, 10, 1, tzinfo=timezone.utc)


def test_failed_runs_are_errors_not_wrong_verdicts():
    failed = Row("e", "tune", "en", "CONTRADICTED", "ERROR", 0.0, False, False, False, 0, 0, 0, 5.0, "RuntimeError: down")
    rows = [row("CONTRADICTED", "CONTRADICTED", 0.9), failed]
    s = summarize(rows)
    assert s["n"] == 1 and s["errors"] == 1 and s["confident_wrong_rate"] == 0.0 and s["accuracy"] == 1.0


def test_replayed_abstention_is_not_counted_correct():
    rows = [row("CONTRADICTED", "CONTRADICTED", 0.7), row("CONTRADICTED", "CONTRADICTED", 0.9)]
    assert summarize(rows)["accuracy"] == 1.0
    s = summarize(rows, 0.8)
    assert s["abstain_rate"] == 0.5 and s["accuracy"] == 0.5
