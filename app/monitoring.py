"""Operational summary over recent stage logs, checks and verdicts (GET /api/monitoring)."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta

from app.db.store import Store
from app.models.schemas import UNVERIFIED_STATUSES


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(q * len(s)))], 1)


def summarize_activity(data: dict, top_n: int = 10) -> dict:
    runs, checks, claims = data["stage_runs"], data["checks"], data["claims"]

    by_stage: dict[str, list] = defaultdict(list)
    for r in runs:
        by_stage[r["stage"]].append(r)
    stages = {
        name: {
            "runs": len(rs),
            "errors": sum(1 for r in rs if r["error"]),
            "latency_ms_p50": _pct([r["latency_ms"] for r in rs], 0.5),
            "latency_ms_p95": _pct([r["latency_ms"] for r in rs], 0.95),
        }
        for name, rs in sorted(by_stage.items())
    }

    models: dict[str, Counter] = defaultdict(Counter)
    for r in runs:
        if r["model_version"]:
            models[r["stage"]][r["model_version"]] += 1
    fallback_runs = sum(1 for r in runs if r["model_version"] and "fallback" in r["model_version"])

    kept = dropped = 0
    for r in runs:
        out = r.get("outputs") if r["stage"] == "verify" else None
        if isinstance(out, dict):
            kept += len(out.get("kept", []))
            dropped += len(out.get("dropped", []))

    statuses = Counter(c["status"] for c in claims)
    unverified = {s.value for s in UNVERIFIED_STATUSES}
    cascades = sorted(
        (c for c in claims if c.get("signals")),
        key=lambda c: -(c["signals"].get("similar_submissions_24h") or 0),
    )
    seen, top = set(), []
    for c in cascades:
        key = c["text_en"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        top.append({
            "claim": c["text_en"], "status": c["status"],
            "submissions_24h": c["signals"].get("similar_submissions_24h", 0) + 1,
            "submissions_7d": c["signals"].get("similar_submissions_7d", 0) + 1,
            "first_seen": c["signals"].get("first_seen"),
            "echo_only": c["signals"].get("echo_only"),
            "distinct_accounts_7d": c["signals"].get("distinct_accounts_7d"),
        })
        if len(top) >= top_n:
            break

    return {
        "checks": {"total": len(checks), "by_status": dict(Counter(c["status"] for c in checks))},
        "claims": {
            "total": len(claims),
            "by_status": dict(statuses),
            "abstain_rate": round(sum(statuses[s] for s in unverified) / len(claims), 3) if claims else None,
            "rechecks_done": sum(1 for c in claims if c.get("rechecked_from")),
        },
        "stages": stages,
        "models": {stage: dict(cnt) for stage, cnt in models.items()},
        "fallback_runs": fallback_runs,
        "nli": {"sentences_kept": kept, "sentences_deleted": dropped,
                "deletion_rate": round(dropped / (kept + dropped), 3) if kept + dropped else None},
        "top_cascades": [t for t in top if t["submissions_24h"] > 1],
    }


async def monitoring_report(store: Store, now: datetime, hours: float) -> dict:
    since = now - timedelta(hours=hours)
    report = summarize_activity(await store.recent_activity(since))
    return {"window_hours": hours, "since": since.isoformat(), "generated_at": now.isoformat(), **report}
