"""Re-run UNVERIFIED claims whose recheck_at has passed (TOO_EARLY after hours, EVIDENCE_MISSING after days).

A recheck runs the claim again (single-claim mode, original post date), stores a new verdict linked
to the old one (verdicts.rechecked_from) and marks the old one superseded. A claim older than
RECHECK_MAX_AGE_DAYS is no longer rechecked. A failed recheck is postponed, not retried in a loop.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from app.models.schemas import CheckInput
from app.pipeline.orchestrator import Pipeline

log = logging.getLogger("fact.recheck")


async def run_rechecks(pipeline: Pipeline, limit: int = 20) -> list[dict]:
    s = pipeline.settings
    now = pipeline.clock()
    due = await pipeline.store.due_rechecks(now, now - timedelta(days=s.recheck_max_age_days), limit)
    results = []
    for d in due:
        try:
            res = await pipeline.run(CheckInput(text=d.text_original, post_date=d.post_date, single_claim=True))
            await pipeline.store.link_recheck(d.verdict_id, res.check_id, "c1")
            new = res.claims[0].status.value if res.claims else "NO_CLAIMS"
            results.append({"verdict_id": d.verdict_id, "old": d.status, "new": new, "check_id": res.check_id})
            log.info("recheck verdict=%s %s -> %s", d.verdict_id, d.status, new)
        except Exception as exc:
            await pipeline.store.postpone_recheck(d.verdict_id, now + timedelta(hours=1))
            results.append({"verdict_id": d.verdict_id, "old": d.status, "error": f"{type(exc).__name__}: {exc}"})
            log.warning("recheck verdict=%s failed: %s", d.verdict_id, exc)
    return results
