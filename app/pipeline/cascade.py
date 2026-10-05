"""Rumour-cascade signals: how widely a claim is being submitted, and whether its only support is echo.

Signals are stored per claim (claims.signals), sent on the `verdict` event and summarised by the
monitoring endpoint. They are context for reviewers and never change the evidence status.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from app.db.store import Store
from app.models.schemas import Passage, PassageJudgment, Stance


def support_by_tier(passages: list[Passage], judgments: list[PassageJudgment], min_prob: float) -> dict[str, int]:
    tier = {p.id: p.tier for p in passages}
    counts = {"1": 0, "2": 0, "3": 0}
    for j in judgments:
        if j.stance == Stance.SUPPORTS and j.probability >= min_prob and j.passage_id in tier:
            counts[str(tier[j.passage_id])] += 1
    return counts


async def cascade_signals(
    store: Store,
    embedding: list[float],
    embedding_model: str,
    now: datetime,
    threshold: float,
    passages: list[Passage] | None = None,
    judgments: list[PassageJudgment] | None = None,
    min_prob: float = 0.5,
    account_handle: str | None = None,
) -> dict:
    stats = await store.similar_claim_stats(
        embedding, embedding_model, now - timedelta(days=7), now, threshold, account_handle=account_handle
    )
    signals: dict = {
        "similar_submissions_24h": stats["count_24h"],
        "similar_submissions_7d": stats["count_7d"],
        "first_seen": stats["first_seen"].isoformat() if stats["first_seen"] else now.isoformat(),
    }
    if stats.get("accounts_7d"):
        # Distinct accounts (from screenshots) posting this rumour in 7 days: many accounts = a spreading cascade.
        signals["distinct_accounts_7d"] = stats["accounts_7d"]
    if passages is not None and judgments is not None:
        tiers = support_by_tier(passages, judgments, min_prob)
        signals["support_by_tier"] = tiers
        # Only aggregators repeat it; no primary source or original reporting backs it.
        signals["echo_only"] = tiers["3"] > 0 and tiers["1"] == 0 and tiers["2"] == 0
    return signals
