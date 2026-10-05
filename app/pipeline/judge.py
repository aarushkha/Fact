"""Stage 6: judge. The classifier gives probabilities; the rules here decide the status.

The non-negotiable rules are enforced in code, not left to the model:
  * missing evidence is UNVERIFIED_*, never CONTRADICTED;
  * a definitive status needs a matching passage from the best tier present;
  * below the confidence threshold we abstain;
  * claim age, not the model, separates UNVERIFIED_TOO_EARLY from UNVERIFIED_EVIDENCE_MISSING.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.adapters.base import Classifier
from app.models.schemas import (
    DEFINITIVE_STATUSES,
    UNVERIFIED_STATUSES,
    Claim,
    ExpectedEvidence,
    Passage,
    PassageJudgment,
    Stance,
    Status,
)


@dataclass
class Thresholds:
    confidence: float
    passage_relevance: float
    too_early_window_hours: float
    same_event: float = 0.6


def apply_same_event_gate(j: PassageJudgment, threshold: float) -> PassageJudgment:
    """A passage about a different incident is irrelevant, however strongly it 'contradicts'.

    Debunks of other viral videos share vocabulary with almost any viral claim; without this gate they
    were the evidence behind most confident errors on real fact-check data.
    """
    if j.same_event is not None and j.same_event < threshold and j.stance != Stance.IRRELEVANT:
        return j.model_copy(update={"stance": Stance.IRRELEVANT, "probability": 1 - j.same_event})
    return j


def claim_age_hours(post_date: datetime | None, now: datetime) -> float | None:
    if post_date is None:
        return None
    return max(0.0, (now - post_date).total_seconds() / 3600)


def effective_judgments(
    judgments: list[PassageJudgment], passages: list[Passage], min_prob: float
) -> list[tuple[PassageJudgment, Passage]]:
    """Relevant judgments from the best (lowest-numbered) tier present.

    Primary sources outrank outlets that repeat social posts: when a tier-1 source speaks,
    lower-tier passages do not count towards (or against) the verdict.
    """
    by_id = {p.id: p for p in passages}
    relevant = [
        (j, by_id[j.passage_id])
        for j in judgments
        if j.stance != Stance.IRRELEVANT and j.probability >= min_prob and j.passage_id in by_id
    ]
    if not relevant:
        return []
    best = min(p.tier for _, p in relevant)
    return [(j, p) for j, p in relevant if p.tier == best]


def unverified_status(age_hours: float | None, window_hours: float) -> Status:
    if age_hours is not None and age_hours < window_hours:
        return Status.UNVERIFIED_TOO_EARLY
    return Status.UNVERIFIED_EVIDENCE_MISSING


FAILS = (Status.CONTRADICTED, Status.MISLEADING_CONTEXT)


def review_disagreement(
    effective: list[tuple[PassageJudgment, Passage]], review_ratings: dict[str, Status]
) -> set[Status]:
    """Distinct mapped ratings among the fact-check reviews that count as evidence ({} or one = agreement).

    Only reviews in `effective` count: a review the same-event gate rejected, or one outranked by a
    primary source, cannot create (or settle) a disagreement.
    """
    ratings = {review_ratings[p.id] for _, p in effective if p.id in review_ratings}
    return ratings if len(ratings) > 1 else set()


def decide_status(
    probabilities: dict[Status, float],
    effective: list[tuple[PassageJudgment, Passage]],
    age_hours: float | None,
    t: Thresholds,
    disagreement: set[Status] = frozenset(),
) -> tuple[Status, float]:
    """`disagreement`: the ratings of whitelisted fact-checkers that disagree about this claim.

    Fact-checkers often split CONTRADICTED vs MISLEADING_CONTEXT on the same claim (most remaining
    real-eval errors). Then the more conservative MISLEADING_CONTEXT is returned: both reviews say the
    claim fails as stated, only one says the event never happened. Reviews split on direction (one says
    it holds, another that it fails) mean abstaining.
    """
    stances = {j.stance for j, _ in effective}
    candidates = sorted(
        ((s, probabilities.get(s, 0.0)) for s in DEFINITIVE_STATUSES), key=lambda sp: -sp[1]
    )
    unverified_p = sum(probabilities.get(s, 0.0) for s in UNVERIFIED_STATUSES)
    status, p = candidates[0]

    evidence_ok = {
        # Conflicting best-tier evidence never yields CONFIRMED.
        Status.CONFIRMED: Stance.SUPPORTS in stances and Stance.CONTRADICTS not in stances,
        Status.CONTRADICTED: Stance.CONTRADICTS in stances,
        Status.MISLEADING_CONTEXT: bool(stances),
    }[status]

    direction_split = Status.CONFIRMED in disagreement and any(r in disagreement for r in FAILS)
    if evidence_ok and p >= t.confidence and p > unverified_p and not direction_split:
        if status == Status.CONTRADICTED and Status.MISLEADING_CONTEXT in disagreement:
            # Confidence stays the judge's: it measures "the claim fails as stated", which both reviews share.
            return Status.MISLEADING_CONTEXT, round(p, 4)
        return status, round(p, 4)
    # Abstaining. Confidence = how sure we are that no definitive status is warranted: the judge's
    # unverified probability, or 1 - p when the judge leaned definitive but was unsure.
    abstain_conf = max(unverified_p, 1 - p)
    return unverified_status(age_hours, t.too_early_window_hours), round(min(1.0, abstain_conf), 4)


@dataclass
class JudgeResult:
    status: Status
    confidence: float
    probabilities: dict[Status, float]  # raw classifier output, logged for calibration
    judgments: list[PassageJudgment]
    expected: list[ExpectedEvidence]
    effective: list[tuple[PassageJudgment, Passage]]
    disagreement: set[Status] = field(default_factory=set)  # fact-check ratings that disagree (logged, SSE)


async def judge_claim(
    claim: Claim,
    passages: list[Passage],
    classifier: Classifier,
    age_hours: float | None,
    t: Thresholds,
    review_ratings: dict[str, Status] | None = None,
) -> JudgeResult:
    """review_ratings: passage id -> mapped rating, for passages that are whitelisted fact-check reviews."""
    raw = await asyncio.gather(*(classifier.judge_passage(claim, p) for p in passages))
    judgments = [apply_same_event_gate(j, t.same_event) for j in raw]
    expected = await classifier.expected_evidence(claim, passages, judgments)
    # The claim-level judge sees only passages judged relevant; irrelevant ones are noise.
    relevant_ids = {j.passage_id for j in judgments if j.stance != Stance.IRRELEVANT}
    relevant = [p for p in passages if p.id in relevant_ids]
    verdict = await classifier.judge_claim(
        claim, relevant, [j for j in judgments if j.passage_id in relevant_ids], expected, age_hours
    )
    effective = effective_judgments(judgments, passages, t.passage_relevance)
    disagreement = review_disagreement(effective, review_ratings or {})
    status, confidence = decide_status(verdict.probabilities, effective, age_hours, t, disagreement)
    return JudgeResult(status, confidence, verdict.probabilities, judgments, expected, effective, disagreement)


def would_change_if(status: Status, expected: list[ExpectedEvidence]) -> str | None:
    """Templated description of what evidence would change the status (states no facts)."""
    missing = [e.item for e in expected if not e.found]
    if status == Status.NOT_CHECKABLE:
        return "The claim is restated as a specific, checkable statement of fact."
    if status == Status.CONFIRMED:
        return "A primary source issues a correction or contradicts the cited reports."
    if status == Status.CONTRADICTED:
        return "A primary source confirms the claim or retracts the cited statement."
    if status == Status.MISLEADING_CONTEXT:
        return "Evidence shows the content does match the time, place or context claimed."
    if missing:
        return "Any of these appears from a whitelisted source: " + "; ".join(missing) + "."
    return "A whitelisted source publishes a statement or report about this claim."


def recheck_at(statuses: list[Status], now: datetime, too_early_hours: float, missing_days: float) -> datetime | None:
    if Status.UNVERIFIED_TOO_EARLY in statuses:
        return now + timedelta(hours=too_early_hours)
    if Status.UNVERIFIED_EVIDENCE_MISSING in statuses:
        return now + timedelta(days=missing_days)
    return None
