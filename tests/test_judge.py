from datetime import timedelta

import pytest

from app.models.schemas import Stance, Status
from app.pipeline.judge import Thresholds, claim_age_hours, decide_status, effective_judgments, recheck_at

from .conftest import NOW
from .helpers import judgment, passage

T = Thresholds(confidence=0.6, passage_relevance=0.5, too_early_window_hours=72)
SUP, CON = Stance.SUPPORTS, Stance.CONTRADICTS


def probs(**kw):
    return {Status[k]: v for k, v in kw.items()}


def eff(*items):
    ps = [passage(pid, tier=tier) for pid, tier, _ in items]
    js = [judgment(pid, stance) for pid, _, stance in items]
    return effective_judgments(js, ps, 0.5)


def test_confirmed_with_support():
    assert decide_status(probs(CONFIRMED=0.9), eff(("a", 1, SUP)), 100, T) == (Status.CONFIRMED, 0.9)


def test_missing_evidence_is_never_contradicted():
    # The model is sure it's false, but nothing contradicts it -> abstain, not CONTRADICTED.
    status, _ = decide_status(probs(CONTRADICTED=0.95), [], 100, T)
    assert status == Status.UNVERIFIED_EVIDENCE_MISSING


def test_confirmed_needs_supporting_passage():
    status, _ = decide_status(probs(CONFIRMED=0.95), eff(("a", 1, CON)), 100, T)
    assert status == Status.UNVERIFIED_EVIDENCE_MISSING


def test_below_threshold_abstains():
    status, conf = decide_status(probs(CONFIRMED=0.55, UNVERIFIED_EVIDENCE_MISSING=0.45), eff(("a", 1, SUP)), 100, T)
    assert status == Status.UNVERIFIED_EVIDENCE_MISSING and conf == 0.45


def test_conflict_at_same_tier_never_confirms():
    status, _ = decide_status(probs(CONFIRMED=0.9), eff(("a", 1, SUP), ("b", 1, CON)), 100, T)
    assert status == Status.UNVERIFIED_EVIDENCE_MISSING


def test_primary_source_outranks_aggregator():
    e = eff(("viral", 3, SUP), ("police", 1, CON))
    assert [p.id for _, p in e] == ["police"]
    assert decide_status(probs(CONTRADICTED=0.9), e, 100, T)[0] == Status.CONTRADICTED


def test_irrelevant_and_low_probability_judgments_ignored():
    ps = [passage("a", tier=1), passage("b", tier=1)]
    js = [judgment("a", Stance.IRRELEVANT), judgment("b", SUP, 0.3)]
    assert effective_judgments(js, ps, 0.5) == []


@pytest.mark.parametrize("age,expected", [
    (3, Status.UNVERIFIED_TOO_EARLY),
    (71.9, Status.UNVERIFIED_TOO_EARLY),
    (72, Status.UNVERIFIED_EVIDENCE_MISSING),
    (None, Status.UNVERIFIED_EVIDENCE_MISSING),  # unknown post date
])
def test_claim_age_splits_unverified(age, expected):
    assert decide_status(probs(UNVERIFIED_TOO_EARLY=0.9), [], age, T)[0] == expected


def test_claim_age_hours():
    assert claim_age_hours(NOW - timedelta(hours=5), NOW) == pytest.approx(5)
    assert claim_age_hours(None, NOW) is None
    assert claim_age_hours(NOW + timedelta(hours=1), NOW) == 0  # future dates clamp to 0


def test_recheck_at():
    assert recheck_at([Status.CONFIRMED, Status.UNVERIFIED_TOO_EARLY], NOW, 6, 7) == NOW + timedelta(hours=6)
    assert recheck_at([Status.UNVERIFIED_EVIDENCE_MISSING], NOW, 6, 7) == NOW + timedelta(days=7)
    assert recheck_at([Status.CONFIRMED], NOW, 6, 7) is None
