from datetime import timedelta

import pytest

from app.models.schemas import Stance, Status
from app.pipeline.judge import (
    Thresholds,
    claim_age_hours,
    decide_status,
    effective_judgments,
    recheck_at,
    review_disagreement,
)

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


def test_abstain_confidence_reflects_split_judgement():
    # Judge split between two definitive statuses: abstain, and say we are ~half sure.
    status, conf = decide_status(probs(CONTRADICTED=0.45, MISLEADING_CONTEXT=0.55), eff(("a", 1, CON)), 100, T)
    assert status == Status.UNVERIFIED_EVIDENCE_MISSING and conf == 0.45
    status, conf = decide_status(probs(UNVERIFIED_EVIDENCE_MISSING=0.9, CONFIRMED=0.1), [], 100, T)
    assert conf == 0.9


async def test_claim_judge_only_sees_relevant_passages():
    from app.adapters.mock import MockClassifier
    from app.pipeline.judge import judge_claim

    seen = {}

    class Spy(MockClassifier):
        async def judge_claim(self, claim, passages, judgments, expected, age):
            seen["ids"] = [p.id for p in passages]
            return await super().judge_claim(claim, passages, judgments, expected, age)

    from .helpers import claim
    rel = passage("rel", tier=1, text="Mumbai airport is closed for a week, officials confirmed.")
    irr = passage("irr", tier=1, text="Cricket scores from Pune today.")
    r = await judge_claim(claim(), [rel, irr], Spy(), 100, T)
    assert seen["ids"] == ["rel"] and r.status == Status.CONFIRMED and r.probabilities


def test_same_event_gate_turns_other_incidents_irrelevant():
    from app.pipeline.judge import apply_same_event_gate

    other = judgment("a", CON).model_copy(update={"same_event": 0.3})
    gated = apply_same_event_gate(other, 0.6)
    assert gated.stance == Stance.IRRELEVANT and gated.probability == 0.7
    same = judgment("b", CON).model_copy(update={"same_event": 0.9})
    assert apply_same_event_gate(same, 0.6) == same
    assert apply_same_event_gate(judgment("c", SUP), 0.6).stance == SUP  # not assessed -> unchanged


async def test_unrelated_debunk_cannot_contradict():
    from app.adapters.mock import MockClassifier
    from app.pipeline.judge import judge_claim
    from .helpers import claim

    class OtherIncident(MockClassifier):
        async def judge_passage(self, c, p):
            j = await super().judge_passage(c, p)
            return j.model_copy(update={"same_event": 0.2})

    p = passage("x", tier=2, text="Mumbai airport video is false, the viral clip is from another airport, fact-check finds.")
    r = await judge_claim(claim(), [p], OtherIncident(), 100, T)
    assert r.status == Status.UNVERIFIED_EVIDENCE_MISSING and r.effective == []


C, M, OK = Status.CONTRADICTED, Status.MISLEADING_CONTEXT, Status.CONFIRMED


def test_factcheckers_split_on_contradicted_vs_misleading_gives_misleading():
    e = eff(("rev_a", 2, CON), ("rev_b", 2, CON))
    split = review_disagreement(e, {"rev_a": C, "rev_b": M})
    assert split == {C, M}
    assert decide_status(probs(CONTRADICTED=0.88), e, 100, T, split) == (M, 0.88)
    # Already the conservative status: unchanged.
    assert decide_status(probs(MISLEADING_CONTEXT=0.85), e, 100, T, split) == (M, 0.85)


def test_factcheckers_split_on_direction_abstains():
    e = eff(("rev_a", 2, CON), ("rev_b", 2, SUP))
    split = review_disagreement(e, {"rev_a": C, "rev_b": OK})
    status, _ = decide_status(probs(CONTRADICTED=0.9), e, 100, T, split)
    assert status == Status.UNVERIFIED_EVIDENCE_MISSING


def test_agreeing_or_gated_reviews_are_not_a_disagreement():
    e = eff(("rev_a", 2, CON), ("rev_b", 2, CON))
    assert review_disagreement(e, {"rev_a": C, "rev_b": C}) == set()
    # rev_b was rejected (irrelevant / other incident), so it is not in `effective` and cannot split.
    assert review_disagreement(eff(("rev_a", 2, CON)), {"rev_a": C, "rev_b": M}) == set()


def test_primary_source_outranks_factchecker_split():
    e = eff(("police", 1, CON), ("rev_a", 2, CON), ("rev_b", 2, CON))
    assert review_disagreement(e, {"rev_a": C, "rev_b": M}) == set()
    assert decide_status(probs(CONTRADICTED=0.9), e, 100, T)[0] == C
