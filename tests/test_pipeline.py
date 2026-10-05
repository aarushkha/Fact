"""End-to-end in MOCK_MODE: every status is reachable, events arrive in order, every stage is logged."""

from datetime import timedelta

import pytest

from app.adapters.mock import MockLLM, make_mock_png
from app.models.schemas import CheckInput, Status

from .conftest import NOW

CASES = [
    ("A footbridge over the Godavari river in Nashik collapsed on Sunday.", None, Status.CONFIRMED),
    ("Mumbai airport is closed for a week.", None, Status.CONTRADICTED),
    ("This video shows floods in Kolhapur this week.", None, Status.MISLEADING_CONTEXT),
    ("A fire broke out at a chemical factory in Thane.", NOW - timedelta(hours=3), Status.UNVERIFIED_TOO_EARLY),
    ("A fire broke out at a chemical factory in Thane.", NOW - timedelta(days=10), Status.UNVERIFIED_EVIDENCE_MISSING),
    ("A fire broke out at a chemical factory in Thane.", None, Status.UNVERIFIED_EVIDENCE_MISSING),
    ("Nashik is the most beautiful city in India.", None, Status.NOT_CHECKABLE),
    ("The state government is giving free laptops to all college students.", None, Status.CONTRADICTED),
    ("नाशिकमध्ये गोदावरी नदीवरील पादचारी पूल रविवारी कोसळला.", None, Status.CONFIRMED),
    ("Mumbai airport ek hafte ke liye band hai.", None, Status.CONTRADICTED),
]


@pytest.mark.parametrize("text,post_date,expected", CASES)
async def test_statuses(pipeline, text, post_date, expected):
    res = await pipeline.run(CheckInput(text=text, post_date=post_date))
    assert [c.status for c in res.claims] == [expected]


async def test_every_summary_sentence_cites_a_returned_source(pipeline):
    res = await pipeline.run(CheckInput(text="Mumbai airport is closed for a week. "
                                             "A footbridge over the Godavari river in Nashik collapsed on Sunday."))
    source_ids = {s.id for s in res.sources}
    for c in res.claims:
        assert c.summary, "mock evidence exists, so a verified summary should survive"
        for s in c.summary:
            assert s.sources and set(s.sources) <= source_ids
        # the mock writer's deliberately unsupported sentence must be deleted
        assert all("under control" not in s.sentence for s in c.summary)


async def test_tier3_aggregator_never_cited_against_primary(pipeline):
    res = await pipeline.run(CheckInput(text="Mumbai airport is closed for a week."))
    assert [s.tier for s in res.sources] == [1]


async def test_no_evidence_returns_status_only(pipeline):
    res = await pipeline.run(CheckInput(text="A fire broke out at a chemical factory in Thane."))
    c = res.claims[0]
    assert c.summary == [] and res.sources == []
    assert c.would_change_if and not any(e.found for e in c.expected_evidence)
    assert res.recheck_at == NOW + timedelta(days=7)


async def test_too_early_gets_short_recheck(pipeline):
    res = await pipeline.run(CheckInput(text="A fire broke out at a chemical factory in Thane.",
                                        post_date=NOW - timedelta(hours=1)))
    assert res.recheck_at == NOW + timedelta(hours=6)


async def test_event_order(pipeline):
    events = [e.name async for e in pipeline.stream(CheckInput(
        text="Mumbai airport is closed for a week. Nashik is the most beautiful city in India."))]
    assert events[0] == "claims_extracted" and events[-1] == "done"
    assert events.count("verdict") == 2
    assert events.index("evidence") < len(events) - 1


async def test_cache_hit_on_repeat(pipeline):
    text = "Mumbai airport is closed for a week."
    await pipeline.run(CheckInput(text=text))
    events = [e async for e in pipeline.stream(CheckInput(text=text))]
    names = [e.name for e in events]
    assert names == ["claims_extracted", "cache_hit", "verdict", "done"]
    assert events[1].data["kind"] == "cache"
    assert events[-1].data["claims"][0]["status"] == "CONTRADICTED"


async def test_factcheck_hit_short_circuits(pipeline):
    events = [e async for e in pipeline.stream(CheckInput(
        text="The state government is giving free laptops to all college students."))]
    assert [e.name for e in events] == ["claims_extracted", "cache_hit", "verdict", "done"]
    assert events[1].data["kind"] == "factcheck"
    assert events[-1].data["sources"][0]["publisher"] == "FactCheck Desk (MOCK)"


async def test_screenshot_end_to_end(pipeline):
    png = make_mock_png({"post_text": "Mumbai airport is closed for a week.", "post_date": "2h"})
    res = await pipeline.run(CheckInput(image=png, image_mime="image/png"))
    assert res.input_type == "screenshot" and res.claims[0].status == Status.CONTRADICTED


async def test_every_stage_logged(pipeline, store):
    res = await pipeline.run(CheckInput(text="Mumbai airport is closed for a week."))
    runs = [r for r in store.stage_runs if r.check_id == res.check_id]
    stages = [r.stage for r in runs]
    for s in ("ingest", "normalize", "extract", "embed", "cache", "factcheck", "retrieve", "judge", "write", "verify", "store"):
        assert s in stages
    assert all(r.latency_ms >= 0 for r in runs)
    verify = next(r for r in runs if r.stage == "verify")
    assert verify.model_version == "mock-1"
    assert verify.outputs["dropped"]  # the unsupported mock sentence
    assert res.model_versions["pipeline"] and res.model_versions["nli"] == "mock-1"
    assert await store.get_check(res.check_id) == res


async def test_claim_failure_abstains(pipeline):
    class Exploding(MockLLM):
        async def write_summary(self, claim, passages, status):
            raise RuntimeError("boom")

    pipeline.a.llm = Exploding()
    events = [e async for e in pipeline.stream(CheckInput(text="Mumbai airport is closed for a week."))]
    names = [e.name for e in events]
    assert "error" in names and names[-1] == "done"
    claim = events[-1].data["claims"][0]
    assert claim["status"] == "UNVERIFIED_EVIDENCE_MISSING" and claim["confidence"] == 0


async def test_fatal_failure_emits_error(pipeline):
    events = [e async for e in pipeline.stream(CheckInput(text="   "))]
    assert [e.name for e in events] == ["error"] and events[0].data["fatal"]


async def test_post_language_summary_stage_runs_and_keeps_verified_sentences(settings, adapters, store, whitelist):
    from app.pipeline.orchestrator import Pipeline

    p = Pipeline(settings.model_copy(update={"summary_language": "post"}), adapters, store, whitelist, clock=lambda: NOW)
    res = await p.run(CheckInput(text="नाशिकमध्ये गोदावरी नदीवरील पादचारी पूल रविवारी कोसळला."))
    assert res.claims[0].status == Status.CONFIRMED and res.claims[0].summary
    cited = {s.id for s in res.sources}
    assert all(set(s.sources) <= cited for s in res.claims[0].summary)
    assert "localize" in {r.stage for r in store.stage_runs}
