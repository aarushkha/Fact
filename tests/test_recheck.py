from datetime import timedelta

from app.models.schemas import CheckInput, Status
from app.pipeline.recheck import run_rechecks

from .conftest import NOW


async def test_too_early_claim_is_rechecked_and_superseded(pipeline, store):
    text = "A fire broke out at a chemical factory in Thane."
    first = await pipeline.run(CheckInput(text=text, post_date=NOW - timedelta(hours=1)))
    assert first.claims[0].status == Status.UNVERIFIED_TOO_EARLY
    assert await run_rechecks(pipeline) == []  # not due yet

    pipeline.clock = lambda: NOW + timedelta(days=4)  # due, and now older than the 72 h window
    results = await run_rechecks(pipeline)
    assert [(r["old"], r["new"]) for r in results] == [("UNVERIFIED_TOO_EARLY", "UNVERIFIED_EVIDENCE_MISSING")]
    old, new = store.claims[0], store.claims[1]
    assert old.superseded and new.rechecked_from == old.id and new.post_date == old.post_date
    assert await run_rechecks(pipeline) == []  # superseded rows are not picked again


async def test_definitive_and_old_claims_are_not_rechecked(pipeline):
    await pipeline.run(CheckInput(text="Mumbai airport is closed for a week."))
    await pipeline.run(CheckInput(text="A fire broke out at a chemical factory in Thane.",
                                  post_date=NOW - timedelta(days=60)))
    pipeline.clock = lambda: NOW + timedelta(days=10)
    assert await run_rechecks(pipeline) == []


async def test_failed_recheck_is_postponed(pipeline, store):
    await pipeline.run(CheckInput(text="A fire broke out at a chemical factory in Thane.", post_date=NOW))
    pipeline.clock = lambda: NOW + timedelta(hours=7)

    async def boom(*a, **k):
        raise RuntimeError("down")

    pipeline.run = boom
    results = await run_rechecks(pipeline)
    assert "error" in results[0] and store.claims[0].recheck_at == NOW + timedelta(hours=8)


async def test_recheck_without_a_new_verdict_keeps_the_old_one(pipeline, store):
    await pipeline.run(CheckInput(text="A fire broke out at a chemical factory in Thane.", post_date=NOW))
    pipeline.clock = lambda: NOW + timedelta(hours=7)
    real_run = pipeline.run

    async def opinion_only(inp):  # e.g. the recheck run classified the text NOT_CHECKABLE: nothing stored
        return await real_run(CheckInput(text="Nashik is the most beautiful city in India.", single_claim=True))

    pipeline.run = opinion_only
    results = await run_rechecks(pipeline)
    assert results[0]["postponed"] and results[0]["new"] == "NOT_CHECKABLE"
    assert not store.claims[0].superseded and store.claims[0].recheck_at == NOW + timedelta(hours=8)
