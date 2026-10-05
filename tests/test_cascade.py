from datetime import timedelta

from app.models.schemas import CheckInput
from app.pipeline.cascade import support_by_tier

from .helpers import judgment, passage


async def verdict_signals(pipeline, text):
    return [e.data["signals"] async for e in pipeline.stream(CheckInput(text=text)) if e.name == "verdict"][0]


async def test_repeat_submissions_are_counted(pipeline):
    text = "A fire broke out at a chemical factory in Thane."
    first = await verdict_signals(pipeline, text)
    assert first["similar_submissions_24h"] == 0
    await verdict_signals(pipeline, text)
    third = await verdict_signals(pipeline, "A fire broke out at a chemical factory in Thane!")
    assert third["similar_submissions_24h"] == 2 and third["similar_submissions_7d"] == 2


async def test_cache_hits_are_counted_too(pipeline):
    text = "Mumbai airport is closed for a week."
    await verdict_signals(pipeline, text)
    second = await verdict_signals(pipeline, text)  # served from cache, still recorded
    assert second["similar_submissions_24h"] == 1


async def test_window_excludes_old_submissions(pipeline):
    text = "A fire broke out at a chemical factory in Thane."
    now = pipeline.clock()
    await verdict_signals(pipeline, text)
    pipeline.clock = lambda: now + timedelta(days=2)
    later = await verdict_signals(pipeline, text)
    assert later["similar_submissions_24h"] == 0 and later["similar_submissions_7d"] == 1
    assert later["first_seen"] == now.isoformat()


async def test_echo_only_when_only_aggregators_support(pipeline):
    # tier-3 aggregator repeats the airport rumour; the tier-1 authority contradicts it
    sig = await verdict_signals(pipeline, "Mumbai airport is closed for a week.")
    assert sig["support_by_tier"] == {"1": 0, "2": 0, "3": 1} and sig["echo_only"] is True
    sig = await verdict_signals(pipeline, "A footbridge over the Godavari river in Nashik collapsed on Sunday.")
    assert sig["echo_only"] is False


def test_support_by_tier_ignores_weak_and_irrelevant():
    from app.models.schemas import Stance

    ps = [passage("a", tier=1), passage("b", tier=3), passage("c", tier=2)]
    js = [judgment("a", Stance.SUPPORTS, 0.4), judgment("b", Stance.SUPPORTS), judgment("c", Stance.IRRELEVANT)]
    assert support_by_tier(ps, js, 0.5) == {"1": 0, "2": 0, "3": 1}


async def test_distinct_accounts_counted_from_screenshots(pipeline):
    from app.adapters.mock import make_mock_png

    async def post(handle):
        img = make_mock_png({"post_text": "A fire broke out at a chemical factory in Thane.", "account_handle": handle,
                             "post_date": "1h"})
        inp = CheckInput(image=img, image_mime="image/png")
        return [e.data["signals"] async for e in pipeline.stream(inp) if e.name == "verdict"][0]

    assert (await post("@alpha"))["distinct_accounts_7d"] == 1
    assert (await post("@Alpha "))["distinct_accounts_7d"] == 1  # same account, normalised
    third = await post("@beta")
    assert third["distinct_accounts_7d"] == 2 and third["similar_submissions_7d"] == 2
    # A text submission has no handle: the count of earlier accounts still shows.
    assert (await verdict_signals(pipeline, "A fire broke out at a chemical factory in Thane."))["distinct_accounts_7d"] == 2


async def test_no_account_signal_without_handles(pipeline):
    assert "distinct_accounts_7d" not in await verdict_signals(pipeline, "A fire broke out at a chemical factory in Thane.")
