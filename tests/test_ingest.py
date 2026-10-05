from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from app.adapters.mock import MockVisionReader, make_mock_png
from app.models.schemas import CheckInput
from app.pipeline.ingest import ingest, parse_post_date

from .conftest import NOW

IST = ZoneInfo("Asia/Kolkata")


@pytest.mark.parametrize(
    "raw,delta",
    [("2h", timedelta(hours=2)), ("45 minutes ago", timedelta(minutes=45)), ("3d", timedelta(days=3)),
     ("1w", timedelta(weeks=1)), ("yesterday", timedelta(days=1)), ("just now", timedelta(0))],
)
def test_relative_dates(raw, delta):
    assert parse_post_date(raw, NOW, IST) == NOW - delta


def test_absolute_dates():
    assert parse_post_date("2026-10-01", NOW, IST) == datetime(2026, 10, 1, tzinfo=IST)
    assert parse_post_date("2026-10-01T10:00:00+00:00", NOW, IST) == datetime(2026, 10, 1, 10, tzinfo=timezone.utc)
    assert parse_post_date("3 March 2026", NOW, IST) == datetime(2026, 3, 3, tzinfo=IST)


def test_date_without_year_never_in_future():
    assert parse_post_date("3 March", NOW, IST).year == 2026
    assert parse_post_date("25 December", NOW, IST).year == 2025


def test_unreadable_date_is_none():
    assert parse_post_date("sometime", NOW, IST) is None
    assert parse_post_date(None, NOW, IST) is None


async def test_text_input_skips_vision():
    out = await ingest(CheckInput(text="  hello  "), MockVisionReader(), NOW, IST)
    assert out.input_type == "text" and out.text == "hello" and out.post_date is None


async def test_screenshot_uses_vision_and_relative_date():
    png = make_mock_png({"post_text": "Mumbai airport is closed.", "account_handle": "@x", "post_date": "2h"})
    out = await ingest(CheckInput(image=png, image_mime="image/png"), MockVisionReader(), NOW, IST)
    assert out.input_type == "screenshot"
    assert out.text == "Mumbai airport is closed."
    assert out.account_handle == "@x"
    assert out.post_date == NOW - timedelta(hours=2)


async def test_user_post_date_overrides_screenshot_date():
    png = make_mock_png({"post_text": "x y z", "post_date": "2h"})
    user = datetime(2026, 9, 1, 9, 0)  # naive -> interpreted in configured timezone
    out = await ingest(CheckInput(image=png, post_date=user), MockVisionReader(), NOW, IST)
    assert out.post_date == user.replace(tzinfo=IST)


async def test_empty_input_rejected():
    with pytest.raises(ValueError):
        await ingest(CheckInput(text="   "), MockVisionReader(), NOW, IST)
