"""Stage 1: ingest. Screenshot -> vision reader; text input passes straight through."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, tzinfo

from app.adapters.base import VisionReader
from app.models.schemas import CheckInput, Ingested

_REL_RE = re.compile(
    r"^(\d+)\s*(s|sec|secs|seconds?|m|min|mins|minutes?|h|hr|hrs|hours?|d|days?|w|wk|wks|weeks?)(\s+ago)?$"
)
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
_ABS_FORMATS = ("%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%d/%m/%Y", "%d-%m-%Y")
_NO_YEAR_FORMATS = ("%d %B", "%d %b", "%B %d", "%b %d")


def ensure_aware(dt: datetime, tz: tzinfo) -> datetime:
    return dt.replace(tzinfo=tz) if dt.tzinfo is None else dt


def parse_post_date(raw: str | None, now: datetime, tz: tzinfo) -> datetime | None:
    """Parse a date as printed on a post: "2h", "3 days ago", "yesterday", "2026-10-01", "3 March".

    Returns None when the date cannot be read; callers must then treat the claim age as unknown.
    """
    if not raw:
        return None
    s = raw.strip().lower().replace("·", "").strip()
    if s in {"just now", "now"}:
        return now
    if s == "yesterday":
        return now - timedelta(days=1)
    m = _REL_RE.match(s)
    if m:
        unit = m.group(2)
        key = "m" if unit.startswith("m") else unit[0]
        return now - timedelta(seconds=int(m.group(1)) * _UNIT_SECONDS[key])
    try:
        return ensure_aware(datetime.fromisoformat(raw.strip()), tz)
    except ValueError:
        pass
    for fmt in _ABS_FORMATS:
        try:
            return datetime.strptime(raw.strip(), fmt).replace(tzinfo=tz)
        except ValueError:
            continue
    for fmt in _NO_YEAR_FORMATS:
        try:
            parsed = datetime.strptime(f"{raw.strip()} {now.year}", f"{fmt} %Y").replace(tzinfo=tz)
        except ValueError:
            continue
        # A post date without a year that lands in the future must be from last year.
        return parsed.replace(year=now.year - 1) if parsed > now else parsed
    return None


async def ingest(inp: CheckInput, vision: VisionReader, now: datetime, tz: tzinfo) -> Ingested:
    user_date = ensure_aware(inp.post_date, tz) if inp.post_date else None
    if inp.image:
        read = await vision.read(inp.image, inp.image_mime)
        text = read.post_text
        if inp.text and inp.text.strip():
            text = f"{text}\n{inp.text.strip()}"
        return Ingested(
            input_type="screenshot",
            text=text,
            account_handle=read.account_handle,
            post_date=user_date or parse_post_date(read.post_date_raw, now, tz),
            image_description=read.image_description,
        )
    if not inp.text or not inp.text.strip():
        raise ValueError("Provide a screenshot or some text.")
    return Ingested(input_type="text", text=inp.text.strip(), post_date=user_date)
