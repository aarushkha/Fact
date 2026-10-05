from datetime import datetime, timezone

from app.models.schemas import Claim, ClaimType, Entity, Passage, PassageJudgment, Stance


def passage(pid: str, tier: int = 2, relevance: float = 0.5, url: str | None = None, text: str = "text",
            published_at: datetime | None = None, kind: str | None = None) -> Passage:
    url = url or f"https://{pid}.example/"
    return Passage(id=pid, source_id=f"src_{pid}", url=url, publisher=pid, tier=tier, relevance=relevance,
                   text=text, published_at=published_at, kind=kind)


def judgment(pid: str, stance: Stance, p: float = 0.9) -> PassageJudgment:
    return PassageJudgment(passage_id=pid, stance=stance, probability=p)


def claim(text: str = "Mumbai airport is closed for a week.", entities=("Mumbai",)) -> Claim:
    return Claim(id="c1", text_original=text, text_en=text, type=ClaimType.CHECKABLE, type_confidence=0.9,
                 entities=[Entity(text=e, kind="place") for e in entities])


def ts(day: int) -> datetime:
    return datetime(2026, 9, day, tzinfo=timezone.utc)
