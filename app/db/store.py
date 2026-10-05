"""Persistence interface. Step 1 ships an in-memory store; step 2 adds Postgres + pgvector."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from app.models.schemas import DEFINITIVE_STATUSES, CheckResponse, ClaimResult, Entity, SourceOut


@dataclass
class CachedVerdict:
    claim: ClaimResult
    sources: list[SourceOut]
    similarity: float
    check_id: str


@dataclass
class DueRecheck:
    verdict_id: int
    check_id: str
    claim_key: str
    text_original: str
    text_en: str
    status: str
    post_date: datetime | None
    recheck_at: datetime


@dataclass
class StageRun:
    check_id: str
    claim_id: str | None
    stage: str
    inputs: Any
    outputs: Any
    latency_ms: float
    model_version: str | None
    error: str | None
    started_at: datetime


class Store(Protocol):
    async def create_check(self, check_id: str, input_type: str, meta: dict) -> None: ...

    async def log_stage(self, run: StageRun) -> None: ...

    async def find_cached(
        self, embedding: list[float], embedding_model: str, entities: list[Entity], threshold: float, now: datetime
    ) -> CachedVerdict | None:
        """Best past DEFINITIVE verdict with cosine >= threshold and at least one shared entity.

        Only claims embedded by the same model are compared.
        """
        ...

    async def save_claim(
        self,
        check_id: str,
        claim_key: str,
        claim: ClaimResult,
        embedding: list[float],
        embedding_model: str,
        entities: list[Entity],
        sources: list[SourceOut],
        model_versions: dict[str, str],
        recheck_at: datetime | None,
        post_date: datetime | None = None,
        signals: dict | None = None,
        created_at: datetime | None = None,
    ) -> None: ...

    async def similar_claim_stats(
        self, embedding: list[float], embedding_model: str, since: datetime, now: datetime, threshold: float
    ) -> dict:
        """{count_24h, count_7d, first_seen} over earlier claims with cosine >= threshold, since `since`."""
        ...

    async def due_rechecks(self, now: datetime, oldest: datetime, limit: int) -> list[DueRecheck]:
        """UNVERIFIED verdicts whose recheck_at has passed, not yet superseded, claim newer than `oldest`."""
        ...

    async def link_recheck(self, old_verdict_id: int, new_check_id: str, claim_key: str) -> None:
        """Mark the old verdict superseded and point the new check's verdict at it."""
        ...

    async def postpone_recheck(self, verdict_id: int, until: datetime) -> None: ...

    async def finish_check(self, response: CheckResponse) -> None: ...

    async def fail_check(self, check_id: str, error: str) -> None: ...

    async def get_check(self, check_id: str) -> CheckResponse | None: ...


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def entity_keys(entities: list[Entity]) -> set[str]:
    return {e.text.strip().lower() for e in entities if e.text.strip()}


@dataclass
class _StoredClaim:
    check_id: str
    claim: ClaimResult
    embedding: list[float]
    embedding_model: str
    entities: set[str]
    sources: list[SourceOut]
    recheck_at: datetime | None
    id: int = 0
    claim_key: str = ""
    post_date: datetime | None = None
    created_at: datetime | None = None
    signals: dict | None = None
    rechecked_from: int | None = None
    superseded: bool = False


@dataclass
class InMemoryStore:
    checks: dict[str, dict] = field(default_factory=dict)
    responses: dict[str, CheckResponse] = field(default_factory=dict)
    stage_runs: list[StageRun] = field(default_factory=list)
    claims: list[_StoredClaim] = field(default_factory=list)

    async def create_check(self, check_id: str, input_type: str, meta: dict) -> None:
        self.checks[check_id] = {"input_type": input_type, **meta}

    async def log_stage(self, run: StageRun) -> None:
        self.stage_runs.append(run)

    async def find_cached(
        self, embedding: list[float], embedding_model: str, entities: list[Entity], threshold: float, now: datetime
    ) -> CachedVerdict | None:
        keys = entity_keys(entities)
        best: CachedVerdict | None = None
        for row in self.claims:
            if row.claim.status not in DEFINITIVE_STATUSES:
                continue  # unverified results depend on claim age and on evidence that may appear later
            if row.embedding_model != embedding_model:
                continue
            if row.superseded or (row.recheck_at is not None and row.recheck_at <= now):
                continue  # stale: superseded or due for a recheck, do not serve
            if not keys & row.entities:
                continue
            sim = cosine(embedding, row.embedding)
            if sim >= threshold and (best is None or sim > best.similarity):
                best = CachedVerdict(claim=row.claim, sources=row.sources, similarity=sim, check_id=row.check_id)
        return best

    async def save_claim(
        self,
        check_id: str,
        claim_key: str,
        claim: ClaimResult,
        embedding: list[float],
        embedding_model: str,
        entities: list[Entity],
        sources: list[SourceOut],
        model_versions: dict[str, str],
        recheck_at: datetime | None,
        post_date: datetime | None = None,
        signals: dict | None = None,
        created_at: datetime | None = None,
    ) -> None:
        self.claims.append(
            _StoredClaim(
                check_id, claim, embedding, embedding_model, entity_keys(entities), sources, recheck_at,
                id=len(self.claims) + 1, claim_key=claim_key, post_date=post_date,
                created_at=created_at or datetime.now(timezone.utc), signals=signals,
            )
        )

    async def similar_claim_stats(
        self, embedding: list[float], embedding_model: str, since: datetime, now: datetime, threshold: float
    ) -> dict:
        from datetime import timedelta

        hits = [
            r.created_at for r in self.claims
            if r.embedding_model == embedding_model and r.created_at and since <= r.created_at <= now
            and cosine(embedding, r.embedding) >= threshold
        ]
        return {
            "count_24h": sum(1 for t in hits if t >= now - timedelta(hours=24)),
            "count_7d": len(hits),
            "first_seen": min(hits) if hits else None,
        }

    async def due_rechecks(self, now: datetime, oldest: datetime, limit: int) -> list[DueRecheck]:
        out = []
        for r in self.claims:
            if r.superseded or r.recheck_at is None or r.recheck_at > now or r.claim.status in DEFINITIVE_STATUSES:
                continue
            if (r.post_date or r.created_at or now) < oldest:
                continue
            out.append(DueRecheck(r.id, r.check_id, r.claim_key, r.claim.text_original, r.claim.text_en,
                                  r.claim.status.value, r.post_date, r.recheck_at))
        return sorted(out, key=lambda d: d.recheck_at)[:limit]

    async def link_recheck(self, old_verdict_id: int, new_check_id: str, claim_key: str) -> None:
        for r in self.claims:
            if r.id == old_verdict_id:
                r.superseded = True
            elif r.check_id == new_check_id and r.claim_key == claim_key:
                r.rechecked_from = old_verdict_id

    async def postpone_recheck(self, verdict_id: int, until: datetime) -> None:
        for r in self.claims:
            if r.id == verdict_id:
                r.recheck_at = until

    async def finish_check(self, response: CheckResponse) -> None:
        self.responses[response.check_id] = response

    async def fail_check(self, check_id: str, error: str) -> None:
        self.checks.setdefault(check_id, {})["error"] = error

    async def get_check(self, check_id: str) -> CheckResponse | None:
        return self.responses.get(check_id)
