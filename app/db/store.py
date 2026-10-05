"""Persistence interface. Step 1 ships an in-memory store; step 2 adds Postgres + pgvector."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from app.models.schemas import DEFINITIVE_STATUSES, CheckResponse, ClaimResult, Entity, SourceOut


@dataclass
class CachedVerdict:
    claim: ClaimResult
    sources: list[SourceOut]
    similarity: float
    check_id: str


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
        self, embedding: list[float], entities: list[Entity], threshold: float, now: datetime
    ) -> CachedVerdict | None:
        """Best past DEFINITIVE verdict with cosine >= threshold and at least one shared entity."""
        ...

    async def save_claim(
        self,
        check_id: str,
        claim: ClaimResult,
        embedding: list[float],
        entities: list[Entity],
        sources: list[SourceOut],
        model_versions: dict[str, str],
        recheck_at: datetime | None,
    ) -> None: ...

    async def finish_check(self, response: CheckResponse) -> None: ...

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
    entities: set[str]
    sources: list[SourceOut]
    recheck_at: datetime | None


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
        self, embedding: list[float], entities: list[Entity], threshold: float, now: datetime
    ) -> CachedVerdict | None:
        keys = entity_keys(entities)
        best: CachedVerdict | None = None
        for row in self.claims:
            if row.claim.status not in DEFINITIVE_STATUSES:
                continue  # unverified results depend on claim age and on evidence that may appear later
            if row.recheck_at is not None and row.recheck_at <= now:
                continue  # stale: due for a recheck, do not serve
            if not keys & row.entities:
                continue
            sim = cosine(embedding, row.embedding)
            if sim >= threshold and (best is None or sim > best.similarity):
                best = CachedVerdict(claim=row.claim, sources=row.sources, similarity=sim, check_id=row.check_id)
        return best

    async def save_claim(
        self,
        check_id: str,
        claim: ClaimResult,
        embedding: list[float],
        entities: list[Entity],
        sources: list[SourceOut],
        model_versions: dict[str, str],
        recheck_at: datetime | None,
    ) -> None:
        self.claims.append(
            _StoredClaim(check_id, claim, embedding, entity_keys(entities), sources, recheck_at)
        )

    async def finish_check(self, response: CheckResponse) -> None:
        self.responses[response.check_id] = response

    async def get_check(self, check_id: str) -> CheckResponse | None:
        return self.responses.get(check_id)
