"""Postgres implementation of Store: checks, stage_runs, claims, verdicts."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.store import CachedVerdict, DueRecheck, StageRun, entity_keys
from app.db.tables import Tables
from app.jsonable import to_jsonable
from app.models.schemas import DEFINITIVE_STATUSES, UNVERIFIED_STATUSES, CheckResponse, ClaimResult, Entity, SourceOut


def _json(value: Any) -> Any:
    """JSON-safe value for a JSONB column (Postgres JSONB rejects NUL characters)."""
    value = to_jsonable(value)
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json(v) for v in value]
    return value


class PgStore:
    def __init__(self, engine: AsyncEngine, tables: Tables):
        self.engine = engine
        self.t = tables

    async def create_check(self, check_id: str, input_type: str, meta: dict) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(insert(self.t.checks).values(id=check_id, input_type=input_type, meta=_json(meta)))

    async def log_stage(self, run: StageRun) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                insert(self.t.stage_runs).values(
                    check_id=run.check_id,
                    claim_id=run.claim_id,
                    stage=run.stage,
                    inputs=_json(run.inputs),
                    outputs=_json(run.outputs),
                    latency_ms=run.latency_ms,
                    model_version=run.model_version,
                    error=run.error,
                    started_at=run.started_at,
                )
            )

    async def find_cached(
        self, embedding: list[float], embedding_model: str, entities: list[Entity], threshold: float, now: datetime
    ) -> CachedVerdict | None:
        keys = sorted(entity_keys(entities))
        if not keys:
            return None
        c, v = self.t.claims, self.t.verdicts
        distance = c.c.embedding.cosine_distance(embedding)
        stmt = (
            select(c.c.check_id, v.c.result, v.c.sources, (1 - distance).label("similarity"))
            .select_from(c.join(v, v.c.claim_id == c.c.id))
            .where(
                v.c.status.in_([s.value for s in DEFINITIVE_STATUSES]),
                v.c.superseded_at.is_(None),
                (v.c.recheck_at.is_(None)) | (v.c.recheck_at > now),
                c.c.embedding_model == embedding_model,
                c.c.entity_keys.overlap(keys),
            )
            .order_by(distance)
            .limit(1)
        )
        async with self.engine.connect() as conn:
            row = (await conn.execute(stmt)).first()
        if row is None or row.similarity < threshold:
            return None
        return CachedVerdict(
            claim=ClaimResult.model_validate(row.result),
            sources=[SourceOut.model_validate(s) for s in row.sources],
            similarity=float(row.similarity),
            check_id=row.check_id,
        )

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
    ) -> None:
        async with self.engine.begin() as conn:
            claim_id = (
                await conn.execute(
                    insert(self.t.claims)
                    .values(
                        check_id=check_id,
                        claim_key=claim_key,
                        text_original=claim.text_original,
                        text_en=claim.text_en,
                        type=claim.type.value,
                        entities=_json(entities),
                        entity_keys=sorted(entity_keys(entities)),
                        embedding=embedding,
                        embedding_model=embedding_model,
                        post_date=post_date,
                        signals=_json(signals) if signals else None,
                    )
                    .returning(self.t.claims.c.id)
                )
            ).scalar_one()
            await conn.execute(
                insert(self.t.verdicts).values(
                    claim_id=claim_id,
                    status=claim.status.value,
                    confidence=claim.confidence,
                    result=_json(claim),
                    sources=_json(sources),
                    model_versions=_json(model_versions),
                    recheck_at=recheck_at,
                )
            )

    async def due_rechecks(self, now: datetime, oldest: datetime, limit: int) -> list[DueRecheck]:
        c, v = self.t.claims, self.t.verdicts
        stmt = (
            select(v.c.id, c.c.check_id, c.c.claim_key, c.c.text_original, c.c.text_en, v.c.status, c.c.post_date,
                   v.c.recheck_at)
            .select_from(v.join(c, c.c.id == v.c.claim_id))
            .where(
                v.c.status.in_([s.value for s in UNVERIFIED_STATUSES]),
                v.c.superseded_at.is_(None),
                v.c.recheck_at <= now,
                func.coalesce(c.c.post_date, c.c.created_at) >= oldest,
            )
            .order_by(v.c.recheck_at)
            .limit(limit)
        )
        async with self.engine.connect() as conn:
            return [DueRecheck(*row) for row in (await conn.execute(stmt)).all()]

    async def link_recheck(self, old_verdict_id: int, new_check_id: str, claim_key: str) -> None:
        c, v = self.t.claims, self.t.verdicts
        async with self.engine.begin() as conn:
            await conn.execute(update(v).where(v.c.id == old_verdict_id).values(superseded_at=func.now()))
            new_claim_ids = select(c.c.id).where(c.c.check_id == new_check_id, c.c.claim_key == claim_key)
            await conn.execute(update(v).where(v.c.claim_id.in_(new_claim_ids)).values(rechecked_from=old_verdict_id))

    async def postpone_recheck(self, verdict_id: int, until: datetime) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(update(self.t.verdicts).where(self.t.verdicts.c.id == verdict_id).values(recheck_at=until))

    async def finish_check(self, response: CheckResponse) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                update(self.t.checks)
                .where(self.t.checks.c.id == response.check_id)
                .values(status="done", finished_at=func.now(), response=_json(response))
            )

    async def fail_check(self, check_id: str, error: str) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                update(self.t.checks)
                .where(self.t.checks.c.id == check_id)
                .values(status="error", finished_at=func.now(), meta=self.t.checks.c.meta.op("||")(_json({"error": error})))
            )

    async def get_check(self, check_id: str) -> CheckResponse | None:
        async with self.engine.connect() as conn:
            row = (
                await conn.execute(select(self.t.checks.c.response).where(self.t.checks.c.id == check_id))
            ).first()
        if row is None or row.response is None:
            return None
        return CheckResponse.model_validate(row.response)
