"""Chains the stages, streams progress events and logs every stage.

Event order per check:  claims_extracted -> (cache_hit | evidence) -> verdict  (per claim) -> done
Claims are processed concurrently, so events of different claims interleave; each carries claim_id.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Awaitable, Callable
from zoneinfo import ZoneInfo

from app.adapters.base import Adapters
from app.config import Settings
from app.db.store import Store
from app.models.schemas import (
    CheckInput,
    CheckResponse,
    Claim,
    ClaimResult,
    ClaimType,
    ExpectedEvidence,
    Passage,
    SourceOut,
    Status,
)
from app.pipeline.context import StageContext, to_jsonable
from app.pipeline.extract import extract_claims
from app.pipeline.ingest import ingest
from app.pipeline.judge import Thresholds, claim_age_hours, judge_claim, recheck_at, unverified_status, would_change_if
from app.pipeline.match import decisive_factcheck, lookup_cache, match_factchecks
from app.pipeline.normalize import normalize
from app.pipeline.retrieve import rank_passages, retrieve
from app.pipeline.write import verify_sentences
from app.sources import Whitelist

log = logging.getLogger("fact.pipeline")
PIPELINE_VERSION = "pipeline-0.1"


@dataclass
class Event:
    name: str
    data: dict[str, Any]


Emit = Callable[[str, dict[str, Any]], Awaitable[None]]


def source_out(p: Passage) -> SourceOut:
    return SourceOut(id=p.source_id, url=p.url, publisher=p.publisher, tier=p.tier, published_at=p.published_at)


def passage_brief(p: Passage) -> dict[str, Any]:
    return {
        "passage_id": p.id,
        "source_id": p.source_id,
        "publisher": p.publisher,
        "tier": p.tier,
        "url": p.url,
        "language": p.language,
        "published_at": p.published_at.isoformat() if p.published_at else None,
        "snippet": p.text[:240],
    }


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        adapters: Adapters,
        store: Store,
        whitelist: Whitelist,
        clock: Callable[[], datetime] | None = None,
    ):
        self.settings = settings
        self.a = adapters
        self.store = store
        self.whitelist = whitelist
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.tz = ZoneInfo(settings.timezone)
        self.thresholds = Thresholds(
            confidence=settings.confidence_threshold,
            passage_relevance=settings.passage_relevance_threshold,
            too_early_window_hours=settings.too_early_window_hours,
        )

    # ------------------------------------------------------------------ public

    async def run(self, inp: CheckInput) -> CheckResponse:
        result: CheckResponse | None = None
        async for ev in self.stream(inp):
            if ev.name == "done":
                result = CheckResponse.model_validate(ev.data)
            elif ev.name == "error" and ev.data.get("fatal"):
                raise RuntimeError(ev.data.get("message"))
        assert result is not None
        return result

    async def stream(self, inp: CheckInput) -> AsyncIterator[Event]:
        queue: asyncio.Queue[Event | None] = asyncio.Queue()

        async def emit(name: str, data: dict[str, Any]) -> None:
            await queue.put(Event(name, to_jsonable(data)))

        async def runner() -> None:
            try:
                await self._run(inp, emit)
            except Exception as exc:
                log.exception("check failed")
                await emit("error", {"fatal": True, "message": f"{type(exc).__name__}: {exc}"})
            finally:
                await queue.put(None)

        task = asyncio.create_task(runner())
        try:
            while (ev := await queue.get()) is not None:
                yield ev
        finally:
            if not task.done():
                task.cancel()

    # ---------------------------------------------------------------- internal

    async def _run(self, inp: CheckInput, emit: Emit) -> None:
        a, s = self.a, self.settings
        now = self.clock()
        check_id = "chk_" + uuid.uuid4().hex[:16]
        ctx = StageContext(check_id, self.store)
        model_versions = {"pipeline": PIPELINE_VERSION, **a.model_versions()}
        await self.store.create_check(check_id, inp.input_type, {"post_date": inp.post_date, "created_at": now})

        ingested = await ctx.run(
            "ingest",
            lambda: ingest(inp, a.vision, now, self.tz),
            inputs={"input_type": inp.input_type, "text": inp.text, "image": inp.image, "post_date": inp.post_date},
            model_version=a.vision.model_version if inp.image else None,
        )
        norm, translator_used = await ctx.run(
            "normalize",
            lambda: normalize(ingested.text, a.translator, a.llm),
            inputs={"text": ingested.text},
            model_version=a.translator.model_version,
        )
        if translator_used == "llm":
            model_versions["translator"] = f"llm-fallback:{a.llm.model_version}"
        claims = await ctx.run(
            "extract",
            lambda: extract_claims(norm, a.llm, a.classifier, s.claim_type_threshold),
            inputs=norm,
            model_version=f"llm={a.llm.model_version};classifier={a.classifier.model_version}",
        )
        await emit(
            "claims_extracted",
            {
                "check_id": check_id,
                "input_type": ingested.input_type,
                "languages": norm.languages,
                "text_original": norm.text_original,
                "text_en": norm.text_en,
                "post": {
                    "account_handle": ingested.account_handle,
                    "post_date": ingested.post_date,
                    "image_description": ingested.image_description,
                },
                "claims": [
                    {"claim_id": c.id, "text_original": c.text_original, "text_en": c.text_en, "type": c.type}
                    for c in claims
                ],
            },
        )

        age = claim_age_hours(ingested.post_date, now)
        outcomes = await asyncio.gather(
            *(self._claim(ctx, c, norm.languages, age, now, model_versions, emit) for c in claims)
        )

        results = [r for r, _ in outcomes]
        sources: dict[str, SourceOut] = {}
        for _, srcs in outcomes:
            for src in srcs:
                sources.setdefault(src.id, src)
        response = CheckResponse(
            check_id=check_id,
            input_type=ingested.input_type,
            languages=norm.languages,
            claims=results,
            sources=list(sources.values()),
            model_versions=model_versions,
            checked_at=now,
            recheck_at=recheck_at(
                [r.status for r in results], now, s.recheck_too_early_hours, s.recheck_evidence_missing_days
            ),
        )
        await ctx.run("store", lambda: self.store.finish_check(response), inputs={"claims": len(results)})
        await emit("done", response.model_dump(mode="json"))

    async def _claim(
        self,
        ctx: StageContext,
        claim: Claim,
        languages: list[str],
        age: float | None,
        now: datetime,
        model_versions: dict[str, str],
        emit: Emit,
    ) -> tuple[ClaimResult, list[SourceOut]]:
        try:
            return await self._claim_inner(ctx, claim, languages, age, now, model_versions, emit)
        except Exception as exc:
            # Abstain on failure; never guess.
            log.exception("claim %s failed", claim.id)
            result = ClaimResult(
                text_original=claim.text_original,
                text_en=claim.text_en,
                type=claim.type,
                status=unverified_status(age, self.thresholds.too_early_window_hours),
                confidence=0.0,
                would_change_if="The check is re-run successfully.",
            )
            await emit("error", {"claim_id": claim.id, "fatal": False, "message": f"{type(exc).__name__}: {exc}"})
            await emit("verdict", {"claim_id": claim.id, "claim": result, "sources": []})
            return result, []

    async def _claim_inner(
        self,
        ctx: StageContext,
        claim: Claim,
        languages: list[str],
        age: float | None,
        now: datetime,
        model_versions: dict[str, str],
        emit: Emit,
    ) -> tuple[ClaimResult, list[SourceOut]]:
        a, s = self.a, self.settings
        cid = claim.id

        def result(**kw: Any) -> ClaimResult:
            return ClaimResult(text_original=claim.text_original, text_en=claim.text_en, type=claim.type, **kw)

        # Stage 3 exit: not-checkable claims stop here.
        if claim.type != ClaimType.CHECKABLE:
            res = result(
                status=Status.NOT_CHECKABLE,
                confidence=round(claim.type_confidence, 4),
                would_change_if=would_change_if(Status.NOT_CHECKABLE, []),
            )
            await emit("verdict", {"claim_id": cid, "claim": res, "sources": []})
            return res, []

        # Stage 4: cache + fact-check match.
        (embedding,) = await ctx.run(
            "embed", lambda: a.embedder.embed([claim.text_en]), inputs={"text": claim.text_en},
            model_version=a.embedder.model_version, claim_id=cid,
            log_output=lambda vecs: {"vectors": len(vecs), "dim": len(vecs[0]) if vecs else 0},
        )
        cached, fc_matches = await asyncio.gather(
            ctx.run(
                "cache", lambda: lookup_cache(claim, embedding, self.store, s.cache_similarity_threshold, now),
                inputs={"entities": claim.entities}, claim_id=cid,
            ),
            ctx.run(
                "factcheck",
                lambda: match_factchecks(
                    claim, embedding, languages, a.factcheck, a.embedder, self.whitelist,
                    s.factcheck_similarity_threshold,
                ),
                inputs={"text_en": claim.text_en, "text_original": claim.text_original},
                model_version=a.factcheck.model_version, claim_id=cid,
            ),
        )
        if cached is not None:
            res = cached.claim.model_copy(
                update={"text_original": claim.text_original, "text_en": claim.text_en, "type": claim.type}
            )
            await emit("cache_hit", {
                "claim_id": cid, "kind": "cache", "similarity": round(cached.similarity, 4),
                "from_check_id": cached.check_id,
            })
            await emit("verdict", {"claim_id": cid, "claim": res, "sources": cached.sources})
            return res, cached.sources

        decisive = decisive_factcheck(fc_matches)
        if decisive is not None and decisive.status is not None and s.factcheck_hit_confidence >= s.confidence_threshold:
            await emit("cache_hit", {
                "claim_id": cid, "kind": "factcheck", "similarity": round(decisive.similarity, 4),
                "publisher": decisive.passage.publisher, "rating": decisive.hit.textual_rating,
                "url": decisive.hit.review_url,
            })
            status, confidence = decisive.status, s.factcheck_hit_confidence
            passages = [decisive.passage]
            expected = [ExpectedEvidence(item="Published fact-check by a whitelisted fact-checker", found=True)]
        else:
            # Stage 5: retrieve.
            passages = await ctx.run(
                "retrieve",
                lambda: retrieve(claim, languages, embedding, a.search, self.whitelist, s.retrieval_top_k, a.web_search),
                inputs={"text_en": claim.text_en, "text_original": claim.text_original},
                model_version=a.search.model_version, claim_id=cid,
            )
            # Whitelisted fact-checks that did not short-circuit still count as evidence.
            passages = rank_passages(passages + [m.passage for m in fc_matches])
            await emit("evidence", {"claim_id": cid, "passages": [passage_brief(p) for p in passages]})

            # Stage 6: judge.
            status, confidence, judgments, expected, effective = await ctx.run(
                "judge",
                lambda: judge_claim(claim, passages, a.classifier, age, self.thresholds),
                inputs={"claim": claim, "passage_ids": [p.id for p in passages], "age_hours": age},
                model_version=a.classifier.model_version, claim_id=cid,
            )
            passages = [p for _, p in effective]

        # Stage 7: write + verify. Only best-tier relevant passages are offered to the writer.
        drafts = await ctx.run(
            "write",
            lambda: a.llm.write_summary(claim, passages, status.value) if passages else _empty(),
            inputs={"status": status, "passage_ids": [p.id for p in passages]},
            model_version=a.llm.model_version, claim_id=cid,
        )
        report = await ctx.run(
            "verify",
            lambda: verify_sentences(drafts, passages, a.nli, s.nli_entailment_threshold),
            inputs={"drafts": drafts}, model_version=a.nli.model_version, claim_id=cid,
        )

        res = result(
            status=status,
            confidence=round(confidence, 4),
            summary=report.kept,
            expected_evidence=expected,
            would_change_if=would_change_if(status, expected),
        )
        sources = [source_out(p) for p in report.cited_passages]
        sources = list({src.id: src for src in sources}.values())

        # Stage 8: store.
        claim_recheck = recheck_at([status], now, s.recheck_too_early_hours, s.recheck_evidence_missing_days)
        await ctx.run(
            "store",
            lambda: self.store.save_claim(ctx.check_id, res, embedding, claim.entities, sources, model_versions, claim_recheck),
            inputs={"status": status}, claim_id=cid,
        )
        await emit("verdict", {"claim_id": cid, "claim": res, "sources": sources})
        return res, sources


async def _empty() -> list:
    return []
