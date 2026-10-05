"""Per-check context: logs every stage's inputs, outputs, latency and model version."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, TypeVar

from app.db.store import StageRun, Store
from app.jsonable import to_jsonable

log = logging.getLogger("fact.pipeline")
T = TypeVar("T")


@dataclass
class StageContext:
    check_id: str
    store: Store

    async def run(
        self,
        stage: str,
        fn: Callable[[], Awaitable[T]],
        *,
        inputs: Any = None,
        model_version: str | Callable[[], str] | None = None,
        claim_id: str | None = None,
        log_output: Callable[[Any], Any] | None = None,
    ) -> T:
        started = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        output: Any = None
        error: str | None = None
        try:
            output = await fn()
            return output
        except Exception as exc:  # logged, then re-raised
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000
            log.debug("stage=%s claim=%s %.1fms error=%s", stage, claim_id, latency_ms, error)
            await self.store.log_stage(
                StageRun(
                    check_id=self.check_id,
                    claim_id=claim_id,
                    stage=stage,
                    inputs=to_jsonable(inputs),
                    outputs=to_jsonable(log_output(output) if log_output and output is not None else output),
                    latency_ms=latency_ms,
                    # Resolved after the call so fallbacks (e.g. another Gemini model) are recorded.
                    model_version=model_version() if callable(model_version) else model_version,
                    error=error,
                    started_at=started,
                )
            )
