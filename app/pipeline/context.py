"""Per-check context: logs every stage's inputs, outputs, latency and model version."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, TypeVar

from pydantic import BaseModel

from app.db.store import StageRun, Store

log = logging.getLogger("fact.pipeline")
T = TypeVar("T")


def to_jsonable(value: Any) -> Any:
    """Make stage inputs/outputs loggable. Raw bytes are logged as a hash + size, never stored."""
    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable({f.name: getattr(value, f.name) for f in fields(value)})
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
    if isinstance(value, dict):
        return {str(k.value if hasattr(k, "value") else k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):  # enums
        return value.value
    return value


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
        model_version: str | None = None,
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
                    model_version=model_version,
                    error=error,
                    started_at=started,
                )
            )
