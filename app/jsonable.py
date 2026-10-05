"""Convert pipeline values (pydantic models, dataclasses, enums, bytes, datetimes) to plain JSON data."""

from __future__ import annotations

import hashlib
from dataclasses import fields, is_dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel


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
