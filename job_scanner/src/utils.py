from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def serialize_dt(dt: Any) -> str:
    if hasattr(dt, "isoformat"):
        return dt.isoformat()
    return str(dt)
