"""Optional paid web-search fallback (off by default: WEB_SEARCH_ENABLED=false).

TODO: no provider has been chosen. Pick one, read its official API docs, and implement `search`
returning Passages for whitelisted URLs only (the pipeline also re-applies the whitelist).
"""

from __future__ import annotations

from datetime import datetime

from app.models.schemas import Passage


class WebSearchStub:
    model_version = "web-search/stub"

    async def search(
        self,
        queries: list[tuple[str, str | None]],
        embedding: list[float] | None,
        k: int,
        published_after: datetime | None = None,
    ) -> list[Passage]:
        raise NotImplementedError("Web-search fallback is a stub (TODO: choose a provider).")
