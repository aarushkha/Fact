"""Google Fact Check Tools API (claims:search).

From developers.google.com/fact-check/tools/api/reference/rest/v1alpha1/claims/search and the Claim /
ClaimReview resource docs, confirmed live:
  GET https://factchecktools.googleapis.com/v1alpha1/claims:search
      ?query=&languageCode=&pageSize=   API key in the x-goog-api-key header (Google's recommended way)
  -> {claims:[{text, claimant, claimDate, claimReview:[{publisher{name,site}, url, title, reviewDate,
     textualRating, languageCode}]}], nextPageToken}   ({} when nothing matches)
"""

from __future__ import annotations

from datetime import datetime

import httpx

from app.adapters.http import request_json
from app.models.schemas import FactCheckHit

URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def _dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def parse_claims(data: dict) -> list[FactCheckHit]:
    hits = []
    for claim in data.get("claims", []) or []:
        for review in claim.get("claimReview", []) or []:
            if not review.get("url") or not review.get("textualRating"):
                continue
            publisher = review.get("publisher") or {}
            hits.append(
                FactCheckHit(
                    claim_text=claim.get("text", ""),
                    claimant=claim.get("claimant"),
                    claim_date=_dt(claim.get("claimDate")),
                    review_url=review["url"],
                    review_title=review.get("title"),
                    publisher_name=publisher.get("name"),
                    publisher_site=publisher.get("site"),
                    textual_rating=review["textualRating"],
                    review_date=_dt(review.get("reviewDate")),
                    language=review.get("languageCode"),
                )
            )
    return hits


class GoogleFactCheckSearch:
    model_version = "google-factcheck/v1alpha1"

    def __init__(self, api_key: str, *, page_size: int = 10, timeout: float = 20, client: httpx.AsyncClient | None = None):
        if not api_key:
            raise ValueError("GOOGLE_FACTCHECK_API_KEY is not set")
        self._headers = {"x-goog-api-key": api_key}
        self.page_size = page_size
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def search(self, query: str, language: str | None = None) -> list[FactCheckHit]:
        params = {"query": query[:500], "pageSize": self.page_size}
        if language and language != "en":
            params["languageCode"] = language.split("-")[0]
        data = await request_json(self._client, "GET", URL, provider="google-factcheck", headers=self._headers, params=params)
        return parse_claims(data)
