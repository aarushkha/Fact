"""Adapter interfaces. Every external model sits behind one of these.

Each adapter exposes `model_version`, which is logged per stage and returned in the response.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from app.models.schemas import (
    Claim,
    ClaimJudgment,
    ClaimType,
    DraftSentence,
    ExpectedEvidence,
    FactCheckHit,
    NLIScore,
    Passage,
    PassageJudgment,
    RawClaim,
    VisionResult,
)


@runtime_checkable
class VisionReader(Protocol):
    model_version: str

    async def read(self, image: bytes, mime: str | None) -> VisionResult: ...


@runtime_checkable
class Translator(Protocol):
    model_version: str

    async def detect(self, text: str) -> list[str]:
        """Language codes present, most prominent first. Romanized Hindi is "hi-Latn"."""
        ...

    async def translate(self, text: str, source: str, target: str = "en") -> str: ...


@runtime_checkable
class LLM(Protocol):
    model_version: str

    async def extract_claims(self, text_original: str, text_en: str, languages: list[str]) -> list[RawClaim]: ...

    async def write_summary(self, claim: Claim, passages: list[Passage], status: str) -> list[DraftSentence]:
        """1-3 sentences, each citing ids of `passages` only."""
        ...

    async def translate(self, text: str, source: str, target: str = "en") -> str:
        """Fallback translator."""
        ...


@runtime_checkable
class Classifier(Protocol):
    model_version: str

    async def claim_type(self, claim: RawClaim) -> tuple[ClaimType, float]: ...

    async def judge_passage(self, claim: Claim, passage: Passage) -> PassageJudgment: ...

    async def expected_evidence(
        self, claim: Claim, passages: list[Passage], judgments: list[PassageJudgment]
    ) -> list[ExpectedEvidence]: ...

    async def judge_claim(
        self,
        claim: Claim,
        passages: list[Passage],
        judgments: list[PassageJudgment],
        expected: list[ExpectedEvidence],
        claim_age_hours: float | None,
    ) -> ClaimJudgment: ...


@runtime_checkable
class Embedder(Protocol):
    model_version: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class NLIVerifier(Protocol):
    model_version: str

    async def score(self, pairs: list[tuple[str, str]]) -> list[NLIScore]:
        """pairs = [(premise, hypothesis)]."""
        ...


@runtime_checkable
class FactCheckSearch(Protocol):
    model_version: str

    async def search(self, query: str, language: str | None = None) -> list[FactCheckHit]: ...


@runtime_checkable
class Search(Protocol):
    """Evidence retrieval over the local index (hybrid keyword + vector)."""

    model_version: str

    async def search(
        self,
        queries: list[tuple[str, str | None]],  # (text, language)
        embedding: list[float] | None,
        k: int,
        published_after: datetime | None = None,
    ) -> list[Passage]: ...


@dataclass
class Adapters:
    vision: VisionReader
    translator: Translator
    llm: LLM
    classifier: Classifier
    embedder: Embedder
    nli: NLIVerifier
    factcheck: FactCheckSearch
    search: Search
    web_search: Search | None = None  # optional paid fallback, off by default

    def model_versions(self) -> dict[str, str]:
        out = {
            name: getattr(self, name).model_version
            for name in ("vision", "translator", "llm", "classifier", "embedder", "nli", "factcheck", "search")
        }
        if self.web_search is not None:
            out["web_search"] = self.web_search.model_version
        return out
