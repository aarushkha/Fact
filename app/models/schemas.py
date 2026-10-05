"""Pydantic models: the public output contract plus the internal types passed between stages."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Status(str, Enum):
    """Evidence status. Describes the evidence we found, never reality."""

    CONFIRMED = "CONFIRMED"
    CONTRADICTED = "CONTRADICTED"
    MISLEADING_CONTEXT = "MISLEADING_CONTEXT"
    UNVERIFIED_TOO_EARLY = "UNVERIFIED_TOO_EARLY"
    UNVERIFIED_EVIDENCE_MISSING = "UNVERIFIED_EVIDENCE_MISSING"
    NOT_CHECKABLE = "NOT_CHECKABLE"


DEFINITIVE_STATUSES = (Status.CONFIRMED, Status.CONTRADICTED, Status.MISLEADING_CONTEXT)
UNVERIFIED_STATUSES = (Status.UNVERIFIED_TOO_EARLY, Status.UNVERIFIED_EVIDENCE_MISSING)


class ClaimType(str, Enum):
    CHECKABLE = "checkable"
    OPINION = "opinion"
    SATIRE = "satire"
    PREDICTION = "prediction"
    UNFALSIFIABLE = "unfalsifiable"


class Stance(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    IRRELEVANT = "irrelevant"


InputType = Literal["screenshot", "text"]
EntityKind = Literal["person", "place", "institution", "date", "other"]


# ---------------------------------------------------------------------------
# Public output contract
# ---------------------------------------------------------------------------


class SummarySentence(BaseModel):
    sentence: str
    sources: list[str]  # source ids


class ExpectedEvidence(BaseModel):
    item: str
    found: bool


class ClaimResult(BaseModel):
    text_original: str
    text_en: str
    type: ClaimType
    status: Status
    confidence: float = Field(ge=0, le=1)
    summary: list[SummarySentence] = []
    expected_evidence: list[ExpectedEvidence] = []
    would_change_if: str | None = None


class SourceOut(BaseModel):
    id: str
    url: str
    publisher: str
    tier: int
    published_at: datetime | None = None


class CheckResponse(BaseModel):
    check_id: str
    input_type: InputType
    languages: list[str]
    claims: list[ClaimResult]
    sources: list[SourceOut]
    model_versions: dict[str, str]
    checked_at: datetime
    recheck_at: datetime | None = None


# ---------------------------------------------------------------------------
# Internal stage types
# ---------------------------------------------------------------------------


class CheckInput(BaseModel):
    text: str | None = None
    image: bytes | None = None
    image_mime: str | None = None
    post_date: datetime | None = None  # user-supplied; overrides anything read from the image

    @property
    def input_type(self) -> InputType:
        return "screenshot" if self.image else "text"


class VisionResult(BaseModel):
    post_text: str
    account_handle: str | None = None
    post_date_raw: str | None = None  # as printed on the screenshot, e.g. "2h", "3 March"
    image_description: str | None = None


class Ingested(BaseModel):
    input_type: InputType
    text: str
    account_handle: str | None = None
    post_date: datetime | None = None
    image_description: str | None = None


class Normalized(BaseModel):
    text_original: str
    text_en: str
    languages: list[str]  # e.g. ["mr"], ["hi-Latn", "en"]


class Entity(BaseModel):
    text: str
    kind: EntityKind = "other"


class RawClaim(BaseModel):
    """What the LLM extractor returns, before type classification."""

    text_original: str
    text_en: str
    entities: list[Entity] = []


class Claim(RawClaim):
    id: str
    type: ClaimType
    type_confidence: float


class Passage(BaseModel):
    id: str
    source_id: str
    url: str
    publisher: str
    tier: int
    kind: str | None = None  # police, court, government, institution, wire, outlet, factchecker, aggregator
    language: str | None = None
    published_at: datetime | None = None
    text: str
    relevance: float = 0.0


class PassageJudgment(BaseModel):
    passage_id: str
    stance: Stance
    probability: float = Field(ge=0, le=1)


class ClaimJudgment(BaseModel):
    """Classifier output for a claim: a probability per checkable status."""

    probabilities: dict[Status, float]


class DraftSentence(BaseModel):
    """Writer output: a sentence citing passage ids (not yet NLI-verified)."""

    sentence: str
    passage_ids: list[str]


class NLIScore(BaseModel):
    entailment: float
    neutral: float
    contradiction: float


class FactCheckHit(BaseModel):
    claim_text: str
    claimant: str | None = None
    claim_date: datetime | None = None
    review_url: str
    review_title: str | None = None
    publisher_name: str | None = None
    publisher_site: str | None = None
    textual_rating: str
    review_date: datetime | None = None
    language: str | None = None
