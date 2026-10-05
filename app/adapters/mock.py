"""Deterministic mock adapters. The whole app runs on these when MOCK_MODE=true.

They are keyword-driven over the fixtures in mock_data/, so tests can drive every status on purpose.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
import zlib
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import yaml

from app.models.schemas import (
    Claim,
    ClaimJudgment,
    ClaimType,
    DraftSentence,
    Entity,
    ExpectedEvidence,
    FactCheckHit,
    NLIScore,
    Passage,
    PassageJudgment,
    RawClaim,
    Stance,
    Status,
    VisionResult,
)
from app.evidence_catalog import mark_found
from app.sources import Whitelist
from app.text import content_words, has_devanagari, overlap_ratio, sentences, source_id_for, tokens

MOCK_VERSION = "mock-1"
DATA_DIR = Path(__file__).parent / "mock_data"


@lru_cache
def _load(name: str) -> dict:
    return yaml.safe_load((DATA_DIR / name).read_text(encoding="utf-8")) or {}


# Cue lists used by the mock classifier. Real classifiers do not work like this.
CONTEXT_CUES = ("old footage", "old video", "not from this week", "misattributed", "out of context", "from the 2019")
CONTRADICTION_CUES = ("false", "fake", "hoax", "denied", "no truth", "operating normally", "not true", "rumour")
TYPE_CUES: dict[ClaimType, tuple[str, ...]] = {
    ClaimType.SATIRE: ("satire", "parody"),
    ClaimType.OPINION: ("most beautiful", "best ", "worst ", "i think", "i feel", "should ", "amazing", "terrible"),
    ClaimType.PREDICTION: (" will ", "going to", "next year", "tomorrow"),
    ClaimType.UNFALSIFIABLE: ("god ", "destiny", "karma", "soul "),
}
PLACES = {"nashik", "mumbai", "pune", "thane", "kolhapur", "godavari", "maharashtra", "india"}
INSTITUTIONS = {"police", "airport", "government", "court", "metro", "corporation", "hospital"}
DATE_WORDS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today", "yesterday"}


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------


def read_png_text_chunks(data: bytes) -> dict[str, str]:
    """Read tEXt / iTXt chunks from a PNG. Mock screenshots carry their post text this way."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return {}
    out: dict[str, str] = {}
    pos = 8
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        ctype = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        if ctype == b"tEXt" and b"\x00" in body:
            k, v = body.split(b"\x00", 1)
            out[k.decode("latin-1")] = v.decode("latin-1")
        elif ctype == b"iTXt" and b"\x00" in body:
            k, rest = body.split(b"\x00", 1)
            compressed = rest[0] == 1
            rest = rest[2:]  # compression flag + method
            _lang, rest = rest.split(b"\x00", 1)
            _tkey, text = rest.split(b"\x00", 1)
            out[k.decode("latin-1")] = (zlib.decompress(text) if compressed else text).decode("utf-8")
        if ctype == b"IEND":
            break
        pos += 12 + length
    return out


class MockVisionReader:
    model_version = MOCK_VERSION

    async def read(self, image: bytes, mime: str | None) -> VisionResult:
        meta = read_png_text_chunks(image)
        if "post_text" in meta:
            return VisionResult(
                post_text=meta["post_text"],
                account_handle=meta.get("account_handle"),
                post_date_raw=meta.get("post_date"),
                image_description=meta.get("image_description"),
            )
        return VisionResult(
            post_text="Mumbai airport is closed for a week.",
            account_handle="@mock_account",
            post_date_raw="2h",
            image_description="MOCK: no embedded post_text in this image; returning a fixed demo post.",
        )


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------

MARATHI_MARKERS = ("आहे", "आणि", "मध्ये", "झाला", "कोसळला", "नाही", "च्या", "आहेत")
HINDI_MARKERS = ("है", "हैं", " के ", " में ", "नहीं", " और ", " की ", " का ")
HINGLISH_MARKERS = {"hai", "hain", "nahi", "kya", "mein", "ke", "ki", "ka", "aur", "lag", "gayi", "gaya", "tha", "liye", "hafte", "band"}


class MockTranslator:
    model_version = MOCK_VERSION

    async def detect(self, text: str) -> list[str]:
        if has_devanagari(text):
            padded = f" {text} "
            mr = sum(padded.count(m) for m in MARATHI_MARKERS)
            hi = sum(padded.count(m) for m in HINDI_MARKERS)
            langs = ["mr" if mr > hi else "hi"]
        else:
            words = re.findall(r"[a-z]+", text.lower())
            langs = ["hi-Latn" if sum(w in HINGLISH_MARKERS for w in words) >= 2 else "en"]
        latin_words = re.findall(r"[A-Za-z]{3,}", text)
        if langs[0] != "en" and has_devanagari(text) and len(latin_words) >= 3:
            langs.append("en")
        return langs

    async def translate(self, text: str, source: str, target: str = "en") -> str:
        if source == target:
            return text
        table: dict[str, str] = _load("translations.yaml").get("translations", {})
        if text.strip() in table:
            return table[text.strip()]
        return " ".join(table.get(s, s) for s in sentences(text))


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


def _entities(text: str) -> list[Entity]:
    seen: dict[str, Entity] = {}
    for word in re.findall(r"[A-Za-z]+", text):
        low = word.lower()
        if low in PLACES:
            seen.setdefault(low, Entity(text=word, kind="place"))
        elif low in INSTITUTIONS:
            seen.setdefault(low, Entity(text=word, kind="institution"))
        elif low in DATE_WORDS:
            seen.setdefault(low, Entity(text=word, kind="date"))
    return list(seen.values())


class MockLLM:
    model_version = MOCK_VERSION

    async def extract_claims(self, text_original: str, text_en: str, languages: list[str]) -> list[RawClaim]:
        en = sentences(text_en)
        orig = sentences(text_original)
        aligned = len(en) == len(orig)
        return [
            RawClaim(text_original=orig[i] if aligned else text_original, text_en=s, entities=_entities(s))
            for i, s in enumerate(en)
        ]

    async def write_summary(self, claim: Claim, passages: list[Passage], status: str) -> list[DraftSentence]:
        drafts = [DraftSentence(sentence=sentences(p.text)[0], passage_ids=[p.id]) for p in passages[:2]]
        if drafts:
            # Deliberately unsupported sentence, so the NLI filter's deletion is exercised end to end.
            drafts.append(
                DraftSentence(
                    sentence="Officials say the situation is completely under control.",
                    passage_ids=[passages[0].id],
                )
            )
        return drafts

    async def translate(self, text: str, source: str, target: str = "en") -> str:
        return await MockTranslator().translate(text, source, target)


# ---------------------------------------------------------------------------
# Classifier / judge
# ---------------------------------------------------------------------------

MOCK_EXPECTED_KEYS = ["police", "institution", "news"]


def _has_cue(text: str, cues: tuple[str, ...]) -> bool:
    low = f" {text.lower()} "
    return any(c in low for c in cues)


class MockClassifier:
    model_version = MOCK_VERSION

    async def claim_type(self, claim: RawClaim) -> tuple[ClaimType, float]:
        for ctype, cues in TYPE_CUES.items():
            if _has_cue(claim.text_en, cues):
                return ctype, 0.9
        return ClaimType.CHECKABLE, 0.9

    async def judge_passage(self, claim: Claim, passage: Passage) -> PassageJudgment:
        shared = content_words(claim.text_en) & content_words(passage.text)
        if len(shared) < 2 or overlap_ratio(claim.text_en, passage.text) < 0.5:
            return PassageJudgment(passage_id=passage.id, stance=Stance.IRRELEVANT, probability=0.9)
        if _has_cue(passage.text, CONTEXT_CUES) or _has_cue(passage.text, CONTRADICTION_CUES):
            return PassageJudgment(passage_id=passage.id, stance=Stance.CONTRADICTS, probability=0.9, same_event=0.9)
        return PassageJudgment(passage_id=passage.id, stance=Stance.SUPPORTS, probability=0.9, same_event=0.9)

    async def expected_evidence(
        self, claim: Claim, passages: list[Passage], judgments: list[PassageJudgment]
    ) -> list[ExpectedEvidence]:
        return mark_found(MOCK_EXPECTED_KEYS, passages, judgments)

    async def judge_claim(
        self,
        claim: Claim,
        passages: list[Passage],
        judgments: list[PassageJudgment],
        expected: list[ExpectedEvidence],
        claim_age_hours: float | None,
    ) -> ClaimJudgment:
        by_id = {p.id: p for p in passages}
        relevant = [(j, by_id[j.passage_id]) for j in judgments if j.stance != Stance.IRRELEVANT and j.passage_id in by_id]
        if relevant:
            best = min(p.tier for _, p in relevant)
            relevant = [(j, p) for j, p in relevant if p.tier == best]
        sup = [p for j, p in relevant if j.stance == Stance.SUPPORTS]
        con = [p for j, p in relevant if j.stance == Stance.CONTRADICTS]
        unverified = (
            Status.UNVERIFIED_TOO_EARLY
            if claim_age_hours is not None and claim_age_hours < 72
            else Status.UNVERIFIED_EVIDENCE_MISSING
        )
        if any(_has_cue(p.text, CONTEXT_CUES) for p in con):
            top, p = Status.MISLEADING_CONTEXT, 0.85
        elif con and not sup:
            top, p = Status.CONTRADICTED, 0.88
        elif sup and not con:
            top, p = Status.CONFIRMED, 0.9 if (len(sup) >= 2 or sup[0].tier == 1) else 0.7
        elif sup and con:
            return ClaimJudgment(
                probabilities={Status.CONFIRMED: 0.4, Status.CONTRADICTED: 0.4, unverified: 0.2}
            )
        else:
            top, p = unverified, 0.9
        rest = [s for s in (Status.CONFIRMED, Status.CONTRADICTED, Status.MISLEADING_CONTEXT, unverified) if s != top]
        return ClaimJudgment(probabilities={top: p, **{s: (1 - p) / len(rest) for s in rest}})


# ---------------------------------------------------------------------------
# Embedder / NLI
# ---------------------------------------------------------------------------


class MockEmbedder:
    """Hashed bag-of-words: identical texts -> cosine 1.0, overlapping texts -> high cosine."""

    model_version = MOCK_VERSION

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in tokens(text):
            h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:8], "big")
            vec[h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


NEGATIONS = {"not", "no", "never", "nor", "isn't", "wasn't", "aren't", "didn't", "doesn't", "नहीं", "नाही"}


def _negated(text: str) -> bool:
    return bool(NEGATIONS & set(re.findall(r"[\w\u0900-\u097F']+", text.lower())))  # tokens() drops "not"


class MockNLIVerifier:
    """Entailment = share of the hypothesis' content words present in the premise, unless exactly one
    of the two is negated: then the overlap counts as contradiction (a real NLI model sees "not")."""

    model_version = MOCK_VERSION

    async def score(self, pairs: list[tuple[str, str]]) -> list[NLIScore]:
        out = []
        for premise, hypothesis in pairs:
            r = overlap_ratio(hypothesis, premise)
            if _negated(premise) != _negated(hypothesis):
                out.append(NLIScore(entailment=0.0, neutral=1 - r, contradiction=r))
            else:
                out.append(NLIScore(entailment=r, neutral=1 - r, contradiction=0.0))
        return out


# ---------------------------------------------------------------------------
# Fact-check search / evidence search
# ---------------------------------------------------------------------------


class MockFactCheckSearch:
    model_version = MOCK_VERSION

    async def search(self, query: str, language: str | None = None) -> list[FactCheckHit]:
        hits = [FactCheckHit(**h) for h in _load("factchecks.yaml").get("hits", [])]
        return [h for h in hits if len(content_words(query) & content_words(h.claim_text)) >= 2]


class MockSearch:
    """Keyword search over mock_data/corpus.yaml. Publisher/tier are looked up in the mock whitelist."""

    model_version = MOCK_VERSION

    def __init__(self, whitelist: Whitelist):
        self.whitelist = whitelist

    async def search(
        self,
        queries: list[tuple[str, str | None]],
        embedding: list[float] | None,
        k: int,
        published_after: datetime | None = None,
    ) -> list[Passage]:
        out = []
        for row in _load("corpus.yaml").get("passages", []):
            text = row["text"]
            scores = [overlap_ratio(q, text) for q, _ in queries]
            shared = max(len(content_words(q) & content_words(text)) for q, _ in queries)
            if shared < 2:
                continue
            entry = self.whitelist.lookup(row["url"])
            out.append(
                Passage(
                    id=row["id"],
                    source_id=source_id_for(row["url"]),
                    url=row["url"],
                    publisher=entry.name if entry else "unknown",
                    tier=entry.tier if entry else 3,
                    kind=entry.kind if entry else None,
                    language=row.get("language"),
                    published_at=row.get("published_at"),
                    text=text,
                    relevance=max(scores),
                )
            )
        out.sort(key=lambda p: -p.relevance)
        return out[:k]


def _png_chunk(ctype: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + ctype + body + struct.pack(">I", zlib.crc32(ctype + body) & 0xFFFFFFFF)


def add_png_text(png: bytes, meta: dict[str, str]) -> bytes:
    """Insert `meta` as iTXt chunks before IEND, so the mock vision reader can 'read' a real screenshot."""
    if not png.startswith(b"\x89PNG\r\n\x1a\n") or not png.endswith(_png_chunk(b"IEND", b"")):
        raise ValueError("not a PNG ending in IEND")
    text = b"".join(
        _png_chunk(b"iTXt", k.encode("latin-1") + b"\x00\x00\x00\x00\x00" + v.encode("utf-8")) for k, v in meta.items()
    )
    return png[:-12] + text + png[-12:]


def make_mock_png(meta: dict[str, str]) -> bytes:
    """Build a 1x1 PNG carrying `meta` as iTXt chunks. Used for synthetic screenshots in tests/evals."""
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", zlib.compress(b"\x00\xff")) + _png_chunk(b"IEND", b"")
    return add_png_text(png, meta)
