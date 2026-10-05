"""Provider-independent prompts and JSON schemas for LLM-backed adapters.

Schemas use only the JSON Schema subset the Gemini docs list as supported (type, properties,
required, items, enum, minimum, maximum, minItems, maxItems, description).
"""

from __future__ import annotations

from app.evidence_catalog import CATALOG

ENTITY_KINDS = ["person", "place", "institution", "date", "other"]

# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------

VISION_SYSTEM = """You read screenshots of social-media posts for a fact-checking service.
Transcribe; do not interpret, judge or translate.
- post_text: the post's own text, verbatim, in its original language and script. Exclude UI chrome
  (like/share counts, buttons, menus) and comments by other users. Keep line breaks.
- account_handle: the poster's handle or name as shown, else "".
- post_date: the post date/time exactly as printed (e.g. "2h", "3 March", "12/09/2026"), else "".
- image_description: one or two neutral sentences describing any photo/video frame in the post
  (who/what is visible, any visible text). "" if there is none."""

VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "post_text": {"type": "string"},
        "account_handle": {"type": "string"},
        "post_date": {"type": "string"},
        "image_description": {"type": "string"},
    },
    "required": ["post_text", "account_handle", "post_date", "image_description"],
}

# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = """You split social-media text into atomic claims for a fact-checking service.
The text may be Marathi, Hindi, Hinglish (romanized Hindi) or English, or a mix.
Rules:
- One claim = one statement that is true or false on its own. Split compound sentences.
- Make each claim self-contained: resolve pronouns and "this/here/today" using the text itself only.
- Include opinions, jokes, predictions and slogans too, as separate claims (another system labels them).
- Never add facts, dates, names or numbers that are not in the text. Never judge truth.
- text_original: the claim in the post's original language and script (a span or close paraphrase).
- text_en: a faithful English rendering of that claim.
- entities: people, places, institutions and dates mentioned in the claim, as written in English.
Return at most 8 claims, most important first."""

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "text_original": {"type": "string"},
                    "text_en": {"type": "string"},
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "kind": {"type": "string", "enum": ENTITY_KINDS},
                            },
                            "required": ["text", "kind"],
                        },
                    },
                },
                "required": ["text_original", "text_en", "entities"],
            },
        }
    },
    "required": ["claims"],
}


def extract_input(text_original: str, text_en: str, languages: list[str]) -> str:
    return (
        f"Detected languages: {', '.join(languages)}\n\n"
        f"ORIGINAL TEXT:\n{text_original}\n\n"
        f"ENGLISH TRANSLATION (machine, may be imperfect):\n{text_en}"
    )


# ---------------------------------------------------------------------------
# Summary writing
# ---------------------------------------------------------------------------

WRITE_SYSTEM = """You write the evidence summary for one claim in a fact-checking service.
Write 1 to 3 short English sentences describing what the provided passages say about the claim.
Rules:
- Every sentence must be directly supported by the passage(s) it cites; paraphrase closely.
- Cite only passage ids from the list, in passage_ids. Never write ids inside the sentence text.
- Use only the passages: no outside knowledge, no speculation, no verdict words like "true", "false",
  "fake" or "misleading" unless a passage itself says so (then attribute it: "<publisher> says ...").
- Prefer primary sources (tier 1). Mention who said it.
Unsupported sentences are deleted automatically, so stay close to the passage wording."""


def write_schema(passage_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "sentences": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "sentence": {"type": "string"},
                        "passage_ids": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "enum": passage_ids},
                        },
                    },
                    "required": ["sentence", "passage_ids"],
                },
            }
        },
        "required": ["sentences"],
    }


def passages_block(passages) -> str:
    return "\n\n".join(
        f"[{p.id}] publisher={p.publisher} tier={p.tier} published_at={p.published_at.isoformat() if p.published_at else 'unknown'}\n{p.text}"
        for p in passages
    )


def write_input(claim_en: str, status: str, passages) -> str:
    return f"CLAIM: {claim_en}\nEVIDENCE STATUS (already decided): {status}\n\nPASSAGES:\n{passages_block(passages)}"


# ---------------------------------------------------------------------------
# Translation (LLM fallback)
# ---------------------------------------------------------------------------

TRANSLATE_SYSTEM = """Translate the user's text faithfully into the target language.
Romanized Hindi (Hinglish) is Hindi. Keep names, numbers and dates exactly. Do not add or omit anything."""

TRANSLATE_SCHEMA = {"type": "object", "properties": {"translation": {"type": "string"}}, "required": ["translation"]}

# ---------------------------------------------------------------------------
# Judge (LLM fallback classifier) and shared label descriptions (also used by Jev)
# ---------------------------------------------------------------------------

CLAIM_TYPE_CRITERIA = {
    "checkable": "A specific statement of fact about the world that public evidence could confirm or refute.",
    "opinion": "A value judgement, preference, praise, insult or feeling.",
    "satire": "An obvious joke, parody or satire.",
    "prediction": "A statement about what will happen in the future.",
    "unfalsifiable": "Cannot be tested with any evidence (supernatural, hopelessly vague, or true by definition).",
}

STANCE_CRITERIA = {
    "supports": "The passage reports that the claim, as stated, is true.",
    "contradicts": "The passage reports that the claim is false, or that it misrepresents time, place, people or context (e.g. old media shared as new).",
    "irrelevant": "The passage does not address this claim, or only mentions related topics.",
}

VERDICT_CRITERIA = {
    "CONFIRMED": "The evidence directly confirms the claim as stated.",
    "CONTRADICTED": "The evidence shows the claimed event or fact did not happen or is false.",
    "MISLEADING_CONTEXT": "The event, photo, video or quote is real, but the claim presents it with the wrong time, place, people or context (e.g. a real old video shared as if it were new).",
    "UNVERIFIED": "The evidence is insufficient to confirm or contradict the claim. Absence of evidence is NOT contradiction.",
}

EVIDENCE_KEYS = [i.key for i in CATALOG]
EVIDENCE_GUIDE = "\n".join(f"- {i.key}: {i.label}" for i in CATALOG)

JUDGE_SYSTEM = """You are a careful evidence judge for a fact-checking service. Judge only from the
provided passages. Give honest, calibrated probabilities: when unsure, spread probability mass."""

CLAIM_TYPE_SCHEMA = {
    "type": "object",
    "properties": {
        "probabilities": {
            "type": "object",
            "properties": {k: {"type": "number", "minimum": 0, "maximum": 1} for k in CLAIM_TYPE_CRITERIA},
            "required": list(CLAIM_TYPE_CRITERIA),
        }
    },
    "required": ["probabilities"],
}

STANCE_SCHEMA = {
    "type": "object",
    "properties": {
        "probabilities": {
            "type": "object",
            "properties": {k: {"type": "number", "minimum": 0, "maximum": 1} for k in STANCE_CRITERIA},
            "required": list(STANCE_CRITERIA),
        }
    },
    "required": ["probabilities"],
}

EXPECTED_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {"type": "string", "enum": EVIDENCE_KEYS}}},
    "required": ["items"],
}

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "probabilities": {
            "type": "object",
            "properties": {k: {"type": "number", "minimum": 0, "maximum": 1} for k in VERDICT_CRITERIA},
            "required": list(VERDICT_CRITERIA),
        }
    },
    "required": ["probabilities"],
}


def criteria_text(criteria: dict[str, str]) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in criteria.items())


EXPECTED_QUESTION = (
    "If this claim were true, which of these public traces would normally exist and be findable within "
    "a few days? Choose only items that clearly apply.\n" + EVIDENCE_GUIDE
)
