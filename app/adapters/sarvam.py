"""Sarvam translator (language ID + translation).

From docs.sarvam.ai (api-reference-docs/text/translate-text and /identify-language), confirmed live:
  POST {base}/text-lid    header api-subscription-key  body {input (<=1000 chars)}
       -> {language_code: "mr-IN", script_code: "Deva"|"Latn"|...}
  POST {base}/translate   body {input, source_language_code ("auto" ok for mayura:v1),
       target_language_code, model: "mayura:v1"|"sarvam-translate:v1"}  -> {translated_text}
  Input limit: 1000 chars for mayura:v1, 2000 for sarvam-translate:v1.
"""

from __future__ import annotations

import asyncio
import re

import httpx

from app.adapters.http import request_json
from app.text import has_devanagari, sentences

INPUT_LIMITS = {"mayura:v1": 1000, "sarvam-translate:v1": 2000}
OUR_TO_SARVAM = {"en": "en-IN", "hi": "hi-IN", "hi-Latn": "hi-IN", "mr": "mr-IN"}


def from_sarvam(language_code: str, script_code: str | None) -> str:
    base = language_code.split("-")[0]
    if base == "hi" and script_code == "Latn":
        return "hi-Latn"  # Hinglish
    return base


def to_sarvam(code: str) -> str:
    return OUR_TO_SARVAM.get(code) or (f"{code}-IN" if re.fullmatch(r"[a-z]{2,3}", code) else "auto")


def split_for_limit(text: str, limit: int) -> list[str]:
    """Split on sentence boundaries into pieces of at most `limit` characters."""
    pieces: list[str] = []
    current = ""
    for s in sentences(text) or [text]:
        while len(s) > limit:  # a single overlong sentence: hard split on whitespace
            cut = s.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            pieces.append(s[:cut])
            s = s[cut:].lstrip()
        if current and len(current) + 1 + len(s) > limit:
            pieces.append(current)
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        pieces.append(current)
    return pieces


class SarvamTranslator:
    def __init__(
        self,
        api_key: str,
        model: str = "mayura:v1",
        *,
        base_url: str = "https://api.sarvam.ai",
        timeout: float = 30,
        client: httpx.AsyncClient | None = None,
    ):
        if not api_key:
            raise ValueError("SARVAM_API_KEY is not set")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._headers = {"api-subscription-key": api_key, "Content-Type": "application/json"}
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self.model_version = f"sarvam/{model}"

    async def detect(self, text: str) -> list[str]:
        data = await request_json(
            self._client, "POST", f"{self.base_url}/text-lid",
            provider="sarvam", headers=self._headers, json={"input": text[:1000]},
        )
        primary = from_sarvam(data.get("language_code") or "en-IN", data.get("script_code"))
        langs = [primary]
        # Text-lid returns one language; flag English mixed into Devanagari text.
        if primary != "en" and has_devanagari(text) and len(re.findall(r"[A-Za-z]{3,}", text)) >= 3:
            langs.append("en")
        return langs

    async def _translate_piece(self, piece: str, source: str, target: str) -> str:
        data = await request_json(
            self._client, "POST", f"{self.base_url}/translate",
            provider="sarvam", headers=self._headers,
            json={
                "input": piece,
                "source_language_code": to_sarvam(source),  # "auto" for unknown codes (mayura:v1 only)
                "target_language_code": to_sarvam(target),
                "model": self.model,
            },
        )
        return str(data.get("translated_text", "")).strip()

    async def translate(self, text: str, source: str, target: str = "en") -> str:
        if source == target:
            return text
        pieces = split_for_limit(text, INPUT_LIMITS.get(self.model, 1000))
        out = await asyncio.gather(*(self._translate_piece(p, source, target) for p in pieces))
        return " ".join(o for o in out if o)
