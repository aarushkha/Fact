"""Gemini adapters (LLM + vision) over the Interactions API.

Request/response format from the Gemini docs (ai.google.dev/gemini-api/docs/interactions,
/structured-output, /image-understanding), confirmed with a live call:
  POST {base}/interactions   header x-goog-api-key
  body: model, input (string or [{type:text,text}|{type:image,data,mime_type}]), system_instruction,
        generation_config{temperature, thinking_level}, response_format{type:text, mime_type, schema}, store
  response: {status, steps:[{type:"model_output", content:[{type:"text", text}]}, ...], model, usage}
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from typing import Any

import httpx

from app.adapters import prompts
from app.adapters.http import ApiError, request_json
from app.models.schemas import Claim, DraftSentence, Entity, Passage, RawClaim, VisionResult

log = logging.getLogger("fact.adapters")
MAX_INLINE_BYTES = 18 * 1024 * 1024  # docs: 20MB total request size for inline data


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        fallback_models: list[str] | None = None,
        *,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        thinking_level: str = "low",
        timeout: float = 60,
        client: httpx.AsyncClient | None = None,
    ):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self.models = [model, *[m for m in (fallback_models or []) if m and m != model]]
        self.base_url = base_url.rstrip("/")
        self.thinking_level = thinking_level
        self._headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self.last_model = model
        self._cooldown_until: dict[str, float] = {}  # model -> monotonic time it may be tried again

    def _cool_down(self, model: str, err: ApiError) -> None:
        """Skip a model for a while after it is overloaded or out of quota (free tier: per-minute/per-day)."""
        msg = str(err).lower()
        seconds = 3600.0 if "per day" in msg else 60.0
        self._cooldown_until[model] = time.monotonic() + seconds

    async def generate_json(self, system: str, parts: str | list[dict], schema: dict, temperature: float = 0.0) -> Any:
        """Structured output. Tries each model in order on transient errors (overload, 429, 5xx)."""
        last_err: Exception | None = None
        now = time.monotonic()
        available = [m for m in self.models if self._cooldown_until.get(m, 0) <= now]
        for model in available or self.models[-1:]:
            body = {
                "model": model,
                "store": False,
                "system_instruction": system,
                "input": parts,
                "generation_config": {"temperature": temperature, "thinking_level": self.thinking_level},
                "response_format": {"type": "text", "mime_type": "application/json", "schema": schema},
            }
            try:
                data = await request_json(
                    self._client, "POST", f"{self.base_url}/interactions",
                    provider=f"gemini/{model}", headers=self._headers, json=body,
                    retries=1,
                )
            except ApiError as exc:
                last_err = exc
                if exc.transient:
                    self._cool_down(model, exc)
                    log.warning("gemini model %s unavailable, trying next: %s", model, exc)
                    continue
                raise
            if data.get("status") not in (None, "completed"):
                raise ApiError(f"gemini/{model}", 200, f"interaction status {data.get('status')}")
            text = "".join(
                c.get("text", "")
                for step in data.get("steps", [])
                if step.get("type") == "model_output"
                for c in step.get("content", [])
                if c.get("type") == "text"
            )
            self.last_model = data.get("model") or model
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise ApiError(f"gemini/{model}", 200, f"invalid JSON output: {exc}") from exc
        raise last_err or ApiError("gemini", 0, "no model configured")

    @property
    def model_version(self) -> str:
        return f"gemini/{self.last_model}"


class GeminiLLM:
    def __init__(self, client: GeminiClient):
        self.client = client

    @property
    def model_version(self) -> str:
        return self.client.model_version

    async def extract_claims(self, text_original: str, text_en: str, languages: list[str]) -> list[RawClaim]:
        data = await self.client.generate_json(
            prompts.EXTRACT_SYSTEM, prompts.extract_input(text_original, text_en, languages), prompts.EXTRACT_SCHEMA
        )
        out = []
        for c in data.get("claims", []):
            if not str(c.get("text_en", "")).strip():
                continue
            out.append(
                RawClaim(
                    text_original=str(c.get("text_original") or c["text_en"]).strip(),
                    text_en=str(c["text_en"]).strip(),
                    entities=[
                        Entity(text=e["text"], kind=e.get("kind") if e.get("kind") in prompts.ENTITY_KINDS else "other")
                        for e in c.get("entities", [])
                        if str(e.get("text", "")).strip()
                    ],
                )
            )
        return out

    async def write_summary(self, claim: Claim, passages: list[Passage], status: str) -> list[DraftSentence]:
        if not passages:
            return []
        data = await self.client.generate_json(
            prompts.WRITE_SYSTEM,
            prompts.write_input(claim.text_en, status, passages),
            prompts.write_schema([p.id for p in passages]),
        )
        ids = "|".join(re.escape(p.id) for p in passages)
        inline_ids = re.compile(rf"\s*[\[(]\s*(?:{ids})(?:\s*,\s*(?:{ids}))*\s*[\])]")
        return [
            DraftSentence(
                sentence=inline_ids.sub("", str(s.get("sentence", ""))).strip(),  # citations live in passage_ids
                passage_ids=[str(i) for i in s.get("passage_ids", [])],
            )
            for s in data.get("sentences", [])
        ]

    async def translate(self, text: str, source: str, target: str = "en") -> str:
        data = await self.client.generate_json(
            prompts.TRANSLATE_SYSTEM,
            f"Source language: {source}\nTarget language: {target}\n\nTEXT:\n{text}",
            prompts.TRANSLATE_SCHEMA,
        )
        return str(data.get("translation", "")).strip()


class GeminiVisionReader:
    def __init__(self, client: GeminiClient):
        self.client = client

    @property
    def model_version(self) -> str:
        return self.client.model_version

    async def read(self, image: bytes, mime: str | None) -> VisionResult:
        if len(image) > MAX_INLINE_BYTES:
            raise ValueError("Screenshot too large for inline upload")
        parts = [
            {"type": "text", "text": "Read this social-media screenshot."},
            {"type": "image", "data": base64.b64encode(image).decode("ascii"), "mime_type": mime or "image/png"},
        ]
        data = await self.client.generate_json(prompts.VISION_SYSTEM, parts, prompts.VISION_SCHEMA)

        def opt(key: str) -> str | None:
            value = str(data.get(key) or "").strip()
            return value or None

        return VisionResult(
            post_text=str(data.get("post_text") or "").strip(),
            account_handle=opt("account_handle"),
            post_date_raw=opt("post_date"),
            image_description=opt("image_description"),
        )
