"""Fallback classifier: the LLM returning JSON with probabilities. Plus a primary->fallback wrapper."""

from __future__ import annotations

import json
import logging

from app.adapters import prompts
from app.adapters.base import Classifier
from app.adapters.gemini import GeminiClient
from app.adapters.jev import normalize_probs, passage_state, verdict_to_status_probs
from app.evidence_catalog import mark_found
from app.models.schemas import (
    Claim,
    ClaimJudgment,
    ClaimType,
    ExpectedEvidence,
    Passage,
    PassageJudgment,
    RawClaim,
    Stance,
)

log = logging.getLogger("fact.adapters")


class LLMClassifier:
    def __init__(self, client: GeminiClient):
        self.client = client

    @property
    def model_version(self) -> str:
        return f"llm-judge/{self.client.model_version}"

    async def _probs(self, question: str, criteria: dict[str, str], state: dict, schema: dict) -> dict[str, float]:
        user = (
            f"{question}\n\nOPTIONS:\n{prompts.criteria_text(criteria)}\n\n"
            f"Return a probability for every option.\n\nINPUT (JSON):\n{json.dumps(state, ensure_ascii=False, default=str)}"
        )
        data = await self.client.generate_json(prompts.JUDGE_SYSTEM, user, schema)
        return normalize_probs(data.get("probabilities") or {}, list(criteria))

    async def claim_type(self, claim: RawClaim) -> tuple[ClaimType, float]:
        probs = await self._probs(
            "What kind of statement is this claim?", prompts.CLAIM_TYPE_CRITERIA,
            {"claim": claim.text_en, "claim_original_language": claim.text_original}, prompts.CLAIM_TYPE_SCHEMA,
        )
        choice = max(probs, key=probs.get)
        return ClaimType(choice), probs[choice]

    async def judge_passage(self, claim: Claim, passage: Passage) -> PassageJudgment:
        user = (
            "Judge only from the passage text. What does the passage say about the claim?\n\nOPTIONS:\n"
            f"{prompts.criteria_text(prompts.STANCE_CRITERIA)}\n\nAlso give same_event: {prompts.SAME_EVENT_QUESTION}\n\n"
            f"INPUT (JSON):\n{json.dumps({'claim': claim.text_en, 'passage': passage_state(passage)}, ensure_ascii=False, default=str)}"
        )
        data = await self.client.generate_json(prompts.JUDGE_SYSTEM, user, prompts.STANCE_SCHEMA)
        probs = normalize_probs(data.get("probabilities") or {}, list(prompts.STANCE_CRITERIA))
        choice = max(probs, key=probs.get)
        same = min(1.0, max(0.0, float(data.get("same_event", 0.0))))
        return PassageJudgment(passage_id=passage.id, stance=Stance(choice), probability=probs[choice], same_event=same)

    async def expected_evidence(
        self, claim: Claim, passages: list[Passage], judgments: list[PassageJudgment]
    ) -> list[ExpectedEvidence]:
        data = await self.client.generate_json(
            prompts.JUDGE_SYSTEM, f"{prompts.EXPECTED_QUESTION}\n\nCLAIM: {claim.text_en}", prompts.EXPECTED_SCHEMA
        )
        return mark_found([str(k) for k in data.get("items", [])], passages, judgments)

    async def judge_claim(
        self,
        claim: Claim,
        passages: list[Passage],
        judgments: list[PassageJudgment],
        expected: list[ExpectedEvidence],
        claim_age_hours: float | None,
    ) -> ClaimJudgment:
        by_id = {j.passage_id: j for j in judgments}
        state = {
            "claim": claim.text_en,
            "claim_age_hours": None if claim_age_hours is None else round(claim_age_hours, 1),
            "evidence": [passage_state(p, by_id.get(p.id)) for p in passages],
            "expected_evidence": [e.model_dump() for e in expected],
        }
        probs = await self._probs(
            "Based only on the whitelisted evidence listed (tier 1 = primary source), which evidence status fits?",
            prompts.VERDICT_CRITERIA, state, prompts.VERDICT_SCHEMA,
        )
        return ClaimJudgment(probabilities=verdict_to_status_probs(probs))


class FallbackClassifier:
    """Calls the primary classifier; on any error, the fallback answers that call instead."""

    def __init__(self, primary: Classifier, fallback: Classifier):
        self.primary = primary
        self.fallback = fallback
        self.used_fallback = False

    @property
    def model_version(self) -> str:
        v = self.primary.model_version
        return f"{v} (fallback used: {self.fallback.model_version})" if self.used_fallback else v

    async def _call(self, name: str, *args):
        try:
            result = await getattr(self.primary, name)(*args)
            self.used_fallback = False  # model_version reflects the most recent call
            return result
        except Exception as exc:
            log.warning("classifier %s failed (%s); using fallback", name, exc)
            self.used_fallback = True
            return await getattr(self.fallback, name)(*args)

    async def claim_type(self, claim):
        return await self._call("claim_type", claim)

    async def judge_passage(self, claim, passage):
        return await self._call("judge_passage", claim, passage)

    async def expected_evidence(self, claim, passages, judgments):
        return await self._call("expected_evidence", claim, passages, judgments)

    async def judge_claim(self, claim, passages, judgments, expected, claim_age_hours):
        return await self._call("judge_claim", claim, passages, judgments, expected, claim_age_hours)
