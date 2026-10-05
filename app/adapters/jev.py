"""Jev (TypeSafe) classifier through OpenRouter's Decisions API.

From openrouter.ai/docs/guides/community/jev and the Decisions API reference, confirmed live:
  POST {base}/alpha/decisions   header Authorization: Bearer <OPENROUTER_API_KEY>
  body: {model: "typesafe/jev-1.13", state: str|object, questions: {name: {type: "choice"|"noul"|"score",
         instructions, criteria}}}
        choice: criteria = {label: description};  noul: criteria = {"true": ..., "false": ...}
  response: {model: "typesafe/jev-1.13-<date>", answers: {name: {type:"choice", choice, probabilities,
             confidence} | {type:"noul", noul}}, usage}
Jev returns typed probabilities only (no generated text).
"""

from __future__ import annotations

import httpx

from app.adapters import prompts
from app.adapters.http import ApiError, request_json
from app.evidence_catalog import CATALOG, mark_found
from app.models.schemas import (
    Claim,
    ClaimJudgment,
    ClaimType,
    ExpectedEvidence,
    Passage,
    PassageJudgment,
    RawClaim,
    Stance,
    Status,
)


def passage_state(p: Passage, stance: PassageJudgment | None = None) -> dict:
    out = {
        "id": p.id,
        "publisher": p.publisher,
        "source_tier": p.tier,
        "source_kind": p.kind,
        "published_at": p.published_at.isoformat() if p.published_at else None,
        "text": p.text,
    }
    if stance is not None:
        out["stance"] = stance.stance.value
        out["stance_probability"] = round(stance.probability, 3)
    return out


def normalize_probs(probs: dict[str, float], labels: list[str]) -> dict[str, float]:
    vals = {k: max(0.0, float(probs.get(k, 0.0))) for k in labels}
    total = sum(vals.values())
    return {k: v / total for k, v in vals.items()} if total > 0 else {k: 1 / len(labels) for k in labels}


def verdict_to_status_probs(probs: dict[str, float]) -> dict[Status, float]:
    """UNVERIFIED goes to EVIDENCE_MISSING; claim age decides TOO_EARLY vs MISSING in code."""
    p = normalize_probs(probs, list(prompts.VERDICT_CRITERIA))
    return {
        Status.CONFIRMED: p["CONFIRMED"],
        Status.CONTRADICTED: p["CONTRADICTED"],
        Status.MISLEADING_CONTEXT: p["MISLEADING_CONTEXT"],
        Status.UNVERIFIED_EVIDENCE_MISSING: p["UNVERIFIED"],
    }


class JevClassifier:
    def __init__(
        self,
        api_key: str,
        model: str = "typesafe/jev-1.13",
        *,
        base_url: str = "https://openrouter.ai/api",
        timeout: float = 30,
        client: httpx.AsyncClient | None = None,
    ):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set (needed for Jev)")
        if "latest" in model:
            raise ValueError("Pin an exact Jev version (JEV_MODEL_VERSION), not a -latest alias")
        self.model = model
        self.url = f"{base_url.rstrip('/')}/alpha/decisions"
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self.resolved_model = model

    @property
    def model_version(self) -> str:
        return f"openrouter/{self.resolved_model}"

    async def _ask(self, state: dict, questions: dict) -> dict:
        data = await request_json(
            self._client, "POST", self.url, provider="jev", headers=self._headers,
            json={"model": self.model, "state": state, "questions": questions},
        )
        self.resolved_model = data.get("model") or self.model
        answers = data.get("answers") or {}
        missing = set(questions) - set(answers)
        if missing:
            raise ApiError("jev", 200, f"missing answers: {sorted(missing)}")
        return answers

    async def claim_type(self, claim: RawClaim) -> tuple[ClaimType, float]:
        ans = await self._ask(
            {"claim": claim.text_en, "claim_original_language": claim.text_original},
            {"type": {"type": "choice", "instructions": "What kind of statement is this claim?",
                      "criteria": prompts.CLAIM_TYPE_CRITERIA}},
        )
        a = ans["type"]
        probs = normalize_probs(a.get("probabilities") or {a["choice"]: 1.0}, list(prompts.CLAIM_TYPE_CRITERIA))
        choice = max(probs, key=probs.get)
        return ClaimType(choice), probs[choice]

    async def judge_passage(self, claim: Claim, passage: Passage) -> PassageJudgment:
        ans = await self._ask(
            {"claim": claim.text_en, "claim_original_language": claim.text_original, "passage": passage_state(passage)},
            {"stance": {"type": "choice",
                        "instructions": "Judge only from the passage text. What does the passage say about the claim?",
                        "criteria": prompts.STANCE_CRITERIA}},
        )
        a = ans["stance"]
        probs = normalize_probs(a.get("probabilities") or {a["choice"]: 1.0}, list(prompts.STANCE_CRITERIA))
        choice = max(probs, key=probs.get)
        return PassageJudgment(passage_id=passage.id, stance=Stance(choice), probability=probs[choice])

    async def expected_evidence(
        self, claim: Claim, passages: list[Passage], judgments: list[PassageJudgment]
    ) -> list[ExpectedEvidence]:
        questions = {
            f"expect_{i.key}": {
                "type": "noul",
                "instructions": f"If this claim were true, would this public trace normally exist and be findable "
                                f"within a few days: {i.label}?",
                "criteria": {"true": "Yes, it would normally exist.", "false": "No, or not applicable to this claim."},
            }
            for i in CATALOG
        }
        ans = await self._ask({"claim": claim.text_en}, questions)
        keys = [i.key for i in CATALOG if float(ans[f"expect_{i.key}"].get("noul", 0.0)) >= 0.5]
        return mark_found(keys, passages, judgments)

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
            "note": "Only these whitelisted passages count as evidence. Tier 1 = primary source.",
        }
        ans = await self._ask(
            state,
            {"verdict": {"type": "choice",
                         "instructions": "Based only on the evidence listed, which evidence status fits the claim?",
                         "criteria": prompts.VERDICT_CRITERIA}},
        )
        a = ans["verdict"]
        return ClaimJudgment(probabilities=verdict_to_status_probs(a.get("probabilities") or {a["choice"]: 1.0}))

