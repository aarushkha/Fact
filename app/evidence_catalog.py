"""Expected-evidence catalog: traces that should exist if a claim were true.

Classifiers choose WHICH items apply to a claim; `mark_found` decides deterministically whether each
was found: an item is found when a passage from a source of a matching kind SUPPORTS the claim.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.models.schemas import ExpectedEvidence, Passage, PassageJudgment, Stance


@dataclass(frozen=True)
class EvidenceItem:
    key: str
    label: str
    kinds: frozenset[str]


CATALOG: tuple[EvidenceItem, ...] = (
    EvidenceItem("police", "Statement or FIR from police / the responsible authority", frozenset({"police", "court", "government"})),
    EvidenceItem("court", "Court order, filing or judgment", frozenset({"court"})),
    EvidenceItem("government", "Government order, notice or press release", frozenset({"government"})),
    EvidenceItem("institution", "Official notice from the institution involved", frozenset({"institution", "government"})),
    EvidenceItem("news", "Report by a credible news outlet or wire service", frozenset({"wire", "outlet", "factchecker"})),
    EvidenceItem("media_origin", "Original source and date of the photo or video", frozenset({"wire", "outlet", "factchecker"})),
)
BY_KEY = {i.key: i for i in CATALOG}


def mark_found(
    keys: list[str], passages: list[Passage], judgments: list[PassageJudgment], min_prob: float = 0.5
) -> list[ExpectedEvidence]:
    by_id = {p.id: p for p in passages}
    supporting_kinds = {
        by_id[j.passage_id].kind
        for j in judgments
        if j.stance == Stance.SUPPORTS and j.probability >= min_prob and j.passage_id in by_id
    }
    items = [BY_KEY[k] for k in dict.fromkeys(keys) if k in BY_KEY]
    return [ExpectedEvidence(item=i.label, found=bool(i.kinds & supporting_kinds)) for i in items]
