"""Stage 7: write 1-3 cited sentences, then delete any sentence NLI cannot verify."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.adapters.base import NLIVerifier
from app.models.schemas import DraftSentence, Passage, SummarySentence
from app.text import overlap_ratio, sentences

MAX_SENTENCES = 3
MAX_WINDOWS = 4  # whole passage + the 3 windows that share most words with the sentence


def best_windows(passage_text: str, hypothesis: str, k: int = MAX_WINDOWS) -> list[str]:
    """Bound NLI cost (CPU): the whole passage plus the k-1 windows most lexically similar to the sentence."""
    windows = premise_windows(passage_text)
    head, rest = windows[0], windows[1:]
    rest.sort(key=lambda w: -overlap_ratio(hypothesis, w))
    return [head, *rest[: k - 1]]


def premise_windows(text: str) -> list[str]:
    """The passage, each sentence, and each pair of adjacent sentences.

    NLI cross-encoders under-score a hypothesis against a long premise (mDeBERTa gave 0.44 for a
    near-verbatim sentence against a 2-sentence passage, 0.85 against the matching sentence), so a
    sentence counts as entailed if any window of its cited passage entails it.
    """
    sents = sentences(text)
    windows = [text, *sents, *(f"{a} {b}" for a, b in zip(sents, sents[1:]))]
    return list(dict.fromkeys(w for w in windows if w.strip()))


@dataclass
class VerifyReport:
    kept: list[SummarySentence] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)  # {sentence, reason}
    cited_passages: list[Passage] = field(default_factory=list)


async def verify_sentences(
    drafts: list[DraftSentence], passages: list[Passage], nli: NLIVerifier, threshold: float
) -> VerifyReport:
    """Keep a sentence only if at least one of its cited passages entails it.

    Citations that do not entail the sentence are removed from it, so every citation shown supports
    its sentence. Citations to passages the writer was not given are ignored.
    """
    by_id = {p.id: p for p in passages}
    report = VerifyReport()
    pairs: list[tuple[int, Passage, str]] = []  # (draft index, cited passage, premise window)
    cited_any: set[int] = set()
    for i, d in enumerate(drafts[:MAX_SENTENCES]):
        cited = [by_id[pid] for pid in dict.fromkeys(d.passage_ids) if pid in by_id]
        if not d.sentence.strip() or not cited:
            report.dropped.append({"sentence": d.sentence, "reason": "no valid citation"})
            continue
        cited_any.add(i)
        pairs.extend((i, p, w) for p in cited for w in best_windows(p.text, d.sentence))
    scores = await nli.score([(w, drafts[i].sentence) for i, _, w in pairs]) if pairs else []

    entailing: dict[int, list[Passage]] = {}
    for (i, p, _), s in zip(pairs, scores):
        if s.entailment >= threshold and p not in entailing.get(i, []):
            entailing.setdefault(i, []).append(p)
    cited_ids: dict[str, Passage] = {}
    for i in sorted(cited_any):
        ok = entailing.get(i, [])
        if not ok:
            report.dropped.append({"sentence": drafts[i].sentence, "reason": "not entailed by cited passage"})
            continue
        source_ids = list(dict.fromkeys(p.source_id for p in ok))
        report.kept.append(SummarySentence(sentence=drafts[i].sentence.strip(), sources=source_ids))
        for p in ok:
            cited_ids.setdefault(p.id, p)
    report.cited_passages = list(cited_ids.values())
    return report

