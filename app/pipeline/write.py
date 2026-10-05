"""Stage 7: write 1-3 cited sentences, then delete any sentence NLI cannot verify.

Under a "fails" status (CONTRADICTED, MISLEADING_CONTEXT) a sentence that itself entails the claim is
also deleted: it restates the claim (e.g. the quote a debunk opens with), and its own passage entails it,
so the citation check alone would keep it and show the false claim as the summary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.adapters.base import NLIVerifier
from app.models.schemas import DraftSentence, Passage, Status, SummarySentence
from app.text import overlap_ratio, sentences

MAX_SENTENCES = 3
MAX_WINDOWS = 4  # whole passage + the 3 windows that share most words with the sentence
FAILS = {Status.CONTRADICTED.value, Status.MISLEADING_CONTEXT.value}
# Debunks quote the claim under a label ("Claim:", "Claim Review :", "दावा:"). On 59 real fact-checks every
# such sentence picked as a summary was the false claim itself, and NLI scored many below the threshold.
CLAIM_LABEL = re.compile(r"(^|[\s\-–:])(claim(\s+review(ed)?)?|दावा)\s*:", re.IGNORECASE)


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
    drafts: list[DraftSentence], passages: list[Passage], nli: NLIVerifier, threshold: float,
    claim_text: str | None = None, status: Status | str | None = None, restatement_threshold: float = 0.8,
) -> VerifyReport:
    """Keep a sentence only if at least one of its cited passages entails it.

    Citations that do not entail the sentence are removed from it, so every citation shown supports
    its sentence. Citations to passages the writer was not given are ignored. With claim_text and a
    "fails" status, a sentence that entails the claim (>= restatement_threshold) is deleted as a restatement.
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
    status_value = getattr(status, "value", status)
    echo_check = sorted(cited_any) if claim_text and status_value in FAILS else []
    nli_pairs = [(w, drafts[i].sentence) for i, _, w in pairs] + [(drafts[i].sentence, claim_text) for i in echo_check]
    all_scores = await nli.score(nli_pairs) if nli_pairs else []
    scores, echo_scores = all_scores[: len(pairs)], all_scores[len(pairs) :]
    restates = {i for i, sc in zip(echo_check, echo_scores) if sc.entailment >= restatement_threshold}

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
        if status_value in FAILS and CLAIM_LABEL.search(drafts[i].sentence):
            report.dropped.append({"sentence": drafts[i].sentence, "reason": f"quotes the claim under {status_value}"})
            continue
        if i in restates:
            report.dropped.append({"sentence": drafts[i].sentence, "reason": f"restates the claim under {status_value}"})
            continue
        source_ids = list(dict.fromkeys(p.source_id for p in ok))
        report.kept.append(SummarySentence(sentence=drafts[i].sentence.strip(), sources=source_ids))
        for p in ok:
            cited_ids.setdefault(p.id, p)
    report.cited_passages = list(cited_ids.values())
    return report



def base_language(code: str | None) -> str | None:
    return code.split("-")[0] if code else None


async def localize_summary(
    report: VerifyReport, translator, nli: NLIVerifier, threshold: float, target: str | None,
    claim_text: str | None = None, status: Status | str | None = None, restatement_threshold: float = 0.8,
) -> tuple[list[SummarySentence], list[dict]]:
    """Translate verified sentences into the post's language; a translation is shown only if it passes the
    same checks as the original (NLI against its own cited passages; under a "fails" status, no claim label
    and no restatement). Otherwise the verified original stays. Returns (summary, log).

    Hinglish ("hi-Latn") gets Devanagari Hindi: Sarvam's romanized output has not been verified.
    """
    target = "hi" if target == "hi-Latn" else target
    by_source = {p.source_id: p for p in report.cited_passages}
    status_value = getattr(status, "value", status)
    out, log = [], []
    for s in report.kept:
        cited = [by_source[sid] for sid in s.sources if sid in by_source]
        source_lang = base_language(cited[0].language) if cited else None
        if not target or not cited or source_lang is None or source_lang == base_language(target):
            out.append(s)
            continue
        try:
            translated = (await translator.translate(s.sentence, source_lang, target)).strip()
        except Exception as exc:  # a failed translation never loses the verified sentence
            log.append({"sentence": s.sentence, "kept": "original", "reason": f"translate failed: {exc}"})
            out.append(s)
            continue
        pairs = [(w, translated) for p in cited for w in best_windows(p.text, translated)]
        check_echo = bool(claim_text) and status_value in FAILS
        scores = await nli.score(pairs + ([(translated, claim_text)] if check_echo else []))
        entailed = any(sc.entailment >= threshold for sc in scores[: len(pairs)])
        restates = check_echo and scores[-1].entailment >= restatement_threshold
        labelled = status_value in FAILS and CLAIM_LABEL.search(translated)
        if translated and entailed and not restates and not labelled:
            out.append(SummarySentence(sentence=translated, sources=s.sources))
            log.append({"sentence": s.sentence, "kept": "translation", "translation": translated})
        else:
            reason = "not entailed" if not entailed else "restates the claim" if restates else "claim label"
            log.append({"sentence": s.sentence, "kept": "original", "translation": translated, "reason": reason})
            out.append(s)
    return out, log
