"""Stage 7: write 1-3 cited sentences, then delete any sentence NLI cannot verify.

Under a "fails" status (CONTRADICTED, MISLEADING_CONTEXT) a sentence that itself entails the claim is
also deleted: it restates the claim (e.g. the quote a debunk opens with), and its own passage entails it,
so the citation check alone would keep it and show the false claim as the summary.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

from app.adapters.base import NLIVerifier, Translator
from app.models.schemas import DraftSentence, Passage, Status, SummarySentence
from app.text import has_devanagari, overlap_ratio, sentences

log = logging.getLogger("fact.write")

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


def needs_english(d: DraftSentence, passages: list[Passage]) -> bool:
    """Devanagari text, or a quote from a non-English passage (catches romanized Hindi too)."""
    by_id = {p.id: p for p in passages}
    return has_devanagari(d.sentence) or any(
        i in by_id and by_id[i].language not in (None, "en") for i in d.passage_ids)


async def to_english(drafts: list[DraftSentence], passages: list[Passage], translator: Translator) -> list[DraftSentence]:
    """Normalize drafts to English before verification and optional localization. A translated draft keeps its
    original as `source_sentence` (used to pick premise windows). NLI then checks the English sentence
    against the original passage: real mDeBERTa scores Marathi→Marathi pairs near chance (a sentence
    against itself: 0.27-0.36 entailment) but a Marathi passage → English sentence well (0.68-0.85).
    A draft whose translation fails is dropped rather than shown untranslated."""
    by_id = {p.id: p for p in passages}

    async def one(d: DraftSentence) -> DraftSentence | None:
        if not needs_english(d, passages):
            return d
        source = next((by_id[i].language for i in d.passage_ids
                       if i in by_id and by_id[i].language and by_id[i].language != "en"), None)
        try:
            if source is None:
                detected = await translator.detect(d.sentence)
                source = next((lang for lang in detected if lang != "en"), None)
            if source is None:
                return None  # cannot tell the language: drop rather than show it untranslated
            english = (await translator.translate(d.sentence, source)).strip()
        except Exception as exc:
            log.warning("summary translation failed (%s): %s", type(exc).__name__, exc)
            return None
        if not english or has_devanagari(english):
            return None
        return DraftSentence(sentence=english, passage_ids=d.passage_ids, source_sentence=d.sentence)

    return [d for d in await asyncio.gather(*(one(d) for d in drafts)) if d is not None]


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
        pairs.extend((i, p, w) for p in cited for w in best_windows(p.text, d.source_sentence or d.sentence))
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
    """Translate verified sentences (always English: see to_english) into the post's language; a
    translation is shown only if it passes the same checks as the English sentence (NLI against its own
    cited passages; under a "fails" status, no claim label and no restatement). Otherwise the verified
    English sentence stays. Returns (summary, log). Marathi rarely passes: mDeBERTa scores Marathi
    hypotheses poorly (8/20 vs 13/20 for Hindi on real evidence), so most Marathi posts keep English.

    Hinglish ("hi-Latn") gets Devanagari Hindi: Sarvam's romanized output has not been verified.
    """
    target = "hi" if target == "hi-Latn" else target
    by_source = {p.source_id: p for p in report.cited_passages}
    status_value = getattr(status, "value", status)
    out, log = [], []
    for s in report.kept:
        cited = [by_source[sid] for sid in s.sources if sid in by_source]
        if not target or not cited or base_language(target) == "en":
            out.append(s)
            continue
        try:
            translated = (await translator.translate(s.sentence, "en", target)).strip()
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
