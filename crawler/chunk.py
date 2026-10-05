"""Split article text into passages of whole sentences."""

from __future__ import annotations

from app.text import sentences


def chunk_text(text: str, max_words: int = 180, overlap_sentences: int = 1) -> list[str]:
    """Pack sentences into chunks of at most ~max_words, carrying `overlap_sentences` into the next chunk.

    A single sentence longer than max_words is split on word boundaries.
    """
    units: list[str] = []
    for para in (p.strip() for p in text.split("\n")):
        for s in sentences(para) if para else []:
            words = s.split()
            if len(words) <= max_words:
                units.append(s)
            else:
                units.extend(" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words))

    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for u in units:
        n = len(u.split())
        if current and count + n > max_words:
            chunks.append(" ".join(current))
            carry = current[-overlap_sentences:] if overlap_sentences and len(current) > overlap_sentences else []
            current, count = list(carry), sum(len(c.split()) for c in carry)
        current.append(u)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks
