"""Gemini-free extraction/writing, eval-only exclusion and single-claim mode."""

from app.adapters.extractive import ExtractiveWriter, SentenceExtractor
from app.adapters.mock import MockEmbedder, MockTranslator
from app.models.schemas import CheckInput, Status
from app.text import guess_entities, url_key

from .helpers import claim, passage


def test_url_key():
    assert url_key("https://www.AltNews.in/a/b/amp/?x=1#f") == url_key("http://altnews.in/a/b")


def test_guess_entities():
    names = {e.text for e in guess_entities("The Mumbai Police arrested two men in Pune in 2026.")}
    assert {"Mumbai Police", "Pune", "2026"} <= names and "The" not in names


async def test_sentence_extractor_translates_when_unaligned():
    ex = SentenceExtractor(MockTranslator())
    out = await ex.extract_claims("Mumbai airport ek hafte ke liye band hai.", "Mumbai airport is closed for a week.", ["hi-Latn"])
    assert [(c.text_original, c.text_en) for c in out] == [("Mumbai airport ek hafte ke liye band hai.", "Mumbai airport is closed for a week.")]
    out = await ex.extract_claims("One claim here. Two claim here.", "Only one sentence.", ["en"])
    assert len(out) == 2 and out[0].text_en == "One claim here."


async def test_extractive_writer_quotes_best_sentence_per_passage():
    p1 = passage("p1", text="Weather was fine. Mumbai airport is operating normally and is not closed for a week.")
    p2 = passage("p2", text="Cricket scores today. The airport is not closed, officials of Mumbai airport said.")
    drafts = await ExtractiveWriter(MockEmbedder(64)).write_summary(claim(), [p1, p2], "CONTRADICTED")
    assert [d.sentence for d in drafts] == [
        "Mumbai airport is operating normally and is not closed for a week.",
        "The airport is not closed, officials of Mumbai airport said.",
    ]
    assert [d.passage_ids for d in drafts] == [["p1"], ["p2"]]


async def test_exclude_urls_removes_labelling_evidence(pipeline):
    text = "The state government is giving free laptops to all college students."
    res = await pipeline.run(CheckInput(text=text))
    assert res.claims[0].status == Status.CONTRADICTED  # via the mock fact-check
    guarded = await pipeline.run(CheckInput(
        text=text, exclude_urls=["https://factcheck-desk.mock.example/2026/09/free-laptop-scheme-false"]))
    assert guarded.claims[0].status == Status.UNVERIFIED_EVIDENCE_MISSING


async def test_single_claim_skips_extraction(pipeline):
    res = await pipeline.run(CheckInput(text="Mumbai airport is closed for a week. Nashik is beautiful.", single_claim=True))
    assert len(res.claims) == 1
