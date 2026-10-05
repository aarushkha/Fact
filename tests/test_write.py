from app.adapters.mock import MockNLIVerifier
from app.models.schemas import DraftSentence
from app.pipeline.write import verify_sentences

from .helpers import passage

P1 = passage("p1", text="Nashik police confirmed the footbridge over the Godavari collapsed on Sunday.")
P2 = passage("p2", text="Mumbai airport is operating normally.")


async def test_unsupported_sentence_is_deleted():
    drafts = [
        DraftSentence(sentence="Nashik police confirmed the footbridge collapsed on Sunday.", passage_ids=["p1"]),
        DraftSentence(sentence="Dozens of people died in the collapse.", passage_ids=["p1"]),
    ]
    r = await verify_sentences(drafts, [P1, P2], MockNLIVerifier(), 0.5)
    assert [s.sentence for s in r.kept] == [drafts[0].sentence]
    assert r.kept[0].sources == ["src_p1"]
    assert r.dropped[0]["sentence"] == drafts[1].sentence


async def test_non_entailing_citations_are_pruned():
    d = DraftSentence(sentence="Nashik police confirmed the footbridge collapsed.", passage_ids=["p1", "p2"])
    r = await verify_sentences([d], [P1, P2], MockNLIVerifier(), 0.5)
    assert r.kept[0].sources == ["src_p1"]
    assert [p.id for p in r.cited_passages] == ["p1"]


async def test_uncited_or_unknown_citations_deleted():
    drafts = [
        DraftSentence(sentence="Nashik police confirmed the footbridge collapsed.", passage_ids=[]),
        DraftSentence(sentence="Nashik police confirmed the footbridge collapsed.", passage_ids=["nope"]),
    ]
    r = await verify_sentences(drafts, [P1], MockNLIVerifier(), 0.5)
    assert r.kept == [] and len(r.dropped) == 2


async def test_nothing_survives_returns_empty_summary():
    d = DraftSentence(sentence="Completely unrelated words here.", passage_ids=["p1"])
    r = await verify_sentences([d], [P1], MockNLIVerifier(), 0.5)
    assert r.kept == [] and r.cited_passages == []


async def test_at_most_three_sentences():
    d = DraftSentence(sentence="Mumbai airport is operating normally.", passage_ids=["p2"])
    r = await verify_sentences([d] * 5, [P2], MockNLIVerifier(), 0.5)
    assert len(r.kept) == 3


async def test_sentence_entailed_by_one_window_of_a_long_passage_survives():
    from app.models.schemas import NLIScore
    from app.pipeline.write import premise_windows

    long_p = passage("lp", text="Police confirmed the bridge collapsed on Sunday. Three people were hurt. Traffic was diverted.")

    class SentenceLevelNLI:
        """Like a real cross-encoder: only a short, focused premise entails."""
        model_version = "fake"

        async def score(self, pairs):
            return [NLIScore(entailment=0.9 if p == "Police confirmed the bridge collapsed on Sunday." else 0.3,
                             neutral=0.1, contradiction=0.0) for p, _ in pairs]

    d = DraftSentence(sentence="Police said the bridge collapsed on Sunday.", passage_ids=["lp"])
    r = await verify_sentences([d], [long_p], SentenceLevelNLI(), 0.5)
    assert [s.sentence for s in r.kept] == [d.sentence] and r.kept[0].sources == ["src_lp"]
    assert premise_windows("A. B. C.") == ["A. B. C.", "A.", "B.", "C.", "A. B.", "B. C."]


def test_best_windows_bounded_and_relevant():
    from app.pipeline.write import best_windows

    text = " ".join(f"Sentence {i} about topic {i}." for i in range(12)) + " The bridge collapsed in Nashik."
    w = best_windows(text, "A bridge collapsed in Nashik.")
    assert len(w) == 4 and w[0] == text and w[1] == "The bridge collapsed in Nashik."


async def test_restated_claim_dropped_under_contradicted():
    debunk = passage("p9", text="Mumbai airport is closed for a week. This claim is false: the airport is not closed.")
    drafts = [DraftSentence(sentence="Mumbai airport is closed for a week.", passage_ids=["p9"]),
              DraftSentence(sentence="This claim is false: the airport is not closed.", passage_ids=["p9"])]
    claim_text = "Mumbai airport is closed for a week."
    r = await verify_sentences(drafts, [debunk], MockNLIVerifier(), 0.5, claim_text=claim_text, status="CONTRADICTED")
    assert [s.sentence for s in r.kept] == ["This claim is false: the airport is not closed."]
    assert r.dropped == [{"sentence": "Mumbai airport is closed for a week.", "reason": "restates the claim under CONTRADICTED"}]
    # Under CONFIRMED a sentence that matches the claim is exactly what the summary should say.
    r = await verify_sentences(drafts[:1], [debunk], MockNLIVerifier(), 0.5, claim_text=claim_text, status="CONFIRMED")
    assert [s.sentence for s in r.kept] == ["Mumbai airport is closed for a week."]


async def test_correction_phrased_as_is_false_is_kept_under_contradicted():
    p = passage("p8", text="The claim that Mumbai airport is closed for a week is false.")
    d = DraftSentence(sentence="The claim that Mumbai airport is closed for a week is false.", passage_ids=["p8"])
    r = await verify_sentences([d], [p], MockNLIVerifier(), 0.5, claim_text="Mumbai airport is closed for a week.",
                               status="CONTRADICTED")
    assert [s.sentence for s in r.kept] == [d.sentence]


async def test_labelled_claim_quote_dropped_under_fails_status_only():
    # Real debunks quote the claim under a label; NLI often scored these below the restatement threshold.
    p = passage("p7", text="Claim: The video shows a flood in Pune last week. Fact: the video is from 2019 in Kerala.")
    d = DraftSentence(sentence="Claim: The video shows a flood in Pune last week.", passage_ids=["p7"])
    other = "Something unrelated entirely."
    r = await verify_sentences([d], [p], MockNLIVerifier(), 0.5, claim_text=other, status="MISLEADING_CONTEXT")
    assert r.kept == [] and r.dropped[0]["reason"] == "quotes the claim under MISLEADING_CONTEXT"
    r = await verify_sentences([d], [p], MockNLIVerifier(), 0.5, claim_text=other, status="CONFIRMED")
    assert len(r.kept) == 1


async def test_marathi_quote_is_shown_in_english_and_verified_against_the_original():
    from app.adapters.mock import MockTranslator
    from app.pipeline.write import to_english

    mr_text = "नाशिक शहर पोलिसांनी गोदावरी नदीवरील जुन्या पादचारी पुलाचा भाग रविवारी संध्याकाळी कोसळल्याची पुष्टी केली."
    p = passage("pmr", text=mr_text).model_copy(update={"language": "mr"})
    from app.adapters.extractive import ExtractiveWriter
    from app.adapters.mock import MockEmbedder

    from .helpers import claim

    quoted = await ExtractiveWriter(MockEmbedder(64)).write_summary(claim("A footbridge in Nashik collapsed."), [p], "CONFIRMED")
    assert [d.sentence for d in quoted] == [mr_text]  # the writer quotes verbatim, in Marathi
    drafts = await to_english(quoted, [p], MockTranslator())
    assert drafts[0].sentence.startswith("Nashik City Police confirmed") and drafts[0].source_sentence == mr_text
    r = await verify_sentences(drafts, [p], MockNLIVerifier(), 0.5, claim_text="A footbridge in Nashik collapsed.",
                               status="CONFIRMED")
    assert [s.sentence for s in r.kept] == [drafts[0].sentence]


async def test_untranslatable_quote_is_dropped_not_shown_in_marathi():
    from app.pipeline.write import to_english

    class Down:
        model_version = "down"

        async def detect(self, text):
            return ["mr"]

        async def translate(self, text, source, target="en"):
            raise RuntimeError("translator down")

    p = passage("pmr", text="काहीतरी मराठी वाक्य येथे आहे.").model_copy(update={"language": "mr"})
    d = DraftSentence(sentence="काहीतरी मराठी वाक्य येथे आहे.", passage_ids=["pmr"])
    assert await to_english([d], [p], Down()) == []


async def test_mock_nli_accepts_a_marathi_sentence_against_itself():
    mr = "नाशिक शहर पोलिसांनी गोदावरी नदीवरील जुन्या पादचारी पुलाचा भाग रविवारी संध्याकाळी कोसळल्याची पुष्टी केली."
    (s,) = await MockNLIVerifier().score([(mr, mr)])
    assert s.entailment == 1.0


async def test_undetectable_language_is_dropped_and_romanized_hindi_is_translated():
    from app.adapters.mock import MockTranslator
    from app.pipeline.write import to_english

    class NoDetect(MockTranslator):
        async def detect(self, text):
            return []

    d = DraftSentence(sentence="काहीतरी मराठी वाक्य येथे आहे.", passage_ids=["x"])
    assert await to_english([d], [passage("x", text=d.sentence)], NoDetect()) == []
    hi = passage("h", text="Mumbai airport ek hafte ke liye band hai.").model_copy(update={"language": "hi"})
    out = await to_english([DraftSentence(sentence=hi.text, passage_ids=["h"])], [hi], MockTranslator())
    assert [x.sentence for x in out] == ["Mumbai airport is closed for a week."]
