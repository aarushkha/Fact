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
