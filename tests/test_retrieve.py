from app.adapters.mock import MockSearch
from app.pipeline.retrieve import apply_whitelist, rank_passages, retrieve
from app.sources import SourceEntry, Whitelist

from .helpers import claim, passage, ts


def test_rank_tier_then_relevance_then_recency():
    ps = [
        passage("t2_hi", tier=2, relevance=0.99),
        passage("t1_lo_old", tier=1, relevance=0.4, published_at=ts(1)),
        passage("t1_lo_new", tier=1, relevance=0.4, published_at=ts(20)),
        passage("t1_hi", tier=1, relevance=0.8),
        passage("t3", tier=3, relevance=1.0),
    ]
    assert [p.id for p in rank_passages(ps)] == ["t1_hi", "t1_lo_new", "t1_lo_old", "t2_hi", "t3"]


def test_whitelist_drops_unknown_and_overrides_tier():
    w = Whitelist([SourceEntry(name="Gov", domain="gov.example", tier=1, kind="government")])
    ps = [passage("a", tier=3, url="https://gov.example/x"), passage("b", url="https://blog.example/y")]
    out = apply_whitelist(ps, w)
    assert [p.id for p in out] == ["a"]
    assert out[0].tier == 1 and out[0].publisher == "Gov" and out[0].kind == "government"


async def test_retrieve_queries_both_languages_and_filters(whitelist):
    seen = []

    class Spy(MockSearch):
        async def search(self, queries, embedding, k, published_after=None):
            seen.extend(queries)
            return await super().search(queries, embedding, k, published_after)

    c = claim("Pune metro bridge collapsed this morning.", entities=("Pune",))
    c = c.model_copy(update={"text_original": "पुणे मेट्रो पूल आज सकाळी कोसळला."})
    out = await retrieve(c, ["mr"], None, Spy(whitelist), whitelist, k=8)
    assert seen[0] == (c.text_original, "mr") and seen[1] == (c.text_en, "en")
    assert all("random-blog" not in p.url for p in out)  # not whitelisted


async def test_excluded_urls_do_not_take_top_k_slots(whitelist):
    from app.text import document_key

    c = claim("A footbridge over the Godavari river in Nashik collapsed on Sunday.", entities=("Nashik",))
    full = await retrieve(c, ["en"], None, MockSearch(whitelist), whitelist, k=8)
    assert len(full) >= 2
    top = full[0]
    out = await retrieve(c, ["en"], None, MockSearch(whitelist), whitelist, k=1, exclude={document_key(top.url)})
    assert len(out) == 1 and out[0].url != top.url  # the next eligible passage fills the slot
