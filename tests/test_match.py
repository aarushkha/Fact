from app.adapters.mock import MockEmbedder, MockFactCheckSearch
from app.db.store import InMemoryStore
from app.models.schemas import ClaimResult, ClaimType, Entity, Status
from app.pipeline.match import decisive_factcheck, map_rating, match_factchecks

from .conftest import NOW
from .helpers import claim


def test_rating_map_is_exact_and_conservative():
    assert map_rating("False") == Status.CONTRADICTED
    assert map_rating(" Missing Context. ") == Status.MISLEADING_CONTEXT
    assert map_rating("गलत") == Status.CONTRADICTED
    assert map_rating("Mostly false") is None
    assert map_rating("Unproven") is None


async def test_whitelisted_similar_factcheck_matches(whitelist):
    c = claim("The state government is giving free laptops to all college students", entities=())
    emb = MockEmbedder()
    (vec,) = await emb.embed([c.text_en])
    matches = await match_factchecks(c, vec, ["en"], MockFactCheckSearch(), emb, whitelist, 0.85)
    assert len(matches) == 1 and matches[0].status == Status.CONTRADICTED
    assert decisive_factcheck(matches) is matches[0]


async def test_non_whitelisted_factcheck_ignored(whitelist):
    c = claim("Pune metro bridge collapsed this morning", entities=())
    emb = MockEmbedder()
    (vec,) = await emb.embed([c.text_en])
    assert await match_factchecks(c, vec, ["en"], MockFactCheckSearch(), emb, whitelist, 0.85) == []


async def test_dissimilar_factcheck_ignored(whitelist):
    c = claim("Free laptops were stolen from a college in Pune last year", entities=())
    emb = MockEmbedder()
    (vec,) = await emb.embed([c.text_en])
    assert await match_factchecks(c, vec, ["en"], MockFactCheckSearch(), emb, whitelist, 0.85) == []


def _result(status):
    return ClaimResult(text_original="x", text_en="x", type=ClaimType.CHECKABLE, status=status, confidence=0.9)


async def test_cache_requires_similarity_entities_and_definitive_status():
    s = InMemoryStore()
    v = [1.0, 0.0]
    mumbai = [Entity(text="Mumbai", kind="place")]
    await s.save_claim("chk1", _result(Status.CONTRADICTED), v, mumbai, [], {}, None)
    await s.save_claim("chk2", _result(Status.UNVERIFIED_TOO_EARLY), v, mumbai, [], {}, None)
    hit = await s.find_cached(v, [Entity(text="mumbai")], 0.9, NOW)
    assert hit is not None and hit.check_id == "chk1"
    assert await s.find_cached(v, [Entity(text="Pune")], 0.9, NOW) is None  # no shared entity
    assert await s.find_cached([0.0, 1.0], mumbai, 0.9, NOW) is None  # not similar
    assert await s.find_cached(v, [], 0.9, NOW) is None  # no entities -> no cache
