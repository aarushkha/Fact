import httpx
import pytest

import app.model_server as ms
from app.adapters.mock import MockEmbedder, MockNLIVerifier
from app.adapters.remote_models import RemoteEmbedder, RemoteNLI
from app.config import Settings


@pytest.fixture
def model_app(monkeypatch):
    class FakeEmbedder(MockEmbedder):
        model_version = "fake-embedder@1"

        def __init__(self, *a, dim=8, **k):
            super().__init__(dim)

    class FakeNLI(MockNLIVerifier):
        model_version = "fake-nli@1"

        def __init__(self, *a, **k):
            pass

    monkeypatch.setattr(ms, "BGEM3Embedder", FakeEmbedder)
    monkeypatch.setattr(ms, "MDebertaNLI", FakeNLI)
    monkeypatch.setattr(ms, "get_settings", lambda: Settings(_env_file=None, embedding_dim=8, model_server_token="t"))
    return ms.create_model_app()


def client_for(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://models")


async def test_remote_adapters_round_trip(model_app):
    c = client_for(model_app)
    emb = RemoteEmbedder("http://models", "t", client=c, configured="bge", dim=8)
    vecs = await emb.embed(["a b c"] * 70)  # more than one server batch
    assert len(vecs) == 70 and len(vecs[0]) == 8 and emb.model_version == "fake-embedder@1"
    nli = RemoteNLI("http://models", "t", client=c, configured="mdeberta")
    (s,) = await nli.score([("the bridge collapsed on sunday", "bridge collapsed")])
    assert s.entailment == 1.0 and nli.model_version == "fake-nli@1"


async def test_token_and_dim_checks(model_app):
    c = client_for(model_app)
    with pytest.raises(Exception, match="401"):
        await RemoteEmbedder("http://models", "wrong", client=c).embed(["x"])
    with pytest.raises(ValueError, match="EMBEDDING_DIM"):
        await RemoteEmbedder("http://models", "t", client=c, dim=1024).embed(["x"])
    r = await c.get("/health")
    assert r.json()["embedder"] == "fake-embedder@1"


def test_factory_uses_model_server_when_configured():
    from app.adapters.factory import build_embedder, build_nli

    s = Settings(_env_file=None, mock_mode=False, model_server_url="http://models:8001")
    assert isinstance(build_embedder(s), RemoteEmbedder) and isinstance(build_nli(s), RemoteNLI)


def test_token_never_sent_in_cleartext_to_a_public_host():
    import pytest

    from app.adapters.remote_models import check_transport

    for ok in ["http://models:8001", "http://localhost:8001", "http://10.0.0.7:8001", "https://models.example.com"]:
        check_transport(ok, "t")
    check_transport("http://models.example.com", "")  # no token: nothing secret to protect
    with pytest.raises(ValueError, match="https"):
        check_transport("http://models.example.com:8001", "t")
    with pytest.raises(ValueError):
        check_transport("http://34.1.2.3:8001", "t")


def test_public_ipv6_literal_is_not_internal():
    import pytest

    from app.adapters.remote_models import check_transport

    with pytest.raises(ValueError):
        check_transport("http://[2606:4700:4700::1111]:8001", "t")
    check_transport("http://[::1]:8001", "t")  # loopback stays allowed
