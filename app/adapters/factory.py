"""Builds the adapter bundle from settings. MOCK_MODE=true forces every adapter to its mock."""

from __future__ import annotations

from app.adapters import mock
from app.adapters.base import Adapters
from app.config import Settings
from app.sources import load_whitelist


def build_adapters(settings: Settings) -> Adapters:
    if settings.mock_mode:
        whitelist = load_whitelist(str(settings.effective_sources_file))
        return Adapters(
            vision=mock.MockVisionReader(),
            translator=mock.MockTranslator(),
            llm=mock.MockLLM(),
            classifier=mock.MockClassifier(),
            embedder=mock.MockEmbedder(settings.embedding_dim),
            nli=mock.MockNLIVerifier(),
            factcheck=mock.MockFactCheckSearch(),
            search=mock.MockSearch(whitelist),
        )
    # Real adapters are added in build step 3, one at a time, each written from the provider's docs.
    raise NotImplementedError("Real adapters are not implemented yet; set MOCK_MODE=true.")
