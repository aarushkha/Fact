"""Builds the adapter bundle from settings. MOCK_MODE=true forces every model adapter to its mock."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine

from app.adapters import mock
from app.adapters.base import Adapters, Classifier, Embedder, Search, Translator
from app.config import Settings
from app.db.tables import Tables
from app.sources import load_whitelist


def build_embedder(settings: Settings) -> Embedder:
    if settings.mock_mode:
        return mock.MockEmbedder(settings.embedding_dim)
    if settings.model_server_url:
        from app.adapters.remote_models import RemoteEmbedder

        return RemoteEmbedder(settings.model_server_url, settings.model_server_token,
                              configured=settings.embedder_model, dim=settings.embedding_dim)
    from app.adapters.local_models import BGEM3Embedder

    return BGEM3Embedder(
        settings.embedder_model, settings.embedder_revision or None,
        max_seq_length=settings.embedder_max_seq_length, dim=settings.embedding_dim,
    )


def build_nli(settings: Settings):
    if settings.model_server_url:
        from app.adapters.remote_models import RemoteNLI

        return RemoteNLI(settings.model_server_url, settings.model_server_token, configured=settings.nli_model)
    from app.adapters.local_models import MDebertaNLI

    return MDebertaNLI(settings.nli_model, settings.nli_revision or None)


def build_search(
    settings: Settings, embedder: Embedder, engine: AsyncEngine | None, tables: Tables | None
) -> Search:
    backend = settings.effective_search_backend
    if backend == "postgres":
        if engine is None or tables is None:
            raise ValueError("SEARCH_BACKEND=postgres needs DATABASE_URL.")
        from app.db.pg_search import PgSearch

        return PgSearch(
            engine, tables, embedder,
            rrf_k=settings.search_rrf_k, min_vector_similarity=settings.search_min_vector_similarity,
        )
    if backend == "mock":
        if not settings.mock_mode:
            raise ValueError("The mock search corpus is only available with MOCK_MODE=true; set DATABASE_URL.")
        return mock.MockSearch(load_whitelist(str(settings.effective_sources_file)))
    raise ValueError(f"Unknown SEARCH_BACKEND={backend!r}")


def _fallbacks(settings: Settings) -> list[str]:
    return [m.strip() for m in settings.llm_fallback_models.split(",") if m.strip()]


def build_adapters(
    settings: Settings,
    engine: AsyncEngine | None = None,
    tables: Tables | None = None,
    *,
    embedder: Embedder | None = None,
    nli=None,
) -> Adapters:
    """`embedder` / `nli` let a settings reload keep models that are already loaded (they take minutes to load)."""
    embedder = embedder or build_embedder(settings)
    search = build_search(settings, embedder, engine, tables)
    if settings.mock_mode:
        return Adapters(
            vision=mock.MockVisionReader(),
            translator=mock.MockTranslator(),
            llm=mock.MockLLM(),
            classifier=mock.MockClassifier(),
            embedder=embedder,
            nli=mock.MockNLIVerifier(),
            factcheck=mock.MockFactCheckSearch(),
            search=search,
        )

    from app.adapters.gemini import GeminiClient, GeminiLLM, GeminiVisionReader
    from app.adapters.google_factcheck import GoogleFactCheckSearch
    from app.adapters.llm_judge import FallbackClassifier, LLMClassifier
    if settings.llm_provider != "gemini" or settings.vision_provider != "gemini":
        raise ValueError("Only the gemini provider is implemented for LLM_PROVIDER / VISION_PROVIDER.")

    def gemini(model: str) -> GeminiClient:
        return GeminiClient(
            settings.gemini_api_key, model, _fallbacks(settings), base_url=settings.gemini_base_url,
            thinking_level=settings.llm_thinking_level, timeout=settings.http_timeout_seconds,
        )

    llm = GeminiLLM(gemini(settings.llm_model))
    judge_llm = LLMClassifier(gemini(settings.llm_model))

    translator: Translator
    if settings.translator_provider == "sarvam":
        from app.adapters.sarvam import SarvamTranslator

        translator = SarvamTranslator(settings.sarvam_api_key, settings.sarvam_translate_model, base_url=settings.sarvam_base_url)
    elif settings.translator_provider == "llm":
        translator = LLMTranslator(llm)
    else:
        raise ValueError(f"Unknown TRANSLATOR_PROVIDER={settings.translator_provider!r}")

    if settings.claim_extractor not in ("llm", "sentences") or settings.summary_writer not in ("llm", "extractive"):
        raise ValueError("CLAIM_EXTRACTOR must be llm|sentences and SUMMARY_WRITER llm|extractive")
    from app.adapters.extractive import ExtractiveWriter, HybridLLM, SentenceExtractor

    pipeline_llm = HybridLLM(
        llm,
        extractor=SentenceExtractor(translator) if settings.claim_extractor == "sentences" else None,
        writer=ExtractiveWriter(embedder) if settings.summary_writer == "extractive" else None,
    )

    classifier: Classifier
    if settings.classifier_provider == "jev":
        from app.adapters.jev import JevClassifier

        jev = JevClassifier(settings.openrouter_api_key, settings.jev_model_version, base_url=settings.openrouter_base_url)
        classifier = FallbackClassifier(jev, judge_llm)
    elif settings.classifier_provider == "llm":
        classifier = judge_llm
    else:
        raise ValueError(f"Unknown CLASSIFIER_PROVIDER={settings.classifier_provider!r}")

    if settings.web_search_enabled:
        # TODO: implement a paid web-search adapter (see app/adapters/web_search.py) after choosing a provider.
        raise ValueError("WEB_SEARCH_ENABLED=true but no web-search provider is implemented yet.")

    return Adapters(
        vision=GeminiVisionReader(gemini(settings.vision_model)),
        translator=translator,
        llm=pipeline_llm,
        classifier=classifier,
        embedder=embedder,
        nli=nli or build_nli(settings),
        factcheck=GoogleFactCheckSearch(settings.google_factcheck_api_key),
        search=search,
    )


class LLMTranslator:
    """TRANSLATOR_PROVIDER=llm: language detection by script heuristics, translation by the LLM."""

    def __init__(self, llm):
        self.llm = llm
        self._detector = mock.MockTranslator()  # pure script/marker heuristics, no fixtures involved

    @property
    def model_version(self) -> str:
        return f"llm-translate/{self.llm.model_version}"

    async def detect(self, text: str) -> list[str]:
        return await self._detector.detect(text)

    async def translate(self, text: str, source: str, target: str = "en") -> str:
        return text if source == target else await self.llm.translate(text, source, target)
