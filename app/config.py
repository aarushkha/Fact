"""All configuration comes from the environment / .env (see .env.example)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
MOCK_SOURCES_FILE = ROOT_DIR / "app" / "adapters" / "mock_data" / "sources.yaml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- mode ---
    mock_mode: bool = True
    log_level: str = "INFO"
    timezone: str = "Asia/Kolkata"  # used to interpret date-only post dates

    # --- storage / search ---
    database_url: str = ""  # empty -> in-memory store (mock search only)
    search_backend: str = "auto"  # auto | mock | postgres ; auto = postgres when DATABASE_URL is set

    # --- crawler ---
    crawler_user_agent: str = "FactCrawler/0.1 (+contact: TODO)"
    crawler_max_articles_per_source: int = 50
    crawler_concurrency: int = 4
    chunk_max_words: int = 180

    # --- sources whitelist ---
    sources_file: Path = ROOT_DIR / "sources.yaml"

    # --- providers (ignored when MOCK_MODE=true) ---
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3.8-flash"
    # Tried in order when the primary model is overloaded (HTTP 429/503) or unavailable.
    llm_fallback_models: str = "gemini-3.5-flash,gemini-3.5-flash-lite"
    llm_thinking_level: str = "low"  # minimal | low | medium | high
    # Gemini-free options (real mode): sentences | llm  and  extractive | llm
    claim_extractor: str = "sentences"
    summary_writer: str = "extractive"
    vision_provider: str = "gemini"
    vision_model: str = "gemini-3.8-flash"
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    translator_provider: str = "sarvam"  # sarvam | llm
    sarvam_api_key: str = ""
    sarvam_translate_model: str = "mayura:v1"
    sarvam_base_url: str = "https://api.sarvam.ai"

    classifier_provider: str = "jev"  # jev (falls back to llm on errors) | llm
    openrouter_api_key: str = ""  # Jev is called through OpenRouter's Decisions API
    jev_model_version: str = "typesafe/jev-1.13"  # pinned; never "~typesafe/jev-latest"
    openrouter_base_url: str = "https://openrouter.ai/api"

    embedder_model: str = "BAAI/bge-m3"
    embedder_revision: str = "5617a9f61b028005a4858fdac845db406aefb181"  # pinned HF commit
    embedding_dim: int = 1024
    embedder_max_seq_length: int = 1024
    nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    nli_revision: str = "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"  # pinned HF commit

    google_factcheck_api_key: str = ""
    web_search_enabled: bool = False  # TODO: no paid web-search provider chosen yet
    http_timeout_seconds: float = 60

    # --- thresholds (untuned defaults; tune with eval/run.py --threshold-sweep) ---
    confidence_threshold: float = 0.6  # below this, abstain (UNVERIFIED_*)
    claim_type_threshold: float = 0.6  # min prob to mark a claim not-checkable
    passage_relevance_threshold: float = 0.5  # min prob for supports/contradicts to count
    same_event_threshold: float = 0.6  # min P(passage is about the same incident) for it to count at all
    cache_similarity_threshold: float = 0.92
    cascade_similarity_threshold: float = 0.85  # "same rumour" for counting repeat submissions
    factcheck_similarity_threshold: float = 0.85
    factcheck_hit_confidence: float = 0.9
    nli_entailment_threshold: float = 0.5
    too_early_window_hours: float = 72
    recheck_too_early_hours: float = 6
    recheck_evidence_missing_days: float = 7
    recheck_max_age_days: float = 30  # stop rechecking claims older than this
    recheck_interval_minutes: float = 15  # worker: how often to look for due rechecks
    recheck_batch_size: int = 20
    crawl_interval_minutes: float = 60  # worker: how often to crawl all sources
    retrieval_top_k: int = 8
    search_min_vector_similarity: float = 0.5  # BGE-M3: unrelated short texts score ~0.4
    search_rrf_k: int = 60
    max_upload_mb: int = 10

    # --- API protection ---
    api_keys: str = ""  # comma-separated; empty = auth disabled (local development only)
    rate_limit_per_minute: int = 10  # checks per client (API key, or IP when auth is off); 0 = off

    @property
    def effective_search_backend(self) -> str:
        if self.search_backend != "auto":
            return self.search_backend
        return "postgres" if self.database_url else "mock"

    @property
    def effective_sources_file(self) -> Path:
        """In mock mode the mock corpus needs its own (fake) whitelist."""
        if self.mock_mode:
            return MOCK_SOURCES_FILE
        return self.sources_file if self.sources_file.is_absolute() else ROOT_DIR / self.sources_file


@lru_cache
def get_settings() -> Settings:
    return Settings()
