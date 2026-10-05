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

    # --- sources whitelist ---
    sources_file: Path = ROOT_DIR / "sources.yaml"

    # --- providers (ignored when MOCK_MODE=true) ---
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3.8-flash"  # TODO: confirm exact model id against Gemini docs (step 3)
    vision_provider: str = "gemini"
    vision_model: str = "gemini-3.8-flash"
    gemini_api_key: str = ""

    translator_provider: str = "sarvam"  # sarvam | llm
    sarvam_api_key: str = ""
    sarvam_translate_model: str = ""  # TODO: set from Sarvam docs (step 3)

    classifier_provider: str = "llm"  # llm | jev | local
    jev_api_key: str = ""
    jev_model_version: str = ""  # pin an exact version, never "latest"

    embedder_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

    google_factcheck_api_key: str = ""
    web_search_enabled: bool = False

    # --- thresholds (untuned defaults; tune with eval/run.py --threshold-sweep) ---
    confidence_threshold: float = 0.6  # below this, abstain (UNVERIFIED_*)
    claim_type_threshold: float = 0.6  # min prob to mark a claim not-checkable
    passage_relevance_threshold: float = 0.5  # min prob for supports/contradicts to count
    cache_similarity_threshold: float = 0.92
    factcheck_similarity_threshold: float = 0.85
    factcheck_hit_confidence: float = 0.9
    nli_entailment_threshold: float = 0.5
    too_early_window_hours: float = 72
    recheck_too_early_hours: float = 6
    recheck_evidence_missing_days: float = 7
    retrieval_top_k: int = 8
    max_upload_mb: int = 10

    @property
    def effective_sources_file(self) -> Path:
        """In mock mode the mock corpus needs its own (fake) whitelist."""
        if self.mock_mode:
            return MOCK_SOURCES_FILE
        return self.sources_file if self.sources_file.is_absolute() else ROOT_DIR / self.sources_file


@lru_cache
def get_settings() -> Settings:
    return Settings()
