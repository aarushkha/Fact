"""Settings that can be changed from the web page (/settings) instead of editing .env / docker compose.

Precedence: UI override > environment / .env > built-in default. Overrides live in the database (so the API
and the worker share them), or in a JSON file / memory when there is no database. Which fields are editable,
how they are grouped and validated is declared here once; the page renders itself from `describe()`.

Deliberately not editable here (infrastructure or pinned decisions, change them in .env): DATABASE_URL,
SOURCES_FILE, provider base URLs, embedder / NLI model identity and EMBEDDING_DIM, SEARCH_BACKEND,
WEB_SEARCH_ENABLED.
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError
from sqlalchemy import delete, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine

from app.config import Settings
from app.db.tables import Tables


@dataclass(frozen=True)
class Field:
    name: str
    label: str
    kind: str  # bool | text | secret | choice | int | float
    help: str = ""
    choices: tuple[str, ...] = ()
    min: float | None = None
    max: float | None = None
    placeholder: str = ""


@dataclass(frozen=True)
class Group:
    id: str
    title: str
    description: str
    fields: tuple[Field, ...]


def _t(name, label, help="", **kw): return Field(name, label, "text", help, **kw)
def _s(name, label, help="", **kw): return Field(name, label, "secret", help, **kw)
def _b(name, label, help="", **kw): return Field(name, label, "bool", help, **kw)
def _c(name, label, choices, help="", **kw): return Field(name, label, "choice", help, tuple(choices), **kw)
def _i(name, label, help="", min=0, max=None, **kw): return Field(name, label, "int", help, min=min, max=max, **kw)
def _f(name, label, help="", min=0.0, max=None, **kw): return Field(name, label, "float", help, min=min, max=max, **kw)
def _p(name, label, help=""): return _f(name, label, help, min=0.0, max=1.0)  # a probability / similarity


GROUPS: tuple[Group, ...] = (
    Group("mode", "Mode", "Mock mode runs every model as a deterministic mock over a fictional corpus: no keys, no models. "
          "Real mode calls the providers below and needs their keys.", (
        _b("mock_mode", "Mock mode", "On: mocks only. Off: real providers, real models, the crawled corpus."),
        _c("log_level", "Log level", ("DEBUG", "INFO", "WARNING", "ERROR")),
        _t("timezone", "Timezone", "Interprets post dates that carry no offset, e.g. 2026-10-01.", placeholder="Asia/Kolkata"),
    )),
    Group("keys", "API keys", "Used only in real mode. Stored in the database (or the overrides file) in plain text, like "
          ".env, and never sent back to the browser: leave a field empty to keep it.", (
        _s("gemini_api_key", "Gemini API key", "LLM and screenshot reading (GEMINI_API_KEY). Required in real mode."),
        _s("google_factcheck_api_key", "Google Fact Check Tools key", "Existing fact-checks (GOOGLE_FACTCHECK_API_KEY). Required in real mode."),
        _s("sarvam_api_key", "Sarvam API key", "Language ID + Marathi/Hindi translation. Required when the translator is Sarvam."),
        _s("openrouter_api_key", "OpenRouter API key (Jev)", "The Jev classifier is called through OpenRouter. Required when the classifier is Jev."),
        _s("model_server_token", "Model server token", "Only if the models service has MODEL_SERVER_TOKEN set."),
    )),
    Group("providers", "Providers and models", "Real mode only. The defaults keep Gemini use near zero for text checks.", (
        _c("translator_provider", "Translator", ("sarvam", "llm"), "llm = script heuristics + Gemini; no Sarvam calls for English posts."),
        _c("classifier_provider", "Classifier", ("jev", "llm"), "jev falls back to the Gemini judge on errors."),
        _c("claim_extractor", "Claim extractor", ("sentences", "llm"), "sentences = no Gemini call."),
        _c("summary_writer", "Summary writer", ("extractive", "llm"), "extractive = verbatim evidence quotes, no Gemini call."),
        _c("summary_language", "Summary language", ("english", "post"), "post = also translated into the post's language and re-verified (untuned)."),
        _t("llm_model", "LLM model", "Gemini model for the LLM stages."),
        _t("llm_fallback_models", "LLM fallback models", "Comma-separated; tried in order when the main model is overloaded."),
        _c("llm_thinking_level", "LLM thinking level", ("minimal", "low", "medium", "high")),
        _t("vision_model", "Vision model", "Gemini model that reads screenshots."),
        _t("sarvam_translate_model", "Sarvam translate model"),
        _t("jev_model_version", "Jev model version", "Pinned; never use a floating alias."),
        _t("model_server_url", "Model server URL", "Run the embedder + NLI as a separate service (compose profile 'models'): "
           "http://models:8001. Empty = load them in this container.", placeholder="http://models:8001"),
    )),
    Group("thresholds", "Thresholds", "Untuned defaults; tune them with the eval harness.", (
        _p("confidence_threshold", "Confidence threshold", "Below this the system abstains (UNVERIFIED_*)."),
        _p("claim_type_threshold", "Not-checkable threshold", "Minimum probability to mark a claim NOT_CHECKABLE."),
        _p("passage_relevance_threshold", "Passage relevance", "Minimum probability for supports/contradicts to count."),
        _p("same_event_threshold", "Same-event gate", "Minimum probability that a passage is about the same incident."),
        _p("cache_similarity_threshold", "Cache similarity"),
        _p("factcheck_similarity_threshold", "Fact-check claim similarity"),
        _p("factcheck_hit_confidence", "Fact-check hit confidence"),
        _p("nli_entailment_threshold", "NLI entailment threshold", "A summary sentence must entail its cited passage at least this much."),
        _p("nli_restatement_threshold", "NLI restatement threshold", "Sentences entailing a CONTRADICTED / MISLEADING claim this strongly are dropped."),
        _f("too_early_window_hours", "Too-early window (hours)", "Posts younger than this get UNVERIFIED_TOO_EARLY instead of EVIDENCE_MISSING."),
        _i("retrieval_top_k", "Passages retrieved per claim", min=1, max=50),
        _p("search_min_vector_similarity", "Minimum vector similarity"),
        _i("search_rrf_k", "RRF k", "Reciprocal-rank-fusion constant of hybrid search.", min=1, max=1000),
    )),
    Group("protection", "API protection", "", (
        _s("api_keys", "API keys for clients", "Comma-separated, sent as X-API-Key. Empty = no auth (local testing only)."),
        _s("admin_token", "Admin token", "Guards this page and its API (X-Admin-Token). Empty = an API key is needed instead, or "
           "nothing when no API keys are set. Ignored here when ADMIN_TOKEN is set in the environment."),
        _i("rate_limit_per_minute", "Checks per minute per client", "0 = unlimited.", min=0, max=100000),
        _i("max_upload_mb", "Max screenshot size (MB)", min=1, max=100),
    )),
    Group("worker", "Worker and crawler", "Picked up by the worker within about 30 seconds.", (
        _f("recheck_interval_minutes", "Recheck every (minutes)", min=1, max=100000),
        _i("recheck_batch_size", "Rechecks per pass", min=1, max=1000),
        _f("recheck_too_early_hours", "Recheck TOO_EARLY after (hours)", min=0.1),
        _f("recheck_evidence_missing_days", "Recheck EVIDENCE_MISSING after (days)", min=0.1),
        _f("recheck_max_age_days", "Stop rechecking after (days)", min=1),
        _f("crawl_interval_minutes", "Crawl every (minutes)", min=1, max=100000),
        _i("crawler_max_articles_per_source", "Articles per source per crawl", min=1, max=10000),
        _i("crawler_concurrency", "Crawler concurrency", min=1, max=64),
        _f("crawler_min_host_interval_seconds", "Seconds between requests to one host", min=0, max=60),
        _t("crawler_user_agent", "Crawler user agent"),
        _i("chunk_max_words", "Passage chunk size (words)", min=20, max=2000),
        _f("http_timeout_seconds", "HTTP timeout (seconds)", min=1, max=600),
    )),
)

FIELDS: dict[str, Field] = {f.name: f for g in GROUPS for f in g.fields}
SECRETS = frozenset(n for n, f in FIELDS.items() if f.kind == "secret")
ENV_WINS = frozenset({"admin_token"})  # a value set in the environment cannot be replaced from the web page


class SettingsInvalid(ValueError):
    """`errors` maps a field name (or "" for the whole configuration) to a message."""

    def __init__(self, errors: dict[str, str]):
        super().__init__("; ".join(f"{k or 'settings'}: {v}" for k, v in errors.items()))
        self.errors = errors


def _check_value(f: Field, v: Any) -> Any:
    if f.kind == "bool":
        if not isinstance(v, bool):
            raise ValueError("must be true or false")
        return v
    if f.kind in ("text", "secret", "choice"):
        if not isinstance(v, str):
            raise ValueError("must be text")
        v = v.strip()
        if f.kind == "choice" and v not in f.choices:
            raise ValueError("must be one of " + ", ".join(f.choices))
        if f.kind == "secret" and any(c in v for c in "\r\n\t"):
            raise ValueError("must not contain line breaks or tabs")
        return v
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError("must be a number")
    if f.kind == "int":
        if v != int(v):
            raise ValueError("must be a whole number")
        v = int(v)
    else:
        v = float(v)
    if f.min is not None and v < f.min:
        raise ValueError(f"must be at least {f.min:g}")
    if f.max is not None and v > f.max:
        raise ValueError(f"must be at most {f.max:g}")
    return v


def clean_overrides(values: dict[str, Any]) -> dict[str, Any]:
    """Validate a {field: value} mapping from the page; raises SettingsInvalid naming every bad field."""
    out, errors = {}, {}
    for name, v in values.items():
        f = FIELDS.get(name)
        if f is None:
            errors[name] = "this setting cannot be changed here"
            continue
        try:
            out[name] = _check_value(f, v)
        except ValueError as exc:
            errors[name] = str(exc)
    if errors:
        raise SettingsInvalid(errors)
    return out


def merge_settings(base: Settings, overrides: dict[str, Any]) -> Settings:
    """The effective settings: `base` (environment) with the overrides on top."""
    overrides = {k: v for k, v in overrides.items() if k in FIELDS}
    if base.admin_token:
        overrides.pop("admin_token", None)
    if not overrides:
        return base
    try:
        return Settings(**{**base.model_dump(), **overrides})
    except ValidationError as exc:
        raise SettingsInvalid({".".join(map(str, e["loc"])) or "": e["msg"] for e in exc.errors()}) from exc


@dataclass
class Readiness:
    errors: list[str] = field(default_factory=list)  # real mode cannot work
    warnings: list[str] = field(default_factory=list)


def check_readiness(s: Settings) -> Readiness:
    r = Readiness()
    if s.mock_mode:
        return r
    if not s.database_url:
        r.errors.append("Real mode needs the database (DATABASE_URL) for the crawled corpus; docker compose sets it for you.")
    missing = []
    if not s.gemini_api_key:
        missing.append("Gemini API key")
    if not s.google_factcheck_api_key:
        missing.append("Google Fact Check Tools key")
    if s.translator_provider == "sarvam" and not s.sarvam_api_key:
        missing.append("Sarvam API key (or switch the translator to llm)")
    if s.classifier_provider == "jev" and not s.openrouter_api_key:
        missing.append("OpenRouter API key (or switch the classifier to llm)")
    if missing:
        r.errors.append("Real mode needs: " + ", ".join(missing) + ".")
    if not s.model_server_url and any(
        importlib.util.find_spec(m) is None for m in ("sentence_transformers", "transformers", "torch")
    ):
        r.errors.append(
            "The embedder and NLI models are not installed in this container. Rebuild the image with "
            "INSTALL_MODELS=true, or run the models service (docker compose --profile models up) and set its URL."
        )
    if not s.api_keys and not s.admin_token:
        r.warnings.append("No API keys are set: anyone who can reach this server can run checks (fine for local testing only).")
    return r


def describe(base: Settings, current: Settings, overrides: dict[str, Any]) -> list[dict]:
    """The groups, with each field's current value and where it comes from. Secrets are never included."""
    groups = []
    for g in GROUPS:
        fields = []
        for f in g.fields:
            locked = f.name in ENV_WINS and bool(getattr(base, f.name))
            if f.name in overrides and not locked:
                source = "ui"
            elif f.name in base.model_fields_set:
                source = "env"
            else:
                source = "default"
            item: dict[str, Any] = {
                "name": f.name, "label": f.label, "kind": f.kind, "help": f.help, "source": source,
                "choices": list(f.choices), "min": f.min, "max": f.max, "placeholder": f.placeholder, "locked": locked,
            }
            value = getattr(current, f.name)
            if f.kind == "secret":
                item["is_set"] = bool(value)
            else:
                item["value"] = value
            fields.append(item)
        groups.append({"id": g.id, "title": g.title, "description": g.description, "fields": fields})
    return groups


# ---------------------------------------------------------------------------------------------- persistence
class OverrideStore(Protocol):
    kind: str  # database | file | memory

    async def load(self) -> dict[str, Any]: ...

    async def save(self, values: dict[str, Any]) -> None:
        """Replace everything stored with `values`."""


class MemoryOverrides:
    kind = "memory"

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    async def load(self) -> dict[str, Any]:
        return dict(self._values)

    async def save(self, values: dict[str, Any]) -> None:
        self._values = dict(values)


class FileOverrides:
    kind = "file"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    async def load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        return data if isinstance(data, dict) else {}

    async def save(self, values: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name, suffix=".tmp")  # 0600
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(values, fh, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            os.unlink(tmp)
            raise


class DbOverrides:
    kind = "database"

    def __init__(self, engine: AsyncEngine, tables: Tables):
        self.engine, self.t = engine, tables.settings_overrides

    async def load(self) -> dict[str, Any]:
        async with self.engine.connect() as conn:
            rows = (await conn.execute(self.t.select())).all()
        return {r.key: r.value for r in rows}

    async def save(self, values: dict[str, Any]) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(delete(self.t).where(self.t.c.key.not_in(list(values) or [""])))
            if values:
                ins = pg_insert(self.t).values([{"key": k, "value": v} for k, v in values.items()])
                await conn.execute(ins.on_conflict_do_update(
                    index_elements=[self.t.c.key],
                    set_={"value": ins.excluded.value, "updated_at": func.now()},
                ))
