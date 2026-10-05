# Fact

Fact-checking backend for social-media posts in Marathi, Hindi, Hinglish and English. A user submits
a screenshot or text; each claim gets an **evidence status** with citations. Statuses describe the
evidence found, never reality: missing evidence is `UNVERIFIED_*`, never `CONTRADICTED`.

Statuses: `CONFIRMED`, `CONTRADICTED`, `MISLEADING_CONTEXT`, `UNVERIFIED_TOO_EARLY`,
`UNVERIFIED_EVIDENCE_MISSING`, `NOT_CHECKABLE`.

## Setup

Needs git plus Docker **or** Python 3.11+.

```bash
git clone https://github.com/aarushkha/Fact.git && cd Fact
cp .env.example .env
```

| How | Command | Then open |
|---|---|---|
| Docker: app + Postgres | `docker compose up --build` | http://localhost:8000/ |
| Python only, in-memory, mock search | `pip install -e '.[dev]'` then `uvicorn app.main:app --reload` | http://localhost:8000/ |
| Python app + Docker Postgres | `docker compose up -d db`, set `DATABASE_URL=postgresql+asyncpg://fact:fact@localhost:5432/fact`, run uvicorn | http://localhost:8000/ |

The test page lets you paste text or upload a screenshot (with an optional post date). It streams
events live and shows claim cards (status, confidence, cited sentences, expected-evidence checklist),
plus a raw-JSON toggle and the model versions.

## MOCK_MODE

`MOCK_MODE=true` (the default) swaps every external model for a deterministic mock and uses a small
**fictional** corpus (`app/adapters/mock_data/`, publishers on `*.mock.example`). It needs no keys
and runs fully offline. Try these:

| Paste | Expected |
|---|---|
| `A footbridge over the Godavari river in Nashik collapsed on Sunday.` | CONFIRMED |
| `Mumbai airport ek hafte ke liye band hai.` | CONTRADICTED |
| `This video shows floods in Kolhapur this week.` | MISLEADING_CONTEXT |
| `A fire broke out at a chemical factory in Thane.` + post date 1 h ago / no date | UNVERIFIED_TOO_EARLY / _EVIDENCE_MISSING |
| `Nashik is the most beautiful city in India.` | NOT_CHECKABLE |

## Real mode (MOCK_MODE=false)

1. Install the self-hosted models (about 3 GB of weights, downloaded on first use):
   `pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install -e '.[models]'`.
   With Docker: `INSTALL_MODELS=true docker compose up --build`.
2. Put the keys in `.env`: `GEMINI_API_KEY`, `SARVAM_API_KEY`, `OPENROUTER_API_KEY` (for Jev) and
   `GOOGLE_FACTCHECK_API_KEY`. Also set `MOCK_MODE=false` and `DATABASE_URL`.
3. Fill `sources.yaml` and run the crawler (below). Until then, every claim is UNVERIFIED.
   To try real models first on the fictional corpus, set `SOURCES_FILE=app/adapters/mock_data/sources.yaml`
   and run `python -m app.db.seed`.

| Adapter | Real implementation (written from the provider's docs, verified live) |
|---|---|
| VisionReader, LLM | Gemini Interactions API with JSON-schema output. Overloaded or rate-limited models are skipped and `LLM_FALLBACK_MODELS` is tried. |
| Translator | Sarvam `/text-lid` + `/translate` (`mayura:v1`). Hinglish is detected as `hi-Latn`. |
| Classifier | Jev `typesafe/jev-1.13` via the OpenRouter Decisions API; falls back to Gemini returning JSON probabilities. |
| Embedder | `BAAI/bge-m3`, in-process, pinned to a commit |
| NLIVerifier | `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, in-process, pinned to a commit |
| FactCheckSearch | Google Fact Check Tools `claims:search` |
| Web search | stub, off by default (see TODOs) |

**Gemini free tier:** `gemini-3.8-flash` allows 5 requests/min and 20/day. A check uses about
1 + (number of claims) Gemini calls, plus 1 per screenshot. The model that actually answered is
logged per stage and returned in `model_versions`.

## How a check works

`ingest` → `normalize` → `extract` → (cache + fact-check match) → `retrieve` → `judge` → `write` →
`verify` → `store`. Every stage logs inputs, outputs, latency and model version to `stage_runs`.

These rules are enforced in code (`app/pipeline/judge.py`, `write.py`), not left to the models:
- A definitive status needs a matching passage from the best (lowest-numbered) tier present. Primary
  sources outrank aggregators.
- Below `CONFIDENCE_THRESHOLD`, the pipeline abstains. Claim age (post date vs. now) alone separates
  TOO_EARLY from EVIDENCE_MISSING; an unknown post date means EVIDENCE_MISSING.
- Only whitelisted sources count, including fact-check reviews.
- Every summary sentence must be entailed by a cited passage according to the NLI check, or it is
  deleted. If nothing survives, the claim carries a status only.
- The cache only reuses definitive verdicts. It requires cosine similarity ≥ threshold, at least one
  shared entity, and the same embedding model.

API: `POST /api/check` (form: `text`, `image`, `post_date`) returns JSON. `POST /api/check/stream`
streams server-sent events: `claims_extracted` → `cache_hit` | `evidence` → `verdict` → `done`.
`GET /api/checks/{id}` returns a stored check; `GET /api/health` reports mode and model versions.

## Filling sources.yaml

Each entry has `name`, `domain`, `rss_url` and/or `sitemap_url`, `language`, `tier` (1 = primary:
police, courts, government, institutions, wire services; 2 = original reporting and fact-checkers;
3 = aggregator) and `kind` (police | court | government | institution | wire | outlet | factchecker |
aggregator). `kind` drives the expected-evidence checklist. Entries with `todo: true` are ignored.
Use only feed URLs you have confirmed.

## Crawler

```bash
python -m crawler.run --dry-run                 # list what would be fetched
python -m crawler.run                           # all sources (needs DATABASE_URL)
python -m crawler.run --source example.org --limit 20 --since-days 7
```

The crawler only keeps URLs on each source's own domain (also after redirects) and obeys robots.txt.
It extracts article text with trafilatura, splits it into sentence chunks, embeds them, and stores them
with url, publisher, tier and published_at. Already-stored URLs are skipped. Search is hybrid: Postgres
full-text (`simple` + `english`) plus pgvector cosine, queried in both the original language and
English, fused with RRF.

## Evaluation

```bash
python -m eval.run --split tune                               # or hidden | all
python -m eval.run --split tune --threshold-sweep --target 0.05
```

`eval/claims.jsonl` holds 10 **synthetic** examples covering all six statuses (7 tune, 3 hidden),
including one rendered screenshot. `post_date` is ISO 8601 or relative to the run (`-2h`, `-10d`).
Each example is scored on its first claim.

The report gives the confident-wrong rate (wrong and not abstained), abstain rate, citation precision
(the share of written sentences that pass the NLI check), calibration buckets with ECE, and p50/p95
latency. It writes a per-example CSV and a JSON summary to `eval/out/`. The sweep runs once with
threshold 0, then replays every threshold exactly.

Current results:
- **Mock mode:** 10/10 correct.
- **Real mode on the fictional corpus:** 9/10, 0 confident-wrong, citation precision 1.0.
  The miss is e05, which only the mock fact-check fixture can answer.

## Tests

```bash
pytest                                                          # offline: unit + mock end-to-end
TEST_DATABASE_URL=postgresql+asyncpg://fact:fact@localhost:5432/fact_test pytest   # + Postgres
RUN_LIVE_TESTS=1 RUN_MODEL_TESTS=1 pytest tests/test_live.py   # real APIs/models (uses .env keys)
```

The Postgres tests drop and recreate their tables; point them at a throwaway database.

## TODOs left

1. `sources.yaml`: 19 verified sources, but almost no tier-1 primary sources (PIB, state police, DGIPR, PTI, ANI were unreachable or feedless from the build environment). Add them from your network.
2. `CRAWLER_USER_AGENT`: add a real contact address.
3. Paid web-search fallback: only a stub (`app/adapters/web_search.py`). No provider was chosen, and
   `WEB_SEARCH_ENABLED=true` fails at startup.
4. Only Gemini is implemented for `LLM_PROVIDER` / `VISION_PROVIDER`.
5. Real evaluation data: the 10 examples are synthetic and the hidden split has 3 rows. All thresholds
   in `.env.example` are untuned defaults.
6. Database migrations: tables are created with `create_all`; switch to Alembic once the schema settles.
7. `recheck_at` is stored, but nothing re-runs checks yet (needs a scheduler/cron).
8. Summaries are English only; consider writing them in the post's language.
9. Fact-check rating map (`app/pipeline/match.py`) is a small, conservative exact-match table.
10. `TRANSLATOR_PROVIDER=llm` detects language with script heuristics; Sarvam's text-lid returns one
    language per text (mixed-language posts are flagged by a heuristic).
11. Crawler runs on demand only (no schedule) and has no per-host rate limit beyond the concurrency limit.
12. The API has no auth or rate limiting.
13. The Docker image build was not run in the development environment (no Docker daemon there); the
    compose file was validated with `docker compose config`.
