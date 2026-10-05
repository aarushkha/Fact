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

**Keeping Gemini use low.** By default (`CLAIM_EXTRACTOR=sentences`, `SUMMARY_WRITER=extractive`)
a text check makes **no Gemini calls**:
- claims are split by sentence, with English from Sarvam;
- summaries quote the most relevant evidence sentence verbatim, and NLI still verifies each one.

Gemini is then used only to read screenshots and as a fallback (if Jev or Sarvam fails).
Set both to `llm` for LLM-quality extraction and writing. The Gemini free tier allows
`gemini-3.8-flash` 5 requests/min and 20/day.

**API protection.** Set `API_KEYS` (comma-separated) to require an `X-API-Key` header on
`/api/check*` and `/api/checks/*`; the test page shows a key field when a key is required.
`RATE_LIMIT_PER_MINUTE` limits checks per key (or per IP when auth is off). The limiter is in-memory
and per process.

## Background worker, rechecks, monitoring, model server

- `python -m app.worker` (compose service `worker`) re-runs UNVERIFIED claims whose `recheck_at` has
  passed (TOO_EARLY after 6 h, EVIDENCE_MISSING after 7 days, never past `RECHECK_MAX_AGE_DAYS`) and
  crawls all sources every `CRAWL_INTERVAL_MINUTES`. A new verdict links `rechecked_from`; the old one
  is marked `superseded_at`. Use `--once` for cron.
- Rumour-cascade signals per claim: repeat submissions in 24 h / 7 d, first seen, supporting passages
  per tier and `echo_only` (only aggregators support it). They are stored in `claims.signals` and sent
  on the `verdict` event; they never change the status.
- `GET /api/monitoring?hours=24` and the `/monitor` page show: checks, statuses and abstain rate,
  per-stage latency and errors, models and fallbacks, NLI deletion rate, rechecks and top cascades.
- Optional model server: `uvicorn app.model_server:app --port 8001` (or
  `docker compose --profile models up`, which publishes it on 127.0.0.1 only) with `MODEL_SERVER_URL`
  set. The API and worker then don't load torch. Set `MODEL_SERVER_TOKEN` before exposing it further.
- Migrations: Alembic (`app/db/migrations`). The app upgrades to head on startup. After changing
  `app/db/tables.py`, run `alembic revision --autogenerate -m "..."`. CI fails on drift (`alembic check`).

## How a check works

`ingest` → `normalize` → `extract` → (cache + fact-check match) → `retrieve` → `judge` → `write` →
`verify` → `store`. Every stage logs inputs, outputs, latency and model version to `stage_runs`.

These rules are enforced in code (`app/pipeline/judge.py`, `write.py`), not left to the models:
- A definitive status needs a matching passage from the best (lowest-numbered) tier present. Primary
  sources outrank aggregators.
- Below `CONFIDENCE_THRESHOLD`, the pipeline abstains. Claim age (post date vs. now) alone separates
  TOO_EARLY from EVIDENCE_MISSING; an unknown post date means EVIDENCE_MISSING.
- Only whitelisted sources count, including fact-check reviews.
- A passage only counts if the judge says it is about the **same incident** as the claim
  (`SAME_EVENT_THRESHOLD`). Without this, debunks of *other* viral videos (similar wording) were the
  evidence behind most confident errors on real data.
- Every summary sentence must be entailed by a cited passage according to the NLI check, or it is
  deleted. Under CONTRADICTED or MISLEADING_CONTEXT, a sentence that itself entails the claim
  (`NLI_RESTATEMENT_THRESHOLD`) is deleted too: it restates the claim, like the quote a debunk opens
  with. If nothing survives, the claim carries a status only.
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
Use only feed URLs you have confirmed. When one source has feeds in several languages, give each
feed that differs from the entry's `language` as `{url: ..., language: ...}`, so its articles are
stored with the right language.

## Crawler

```bash
python -m crawler.run --dry-run                 # list what would be fetched
python -m crawler.run                           # all sources (needs DATABASE_URL)
python -m crawler.run --source example.org --limit 20 --since-days 7
python -m crawler.run --reindex                 # also re-extract and re-embed already-stored articles
```

The crawler only keeps URLs on each source's own domain and obeys robots.txt. It follows redirects
itself and never off the source's domain, ignores child sitemaps on other hosts, and refuses hosts that
resolve to private, loopback or link-local addresses. That check is not pinned to the connection (DNS
rebinding), so in production also restrict the worker's egress to the public internet.
It extracts article text with trafilatura, splits it into sentence chunks, embeds them, and stores them
with url, publisher, tier and published_at. Already-stored URLs are skipped (compared without scheme,
`www.`, query or trailing slash, since feeds may link a URL that redirects) unless `--reindex` is
given (use it after changing chunking, embedding or ClaimReview handling). A feed that fails is logged
and skipped; the source only fails when all of its feeds and sitemaps do. Search is hybrid: Postgres
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
latency. It writes a per-example CSV and a JSON summary to `eval/out/`. Examples whose run failed are
reported as `errors` and left out of every rate. The sweep runs once with threshold 0, then replays
every threshold exactly; an answer the replay turns into an abstention no longer counts as correct.
Date-only `post_date` values are read in `TIMEZONE`, as in the app.

**Real fact-check set**: `eval/factchecks.jsonl` holds 200 real claims (en 80, hi 70, mr 50),
labelled by published fact-checks (Alt News, Factly, Vishvas, BOOM, The Quint, Aaj Tak, Fact
Crescendo, Lokmat) via the Fact Check API. Rebuild it with `python -m eval.build_factcheck_set`.
- Each row excludes its own labelling review from the evidence, so the answer can't simply be looked up.
- Without other evidence the right behaviour is to abstain, so accuracy is low by design. The number
  to watch is the **confident-wrong rate**.
- Run it with `python -m eval.run --file eval/factchecks.jsonl --split all`.

On the real set, confident errors came almost entirely from two causes. Unrelated fact-check articles
were used as evidence; this is now fixed by the same-event gate, and 19 of 23 such rows now abstain.
The rest come from fact-checkers disagreeing with each other on CONTRADICTED vs MISLEADING (label
noise). Re-run the full set after a longer crawl.

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
5. Evaluation (data is the main gap; calibrate Jev once there is enough labelled data): the real fact-check set is mostly false claims (CONFIRMED is rare in fact-checks) and
   has no TOO_EARLY / NOT_CHECKABLE rows; those come only from the 10 synthetic examples. All thresholds
   in `.env.example` are untuned defaults.
6. Rate limiter and API keys are in-memory/env based; move to a shared store when running several workers.
7. LLM-written summaries are English only (extractive quotes keep the source language); consider writing them in the post's language.
8. Fact-check rating map (`app/pipeline/match.py`) is a small, conservative exact-match table.
9. `TRANSLATOR_PROVIDER=llm` detects language with script heuristics; Sarvam's text-lid returns one
    language per text (mixed-language posts are flagged by a heuristic).
10. Crawler has no per-host rate limit beyond the concurrency limit.
11. The Docker image build was not run in the development environment (no Docker daemon there); the
    compose file was validated with `docker compose config`.
