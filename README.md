# Fact

Fact-checking backend for social-media posts in Marathi, Hindi, Hinglish and English. A user submits
a screenshot or text; each claim gets an **evidence status** with citations. Statuses describe the
evidence found, never reality: missing evidence is `UNVERIFIED_*`, never `CONTRADICTED`.

Statuses: `CONFIRMED`, `CONTRADICTED`, `MISLEADING_CONTEXT`, `UNVERIFIED_TOO_EARLY`,
`UNVERIFIED_EVIDENCE_MISSING`, `NOT_CHECKABLE`.

## Setup

Choose **Python** for a quick offline demo, or **Docker** for the app with persistent Postgres
storage. Both start in mock mode and need no API keys. Downloads during installation need internet
access; the installed mock app runs offline.

### 1. Get the source and configuration

Install Git, then run:

```sh
git clone https://github.com/aarushkha/Fact.git
cd Fact
```

Create `.env` once (keep an existing file if you have already configured it):

**macOS / Linux / Bash:**

```sh
cp .env.example .env
```

**Windows PowerShell:**

```powershell
Copy-Item .env.example .env
```

Run all commands below from the `Fact` repository root. Edit `.env` in a text editor; the app
loads it automatically. **Do not run `source .env`**: it is a dotenv file, and some values contain
spaces and parentheses that are not shell syntax. Variables already set in your terminal override
`.env`. Values saved on the [settings page](#settings-page-no-env-editing-no-docker-restarts) override
both; use the field’s revert button and save to return to the environment value. Restart the app after
editing `.env`; for Docker, rerun `docker compose up -d app` to recreate the container.
Changes saved on the settings page apply immediately without a restart.

### 2A. Python: offline demo without a database

Install **Python 3.11 or newer** with pip and venv support. Create and activate a virtual environment
so installation and startup use the same interpreter.

**macOS / Linux / Bash** (use your installed Python 3.11+ executable if it has a different name):

```sh
python3 --version
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell** (with Python 3.11 installed via the Python launcher):

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Then, in either shell:

```sh
python -m pip install -e '.[dev]'
python -m uvicorn app.main:app --reload
```

Keep these defaults in `.env` for this route:

```dotenv
MOCK_MODE=true
DATABASE_URL=
SEARCH_BACKEND=auto
```

Leave the server running and open <http://localhost:8000/>. Stop it with Ctrl+C. Data is held in
memory and lost when the server restarts. In a new terminal, activate `.venv` again before running
Python commands.

### 2B. Docker: app and Postgres

Install Docker Engine or Docker Desktop and start it. Use **Docker Compose 2.24.0 or newer**;
the compose file uses [optional `env_file` support](https://docs.docker.com/compose/how-tos/environment-variables/set-environment-variables/).
Install the Docker Buildx plugin along with Compose (Docker Desktop includes both).
Host Python is not needed for this route.

```sh
docker compose version
docker buildx version
docker compose config --quiet
docker compose up -d --build --wait db app
docker compose logs -f app
```

Open <http://localhost:8000/> once the logs show application startup is complete. Ctrl+C stops
following logs; the containers keep running. The app migrates the database and seeds the fictional
corpus automatically in mock mode. Compose supplies the database URL using the hostname `db`.

```sh
docker compose down
```

This stops the containers and keeps database/model-cache volumes for the next run.

### 2C. Python app with Docker Postgres

Complete the Python environment and dependency installation in **2A**, and install Docker as in
**2B**. Stop any app already using port 8000, then start just the database:

```sh
docker compose up -d --wait db
```

Edit the existing settings in `.env`:

```dotenv
MOCK_MODE=true
DATABASE_URL=postgresql+asyncpg://fact:fact@localhost:5432/fact
SEARCH_BACKEND=auto
```

Start the local app from the activated virtual environment:

```sh
python -m uvicorn app.main:app --reload
```

Use `localhost` for Python running on your machine and `db` for processes inside Compose. The
included database is Postgres 16 **with pgvector**; a plain Postgres installation needs the vector
extension installed separately. The app runs migrations and seeds mock data on startup.

### 3. Check that it works

Open <http://localhost:8000/api/health>; it should return `ok: true` and `mock_mode: true` for
the default setup. On <http://localhost:8000/>, submit:

```text
Mumbai airport is closed for a week.
```

Expect `CONTRADICTED` from the fictional demo corpus. The test page also accepts screenshots and an
optional post date, streams events, and displays claim cards, citations and raw JSON. Interactive
API documentation is at <http://localhost:8000/docs>.

### Installation troubleshooting

| Symptom | Fix |
|---|---|
| Python version error, or `No module named uvicorn` | Confirm Python is 3.11+, activate `.venv`, and rerun `python -m pip install -e '.[dev]'`. |
| `externally-managed-environment` from pip | Create and use the virtual environment in 2A. |
| PowerShell blocks `Activate.ps1` | Activation is optional: use `.\.venv\Scripts\python.exe -m pip install -e '.[dev]'` and `.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload`. |
| Syntax errors around `VAR=value command` or `source` | Those are Bash commands. Edit settings in `.env`, and use the PowerShell setup block on Windows. |
| Compose rejects `env_file` / `required` | Upgrade to Docker Compose 2.24.0+ and use `docker compose` (with a space). |
| Compose says `build requires buildx` | Install or update Docker Buildx to the version requested by your Compose release. |
| Cannot connect to the Docker daemon | Start Docker Engine / Docker Desktop, then retry. |
| Database connection refused | For the Python-only demo, leave `DATABASE_URL` empty. For persistence, start `db` and use the URL from 2C. |
| Port 8000 or 5432 is already allocated | Stop the conflicting local service or container before starting this setup. |
| Real mode says the mock corpus is unavailable | Real mode needs a pgvector database and `SEARCH_BACKEND=auto` (or `postgres`); follow the real-mode steps below. |
| Crawler prints no URLs in mock mode | The mock whitelist has no live feeds. Configure real mode and `SOURCES_FILE=sources.yaml` before crawling. |

## Settings page (no .env editing, no Docker restarts)

Open `/settings` to switch between mock and real mode, enter the API keys, pick providers and models, tune the
thresholds, API protection and worker/crawler options. **Save and apply** takes effect immediately in the API and
on the worker’s next settings poll (normally within ~30 s when idle); already loaded models are kept.
A change that cannot work (real mode without keys, or without the models installed) is refused with
the reason and the running configuration stays as it was.

- Precedence: value saved on the page > `.env` / environment > default. Each field shows where its value comes from,
  with a button to revert it. Changes are stored in the `settings_overrides` table (or `SETTINGS_OVERRIDES_FILE`
  without a database). With neither configured, changes are held in memory and lost on restart. For the
  Python-only demo, set `SETTINGS_OVERRIDES_FILE=.settings-overrides.json` in `.env` before startup to keep
  them across restarts. API keys are stored in plain text, like in `.env`, and are never sent back to the browser.
- Who may open it: `ADMIN_TOKEN` if set (always from the environment wins), else any API key from `API_KEYS`, else
  anyone who can reach the server. **Set an admin token before exposing the server**; you can do it on the page.
- Not editable there (change `.env` and restart): `DATABASE_URL`, `SOURCES_FILE`, provider base URLs, embedder/NLI model
  pins and `EMBEDDING_DIM`, Docker-level choices (ports, `INSTALL_MODELS` build arg, the `models` profile). Real mode
  needs the image built with `INSTALL_MODELS=true` (or a running models service whose URL you enter on the page).
- With multiple API processes or replicas, a save updates the process handling that request. Restart
  the other API processes to load the persisted values. The background worker polls the shared database.
- The separate models service (`app.model_server`), `crawler.run`, `app.db.seed` and evaluation commands
  still read only the environment / `.env`. Keys or mode changes saved on the page do not configure these
  commands; keep their `.env` configuration in sync when following the real-mode steps below.

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

Real mode needs a **pgvector database**, provider credentials, internet access and the optional
model dependencies (about 3 GB of weights downloaded on first use). Start with setup **2B** or
**2C** above, and stop the app while changing modes.

1. Install the models in your activated Python environment. The CPU-only torch command below is
   for Linux/Windows; on macOS, install torch from the default index with `python -m pip install torch`.

   ```sh
   python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
   python -m pip install -e '.[models]'
   ```

   For Docker, add `INSTALL_MODELS=true` to `.env` instead. Compose uses it as a build argument;
   you must rebuild the image after changing it.

2. Edit `.env`: set `MOCK_MODE=false`, `SEARCH_BACKEND=auto` and `SOURCES_FILE=sources.yaml`.
   Fill `GEMINI_API_KEY`, `SARVAM_API_KEY`, `OPENROUTER_API_KEY` (for Jev) and
   `GOOGLE_FACTCHECK_API_KEY`. For a local Python process, set `DATABASE_URL` as in **2C**;
   Compose supplies its own database URL. Keep `WEB_SEARCH_ENABLED=false` (no implementation yet).
   Provider model names and account access must also match your provider configuration.

3. Build the Docker image if using containers:

   ```sh
   docker compose build app
   docker compose up -d --wait db
   ```

4. Review the existing entries in `sources.yaml`, then populate the search index. For local Python:

   ```sh
   python -m crawler.run --source factly.in --limit 20 --since-days 7
   python -m uvicorn app.main:app --reload
   ```

   For Docker:

   ```sh
   docker compose run --rm app python -m crawler.run --source factly.in --limit 20 --since-days 7
   docker compose up -d app
   docker compose logs -f app
   ```

   Run the crawler without `--source`, `--limit` and `--since-days` to crawl all configured sources
   with the default limits. A claim can remain UNVERIFIED when no relevant evidence is found;
   filling the index does not guarantee a definitive status.

To try real models on the **fictional** corpus instead, set
`SOURCES_FILE=app/adapters/mock_data/sources.yaml` while keeping `MOCK_MODE=false` and the database
configured. Run `python -m app.db.seed` locally, or `docker compose run --rm app python -m app.db.seed`,
then restart the app. Restore `SOURCES_FILE=sources.yaml` before crawling real sources. Mock and real
embeddings are model-specific; seeding/crawling must use the mode you intend to run.

| Adapter | Implementation in this repository |
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
Set both to `llm` for LLM-based extraction and writing. Check your provider account for available
models and quotas; the values in `.env.example` are configuration defaults.

**API protection.** Set `API_KEYS` (comma-separated) to require an `X-API-Key` header on
`/api/check*` and `/api/checks/*`; the test page shows a key field when a key is required.
`RATE_LIMIT_PER_MINUTE` limits checks per key (or per IP when auth is off). With `DATABASE_URL` set the
limiter lives in Postgres (`rate_limit_hits`), so all API workers share one budget per client; without a
database it is in-memory and per process. API keys can come from the environment or the settings page;
see the settings-page note above when running multiple API processes.

## Background worker, rechecks, monitoring, model server

- The worker requires `DATABASE_URL`. Start it in another activated terminal with `python -m app.worker`,
  or with `docker compose up -d --build worker`. It re-runs UNVERIFIED claims whose `recheck_at` has
  passed (TOO_EARLY after 6 h, EVIDENCE_MISSING after 7 days, never past `RECHECK_MAX_AGE_DAYS`) and
  crawls all sources every `CRAWL_INTERVAL_MINUTES` in real mode. A new verdict links `rechecked_from`; the old one
  is marked `superseded_at`. Use `--once` for cron.
- Rumour-cascade signals per claim: repeat submissions in 24 h / 7 d, first seen, supporting passages
  per tier, `echo_only` (only aggregators support it) and `distinct_accounts_7d` (distinct poster handles
  read from screenshots of the same rumour; stored normalised in `claims.account_handle`). They are stored
  in `claims.signals` and sent on the `verdict` event; they never change the status.
- `GET /api/monitoring?hours=24` and the `/monitor` page show: checks, statuses and abstain rate,
  per-stage latency and errors, models and fallbacks, NLI deletion rate, rechecks and top cascades.
- Optional model server (requires the model dependencies):
  `python -m uvicorn app.model_server:app --port 8001`, or
  `docker compose --profile models up -d --build models` (published on 127.0.0.1 only). Set
  `MODEL_SERVER_URL=http://localhost:8001` for a local Python app, or
  `MODEL_SERVER_URL=http://models:8001` for the Compose app/worker. Restart the app and worker
  after changing the URL. In real mode they then use the model server for embeddings and NLI.
  Set `MODEL_SERVER_TOKEN` before exposing it further;
  with a token, a public `MODEL_SERVER_URL` must be `https://` (plain HTTP is allowed only to internal hosts).
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
  (`NLI_RESTATEMENT_THRESHOLD`) or is labelled as the claim ("Claim: …") is deleted too: it restates
  the claim, like the quote a debunk opens with. If nothing survives, the claim carries a status only.
- Summaries default to English. A quote from Marathi or Hindi evidence is translated, and NLI
  checks the English sentence against the original passage (the NLI model can't compare Marathi with
  Marathi reliably, but handles Marathi evidence → English sentence well).
  Fact-check passages are stored as one verdict-first sentence (`Fact-check verdict False on the claim
  "…"`); run `python -m crawler.run --reindex` to rewrite passages crawled before this change.
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

These commands assume the local Python environment and real-mode configuration above. For Docker,
replace `python -m crawler.run` with `docker compose run --rm app python -m crawler.run`.
`--dry-run` still fetches feeds/sitemaps over the network, but does not store articles or require a
database. A normal crawl requires `DATABASE_URL` and the configured embedder.

```sh
python -m crawler.run --dry-run                 # fetch feeds/sitemaps and list discovered URLs
python -m crawler.run                           # all sources (needs DATABASE_URL)
python -m crawler.run --source factly.in --limit 20 --since-days 7
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

Run these from the repository root in the local Python environment. For the synthetic mock
baseline, use `MOCK_MODE=true`, an empty `DATABASE_URL` and `SEARCH_BACKEND=auto` in `.env`.
The Docker image does not include the `eval/` directory.

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

**Real fact-check set**: `eval/factchecks.jsonl` holds 300 real claims (en 120, hi 105, mr 75),
labelled by published fact-checks (Alt News, Factly, Vishvas, BOOM, The Quint, Aaj Tak, Fact
Crescendo, Lokmat) via the Fact Check API. Grow it with `python -m eval.build_factcheck_set --append --n 100`
(keeps every existing row; without `--append` the file is rebuilt from scratch).
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

Run from the repository root with the virtual environment and `.[dev]` dependencies from **2A**.
The standard suite uses mocks; Postgres, live-provider and model tests are opt-in:

```sh
python -m pytest
```

For Postgres tests, create a dedicated disposable database first:

```sh
docker compose up -d --wait db
docker compose exec db createdb -U fact fact_test
```

If `fact_test` already exists, reuse it only if it is disposable. The Postgres tests **drop and
recreate tables**; never point them at a database whose data you want to keep.

**Bash:**

```bash
TEST_DATABASE_URL=postgresql+asyncpg://fact:fact@localhost:5432/fact_test python -m pytest
```

**PowerShell:**

```powershell
$env:TEST_DATABASE_URL = "postgresql+asyncpg://fact:fact@localhost:5432/fact_test"
python -m pytest
Remove-Item Env:TEST_DATABASE_URL
```

Optional live API/model tests require the real-mode dependencies and provider keys in `.env` and
may incur provider charges. These flags are read from the **process environment**, not `.env`:

**Bash:**

```bash
RUN_LIVE_TESTS=1 RUN_MODEL_TESTS=1 python -m pytest tests/test_live.py
```

**PowerShell:**

```powershell
$env:RUN_LIVE_TESTS = "1"
$env:RUN_MODEL_TESTS = "1"
python -m pytest tests/test_live.py
Remove-Item Env:RUN_LIVE_TESTS, Env:RUN_MODEL_TESTS
```

## TODOs left

1. `sources.yaml`: 19 verified sources, but almost no tier-1 primary sources (PIB, state police, DGIPR, PTI, ANI were unreachable or feedless from the build environment). Add them from your network.
2. `CRAWLER_USER_AGENT`: add a real contact address.
3. Paid web-search fallback: only a stub (`app/adapters/web_search.py`). No provider was chosen, and
   `WEB_SEARCH_ENABLED=true` fails at startup.
4. Only Gemini is implemented for `LLM_PROVIDER` / `VISION_PROVIDER`.
5. Evaluation (data is the main gap; calibrate Jev once there is enough labelled data): the real fact-check set is mostly false claims (CONFIRMED is rare in fact-checks) and
   has no TOO_EARLY / NOT_CHECKABLE rows; those come only from the 10 synthetic examples. All thresholds
   in `.env.example` are untuned defaults.
6. API keys live in `API_KEYS` (no per-key quotas, revocation means a restart). The rate limiter is shared through Postgres.
7. Summaries are English by default. `SUMMARY_LANGUAGE=post` translates each verified English sentence into the
   post's language (Sarvam) and shows it only if it passes NLI against its own passage again; otherwise the
   English sentence stays. Measured on 20 real English evidence sentences: Hindi translations pass 13/20, Marathi
   8/20 (English 15/20), so the default stays `english`. Hinglish posts get Devanagari Hindi.
8. Fact-check rating map (`app/pipeline/match.py`) is a small, conservative exact-match table.
9. `TRANSLATOR_PROVIDER=llm` detects language with script heuristics; Sarvam's text-lid returns one
    language per text (mixed-language posts are flagged by a heuristic).
10. Crawler politeness: at most one request per host every `CRAWLER_MIN_HOST_INTERVAL_SECONDS` (raised by a robots.txt `Crawl-delay`, capped at 60 s), within one crawler process.
11. The Docker image is built in CI: the `image` job starts app + pgvector with compose in mock mode and
    runs one check. Model weights (`INSTALL_MODELS=true`) are not built in CI.
