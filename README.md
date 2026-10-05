# Fact

Evidence-status fact-checking backend for Marathi, Hindi, Hinglish and English posts.
Statuses describe the evidence found, never reality.

> Work in progress. Done so far: step 1 (schemas, mock adapters, pipeline, test page) and step 2
> (Postgres + pgvector, crawler, hybrid search). The full README (real adapters, evals, TODO list)
> lands at the end of the build.

## Run it on your machine

You need [git](https://git-scm.com/) and either Docker Desktop **or** Python 3.11+.
Nothing below needs an API key: `MOCK_MODE=true` (the default) runs every model as a deterministic mock.

```bash
git clone https://github.com/aarushkha/Fact.git
cd Fact
git checkout ccr-c6fc4224-mqd25r     # until this branch is merged
cp .env.example .env
```

### Option A: Docker (app + Postgres in one command)

```bash
docker compose up --build
```

Open http://localhost:8000/. The app creates its tables on startup and, in mock mode, loads the
small fictional mock corpus into Postgres so hybrid search has something to find.

Stop with `Ctrl+C`; `docker compose down -v` also deletes the database.

### Option B: Python only (no database)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
uvicorn app.main:app --reload
```

Open http://localhost:8000/. With `DATABASE_URL` empty the app keeps everything in memory and searches
the built-in mock corpus.

### Option C: Python app + Postgres from Docker

```bash
docker compose up -d db
# in .env:  DATABASE_URL=postgresql+asyncpg://fact:fact@localhost:5432/fact
uvicorn app.main:app --reload
```

### Things to try on the test page

| Paste | Expected status |
|---|---|
| `A footbridge over the Godavari river in Nashik collapsed on Sunday.` | CONFIRMED |
| `Mumbai airport ek hafte ke liye band hai.` (Hinglish) | CONTRADICTED |
| `This video shows floods in Kolhapur this week.` | MISLEADING_CONTEXT |
| `A fire broke out at a chemical factory in Thane.` + post date = 1 hour ago | UNVERIFIED_TOO_EARLY |
| same text, no post date | UNVERIFIED_EVIDENCE_MISSING |
| `Nashik is the most beautiful city in India.` | NOT_CHECKABLE |
| `The state government is giving free laptops to all college students.` | CONTRADICTED (via fact-check match) |

Paste the same claim twice to see a `cache_hit`. All mock publishers and events are fictional.

## API

| Method | Path | |
|---|---|---|
| GET | `/` | test page |
| GET | `/api/health` | mode + model versions |
| POST | `/api/check` | form fields `text`, `image` (file), `post_date` (ISO 8601) → final JSON |
| POST | `/api/check/stream` | same form → server-sent events: `claims_extracted` → `cache_hit` \| `evidence` → `verdict` (per claim) → `done` |
| GET | `/api/checks/{check_id}` | a stored result |

## Evidence index

Sources: fill in `sources.yaml` (the three `todo: true` entries are placeholders and are ignored).
Only add real feed URLs (`rss_url` and/or `sitemap_url`).

Crawl (needs `DATABASE_URL`):

```bash
python -m crawler.run --dry-run            # list what would be fetched
python -m crawler.run                      # all sources
python -m crawler.run --source example.org --limit 20 --since-days 7
```

The crawler keeps only URLs on each source's own domain (also after redirects), respects robots.txt,
extracts text with trafilatura, chunks it, embeds it and stores it with url, publisher, tier and
published_at. Re-running skips URLs already stored.

Search is hybrid: Postgres full-text search (`simple` config for every language plus `english`
stemming) and pgvector cosine similarity, queried in both the original language and English, fused
with reciprocal rank fusion. Passages are only compared with vectors from the same embedding model.

## Tests

```bash
pytest                                   # unit + mock end-to-end (no database needed)
TEST_DATABASE_URL=postgresql+asyncpg://fact:fact@localhost:5432/fact_test pytest   # + Postgres tests
```

The Postgres tests drop and recreate their tables, so point `TEST_DATABASE_URL` at a throwaway database
(`createdb -h localhost -U fact fact_test` with the compose database running).
