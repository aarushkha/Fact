# CLAUDE.md: working notes for this repo

Fact-checking backend for social-media posts (screenshot or text) in **Marathi, Hindi, Hinglish and
English**. Each claim gets an **evidence status** with citations. README.md is the user-facing guide;
this file is for whoever (human or Claude) works on the code next.

## Non-negotiable rules (from the original brief; never weaken these)

1. Statuses describe **evidence, never reality**. Missing evidence → `UNVERIFIED_*`, never `CONTRADICTED`.
2. Every sentence shown must cite a retrieved passage and **pass NLI** against it, or it is deleted.
   If nothing survives, return the status only.
3. Below the confidence threshold the system **abstains** (`UNVERIFIED_*`).
4. **Only whitelisted sources** (`sources.yaml`) count. Primary sources (tier 1) outrank aggregators.
5. Every external model sits behind an **adapter with a mock**; `MOCK_MODE=true` runs everything with no keys.
6. **Never guess an API format.** Read the official docs first, then confirm with one live call. If the docs
   can't be read, write a stub with a TODO and say so. **Never invent feed URLs**; add only URLs that were
   fetched and parsed.

The six statuses are exactly: `CONFIRMED, CONTRADICTED, MISLEADING_CONTEXT, UNVERIFIED_TOO_EARLY,
UNVERIFIED_EVIDENCE_MISSING, NOT_CHECKABLE`. The public JSON contract (`CheckResponse` in
`app/models/schemas.py`) is fixed. Extra data goes on SSE events or in the database, never into the contract.

## Commands

```bash
pip install -e '.[dev]'                       # + '.[models]' for BGE-M3 / mDeBERTa (CPU torch: install from
                                              #   https://download.pytorch.org/whl/cpu first)
uvicorn app.main:app --reload                 # test page /, monitor /monitor
pytest                                        # offline: unit + mock end to end (~1 s)
TEST_DATABASE_URL=postgresql+asyncpg://...  pytest     # + Postgres tests (they DROP tables: throwaway DB)
RUN_LIVE_TESTS=1 RUN_MODEL_TESTS=1 pytest tests/test_live.py   # real APIs / models
python -m eval.run --split all                            # synthetic set (10 rows, all six statuses)
python -m eval.run --file eval/factchecks.jsonl --split hidden [--ids a,b] [--trace] [--threshold-sweep --target 0.05]
python -m eval.build_factcheck_set            # rebuild the 200 real labelled rows (Fact Check API key)
python -m crawler.run [--source d] [--limit n] [--dry-run] [--reindex]   # --reindex: re-embed stored articles
python -m app.worker [--once] [--no-crawl]    # rechecks + scheduled crawl
python -m app.db.migrate                      # upgrade DB to head (app does this at startup)
alembic revision --autogenerate -m "..."      # after editing app/db/tables.py; CI runs `alembic check`
uvicorn app.model_server:app --port 8001      # optional embed/NLI service (MODEL_SERVER_URL)
```

CI (`.github/workflows/ci.yml`) runs tests with pgvector Postgres, `alembic check`, the mock eval, and
`docker compose config`.

## Architecture (where things live)

Pipeline (`app/pipeline/orchestrator.py` chains the stages; every stage is logged through
`StageContext.run` → `stage_runs`):

`ingest` (vision for screenshots, post-date parsing) → `normalize` (language ID + English translation) →
`extract` (claims + type; `single_claim` skips it) → `embed` → `cache` + `factcheck` (Google API) →
`retrieve` (hybrid search, whitelist, tier → relevance → recency) → `judge` → `write` → `verify` (NLI) →
`cascade` signals → `store`. SSE order: `claims_extracted → cache_hit | evidence → verdict → done`.

- `app/pipeline/judge.py`: **the rules live here, in code**. `apply_same_event_gate`,
  `effective_judgments` (best tier only), `decide_status` (evidence guards, threshold, age split).
- `app/pipeline/write.py`: English summaries (`to_english`), NLI verification over premise windows (`best_windows`). Under a "fails"
  status it also drops sentences that entail the claim (`NLI_RESTATEMENT_THRESHOLD`): restatements.
- `app/adapters/`: one file per provider. `factory.py` wires them from `Settings`. `mock.py` and
  `mock_data/` hold deterministic, keyword-driven mocks over a fictional corpus (`*.mock.example`).
- `app/adapters/prompts.py`: all LLM prompts, JSON schemas, and label definitions shared with Jev.
- `app/evidence_catalog.py`: classifiers choose which expected-evidence items apply; whether each is
  *found* is computed in code (a supporting passage from a source of matching `kind`).
- `app/db/`: `tables.py` (SQLAlchemy Core), `migrations/` (Alembic), `store.py` (Store protocol +
  `InMemoryStore`), `pg_store.py`, `pg_search.py` (FTS `simple`+`english` ⊕ pgvector, RRF), `index.py`, `seed.py`.
- `crawler/`: feeds/sitemaps → robots.txt → same-domain check (also after redirects) → trafilatura →
  sentence chunks → embed `title + chunk` → upsert; also ClaimReview JSON-LD → extra passage.
- `app/worker.py` + `app/pipeline/recheck.py`: rechecks UNVERIFIED claims, scheduled crawl.
- `app/monitoring.py` (+ `web/monitor.html`), `app/api/security.py` (API keys, sliding-window rate limit),
  `app/model_server.py` (+ `adapters/remote_models.py`).
- `eval/`: `claims.jsonl` (10 synthetic rows), `factchecks.jsonl` (200 real rows), `run.py`.

## Providers (all verified live on 2026-10-05)

| Role | Provider | Notes |
|---|---|---|
| LLM + vision | Gemini **Interactions API** (`/v1beta/interactions`, `x-goog-api-key`) | Response text is in `steps[type=model_output].content[].text`. JSON via `response_format{type:text, mime_type, schema}`. Free tier: `gemini-3.8-flash` allows **5 req/min, 20 req/day** and is often overloaded (503). The client cools models down and walks `LLM_FALLBACK_MODELS`. |
| Translator | Sarvam `/text-lid` + `/translate` `mayura:v1` | Hinglish = `hi-IN` + script `Latn` → our code `hi-Latn`. `sarvam-translate:v1` rejects `source=auto`. 1000-char limit, so split by sentence. |
| Classifier | Jev `typesafe/jev-1.13` via OpenRouter `POST /api/alpha/decisions` | Typed answers only (`choice` / `noul` / `score`). Several questions can go in one request. Probabilities are often exactly 0/1 (uncalibrated). Falls back to a Gemini JSON judge on error. |
| Embedder | BAAI/bge-m3 (pinned commit) | Weights ship only as `pytorch_model.bin`. ST config normalizes. Unrelated short texts still score ~0.4 cosine. |
| NLI | mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (pinned) | `0=entailment,1=neutral,2=contradiction`, 512 tokens. Trained on hi and mr. |
| Fact checks | Google Fact Check Tools `claims:search` | Key in the header, not the URL. Returns `{}` when there are no hits. Marathi ratings: `चूक`, `दिशाभूल`. |

Defaults keep Gemini use near zero for text checks: `CLAIM_EXTRACTOR=sentences`,
`SUMMARY_WRITER=extractive` (verbatim evidence quotes). Gemini is used for screenshots and fallbacks.

## What we learned (read before changing judging or retrieval)

1. **Wrong evidence was the biggest real-world failure, not wrong labels.** On real fact-checked claims,
   16 of 20 confident CONTRADICTED↔MISLEADING errors cited *another* fact-check. Debunks share vocabulary
   ("viral video… old… falsely shared"), so search and Jev treated them as relevant. Fix: the
   **same-event gate** (Jev `noul`, `SAME_EVENT_THRESHOLD=0.6`, asked in the same call as the stance).
   After it, 19 of 23 problem rows abstain correctly. The "similar event elsewhere" bug (Pune vs Nashik
   bridge) was the same failure in miniature.
2. **Chunks need their article title.** A chunk like "Hence, the claim made in the post is FALSE" matched
   everything (gate score 0.8). With the title in the judge state, 18 of 19 wrong pairs scored ≤ 0.5.
   The crawler now embeds `title + chunk`.
3. **Label wording drives Jev.** Splitting CONTRADICTED ("the event didn't happen") from MISLEADING ("real
   media, wrong time, place or context") took a case from 0.63/0.37 to 0.97. Change criteria in
   `prompts.py` only with an A/B run on real pairs.
4. **NLI under-scores long premises.** A near-verbatim sentence got 0.44 against a 2-sentence passage
   and 0.85 against the matching sentence. Hence premise windows. Scoring *all* windows made rows take
   30–140 s on CPU, so it's capped at the whole passage plus the 3 most similar windows.
5. **The claim-level judge must only see relevant passages.** Irrelevant ones add noise.
6. **Abstention confidence** = max(unverified probability, 1 − top definitive probability). Raw unverified
   mass gave a meaningless "0.00".
7. **The cache must only serve definitive verdicts.** A TOO_EARLY verdict was otherwise served to the
   same claim posted 10 days later. The cache also requires the same embedding model, skips superseded
   verdicts, and is bypassed when the eval excludes evidence.
8. **Eval leakage.** Each real eval row excludes its own labelling review (`exclude_urls`, compared via
   `document_key`). The remaining errors are mostly fact-checkers disagreeing with each other on
   CONTRADICTED vs MISLEADING (label noise). The real set has no TOO_EARLY or NOT_CHECKABLE rows, and
   CONFIRMED is rare.
9. **Watch direction-wrong, not just exact-label accuracy.** Saying "holds" for a false claim (or the
   reverse) is the dangerous error. It was 0 on the rows checked.
10. **Unicode and data quirks:** `\w` splits Devanagari at vowel signs, so the token regex includes the
    block but excludes the danda. JSONB columns need `none_as_null=True`. Feed URLs carry
    `#fragments` that create duplicates, and may differ from the stored (post-redirect) URL by a trailing
    slash (Newschecker), so known URLs are compared by `document_key` (url_key keeps no query, which
    made every RBI `...aspx?prid=N` release look like one page). Postgres FTS handles Devanagari with a UTF-8 locale.
11. **Sources.** Most Indian government and wire feeds (PIB, PTI, ANI, DGIPR, DD) were 403 or timed out
    from the build environment, so there's almost no tier-1 coverage yet. NDTV and RBI block article
    fetches. Alt News and Factly don't embed ClaimReview (Vishvas does); their verdicts come through
    the Google API.
12. **A summary can cite its passage and still be wrong.** A debunk quotes the claim; its passage entails
    that sentence, so the citation check kept the false claim as the summary of a CONTRADICTED verdict.
    Measured on 59 real fact-check articles (real BGE-M3 + mDeBERTa, extractive writer): 20 rows (34%)
    would have shown the false claim. Two guards in `write.py` under CONTRADICTED / MISLEADING_CONTEXT:
    NLI sentence ⇒ claim ≥ `NLI_RESTATEMENT_THRESHOLD` (0.8: 21 drops, all true restatements; 0.5–0.8
    also held real context, so don't lower it blindly), and a claim-label rule ("Claim:", "Claim
    Review :", "दावा:"; 30 more, all true restatements). 25 of those 30 came from our own ClaimReview
    passage, which split as `Claim reviewed: "X". Rating: False.`; it is now one verdict-first sentence
    (`verdict_sentence`). The mock NLI is word overlap plus a negation/"is false" check.
15. **mDeBERTa can't do Marathi→Marathi NLI; summaries are English.** A Marathi sentence against itself
    scored 0.27-0.36 entailment (Hindi 0.59, English 0.94), so every verbatim Marathi quote failed the
    citation check (6 of 10 Marathi rows had empty summaries). A Marathi passage → English sentence
    scored 0.68-0.85. So `write.to_english` translates a non-English quote (Sarvam) and NLI checks the
    English sentence against the *original* passage (windows picked by `source_sentence`). Summaries
    are always shown in English; a quote whose translation fails is dropped, never shown in Marathi.
13. **Feeds carry their own language.** Newschecker, Vishvas and Fact Crescendo serve several languages
    from one domain; a feed whose language differs from its entry is written `{url, language}` in
    `sources.yaml` (checked against the feed's articles, not just its `<language>` tag: Fact
    Crescendo's Marathi feed says `en-US`).
14. **Build environment.** No Docker daemon (compose validated with `config`, image never built). Local
    Postgres 16 + pgvector on port 5433 can stop when the container recycles. Background jobs are capped
    at 1 h (the 200-row eval doesn't fit; use `--ids` / `--limit`). The CPU is slow: BGE crawl ≈ a few
    articles per minute.

## Conventions

- Keep stages pure-ish async functions. Rules belong in `judge.py` and `write.py`, not in prompts.
- New external model → protocol in `adapters/base.py`, mock, real implementation written from the docs,
  offline test with `httpx.MockTransport` that replays a *recorded* response shape, plus an opt-in live test.
- Every new threshold → `Settings` + `.env.example`, marked untuned until an eval says otherwise.
- Schema change → edit `tables.py` → `alembic revision --autogenerate` → name constraints explicitly.
- Never commit keys. `.env` is gitignored. Before committing, grep the tree for anything key-like.
- Commits: small, one feature each, with tests. The real eval runs before and after any judging change.

## What I'd do next (in order)

1. **Build a bigger index, then re-run the full 200-row real eval** with the same-event gate (it has
   only been re-run on the 23 problem rows). Track confident-wrong, direction-wrong, abstain and
   citation precision per language. Use a GPU or the model server for embedding; the CPU crawl is too slow.
2. **Add tier-1 sources from a network that can reach them** (PIB, state police, DGIPR, PTI/ANI, courts),
   then measure how many abstentions turn into correct answers.
3. **Grow the labelled data**: run `build_factcheck_set` regularly (it dedupes), add CONFIRMED rows from
   real news (fact-checks are ~95% false claims), and add TOO_EARLY and NOT_CHECKABLE rows. Once there
   are a few hundred per status, calibrate Jev per question (isotonic or temperature) and tune
   thresholds with `--threshold-sweep`.
4. **Handle fact-checker disagreement explicitly.** When reviews of the same claim disagree on
   CONTRADICTED vs MISLEADING, return the more conservative status (or abstain) and show both reviews.
   Consider a "false or misleading" display bucket for the UI.
5. **Media claims**: perceptual hashes or reverse image search for "old video shared as new", the most
   common real pattern.
6. **Account-level cascade signals** (same rumour from many handles) once screenshots are stored with handles.
7. Production hardening: shared rate limiter and keys (Redis or DB) for several workers, per-host crawl
   rate limits, a real Docker build in CI.
