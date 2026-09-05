# Developer guide (for Claude Code and humans)

YouTube auto-summarizer: fetches each channel's new videos, gets their transcripts,
summarizes them with an LLM, and posts the summaries to Telegram. Runs daily via
GitHub Actions (`.github/workflows/daily-summary.yml`); tests run on every PR
(`.github/workflows/ci.yml`).

## Module map
- `scraper.py` — entry point / orchestration; RSS feed (primary) + YouTube Data API
  (fallback); candidate selection against the dedup watermark; per-video outcomes
  (send / defer / give up); digest mode; `--video-url` on-demand path.
- `transcript.py` — transcript fetch, cheapest source first: Supadata → Gemini
  from the YouTube URL → youtube-transcript-api. Meters Supadata free-tier
  credits in `data/supadata_usage.json`, rotates across multiple keys, and paces
  a monthly budget over the days left in the month. Gemini transcription rotates
  across `GEMINI_TRANSCRIPT_MODELS` (free quota is 20 requests/day *per model*);
  when every source is spent, videos defer silently via `budget_exhausted`
  rather than being written off.
- `summarizer.py` — provider-agnostic LLM summarization (OpenAI-compatible API);
  also exposes `complete()` for generic calls over the same provider chain.
  Gemini runs `GEMINI_MODEL` then `GEMINI_FALLBACK_MODELS` — each model is a
  separate daily quota, and the list must stay disjoint from
  `GEMINI_TRANSCRIPT_MODELS` so video calls can't spend the summary budget.
  **The complete transcript is always sent.** There is no character cap and no
  head/tail cut any more (the old `LLM_MAX_TRANSCRIPT_CHARS` dropped the middle
  of long videos); every request is measured in tokens against the selected
  model's own input limit (`token_budget`), and one that does not fit is
  summarized in complete-coverage chunks and merged (`_summarize_chunked`,
  resumable via `data/research/partials/`, quota-checked before it starts).
  `SUMMARY_MAX_OUTPUT_TOKENS` caps what the model writes, never what it reads.
- `model_capabilities.py` — per-model input/output token limits: live Gemini
  `models.get` (cached in `data/model_capabilities.json`), `MODEL_CAPABILITIES_JSON`
  override, then the dated registry, then a small default for unknown models.
  Each fallback model is judged on its own window.
- `token_budget.py` — sizes the exact request (system + user + title +
  transcript + envelope) with Gemini `countTokens` when available, else a
  conservative estimate (3 chars/token), and computes the available input as
  `min(input_limit, context - output, TPM - output) - CONTEXT_SAFETY_MARGIN_TOKENS`.
- `transcript_normalize.py` — deterministic normalization (formatting only:
  line endings, whitespace, caption-line joins, auto-caption overlap), stable
  segments with offsets/seconds/speaker/category, and complete-coverage
  overlapping chunks with a coverage validator. Raw text and an offset map
  are kept so evidence traces back to the source.
- `transcript_store.py` — raw transcripts persisted gzip'd under
  `data/transcripts/` before any cleaning, indexed in
  `data/research/transcript_records.jsonl`; hash-idempotent.
- `claims.py` — the canonical research unit: atomic, evidence-backed claims
  (schema v2), the extraction prompt with few-shots, deterministic validation
  (evidence located in the transcript, numbers present in evidence, spoken
  tickers only, curated entity resolution plus **local coreference** for
  "the stock"/"it" with `entity_resolution_method`/`_confidence`, documented
  horizon rules, attribution rules for questions / third-party views with
  `host_position` / retrospectives / praise, **portfolio disclosures are never
  views** (`carries_view` is the one stance filter), three-valued
  `testability_type` with observable-condition detection, cue-level evidence
  timestamps, the deterministic **language-aware suspicious-empty check**
  (English and German rule sets; any other language → `needs_review` /
  `empty_extraction_language_guard_unavailable` unless provably asset-free),
  **non-view normalization** (a question / disclosure / retrospective /
  reported view / hypothetical / fact keeps no forecast slots: values move
  to `reported_*`, `hypothetical_*` or `displaced_fields`)), and
  `claims_to_legacy_signals`, the documented reduction that keeps the
  `signals.jsonl` compatibility view working.
- `language_detect.py` — deterministic stop-word language detection (en / de /
  unknown) for the suspicious-empty guard; stored as `transcript_language`.
- `research_budget.py` — keeps research from spending summary requests:
  `scraper.main` queues every separate claim extraction until all eligible
  videos are delivered, then runs each only when
  `remaining − estimated ≥ SUMMARY_REQUEST_RESERVE`; the retry job applies
  the same check (`quota_deferred` / `summary_reserve_protected` otherwise).
- `instruments.py` — canonical instrument metadata (id, ticker, symbol,
  exchange/MIC, asset type, country, currency, GICS sector, benchmark rule)
  from a curated registry or the provider-verified ticker map. The scorecard
  resolves exchange and benchmark from here only; a claim's model-written
  sector is recorded as `speaker_sector` and never chooses either; an
  unresolved instrument is excluded, never guessed.
- `canonical_claims.py` — **the one loader every production analytics job
  reads**: active runs only, no legacy rows, no repeats, condition outcomes
  overlaid; `view_claims`, `aggregate_views` (per asset AND horizon bucket,
  one current view per source), `video_tone`, `portfolio_disclosures`.
  `data/signals.jsonl` is a backward-compatible view read by nothing here.
- `scorecard_pricing.py` — scorecard methodology: exchanges/timezones/
  calendars (NYSE holiday rules), the publication-time **next-close entry
  rule** (never a close that printed before the video; crypto = UTC-day
  close), `PriceSeries` provenance (provider, adjustment — split-adjusted at
  minimum, `PRICE_ADJUSTMENT=all` for total return — corporate-action status,
  currency, requested/resolved dates, daily highs/lows when the provider
  returned bars), per-claim benchmark resolution with an honest null,
  **calendar confidence** (only NYSE closures are modelled; weekday-only
  exchanges are scored but excluded from rankings unless
  `SCORECARD_RANK_WEEKDAY_CALENDARS=true`, and no error bound is claimed),
  **price-target methods** (`evaluate_target`: intraday_touch with bars,
  daily_close otherwise, horizon_close beside them; intraday reach is null,
  not false, without bars), and `SCORECARD_RANKINGS` (off by default, and
  it stays off until the live extraction benchmark has been run and
  reviewed).
- `partial_cache.py` — versioned per-chunk partials for both chunked paths:
  a record is reused only when task type, transcript hash, normalization and
  chunking versions, chunk boundaries, prompt and schema versions and the
  provider/model policy all match.
- `claims_eval.py` + `evals/claims/` — the optional extraction-quality
  harness (precision/recall, numeric, grounding, attribution, entity, stance,
  horizon, recommendation accuracy, false no-claims and duplicate rates;
  errors by category; breakdowns by length / source / language / quality;
  `--compare-chunked`; a real-benchmark specification check that prints
  "NOT ESTABLISHED" until ≥ 20 real videos / ≥ 200 labelled claims have
  been evaluated live; `--export-transcripts` writes labelling skeletons
  from stored transcripts). Offline replay by default; live only with
  `--live` and `CLAIMS_EVAL_LIVE=1`. Never runs in CI. The real benchmark
  has not been built yet (no stored transcripts, no live run).
- `research_state.py` — research state independent of delivery
  (`data/research/research_state.json`) plus the append-only products:
  `claims.jsonl`, `extraction_runs.jsonl`, `review_queue.jsonl`,
  `segments.jsonl`, `gate_outcomes.jsonl`, `video_records.jsonl`. Runs are
  keyed by their complete identity — transcript hash, normalization / prompt
  / schema / chunking versions, extraction mode, chunk policy, model policy
  and generation settings (`RUN_KEY_FORMAT`) — so a rerun appends nothing
  while a changed model or chunk size is a new run that supersedes the old.
- `research_backfill.py` — `--retry` (pending / failed / quota-deferred /
  partial research from stored transcripts), `--reprocess` (stale versions),
  `--import-legacy` (old signals rows as `schema_version="legacy"`,
  review-required, no fabricated evidence). Bounded, quota-aware, resumable.
- `research_analytics.py` — analytics over canonical claims: data-quality
  header with denominators, descriptive counts, consensus of one current
  **view** per source/asset/horizon bucket, flips (same source, asset and
  bucket), the portfolio-disclosure and conditional-forecast reports, and the
  experimental scorecard over matured unconditional forecasts under
  `scorecard_pricing` (conditional ones only with `include_conditional` and
  `condition_status=met`). `market_pulse` appends the quality header.
- `gemini_quota.py` — per-model daily free-tier accounting in
  `data/gemini_usage.json` (20 requests/day per model, resets midnight Pacific);
  also classifies a 429 as a per-day or per-minute limit, which decides whether
  a model is retired for the day or merely retried. A per-day verdict is
  **provisional**: it is honored for `GEMINI_SPENT_RECHECK_MINUTES` (doubling on
  each repeat) and then re-probed, because the API has written a model off after
  three requests — see `docs/tdd-gemini-transcripts.md` § 9. Only the locally
  counted cap is final.
- `signals.py` — the combined summary+claims fast path (one request, two
  products, validated independently: a malformed claims array never costs the
  summary and is recorded as `failed_retryable`, never as an empty result).
  A combined response **truncated by its claims array** returns
  `(None, research)` so the scraper makes a summary-only call, delivers it,
  and extracts the claims separately in smaller chunks (never deferring the
  video, never `no_claims_found`). An empty claims array against a
  claim-bearing transcript is `suspicious_empty_extraction`
  (`failed_retryable` from the combined call, `needs_review` from a
  standalone one). Standalone/chunked `extract_research`, and the legacy
  summary-based extractor kept for compatibility. `data/signals.jsonl` rows
  are derived from validated claims (`research_status` says how).
- `signals_data.py` — the dataset layer every analytics module reads:
  `load_signals`, asset identity (`ASSET_ALIASES`, `TICKER_ALIASES`,
  `UNPRICEABLE_TICKERS`, the learned map, `canonical_ticker`), date windows and
  `aggregate_assets` / `net_stance`. A leaf: it imports none of the modules
  below, which is what lets `market_pulse`, `channel_scorecard`, `pulse_charts`
  and `ticker_resolver` import each other at the top level instead of lazily.
  `market_pulse` re-exports its names for existing callers.
- `market_pulse.py` — the weekly pulse (`weekly-pulse.yml`) over **canonical
  claims** via `canonical_claims` (top assets per horizon bucket, flips,
  new-on-radar, disclosures, tone from each video's own views). The legacy
  pulse over `data/signals.jsonl` survives for compatibility and runs only
  with `PULSE_DATA_SOURCE=legacy`.
- `pulse_charts.py` — the pulse's companion PNG charts (consensus board, flip
  slope, weekly tone, bull-bear spread line, agreement-vs-attention map,
  target-upside ladder), styled for non-technical readers and sent as a Telegram
  photo album after the text pulse. Data prep is pure/testable; matplotlib
  imports lazily and every chart is best-effort — chart failures never block the
  text pulse. A chart with too little history to mean anything is either skipped
  (`MIN_SPREAD_WEEKS`) or labels itself as early days (`MATURE_SPREAD_WEEKS`).
- `ticker_resolver.py` — learns the real ticker for assets the transcript named
  badly, into `data/ticker_map.json` (filled by `warm_prices.py --resolve`).
  The provider's catalogue is tried first; only when it can't match the spelling
  does an LLM propose candidates, and **every suggestion is verified against the
  catalogue before it is kept** — the ticker must exist *and* the listing's
  company name must match the asset. This is deliberately narrower than the
  LLM-supplied tickers the extractor still forbids (see `ASSET_ALIASES`): a model
  guess alone never enters the data, and each entry records how it was resolved.
  `canonical_ticker` consults the curated tables first, so a learned entry can
  never override a hand-checked one.
- `price_cache.py` / `warm_prices.py` — the daily-close cache (`data/prices.json`)
  and the Sunday job that fills it (`warm-prices.yml`). Closes are immutable
  history, so a covered range is never refetched; this is what lets the weekly
  jobs finish at 8 requests/minute over ~180 symbols. `channel_scorecard.fetch_prices`
  reads the cache and only goes live for gaps, so every caller inherits it. The
  cache is process-wide (`price_cache.active`) — tests isolate it via the autouse
  fixture in `tests/conftest.py`, or one test's lookup answers another's mock.
- `channel_scorecard.py` — Friday scorecard (`weekly-scorecard.yml`):
  `generate_canonical_scorecard` scores canonical claims under
  `scorecard_pricing` (experimental, unranked unless `SCORECARD_RANKINGS=true`);
  the legacy 7/30-day directional hit rate over `signals.jsonl` remains behind
  `--legacy`. Twelve Data is requested with `adjust=$PRICE_ADJUSTMENT` and
  `fetch_prices.price_provenance` declares what every series is.
  Prices come from **Twelve Data** when `TWELVEDATA_API` is set (free tier: 800
  requests/day, 8/min — `_twelvedata_pace` respects the per-minute budget so a
  run doesn't turn into 429s), served through the `price_cache` (see below).
  `TWELVEDATA_MAX_REQUESTS` (120) still caps any single run, and callers spend
  it in priority order — `_pulse_inputs` fetches the reader-visible latest
  prices before the optional track-record weighting. Internal symbols are
  provider-independent (`nvda.us`, `btcusd`) and translated per provider by
  `twelvedata_symbol`. Without a key there is no price source: the keyless
  Stooq path it replaced sat behind a JavaScript challenge (verified 2026-08-17)
  and silently disabled implied upside, track-record weighting, the scorecard
  and the price-target chart from the first scheduled run (2026-07-27) until
  Twelve Data replaced it, so it has been removed rather than kept as a fallback
  that only ever returned nothing. `market_pulse.fetch_latest_prices` logs one
  loud "no prices for any ticker" line when the source is down — the first
  thing to check when prices look wrong.
- `sendToTelegram.py` — Telegram delivery (HTML, with plain-text fallback);
  digest builder. Transient failures retry with backoff (429 honours
  `retry_after`); the plain-text fallback runs only when Telegram *rejected*
  the HTML, so a network failure never duplicates already-delivered chunks.
  `scraper._Outbox` is the one place that decides single message vs digest and
  premium vs free, and reports what Telegram accepted.
- `helpers.py` — channel file parsing (`<id|@handle> [digest] [max=N] [only=a,b]` per
  line; handles resolved at run time by `scraper.resolve_channel_handle`; `only=`
  is a whole-word title filter applied before any transcript fetch), dedup
  state (v2 schema + v1 migration), `write_json_atomic` (used by every committed
  JSON state file), env parsing (`env_int` / `env_float` / `env_flag` — always
  use these: an unconfigured Actions variable arrives as `""`), summary cleaning.
- `log.py` — colored logging helpers; `redact()` / `describe_error()` keep bot
  tokens and API keys out of request-error lines.

## Dedup state model
`seen_videos.json`: `channels` maps channel-id → watermark (`last_video_id`,
`last_published`); videos published after the watermark are candidates, oldest
first, capped at `MAX_VIDEOS_PER_RUN` (0 = no cap, the default). `pending` maps
video-id → retry record for
deferred videos (captions not up yet → retried until `NO_TRANSCRIPT_MAX_ATTEMPTS`
*and* `NO_TRANSCRIPT_MIN_HOURS` are both exceeded, and no more often than
`PENDING_RETRY_MIN_HOURS`; LLM quota exhausted). Deciding a video advances the
watermark; deferring does not. **Delivery is part of deciding**: a final
outcome whose message Telegram has not accepted yet is held in the record's
`undelivered` block (the finished text plus the entry fields) and re-sent on a
later run without refetching the transcript or calling the LLM. Held records
survive the video leaving the RSS feed. A digest's videos are finalized only
once the digest itself has been accepted. Runs stop starting new videos after
`RUN_DEADLINE_MINUTES` so buffered digests always get flushed before the
workflow timeout.

## Design docs
- `docs/tdd-transcript-budget.md` — measured findings on transcript-credit
  efficiency (≈2.9 credits per delivered summary), the budget/pacing design and
  its critique, options considered, and the sequenced next steps. Read before
  changing `transcript.py` budget logic or channel caps.
- `docs/tdd-gemini-transcripts.md` — the August 2026 two-day outage, the real
  per-model free-tier limits (20 requests/day, not 1,500), measured cost of a
  Gemini video transcript, and why the transcript and summary model pools must
  stay disjoint. Read before changing either model list.
- `docs/tdd-full-transcript-claims.md` — why the 120k-char head/tail cut was
  removed, the token-budget algorithm, capability discovery, the chunked
  paths, the claim schema and prompt, the research-state design, the
  `signals.jsonl` compatibility view and which job reads what (§ 11), the
  scorecard methodology, conditional-forecast model, suspicious-empty rules
  and evaluation harness (§ 13), migration, and limitations. Read before
  touching request sizing, `claims.py`, the analytics jobs or the research
  data products.

## Dev workflow
```bash
pip install -r requirements-dev.txt   # runtime + pytest (the SessionStart hook does this on web)
./scripts/check.sh                    # byte-compile + run tests — the before/after gate
python -m pytest -q                   # tests only
```

Run `./scripts/check.sh` **before** starting a change to confirm a green baseline,
and **after** finishing to confirm no regressions. Add tests under `tests/` for any
new pure logic; network calls are mocked (the sandbox/CI can't reach YouTube/LLMs).

## Feature brainstorming
`/brainstorm` (`.claude/commands/brainstorm.md`) runs a multi-agent pipeline that
proposes, debates, and ranks feature ideas, then lets you pick one to implement.

## Conventions
- Functions never raise to the caller for expected failures; they return ""/None and log.
- Network calls retry transient errors with backoff; keep that pattern.
- Summaries are plain text (Telegram HTML-escapes them) with `• ` bullets.
- Never hardcode chat/channel identifiers (or any deployment value) in the repo or
  workflows — always read them from GitHub secrets/variables, even for public channels.
