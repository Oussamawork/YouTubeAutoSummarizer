# Developer guide (for Claude Code and humans)

YouTube auto-summarizer: fetches each channel's new videos, gets their transcripts,
summarizes them with an LLM, and posts the summaries to Telegram. Runs daily via
GitHub Actions (`.github/workflows/daily-summary.yml`); tests run on every PR
(`.github/workflows/ci.yml`).

## Module map
- `scraper.py` — entry point / orchestration; RSS feed (primary) + YouTube Data API
  (fallback); candidate selection against the dedup watermark; per-video outcomes
  (send / defer / give up); digest mode; `--video-url` on-demand path. A video
  that comes back after a summary-quota / truncation / delivery deferral
  **reuses its stored transcript** (`_stored_transcript`, reason `stored`,
  provenance kept from the original capture) once it re-verifies as the
  channel's language, instead of spending a second transcript request.
- `transcript.py` — transcript fetch, cheapest source first: Supadata → Gemini
  from the YouTube URL → youtube-transcript-api. Meters Supadata free-tier
  credits in `data/supadata_usage.json`, rotates across multiple keys, and paces
  a monthly budget over the days left in the month. The meter and the plan can
  disagree (Sept 2026: every key answered `limit-exceeded` with 184 local
  credits "left"), so once **every** key reports its plan limit the verdict is
  remembered (`provider_spent`), Supadata is skipped for the rest of the day
  and re-probed with the first video of each following day; a served request,
  a changed key set or the cycle reset clears it. Gemini transcription rotates
  across `GEMINI_TRANSCRIPT_MODELS` (free quota is 20 requests/day *per model*);
  when every source is spent, videos defer silently via `budget_exhausted`
  rather than being written off. **A transcript must be in the channel's
  language**: YouTube keeps translated and auto-dubbed caption tracks beside
  the original and Supadata served Arabic for English videos and English for
  German ones (Sept 2026). `get_transcript_from_video(url, languages)` asks
  Supadata for the channel's track (`lang=` when one language is
  configured), verifies every source's text with
  `language_detect.verify_language` (provider tag, writing system, stop-word
  detection), swaps a wrong Supadata track for an acceptable one the video
  has (one extra credit), and otherwise refuses it (`language_mismatch`) and
  moves to the next source — Gemini is told the language and transcribes the
  audio itself. The result carries `language` / `language_check`.
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
  **5xx answers are billed by the free tier** (2026-10: "503 high demand"
  retries emptied both summary models' daily quota with 2–10 requests
  served), so `_call_provider` retries a 5xx at most `LLM_OVERLOAD_RETRIES` (2)
  times per call and `LLM_OVERLOAD_MAX_FAILED_PER_RUN` (3) times per run, then
  skips the model for the run; a call that ends on 5xx returns the quota
  sentinel so the video DEFERS — never `""`, which reads as a failed summary
  (`manual_review`, final). See `docs/tdd-gemini-transcripts.md` § 10.
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
  are kept so evidence traces back to the source. Free-text asset mentions
  (`alias_mentions`, used by coreference and the suspicious-empty check) skip
  a curated name that is also an ordinary word (`signals_data.COMMON_WORD_NAMES`:
  "upstart", "avalanche", "Pendel") unless capitalised or followed by "stock".
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
  `signals.jsonl` compatibility view working. **Entity resolution: the name
  decides** (`resolve_entity`): the curated tables, plus `CHANNEL_ALIASES`
  (one channel's own spellings: Couch Investor "New" → NU, "Core" → CRWV;
  HKCM "XP" / "Joo Peng" → XPEV; TheStreet "Merck" → MRK, never global since
  the German channels' Merck is Merck KGaA), resolve the subject; the model's ticker only
  confirms it, or contradicts it (`ticker_subject_mismatch`, reviewed only
  when both tickers are real). On its own a model ticker counts only as
  `$T`, in capitals, after "ticker"/"symbol", in the title, or as a subject
  written as a registered ticker ("STX"). `MARKET_SUBJECTS` folds index,
  sector, commodity, bond and macro spellings to the canonical market keys
  `creator_tracker.PROXIES` prices (BONDS is not US TREASURIES); a subject is
  an instrument, market, basket (incl. coin pairs without a slash, "ETH BTC")
  or metric (`subject_kind`); a German article never hides a name ("Der
  Bitcoin"). Dropped model tickers leave non-blocking `resolution_notes`
  (`ticker_not_in_evidence` is retired). `RESOLUTION_VERSION` is stored on
  every claim and is not part of the run key — bump it when these rules or
  the channel/market tables change. A condition's "price level" needs a
  number that is not a span of time ("over the next 6 months" is not one).
- `language_detect.py` — deterministic language handling: a writing-system
  census (`script_of`), stop-word detection (en / de / unknown; floors
  measured on the stored transcripts) and `verify_language(text, accepted,
  reported)`, the one rule for whether a caption track may be used.
  `DEFAULT_LANGUAGES` (`TRANSCRIPT_LANGUAGES`, default en,de) applies to
  channels without a `lang=` option. The verified language is stored as
  `transcript_language`.
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
  unresolved instrument is excluded, never guessed. The registry covers every
  alias output and every ticker with 3+ canonical claims (DPZ's MIC is on the
  audit's verify list), 15 coins, and the SPDR sector funds; WMT is on XNAS
  since 2025-12-09. Stock and ETF are one kind (a claim typed "stock" on an
  ETF row resolves; a coin never does). `Instrument.name` and
  `display_label(ticker)` name the unclear tickers ("Planet Labs (PL)",
  "SpaceX (SPCX)") in every report.
- `canonical_claims.py` — **the one loader every production analytics job
  reads**: active runs only, no legacy rows, no repeats, condition outcomes
  overlaid, and every stored row **re-resolved at load** (`reresolve`: alias
  and rule changes correct stored claims without re-extraction; testability
  is not recomputed); `view_claims`, `aggregate_views` (per asset AND horizon
  bucket, one current view per source — research analytics),
  `aggregate_views_by_asset` (per asset, one vote per creator, `target_by` —
  the pulse), `lean`, `video_tone`, `portfolio_disclosures` (sold / trimmed /
  added read from the evidence, `disclosed_action`). `asset_key` gives basket
  and metric subjects no key and market subjects their canonical key;
  `resolved_ticker` applies `CHANNEL_ALIASES`; `priceable_symbol` is
  registry-first (`COLLIDING_TICKERS`: a coin that is also a NYSE listing —
  STX, LINK, SUI, VET, ... — follows the claim's explicit type).
  `data/signals.jsonl` is a backward-compatible view read by nothing here.
- `scorecard_pricing.py` — scorecard methodology: exchanges/timezones/
  calendars (NYSE holiday rules and 13:00 half days, `nyse_early_closes`), the
  publication-time **next-close entry rule** (never a close that printed
  before the video; crypto = UTC-day close), `PriceSeries` provenance (provider, adjustment — split-adjusted at
  minimum, `PRICE_ADJUSTMENT=all` for total return — corporate-action status,
  currency, requested/resolved dates, daily highs/lows when the provider
  returned bars), per-claim benchmark resolution with an honest null,
  **calendar confidence** (only NYSE closures are modelled; weekday-only
  exchanges are scored but excluded from rankings unless
  `SCORECARD_RANK_WEEKDAY_CALENDARS=true`, and no error bound is claimed),
  **price-target methods** (`evaluate_target`: intraday_touch with bars,
  daily_close otherwise, horizon_close beside them; intraday reach is null,
  not false, without bars; callers pass a window that starts after the
  publication), crypto benchmarks under the asset-class rule (BTC for
  other coins), and `SCORECARD_RANKINGS` (off by default, and it stays off
  until the live extraction benchmark has been run and reviewed).
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
  review-required, no fabricated evidence), `--reject-foreign-transcripts`
  (offline: retire every stored capture not in its channel's language —
  `transcript_store.reject_transcript` moves the file aside, the video's run
  is superseded so its claims leave the canonical set, status
  `transcript_rejected`) and `--refetch` (re-capture those videos in the
  channel's language through the daily job's source chain and budget, then
  extract; `RESEARCH_REFETCH_MAX_VIDEOS`, default 3). The daily job runs the
  last two after the scraper. `--adopt-stored` (offline, idempotent) gives a
  `pending` research record (`delivery_status="not_sent"`,
  `adopted_from="stored_transcript"`) to every stored transcript that never
  got one — research used to start only beside a delivered summary, so a
  video whose summary was quota-deferred until it left the feed or failed
  into manual review kept its transcript but never reached the claims or the
  tracker (40 such on 2026-10-16). Videos still pending a summary in
  `seen_videos.json`, rejected captures and wrong-language text are skipped;
  the daily job runs `--adopt-stored --import-candidates --retry`, so `--retry`
  extracts them under the summary reserve. `--import-candidates` stores
  committed offline extractions (`data/research/candidates/<video_id>.json`:
  `video_id`, `transcript_hash`, `extraction_prompt_version`, `source`, and a
  model-shaped `response`) through `process_video` with no model request:
  validated by `signals.build_research` exactly like a model response, run
  identity `offline:<source>` (`offline_identity`), `extraction_model`
  "offline-extraction"; a changed transcript or prompt version is skipped and
  a finished video is never overridden. The 63-transcript backlog of
  2026-10-17 was cleared that way. Bounded, quota-aware, resumable.
- `research_analytics.py` — analytics over canonical claims: data-quality
  header with denominators, descriptive counts, consensus of one current
  **view** per source/asset/horizon bucket, the portfolio-disclosure and
  conditional-forecast reports, the creators-vs-market summary
  (`creator_tracker.format_summary_lines`, cache only), and the experimental
  scorecard over matured unconditional forecasts under `scorecard_pricing`
  (conditional ones only with `include_conditional` and
  `condition_status=met`). Identity is canonical (`asset_key`, `source_of`,
  `direction_of`; placeholder, basket and metric subjects have no key).
  **View changes** are creator_tracker video calls (one lean per video)
  reversing between different videos at least `RESEARCH_FLIP_MIN_GAP_DAYS`
  (7) apart; quicker reversals are only counted per creator, same-video
  scenario talk never flips, guest-panel flips are tagged. The conditional
  block is compact (two summary lines, at most 6 examples, observability
  recomputed under the current rules). The strict scorecard scores price
  forecasts only (`non_price_metric`), counts ticker-less macro / index /
  sector / commodity / bond subjects as `not_an_instrument`, tests targets
  only via `price_target_of` on the called side within 0.2–5× of the entry,
  and counts an entry day's high/low only when the video came before that
  session. `market_pulse` appends the quality header.
- `gemini_quota.py` — per-model daily free-tier accounting in
  `data/gemini_usage.json` (20 requests/day per model, resets midnight Pacific);
  also classifies a 429 as a per-day or per-minute limit, which decides whether
  a model is retired for the day or merely retried. A per-day verdict is
  **provisional**: it is honored for `GEMINI_SPENT_RECHECK_MINUTES` (doubling on
  each repeat) and then re-probed, because the API has written a model off after
  three requests — see `docs/tdd-gemini-transcripts.md` § 9. Only the locally
  counted cap is final. Attempts answered with a 5xx are counted apart
  (`record_failed`, `failed` in the usage file): they spend the budget line's
  remaining figure ("3/20 left (2 served + 15 failed 5xx attempt(s))") but
  never the served count or `is_exhausted` (§ 10).
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
  `market_pulse` re-exports its names for existing callers. `canonical_ticker`
  rules, in order: a recorded index name (`NOT_A_TICKER`: NASDAQ, SPX, DAX,
  DXY, DOW, ...) is dropped and the name decides; a curated name outranks a
  contradicting recorded ticker (SY → SONY, CRWE → CRWV, RVN → RIVN) unless
  that ticker is itself a registered instrument or UNPRICEABLE
  (`FORMER_TICKERS`: Barrick's old GOLD is B); the learned map only fills
  gaps. Name-shaped `TICKER_ALIASES` keys (SPACEX, MASTEC, the big banks)
  resolve as names but are no free-text mentions (speakers quote banks as
  sources); `LISTED_TICKER_KEYS` (FORD = Forward Industries) are never folded
  from a ticker field. SpaceX is SPCX (listed June 2026; evidence beside the
  alias). UNPRICEABLE holds GOLD (NYSE GOLD is neither Barrick nor the
  metal), MOGU and RCI (known-wrong legacy symbols), the Korean / Taiwanese /
  Shanghai codes and private companies. `COMMON_WORD_NAMES` and
  `NAME_SUFFIXES` are shared with `claims` and `transcript_normalize`.
- `market_pulse.py` — the Monday pulse (`weekly-pulse.yml`) over **canonical
  claims**, aggregated **per asset with one vote per creator**
  (`canonical_claims.aggregate_views_by_asset`: a creator's vote is the
  `lean` — 2/3 rule — of their latest in-window video's claims; conflicting
  horizons vote `mixed`, never averaged). Sent as Telegram HTML (plain
  fallback), never an LLM call, in this order: the takeaway (creator-level
  lean; the strongest agreement, ranked by voters then agreement share; the
  S&P 500 / Bitcoin move "in the week to {close}"; last week's picks h of p)
  → Market (S&P 500, Nasdaq-100, Bitcoin, Ether: week and 4 weeks from the
  tracker's market rows; a series over 3 days old is left out, and a stale
  cache says so) → Where creators agree (move "since {date}" from the
  consensus side's first call vs the S&P 500 — crypto vs Bitcoin — via
  `creator_tracker.asset_outcome`, or "new call, no close yet"; actions and
  the average target only when 2+ consensus-side creators share them; one
  row kept for the strongest bearish agreement) → one pooled creators-vs-
  market line (10+ marked calls, labelled mark-to-market, never per creator)
  → Where they disagree → Last week's picks, one week on (rebuilt from claims
  extracted by Monday 08:00 UTC, priced with `instrument_for` / `price_move`,
  ✅/➖/❌ with a legend; the average is in the called direction) → consensus
  changes → new on the radar → mood → disclosures → footer ("Prices: daily
  closes to {date}"). Every price is a cached close at ONE as-of (the
  tracker's: `default_as_of(today)`, narrowed to the last cached S&P 500
  close); the canonical pulse never fetches live. Wherever 2+ creators must
  agree, two channels of one house count once (`creator_tracker.creator_houses`).
  The text's average target and the upside chart use one list
  (`_row_targets`: each creator's latest in-window target on their own side
  of the close, the consensus side's only), plausible within 0.2×–5×.
  Reference phrases and non-instruments are not assets
  (`is_placeholder_asset`); disclosures need an ownership cue, no negation, a
  resolved ticker and a clean review. Labels go through
  `instruments.display_label`. The full data-quality header goes to the
  workflow log, not Telegram. The legacy pulse over `data/signals.jsonl`
  survives for compatibility and runs only with `PULSE_DATA_SOURCE=legacy`.
- `pulse_charts.py` — the pulse's companion PNG charts, drawn 6.4in wide so
  text survives Telegram's ~380px album preview, each titled with its
  takeaway (`consensus_title`, `tone_title`, `upside_title`). The album is
  consensus board (assets with 2+ creators), weekly mood, and price-target
  ladder (single-creator targets faded); the flip slope, agreement map and
  optimism-gap line join only when `_chart_ready` says they add something
  (2+ well-attended reversals; 6+ points incl. a bearish/contested one;
  `MIN_SPREAD_WEEKS` = 9 weeks of history). Data prep is pure/testable;
  matplotlib imports lazily and every chart is best-effort — chart failures
  never block the text pulse. The upside chart leads with shared targets
  ("Best-supported target: …", never a lone creator's). It also draws the
  Friday creators-vs-market album from a `creator_tracker` record
  (`build_tracker_chart_data` → `render_tracker_charts`, separate from
  `CHARTS`): `tracker-1-creators.png`, each creator's median call vs the
  market in the called direction (rows by number of calls, never by result;
  faded under `TRACKER_MIN_CREATOR_SAMPLE`; grey within ±0.5 points, "level";
  title counts only the full samples, "1 of 5 with 10+ calls ahead of the
  market"), and `tracker-2-consensus.png`, the move since the first call on
  assets creators from two houses agree on, against a grey market tick. Each
  needs 3+ rows; both carry "Mark-to-market, not a verdict". pulse_charts
  imports creator_tracker; creator_tracker never imports pulse_charts.
- `creator_tracker.py` — **creators vs the market**: the mark-to-market layer
  the Friday report and the Monday pulse lead with, separate from the strict
  scorecard (the verdict layer, 1 scored claim today: horizons run to
  2027-28). One call = one creator × asset × change of view, from canonical
  view claims that are directional, a view on the security, not
  trigger-conditional (a qualifier such as "as long as support holds" still
  counts) and on something tradable (`not_an_instrument`: themes, sectors
  without a fund, macro, unlisted companies). Each video votes once per asset
  (`lean`); a reiteration keeps the entry, a reversal closes the call at its
  own entry close, a call matures at its horizon (the call's own claims').
  Instruments: `PROXIES` for market keys (S&P 500 → SPY, sector funds, GLD,
  USO, ...; spellings of one fund are one asset), else
  `instruments.resolve_instrument` on the call's resolved ticker — a spoken
  name is never a ticker. Entry = `scorecard_pricing.entry_point` (never
  shifted when the cache starts late: `no_entry_close`); ONE effective as-of,
  the last spy.us close on or before `default_as_of(today)`, crypto marked
  the same day; benchmark over identical dates (SPY; BTC for other coins;
  none for SPY/VOO/IVV/BTC). Metrics: ret, dir_ret, right, excess,
  dir_excess, beat (in the called direction), w28. Targets: the latest level
  stated before the mark, beyond both the entry and the statement close,
  tested from the day after the statement. Every call is scored or carries
  one reason (no_ticker, unresolved_instrument, no_prices, too_recent,
  no_entry_close, no_post_call_close, stale_mark, younger_than_min_age,
  reversed_within_min_age); with no S&P 500 close in 14 days nothing is
  marked and the report says the prices are stale. Two channels of one house
  (`creator_houses`: a name containing another channel's name,
  `TRACKER_CREATOR_HOUSES` overrides) count once wherever two creators must
  agree. Output is labelled "mark-to-market, not a verdict", prints n beside
  every rate, orders creators by calls and never ranks them; messages ≤ 3,900
  characters. Prices only through `cache_fetcher` (never live);
  `needed_price_ranges(claims, today)` returns exactly the requests
  `track_record` makes (a closed call's range ends soon after it closes) and
  `warm_prices` covers them. Env: `TRACKER_LOOKBACK_DAYS` (180, bounds the
  warm cost), `TRACKER_MIN_CALL_AGE_DAYS` (7), `TRACKER_MIN_CREATOR_SAMPLE`
  (10). CLI: `python creator_tracker.py [--as-of D] [--json PATH]`, offline.
- `ticker_audit.py` — the offline, read-only ticker audit (`python
  ticker_audit.py [--json PATH|-] [--strict] [--top N] [--today D]`; ~1.5 s,
  no network; `--strict` exits 1 on errors). Nothing it finds is applied.
  Sections: alias outputs with no registry row or unpriceable decision
  [error]; `KNOWN_WRONG_SYMBOLS` a claim or legacy row still produces
  [error] or that are only cached [info]; orphaned cache series; prices the
  speakers stated vs the symbol's first close after publication (0.67–1.5×;
  all 91 matched on 2026-10-16) [warning]; resolved tickers without a
  registry row; the unresolved-name worklist (legacy and spoken hints shown,
  never applied); learned-map checks [error]; series anomalies (40% days,
  flat runs, weekend bars, partial crypto candles, stale series the tracker
  needs) [warning]; and `NEEDS_VERIFICATION`, the medium/low-confidence
  corrections, each with its evidence and the exact `warm_prices.py
  --find/--probe` step. The warm workflow runs it as an informational step.
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
  and the job that fills it (`warm-prices.yml`, Sundays and Fridays at 06:00
  UTC, ahead of the Monday pulse and the Friday report). Closes are immutable
  history, so a covered range is never refetched. `needed_ranges` = legacy
  ranges + `canonical_ranges` (testable forecasts and their benchmarks, the
  pulse's targets over `[as_of − 10, as_of]`) + `tracker_ranges`
  (`creator_tracker.needed_price_ranges`: every call symbol from the day
  before the call, the SPY/BTC benchmarks, context SPY/QQQ/BTC/ETH). Each
  request is capped at the last completed session (`last_complete_day`:
  crypto yesterday UTC, else the last NYSE session) and bars past it are
  dropped, so no partial candle is stored and an open-range refetch repairs
  one stored before. A run is bounded: `warm_order` (S&P 500 and Bitcoin
  first, then the tracker's symbols, stalest first), no request after
  `WARM_DEADLINE_MINUTES` (75), a save every `SAVE_EVERY` (50) symbols, and
  `save(keep_from=)` never trims history a needed range still starts at.
  `channel_scorecard.fetch_prices` reads the cache and only goes live for
  gaps; the tracker and the pulse never go live. The cache is process-wide
  (`price_cache.active`) — tests isolate it via the autouse fixture in
  `tests/conftest.py`, or one test's lookup answers another's mock.
- `channel_scorecard.py` — the Friday report (`weekly-scorecard.yml`, 16:00
  UTC), `generate_weekly_report` in this order: (1) creators vs the market
  (`creator_tracker.format_report` at `default_as_of(today)`, the Thursday
  close, through `cache_fetcher` only); (2) the two tracker charts as one
  album, best-effort; (3) the strict block (`generate_canonical_scorecard`:
  forecasts matured by the last completed session, experimental and unranked
  unless `SCORECARD_RANKINGS=true`; below the sample minimum it is compact —
  one plain line per scored forecast, when the next ones mature, and a
  one-line count of the conditional calls of the last
  `CONDITIONAL_WINDOW_DAYS`, 30). Only the strict block's matured forecasts
  may fetch live. `--dry-run` prints and writes the PNGs to
  `charts/scorecard`; `--no-charts`; exit 1 only when a text fails; the
  maintainer's ticker worklist goes to the log. `symbol_for` is
  registry-first with the row's channel (Bitcoin typed stock is btcusd,
  Couch Investor's "New" is nu.us) and prices only US-shaped tickers by type.
  The legacy 7/30-day directional hit rate over `signals.jsonl` remains behind
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
  that only ever returned nothing. `market_pulse.fetch_latest_prices` reads
  the cache and logs one loud "no prices for any ticker" line when it is
  unwarmed — the first thing to check when prices look wrong (then the
  warm-prices run).
- `sendToTelegram.py` — Telegram delivery (HTML, with plain-text fallback);
  digest builder. Transient failures retry with backoff (429 honours
  `retry_after`); the plain-text fallback runs only when Telegram *rejected*
  the HTML, so a network failure never duplicates already-delivered chunks.
  `scraper._Outbox` is the one place that decides single message vs digest and
  premium vs free, and reports what Telegram accepted.
- `helpers.py` — channel file parsing (`<id|@handle> [digest] [max=N] [only=a,b]
  [lang=xx]` per line; handles resolved at run time by
  `scraper.resolve_channel_handle`; `only=` is a whole-word title filter applied
  before any transcript fetch; `lang=` the language its transcripts must be in), dedup
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
- `docs/architecture.md` — the end-to-end pipeline: discovery and gating, the
  transcript source cascade and its budget gates, how the summarizer is instructed,
  what each outcome does to the video, and the fan-out after a summary. Start here.
- `docs/tdd-transcript-budget.md` — measured findings on transcript-credit
  efficiency (≈2.9 credits per delivered summary), the budget/pacing design and
  its critique, options considered, and the sequenced next steps. Read before
  changing `transcript.py` budget logic or channel caps.
- `docs/tdd-gemini-transcripts.md` — the August 2026 two-day outage, the real
  per-model free-tier limits (20 requests/day, not 1,500), measured cost of a
  Gemini video transcript, why the transcript and summary model pools must
  stay disjoint, and § 10: 503 retries are billed, how a 503 storm became 63
  unprocessed transcripts, and the owner options (billing, a third model, a
  second provider). Read before changing either model list or retry logic.
- `docs/tdd-full-transcript-claims.md` — why the 120k-char head/tail cut was
  removed, the token-budget algorithm, capability discovery, the chunked
  paths, the claim schema and prompt, the research-state design, the
  `signals.jsonl` compatibility view and which job reads what (§ 11), the
  scorecard methodology, the creators-vs-market tracker beside it,
  conditional-forecast model, suspicious-empty rules and evaluation harness
  (§ 13), ticker verification, migration, and limitations. Read before
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
- Only certain or high-confidence ticker corrections go into the
  `signals_data` / `instruments` tables (with their evidence in a comment).
  Medium and low ones go into `ticker_audit.NEEDS_VERIFICATION` with their
  evidence and a verify step; never guess a ticker.
- Reports never rank creators and label mark-to-market results "not a
  verdict"; every rate prints its denominator.
