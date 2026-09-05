# TDD — Full-transcript summarization and the claim-level research dataset

**Status:** implemented 2026-09-05. **Scope:** request sizing for the LLM
calls (input vs output), the removal of transcript truncation, and the upgrade
of the research output from per-asset signals to auditable atomic claims.

---

## 1. The problem

`summarizer.py` capped every transcript at `LLM_MAX_TRANSCRIPT_CHARS = 120000`
(`summarizer.py:52`), with per-provider twins `GEMINI_MAX_INPUT_CHARS`
(`:57`) and `GROQ_MAX_INPUT_CHARS` (`:82`). Over the cap, `_head_and_tail`
(`summarizer.py:439-454`) kept the first 60 % and the last 40 % of the budget
and dropped the middle with a `...[transcript truncated]` marker. It was
applied twice: by `_truncate_transcript` (`:423`) at the top of
`summarize_transcript` and inside `signals.summarize_with_signals`
(`signals.py:255`), and again per provider by `_fit_to_provider` (`:393`) in
`_call_provider` (`:503`) — so it affected summary-only calls, the combined
summary+signals call, every escalation retry and every fallback provider. It
did not affect the legacy summary-based signal fallback, which read the
summary rather than the transcript.

The number was an application decision (the comment reasoned from the free
tier's tokens-per-minute limit), not a model limit: gemini-3.7-flash reads
1,048,576 input tokens and writes 65,536. The middle of a long video is where
the second asset, the price target, the condition on the forecast, the guest's
disagreement and the change of mind live; silently dropping it is wrong for
the summary and disqualifying for a research dataset.

## 2. What changed (files)

New: `model_capabilities.py`, `token_budget.py`, `transcript_normalize.py`,
`transcript_store.py`, `claims.py`, `research_state.py`,
`research_backfill.py`, `research_analytics.py`; tests
`tests/test_model_capabilities.py`, `tests/test_token_budget.py`,
`tests/test_transcript_normalize.py`, `tests/test_claims.py`,
`tests/test_research_state.py`, `tests/test_research_analytics.py`,
`tests/test_summarizer_chunked.py`; this document.

Modified: `summarizer.py` (truncation removed; token-aware admission;
`INPUT_TOO_LARGE_SENTINEL`; chunked summarization; telemetry), `signals.py`
(combined envelope now carries claims; independent validation; standalone and
chunked `extract_research`), `scraper.py` (normalize + persist before
summarizing; research context; research persistence separate from delivery;
gate outcomes), `market_pulse.py` (quality header), `tests/conftest.py`
(offline capability/token lookups, scratch research dirs), existing tests in
`tests/test_summarizer.py`, `tests/test_signals.py`, `tests/test_scraper.py`,
`.github/workflows/daily-summary.yml` (research retry step; persist
`data/research`, `data/transcripts`, `data/model_capabilities.json`),
`CLAUDE.md`, `README.md`.

## 3. Token budgeting

For every candidate request (`summarizer._fits_provider`):

1. Build the exact messages: system prompt (summary rules, or summary rules +
   claims envelope, or the claims prompt), user message (title line +
   complete transcript, or the chunk / merge input).
2. Measure them (`token_budget.measure_request`): Gemini `countTokens` with
   the same system instruction and user content when `LLM_LIVE_COUNT_TOKENS`
   is on (memoized per identical request, so an escalation retry does not
   count again); otherwise a conservative estimate of
   `ceil(len / 3.0)` per part plus 8 tokens of per-message overhead. The
   method is recorded (`count_tokens` vs `estimate`). `countTokens` is a
   separate endpoint with its own per-minute rate limit and no charge; it
   does not consume a model's 20 daily `generateContent` requests. Any
   failure falls through to the estimate. No generative call counts tokens.
3. Compute the budget (`token_budget.context_budget`) from the model's own
   `ModelCapabilities`:

   ```
   reserved_output = min(max_output_tokens_for_this_call, output_token_limit)
   available_input = min(input_token_limit,
                         context_window_limit - reserved_output   # shared windows only
                         tokens_per_minute_limit - reserved_output # metered tiers only
                        ) - CONTEXT_SAFETY_MARGIN_TOKENS (2048)
   ```
   Gemini reports input and output limits separately (no shared window); the
   free tier's 250k TPM is applied as a per-request bound because a single
   request larger than the minute's allowance is rejected outright
   (`GEMINI_FREE_TIER_TPM`, 0 disables). Groq/OpenAI-style models declare a
   shared window and subtract the reserved output.
4. `fits = input_tokens <= available_input`. If it fits, the complete request
   is sent (coverage `full`). If not, `_call_provider` returns
   `INPUT_TOO_LARGE_SENTINEL` **without sending or trimming anything**.

Output caps stay separate: `SUMMARY_MAX_OUTPUT_TOKENS` (4000),
`COMBINED_MAX_OUTPUT_TOKENS` (8000), `CLAIMS_MAX_OUTPUT_TOKENS` (6000); the
escalation ladder (`LLM_MAX_TOKENS_CEILING`, `LLM_MAX_ESCALATIONS`) and the
`finish_reason` check are unchanged, so output truncation is still detected
from the response and never delivered.

Telemetry per call (`summarizer.LAST_CALL_TELEMETRY`, logged and stored in
`extraction_runs.jsonl`): model, provider, transcript chars, input tokens and
method, available input, model input limit and capability source, output cap,
provider-reported prompt/completion tokens, finish reason, and for chunked
runs coverage status, chunk counts (total / cached / succeeded / failed).

## 4. Capability discovery

`model_capabilities.capabilities_for(provider)`, in order of trust: (1)
`MODEL_CAPABILITIES_JSON` operator override; (2) Gemini `models.get`
(`inputTokenLimit`, `outputTokenLimit`), cached in
`data/model_capabilities.json` for `MODEL_CAPABILITIES_TTL_HOURS` (7 days) and
committed by the daily job; (3) the dated registry (`REGISTRY_VERSION`);
(4) a deliberately small default (32k / 4k) for an unknown model, logged. Every
provider in the chain is evaluated on its own record; nothing inherits
gemini-3.7-flash's window.

## 5. Full and chunked paths

**Fast path (ordinary videos).** `scraper._summarize_video` normalizes the
transcript, persists the raw text, and calls `signals.summarize_with_signals`
with the complete normalized transcript: one request returns
`{"summary", "claims", "extraction_metadata"}`. The summary and the claims
are validated independently (§8). If the envelope is unusable the existing
fallback runs (`summarize_transcript`, then `extract_research` separately).

**Summary over the limit.** `summarize_transcript` gets
`INPUT_TOO_LARGE_SENTINEL` from a provider and calls `_summarize_chunked`:
normalize → `chunk_transcript` at that provider's available input (segment
edges, then sentence edges, then a token-safe cut; overlap
`CHUNK_OVERLAP_TOKENS`, capped at a quarter of a chunk) → `validate_coverage`
(first chunk at 0, last at the end, no gaps, strictly advancing) → **quota
pre-check**: `len(chunks to run) + 1` requests against the model's remaining
daily requests; too few returns `QUOTA_EXHAUSTED_SENTINEL` before any call →
one `CHUNK_NOTES_SYSTEM_PROMPT` call per chunk (constrained JSON notes),
each persisted to `data/research/partials/<transcript_hash>/<chunk_id>.json`
→ one merge call with the summary prompt over all notes in order. A chunk
stopping on quota/truncation/failure returns the corresponding sentinel; the
merge never runs on partial coverage; the next run resumes from the saved
partials. Existing retry semantics are preserved: quota defers without a
watermark move, a hard failure moves to the next provider.

**Research over the limit / exhaustive mode.** `signals.extract_research`
tries one request; `INPUT_TOO_LARGE` (or `EXHAUSTIVE_RESEARCH_MODE`) runs
`_extract_research_chunked`: chunks sized by the widest usable provider (or
`RESEARCH_CHUNK_TOKENS`), one claims call per chunk with per-chunk partials,
`dedupe_across_chunks` for overlap repeats, coverage `chunked_full` only when
every chunk succeeded; otherwise `partial` (or `quota_deferred`) with
`processed_chunk_ids` / `failed_chunk_ids` recorded for resume.

## 6. Normalization and segmentation

`transcript_normalize.normalize_transcript` repairs formatting only: CRLF,
whitespace runs, caption-line joins, SRT/WebVTT/bracket timestamps parsed to
segment seconds, recurring `Label:` speaker turns, adjacent-cue overlap
(a cue repeating the previous cue's tail of ≥3 words / ≥12 chars) and
identical adjacent cues removed **only** for cue-shaped input; repetition
inside a cue or in plain prose is untouched. Every number, %, currency,
range, date, negation and hedge passes through verbatim; unintelligible
markers are flagged, not deleted; an offset map traces every normalized
character to the raw text. Segments are contiguous spans that start on
speaker change, new asset mention, paragraph break or length, carry
offsets/seconds/speaker/raw text, and get a keyword-heuristic category
(sponsor, disclaimer, intro/outro etc. are stored but marked
`excluded_from_headline`).

## 7. Claim schema (v1)

Every field of the requested schema is present on each record
(`claims.validate_claims`), plus `horizon_bucket` and `run_key`. Application
code sets ids (`claim_id` = sha1 of video, run key, chunk, evidence offset and
content), versions, coverage, timestamps, entity resolution, dates and
testability; the model returns the compact subset in `CLAIM_FIELDS_SPEC` and
omits nulls. Segment reference: the evidence is located in the normalized
transcript and the owning segment (and its seconds) is derived from the
offset, so a claim can never reference a segment its evidence is not in.

Validation (deterministic, `claims.py`): JSON shape; enums; evidence located
under a normalization-aware match (case, quotes, dashes, whitespace,
punctuation — never a paraphrase; ≥12 chars); every numeric field present in
the evidence (with k/m/b/thousand multipliers, distributed across ranges);
`ticker_spoken` kept only if it appears in the evidence or title, else
resolved through `ASSET_ALIASES` / `TICKER_ALIASES` / the catalogue-verified
learned map, ambiguous names never resolved; questions → `question`, not
forward-looking; retrospective markers → `historical_claim`; third-party
markers without adoption → review; recommendation verbs required for any
action; ownership language → `portfolio_disclosure`; horizons resolved only
under the rules in `resolve_horizon`'s docstring (vague words get no dates);
testability requires forward-looking + resolved instrument + metric/direction
+ end date + full coverage + located evidence + no condition. Unverifiable
candidates stay in the record with `review_required=True` and go to
`review_queue.jsonl`; nothing unsupported is silently kept and nothing is
silently dropped.

### 7.1 Schema v2 additions (review corrections)

Schema and prompt versions are now `2`. New fields per claim:

| Field | Values | Set by |
| --- | --- | --- |
| `host_position` | adopted, rejected, neutral, not_applicable | evidence regexes first (`_REJECT_RE` beats `_ADOPT_RE`), the model's word only when the excerpt is silent — and a model-asserted adoption without evidence is downgraded to neutral |
| `third_party_origin` | organisation | kept when an adopted third-party view is promoted to the speaker's own |
| `entity_resolution_method` | explicit_mention, local_coreference, title, curated_mapping, unresolved | `validate_claims` |
| `entity_resolution_confidence` | 1.0 / 0.8 explicit, 0.7 local coreference, 0.6 title, 0.5 curated-only, 0.0 unresolved | `validate_claims` |
| `testability_type` | unconditional_testable, conditional_testable, not_testable | `validate_claims` |
| `condition_text`, `condition_observable`, `condition_kind`, `condition_status`, `condition_evaluation_date`, `condition_evidence`, `condition_data_source` | see § 13.2 | `condition_observability`; outcomes from `condition_evaluations.jsonl` |
| `evidence_start_seconds` / `evidence_end_seconds` | the CUE containing the evidence | `NormalizedTranscript.seconds_at` / `end_seconds_at` (null for plain text) |

**Portfolio disclosures are not views.** "I own Tesla" — whatever claim
type, stance or action the model attached — becomes `claim_type =
portfolio_disclosure`, `stance = not_applicable`, `stance_basis =
not_applicable`, `recommendation_action = none`, `is_forward_looking =
false`, `portfolio_disclosure = owns_unspecified` or the position the
evidence itself supports (`long`, `short`, `no_position`; a model-supplied
`long` without supporting words is downgraded). Only a statement that also
forecasts or recommends keeps its view, with the disclosure noted beside
it. `claims.carries_view` is the single filter every stance path uses
(consensus, net stance, flips, tone, the legacy reduction, the pulse, the
scorecard): it requires `claim_type ∈ VIEW_CLAIM_TYPES`, so a disclosure —
or a question, fact, news report, third-party view, retrospective or
hypothetical — never votes, even if its `stance` field says bullish. The
disclosures go to their own report (`canonical_claims.portfolio_disclosures`,
appended to the pulse and the research report).

**Third-party attribution (rule and implementation now agree).** A reported
view (`speaker_quoting_third_party`, `speaker_describing_market_consensus`,
`speaker_reporting_news`) is `third_party_view` with `stance =
not_applicable` and `host_position` recorded; it needs NO review — it is
excluded from own-view analytics by attribution, not by a reviewer. A view
the speaker explicitly adopts ("I agree with that", "that's my target too")
is promoted to `speaker_personal_view` (claim type `price_target` /
`forecast`, `third_party_origin` kept) and counts as their own. What DOES go
to review is an own-view claim whose evidence reads like a quotation without
adoption (`possible_third_party_view`).

**Local coreference (no outside knowledge).** A generic subject ("the
stock", "it", "the company", "shares") resolves to the ACTIVE local subject:
the curated asset names in the evidence itself, else in the previous two
sentences of the same segment (at most 400 characters back). Exactly one
distinct asset → resolved (`local_coreference`, 0.7); two or more →
`ambiguous_coreference`, unresolved, review; none → unresolved. Only names
the curated tables or the catalogue-verified learned map already know can
be the referent.

## 8. Combined envelope outcomes

| Response | Delivery | Research |
| --- | --- | --- |
| valid summary + valid claims | sent | `complete` (or `needs_review` when every claim is flagged) |
| valid summary + malformed `claims` | sent | `failed_retryable`, reason `malformed_claims`, `signals: null`, retried later |
| valid summary + `claims: []`, transcript plausibly claim-free | sent | `no_claims_found` (after validation AND the suspicious-empty check), `signals: {"assets": []}` |
| valid summary + `claims: []`, transcript carries claim language | sent | `failed_retryable`, reason `suspicious_empty_extraction`, `signals: null`; a standalone pass runs (same run when budget permits, else the retry job); a standalone pass that is empty again → `needs_review`. Never `no_claims_found`. |
| malformed envelope / no summary | fallback: plain summary call, then separate `extract_research` | as that call decides |
| output truncated (`finish_reason=length`) after escalation | `complete()` reports `TRUNCATED_SENTINEL` (truncation wins over quota and "" so the cause is not hidden); `summarize_with_signals` returns `(None, research)`; the scraper makes a **summary-only call** and delivers it; the watermark advances with that delivery exactly as for any sent summary | `failed_retryable` / `combined_output_truncated` with `retry_separately`; the scraper immediately runs `extract_research(prefer_chunked=True)` — claims in `RESEARCH_CHUNK_TOKENS` chunks, each writing fewer claims — and records its outcome (`complete`, `quota_deferred`, …); if that cannot finish, the ledger keeps the truncation reason and `research_backfill --retry` picks the video up. Never `no_claims_found`. |
| standalone claims call truncated | n/a | re-run in smaller chunks (`_extract_research_chunked(max_chunk_tokens=RESEARCH_CHUNK_TOKENS)`) |
| input too large for every model | `summarize_transcript` chunked path | `extract_research` chunked path |
| partial coverage | n/a (a merge never runs) | `partial`, claims excluded from headline analytics |

## 9. Runtime extraction prompt

`claims.CLAIMS_SYSTEM_PROMPT` = `CLAIM_RULES` + `CLAIM_EXAMPLES` +
`CLAIM_FIELDS_SPEC` + the JSON contract; the combined envelope
(`claims.COMBINED_ENVELOPE`) appends the same rules, examples and field spec
after the summary rules. The rules text (verbatim from `claims.py`):

```
You are performing EXTRACTION for a research dataset, not giving investment advice. Use ONLY the transcript and the metadata supplied; never use outside knowledge to fill a field. Return ONE record per ATOMIC claim: split a statement whenever asset, metric, direction, target, recommendation, horizon, condition, catalyst or risk differs. Rules:
- Preserve conditions, negations, hedges (certainty_original), targets and the speaker's exact time wording (horizon_original). Never convert a possibility into a certainty.
- A question is not a forecast (attribution_type=interviewer_question, claim_type=question, is_forward_looking=false).
- A reported analyst/bank/consensus target is a third-party view (speaker_quoting_third_party / speaker_describing_market_consensus) unless the speaker explicitly adopts it.
- "Last year I said X" is retrospective_claim / historical_claim, not a new forecast.
- Owning a stock is portfolio_disclosure, not a recommendation; praise without an explicit buy/sell/hold instruction is opinion with recommendation_action=none. Only an explicit instruction gets buy/sell/hold/etc.
- Keep short-term and long-term views as separate claims. Do not assign a market-wide statement to every company mentioned elsewhere.
- ticker_spoken only when the speaker SAYS the ticker or it appears in the video title; otherwise omit it. Never invent a ticker, a number, a date, a company name, a speaker identity or a timestamp.
- evidence_text must be copied verbatim from the transcript (one or two sentences) and must contain the numbers and condition you extracted.
- Never silently drop an uncertain claim: include it with extraction_confidence=low and a review_reasons entry.
- Omit any field you would set to null or []. Valid JSON only.
```

Ten few-shot examples follow (short-term bearish + long-term bullish; a
question; a quoted analyst target; a retrospective; a company without a
ticker; an explicit buy; praise without a recommendation; a conditional
forecast; a target range; two metrics in one sentence). The whole prompt is
~2k tokens.

## 10. Research state (independent of delivery)

`data/research/research_state.json` → `videos[video_id]` with
`delivery_status`, `research_status` (pending, extracting, complete,
no_claims_found, partial, needs_review, quota_deferred, failed_retryable,
failed_final, superseded), transcript hash/source/stored flag, schema /
normalization / prompt versions, extraction model, attempt count (a quota
deferral never counts), last attempt, next eligible attempt, processed and
failed chunk ids, coverage, failure reason, `active_run_key`,
`superseded_run_keys`, timestamps. Written atomically after each video by
`scraper._persist_research` and by the backfill job. Delivery state
(`seen_videos.json`) is untouched by any research outcome; a valid summary
with failed claims ends as delivery `sent` / research `failed_retryable`
with the watermark advanced exactly as before.

Idempotency: `run_key = hash16:nN:pP:sS`. `store_claims` appends nothing for a
run key already active and skips claim ids already stored; a new run key
becomes active and the previous one is listed as superseded, so
`load_active_claims` counts one version per video while the history stays in
`claims.jsonl`.

## 11. `signals.jsonl` compatibility and who reads what

`data/signals.jsonl` is a **backward-compatible view only**. Rows keep the
legacy shape and are derived from validated claims
(`claims.claims_to_legacy_signals`): only `carries_view` claims contribute;
one entry per asset; `stance` = the shortest-horizon directional claim's
stance with `reduced: "conflicting_horizons:…"` when horizons disagree (the
legacy enum has no "mixed" and no horizon axis); conviction / action /
price-target reductions as before. A failed or suspicious-empty extraction
is written as `signals: null`, never as an empty asset list. That reduction
controls nothing new.

**Canonical data source.** `canonical_claims.load_canonical_claims` is the
one loader: active run per video (superseded runs out), no
`schema_version="legacy"` rows, no cross-chunk repeats, the latest recorded
condition outcome overlaid. `headline_claims` and `view_claims` narrow it
further; `aggregate_views` groups per (asset, horizon bucket) with one
current view per source.

| Scheduled job | Entry point | Data source |
| --- | --- | --- |
| `weekly-pulse.yml` (Mon) | `market_pulse.main` → `_canonical_pulse_inputs` | canonical claims via `load_canonical_claims`; tone from each video's own view claims (`video_tone`); charts from the same views (`pulse_charts.build_chart_data(..., tone=…)`); latest prices for USD targets; track-record weights only when `SCORECARD_RANKINGS=true` |
| `weekly-scorecard.yml` (Fri) | `channel_scorecard.main` → `generate_canonical_scorecard` | canonical claims scored by `research_analytics.scorecard` under `scorecard_pricing`; experimental / unranked by default; conditional-forecast report appended |
| research report (`research_analytics.main`) | `build_report` | canonical claims; quality header, consensus, flips, disclosures, conditional report, scorecard |
| `warm-prices.yml` (Sun) | `warm_prices.needed_ranges` | canonical forecast + benchmark ranges (`canonical_ranges`) **plus** the legacy ranges, so both the production jobs and the legacy consumers find their closes cached |
| `daily-summary.yml` | `scraper.py`, `research_backfill.py --retry` | writes claims, state and the compatibility row; reads neither |

**Legacy consumers that remain** (compatibility only, never the headline
result): `market_pulse.build_pulse` / `aggregate_assets` (run only with
`PULSE_DATA_SOURCE=legacy`), `channel_scorecard.evaluate` /
`build_scorecard` / `generate_scorecard` (`--legacy`), `pulse_charts` data
prep over legacy records, `warm_prices` legacy ranges, `ticker_resolver`
(learns tickers from asset names in the legacy rows), and
`research_backfill --import-legacy` (turns the old rows into review-required
legacy claims). Nothing in `canonical_claims` imports `load_signals`
(asserted by `tests/test_canonical_analytics.py`).

## 12. Migration and backfill

`python research_backfill.py --import-legacy` imports every legacy row's
assets as `schema_version="legacy"` claims: no evidence (`evidence_text:
null`, not fabricated), `review_required=True`, `testable=False`, excluded from
evidence-dependent analytics, retained for history; idempotent by claim id.
`--retry` reprocesses pending/failed/deferred/partial videos from stored
transcripts only (a missing transcript stays `failed_retryable:
transcript_not_stored`; re-fetching remains the daily job's budgeted
decision), at most `RESEARCH_BACKFILL_MAX_VIDEOS` per run, stopping on quota.
`--reprocess` re-extracts videos whose active run predates the current
versions and supersedes the old run. The daily workflow runs `--retry` after
the scraper and commits `data/research/` and `data/transcripts/`.

## 13. Analytics

`research_analytics.py`: quality header (unchanged), descriptive counts,
consensus = latest **view** claim per source per asset per horizon bucket
(`claims.carries_view`; buckets never merged; `net_stance = (bull -
bear)/(bull + bear)` only when the denominator is non-zero), flips (same
source/asset/bucket), the portfolio-disclosure report, the conditional-
forecast report, and the scorecard below.

### 13.1 Scorecard methodology (`scorecard_pricing.py`)

Status: **experimental**. Source rankings are disabled unless
`SCORECARD_RANKINGS=true`; the block is labelled `EXPERIMENTAL — unranked`
and every source is listed alphabetically with `(unranked)`.

*Price data.* Twelve Data `time_series` is requested with
`adjust=$PRICE_ADJUSTMENT` (`splits` by default = split-adjusted, the
minimum accepted; `all` = dividends folded in, recorded as
`total_return_adjusted`). `price_cache` stores the adjustment per symbol and
never serves a series fetched under another adjustment. Every scored claim
records `price_provider`, `adjustment_type`, `corporate_action_status`,
`currency`, `requested_start/end`, the entry's requested date and resolved
trading date, and the evaluation's. A series whose adjustment is unknown or
unadjusted is refused (`unadjusted_or_unknown_prices`), so a fetcher without
`price_provenance` cannot score anything.

*Publication-time rule* (`entry_point`). The instrument's exchange is
resolved from the symbol suffix (`EXCHANGES`: XNYS with full NYSE holiday
rules; XLON/XETR/XAMS/XTKS/XHKG/XKRX/XSHG/XSHE with weekday-only calendars,
`calendar_confidence=weekdays_only`; crypto = continuous). The publication
instant is placed in the exchange's timezone against the session:
before open / during session → that day's close; after close, weekend,
holiday → the next trading day's close ("next-close" convention); a
publication with no time of day → the next trading day's close
(conservative). A close dated the publication day is never used when the
video went up after that close, and a close the series carries on a holiday
is never used. Crypto: no session; entry = the close of the publication's
UTC day (the first daily close after publication). Evaluation = the first
trading-day close on or after `forecast_end_date`; both within 5 days.

*Benchmarks* (`resolve_benchmark`). Per claim, from (asset type, exchange
country, sector) through `BENCHMARKS` (sector ETFs for US sectors, SPY for
US stocks/ETFs, BTC for crypto other than BTC itself; `BENCHMARKS_JSON`
overrides). No defensible benchmark (non-US venue, unresolved exchange,
self-benchmark) → raw performance only, `excess_return = null`,
`benchmark_method` says why. `rankable_sources` ranks only sources that
share one benchmark method and meet `MIN_SCORECARD_SAMPLE`, and only when
rankings are enabled.

*Unresolved exchange* → excluded (`unresolved_exchange`). *Conditional
forecasts* → excluded from the unconditional scorecard (`conditional`);
scored only with `include_conditional=True` and `condition_status == met`.

### 13.2 Conditional-forecast model

`testability_type`: `unconditional_testable` (forward-looking, resolved
instrument, metric/direction, dated horizon, located evidence, full
coverage, no condition), `conditional_testable` (the same, with an
objectively observable condition), `not_testable`. A condition is
observable when `claims.condition_observability` recognises it: a price
level (`daily_close`), a policy-rate decision (`fomc_decisions`), a macro
release (`official_statistics_release`), an earnings release
(`company_filings`), or a dated public event (`public_announcements`).
Anything else is `subjective` and blocks testability with the specific
issue `condition_not_objectively_observable` — the existence of a condition
is never by itself the reason. Fields: `condition_text`, `condition_status`
(met / not_met / partially_met / unknown / not_evaluated),
`condition_evaluation_date`, `condition_evidence`, `condition_data_source`.
Outcomes are appended to `data/research/condition_evaluations.jsonl`
(`research_state.record_condition_evaluation`) and overlaid by the loader;
the latest wins. The separate report:
`research_analytics.conditional_forecast_report` (in the research report
and the scorecard message). Conditional forecasts stay out of unconditional
rankings by default.

### 13.3 Suspicious-empty detection (`claims.suspicious_empty_check`)

Deterministic, high-recall, conservative. The transcript is split into
sentences (unpunctuated caption runs longer than 40 words are scanned in
25-word windows). A sentence is skipped when it ends with "?" or carries an
educational / historical / promotional marker (historically, for example,
let's say, imagine, means that, on average, last year, back in, in 20xx,
returned, was/were, sponsor, use code, …). A remaining sentence must name an
identifiable asset (curated alias, learned ticker, ticker-like token when
the transcript is not mostly upper-case, or "the stock"/"bitcoin"/…). Then:

- **strong**: recommendation wording (I'm buying, I'd sell, this is a buy,
  avoid it, stay away, load up, take profits, …), or prediction wording
  (will, expect, going to, could hit, should reach, price target, by end of,
  next year, over the next, heading to, will double/crash/rally, …) together
  with a number/percentage/currency, bullish/bearish wording, or a
  directional verb (fall, rally, reach, hit, …);
- **moderate**: bullish/bearish/overvalued/undervalued/upside/downside
  wording, or prediction wording alone.

Suspicious = at least one strong sentence or two moderate ones. On a
suspicious empty result: combined path → `failed_retryable` /
`suspicious_empty_extraction` (a standalone pass follows); standalone path
→ `needs_review` / `suspicious_empty_extraction`; `signals: null`; the
matching sentences are logged in the run's warnings. Never
`no_claims_found`.

### 13.4 Evidence timestamps

`NormalizedTranscript.cues` keeps every caption cue's normalized span and
seconds; `seconds_at(offset)` returns the cue containing the offset (an
untimed cue inherits the previous timestamp), `end_seconds_at` the next
cue's start. Claims in one segment therefore carry their own cue times
(15 / 19 / 23 s in the example below, not the segment's 12 s). Plain-text
transcripts keep null.

### 13.5 Partial-cache versioning (`partial_cache.py`)

Both chunked paths store one record per chunk under
`data/research/partials/<hash>/<task>-<chunk>.json` with the full key —
task type (`summary_notes` vs `research_claims`), transcript hash,
normalization version, `CHUNKING_VERSION`, chunk id and boundaries, prompt
version, schema version, provider/model policy — and a record is reused only
when every key field matches the current run; otherwise it is ignored and
overwritten.

### 13.6 Extraction-quality evaluation (`claims_eval.py`)

`python claims_eval.py --fixtures evals/claims [--live] [--report out.json]`.
Not part of CI. Offline mode replays each fixture's stored `model_output`
through the real deterministic pipeline and scores it against hand-labelled
`expected_claims`; live mode (needs `--live` AND `CLAIMS_EVAL_LIVE=1` AND
provider credentials, and warns that it spends quota) calls the current
prompt. Metrics: atomic precision/recall, numerical-value, evidence-
grounding, attribution, entity-resolution, stance, horizon and
recommendation accuracy, false no-claims rate, duplicate rate; errors
grouped by category with representative false positives/negatives.
Matching is stable (evidence overlap or same segment, same asset, claim-type
family, direction/target/bucket where labelled), never exact JSON. The
fixture format and expansion steps are in `evals/claims/README.md`. The
five shipped fixtures are hand-written caption-style segments with
hand-written model outputs; **no live model quality has been measured** by
this work.

## 14. Tests

`./scripts/check.sh` → 575 passed. Review corrections:
`test_portfolio_disclosures.py` (1), `test_truncation_fallback.py` (2, 10),
`test_canonical_analytics.py` (3), `test_scorecard_pricing.py` (4),
`test_conditional_forecasts.py` (5), `test_attribution_coreference.py` (6),
`test_suspicious_empty.py` (7), `test_claims_eval.py` (8),
`test_evidence_timestamps.py` (9). Original coverage: 511 passed; Coverage of the required list: full
transcript (1–7: `test_summarizer.py` new tests, `test_token_budget.py`,
`test_model_capabilities.py`), chunking (8–12: `test_transcript_normalize.py`,
`test_summarizer_chunked.py`, `test_signals.py`), normalization (13–18:
`test_transcript_normalize.py`), claims (19–30: `test_claims.py`), state and
delivery (31–40: `test_signals.py`, `test_scraper.py`,
`test_research_state.py`, existing Telegram/analytics suites), analytics
(41–46: `test_research_analytics.py`).

## 15. End-to-end example (offline, model mocked; regenerated 2026-09-05)

Same raw excerpt as before (five caption cues, 12–31 s), published
`2026-09-01T14:00:00+00:00`. The model output is a stored candidate list that
deliberately mislabels two claims: "I own Tesla" as a neutral stance with
`recommendation_action=buy`, and the Apple question as a forecast.

```
VALIDATION WARNINGS:
  claim 2: 'the stock' resolved to 'Nvidia' from local context
  claim 3: question reclassified from forecast
  claim 4: recommendation 'buy' not stated in evidence; set to none
  claim 4: ownership-only statement reclassified from stance to portfolio_disclosure

{"claim_id": "clm_a4babdd5b0cde12cf5b0", "segment_id": "5f3cb3c79a9f-s001", "attribution_type": "speaker_personal_view", "host_position": "not_applicable", "claim_type": "forecast", "is_forward_looking": true, "subject_mention": "Nvidia", "canonical_entity_name": "Nvidia", "entity_resolution_method": "explicit_mention", "entity_resolution_confidence": 1.0, "ticker": "NVDA", "stance": "bearish", "recommendation_action": "none", "forecast_direction": "decrease", "target_value": null, "horizon_bucket": "short", "forecast_end_date": "2026-12-01", "portfolio_disclosure": "not_stated", "evidence_text": "I expect Nvidia to fall over the next three months", "evidence_start_seconds": 15, "evidence_end_seconds": 23, "testable": true, "testability_type": "unconditional_testable", "testability_issues": [], "review_required": false, "review_reasons": []}
{"claim_id": "clm_f0c72e87573bba0df015", "segment_id": "5f3cb3c79a9f-s001", "attribution_type": "speaker_personal_view", "host_position": "not_applicable", "claim_type": "stance", "is_forward_looking": true, "subject_mention": "Nvidia", "canonical_entity_name": "Nvidia", "entity_resolution_method": "local_coreference", "entity_resolution_confidence": 0.7, "ticker": "NVDA", "stance": "bullish", "recommendation_action": "none", "forecast_direction": null, "target_value": null, "horizon_bucket": "long", "forecast_end_date": "2031-09-01", "portfolio_disclosure": "not_stated", "evidence_text": "I remain bullish over five years", "evidence_start_seconds": 19, "evidence_end_seconds": 23, "testable": false, "testability_type": "not_testable", "testability_issues": ["missing_metric", "missing_direction"], "review_required": false, "review_reasons": []}
{"claim_id": "clm_4dffdb532129c4276e69", "segment_id": "5f3cb3c79a9f-s001", "attribution_type": "speaker_quoting_third_party", "host_position": "rejected", "claim_type": "third_party_view", "is_forward_looking": true, "subject_mention": "the stock", "canonical_entity_name": "Nvidia", "entity_resolution_method": "local_coreference", "entity_resolution_confidence": 0.7, "ticker": "NVDA", "stance": "not_applicable", "recommendation_action": "none", "forecast_direction": null, "target_value": 200.0, "horizon_bucket": "unspecified", "forecast_end_date": null, "portfolio_disclosure": "not_stated", "evidence_text": "Goldman expects the stock to reach $200 but that is their call not mine", "evidence_start_seconds": 23, "evidence_end_seconds": 27, "testable": false, "testability_type": "not_testable", "testability_issues": ["missing_horizon"], "review_required": false, "review_reasons": []}
{"claim_id": "clm_8f6a5e10a6abf15703dc", "segment_id": "5f3cb3c79a9f-s002", "attribution_type": "interviewer_question", "host_position": "not_applicable", "claim_type": "question", "is_forward_looking": false, "subject_mention": "Apple", "canonical_entity_name": "Apple", "entity_resolution_method": "explicit_mention", "entity_resolution_confidence": 1.0, "ticker": "AAPL", "stance": "not_applicable", "recommendation_action": "none", "forecast_direction": "decrease", "target_value": null, "horizon_bucket": "unspecified", "forecast_end_date": null, "portfolio_disclosure": "not_stated", "evidence_text": "could Apple fall 30 percent from here?", "evidence_start_seconds": 27, "evidence_end_seconds": null, "testable": false, "testability_type": "not_testable", "testability_issues": [], "review_required": false, "review_reasons": []}
{"claim_id": "clm_02ee7a8f806656ddd46b", "segment_id": "5f3cb3c79a9f-s002", "attribution_type": "speaker_personal_view", "host_position": "not_applicable", "claim_type": "portfolio_disclosure", "is_forward_looking": false, "subject_mention": "Tesla", "canonical_entity_name": "Tesla", "entity_resolution_method": "explicit_mention", "entity_resolution_confidence": 1.0, "ticker": "TSLA", "stance": "not_applicable", "recommendation_action": "none", "forecast_direction": null, "target_value": null, "horizon_bucket": "unspecified", "forecast_end_date": null, "portfolio_disclosure": "owns_unspecified", "evidence_text": "I own Tesla by the way", "evidence_start_seconds": 27, "evidence_end_seconds": null, "testable": false, "testability_type": "not_testable", "testability_issues": [], "review_required": false, "review_reasons": []}

COMPATIBILITY SIGNAL (signals.jsonl row 'signals' field — the legacy VIEW, read by no canonical analytics):
{"assets": [{"name": "Nvidia", "ticker": "NVDA", "type": "stock", "stance": "bearish", "conviction": "medium", "action": "none", "catalysts": [], "price_target": null, "horizon": "short", "claim_ids": ["clm_a4babdd5b0cde12cf5b0", "clm_f0c72e87573bba0df015"], "reduced": "conflicting_horizons:short=bearish,long=bullish"}], "market_sentiment": "mixed", "topics": [], "derived_from": "claims"}

CONSENSUS (canonical, one view per source/asset/horizon):
('NVDA', 'short') {'bullish': 0, 'bearish': 1, 'neutral': 0, 'sources': 1, 'net_stance': -1.0}
('NVDA', 'long')  {'bullish': 1, 'bearish': 0, 'neutral': 0, 'sources': 1, 'net_stance': 1.0}

PULSE VIEWS (canonical_claims.aggregate_views — what the weekly pulse renders):
('NVDA', 'short') {'label': 'NVDA [short]', 'bull': 0, 'bear': 1, 'neutral': 0, 'mentions': 1}
('NVDA', 'long')  {'label': 'NVDA [long]',  'bull': 1, 'bear': 0, 'neutral': 0, 'mentions': 1}

PORTFOLIO DISCLOSURES (separate report):
[{"source": "Demo Channel", "asset": "TSLA", "ticker": "TSLA", "position": "owns_unspecified", "date": "2026-09-01", "evidence": "I own Tesla by the way", "claim_id": "clm_02ee7a8f806656ddd46b", "review_required": false}]

SUSPICIOUS-EMPTY CHECK on this transcript: suspicious=True
  strong: ["Host: welcome back everyone today we talk nvidia and honestly I expect Nvidia to fall over the next three months, but I remain bullish over"]
EMPTY MODEL RESULT ({"claims": []}) -> combined path: failed_retryable / suspicious_empty_extraction, signals: null
                                     -> standalone path: needs_review / suspicious_empty_extraction

TRUNCATED COMBINED CALL (complete() -> TRUNCATED_SENTINEL after escalation)
  summarize_with_signals -> (None, {status: failed_retryable, failure_reason: combined_output_truncated, retry_separately: true})
  scraper -> summary-only call -> delivered -> watermark advanced -> extract_research(prefer_chunked=True)
  (tests/test_truncation_fallback.py drives this through main() with finish_reason=length responses)
```

What changed against the previous example:

- **"I own Tesla" is not a neutral stance.** It is a `portfolio_disclosure`
  with `stance=not_applicable`; there is no `('TSLA', 'unspecified')`
  consensus row any more, no Tesla entry in the compatibility signal, and the
  disclosure appears only in the disclosure report.
- **Short and long term stay separate** in consensus and in the pulse
  (`NVDA [short]` bearish, `NVDA [long]` bullish); the legacy row's single
  `bearish` with `reduced: conflicting_horizons` shapes none of it.
- **"The stock" resolves to Nvidia** through local coreference
  (`entity_resolution_method=local_coreference`, 0.7, ticker NVDA), the host's
  "that is their call not mine" is recorded as `host_position=rejected`, and
  the claim needs no review — consistent with § 7.1.
- **Timestamps are per cue**: 15 s, 19 s, 23 s inside one segment that
  starts at 12 s.
- **A suspicious empty extraction enters review** instead of
  `no_claims_found`.
- **Canonical analytics never read the reduced legacy signal.**

## 16. Limitations

- Segment categories are keyword heuristics; they gate exclusion (sponsor,
  disclaimer, intro) and never decide meaning. The LLM extractor does.
- Evidence timestamps are cue-level (source cues carry no per-word times);
  Supadata is fetched as plain text, so most segments have no seconds at
  all — `missing_timestamps` is reported in the quality header.
- Non-US exchange calendars are weekday-only (`calendar_confidence=
  weekdays_only`); a local holiday there shifts an entry by a day at most,
  and the scored record says which calendar was used.
- The suspicious-empty check is a keyword scan: it is tuned to be quiet on
  explainers and loud on forecasts, but it cannot read meaning. A flagged
  video costs one standalone extraction request or a review, never a
  fabricated claim.
- Local coreference only resolves to names the curated tables or the
  verified learned map already know; a company the dataset has never seen
  stays unresolved even when the context is unambiguous.
- The evaluation fixtures are hand-authored; the harness measures live
  model quality only when run with `--live` and `CLAIMS_EVAL_LIVE=1`, which
  this work did not do.
- `countTokens` measures the native request shape; the OpenAI-compatible
  layer may add a few tokens of scaffolding. The 2,048-token margin covers
  it. The 250k TPM value is this project's measured free-tier limit, not a
  published constant; paid tiers should set `GEMINI_FREE_TIER_TPM=0`.
- Chunked summarization costs one request per chunk plus a merge, and both
  chunked paths draw on the summary model chain. The quota pre-check keeps
  a run from starting what it cannot finish, but a very long backlog of
  over-limit videos will defer.
- Raw transcripts are committed gzip'd (roughly 10–30 KB each); at the
  current publishing rate that is on the order of 100 MB a year in git
  history. `PERSIST_TRANSCRIPTS=false` turns storage off (research retries
  then need a re-fetch under the normal budget).
- Legacy rows cannot gain evidence retroactively; the compatibility view for
  new videos reduces information by design and says so in `reduced`.
- The claim's `sector` comes from the model's wording and is only used to
  pick a sector benchmark when it matches a `BENCHMARKS` key; `exchange`
  on the claim stays null (the scorecard resolves it from the symbol at
  scoring time and records it on the scored row).
