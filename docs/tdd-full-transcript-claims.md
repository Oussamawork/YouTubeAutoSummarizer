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

## 8. Combined envelope outcomes

| Response | Delivery | Research |
| --- | --- | --- |
| valid summary + valid claims | sent | `complete` (or `needs_review` when every claim is flagged) |
| valid summary + malformed `claims` | sent | `failed_retryable`, reason `malformed_claims`, `signals: null`, retried later |
| valid summary + `claims: []` | sent | `no_claims_found` (after validation), `signals: {"assets": []}` |
| malformed envelope / no summary | fallback: plain summary call, then separate `extract_research` | as that call decides |
| output truncated (`finish_reason=length`) | escalate, then `TRUNCATED_SENTINEL` → deferred, never half-delivered | untouched |
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

## 11. `signals.jsonl` compatibility

Rows keep the legacy shape and are now derived from validated claims
(`claims.claims_to_legacy_signals`), with `research_status` and
`coverage_status` added. Reduction rules: only headline-eligible own-view
claims contribute; one entry per asset; `stance` = the shortest-horizon
directional claim's stance, with `reduced: "conflicting_horizons:…"` and every
`claim_id` listed when horizons disagree (the legacy enum has no "mixed");
`conviction` from `certainty_level`; `action` from `recommendation_action`
(add/accumulate→buy, reduce/short→sell, avoid→none); `price_target` from an
absolute USD target or a range midpoint (`reduced: "range_midpoint"`);
`horizon` = bucket; `market_sentiment` from the bullish/bearish mix. A failed
extraction is written as `signals: null` (as before) — never as an empty
asset list — and a later successful retry appends a `backfilled` row for the
same `video_id`. Existing consumers (`signals_data`, `market_pulse`,
`channel_scorecard`, `pulse_charts`, `warm_prices`) are unchanged and
tested against the derived shape.

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

`research_analytics.py`: quality header (discovered / included / transcribed /
fully processed / partial / awaiting retry; claims, forward-looking,
testable, review-required, unresolved entities, missing timestamps; sources;
date range; the universe sentence), descriptive counts with distinct units,
consensus = latest headline claim per source per asset per horizon bucket
(buckets never merged; `net_stance = (bull - bear)/(bull + bear)` only when
the denominator is non-zero), flips (same source/asset/bucket, both claims
and both evidences kept, elapsed days, changed catalysts/risks), scorecard
(matured, testable, evidence-backed, resolved instrument; entry = first close
on/after publication, evaluation = first close on/after the end date, UTC,
5-day lag, unadjusted provider closes, SPY benchmark, conditional forecasts
excluded; direction / target / raw / excess / MFE / MAE kept separate; sources
under `MIN_SCORECARD_SAMPLE` shown but unranked). `market_pulse` appends the
quality header after the text pulse.

## 14. Tests

`./scripts/check.sh` → 511 passed. Coverage of the required list: full
transcript (1–7: `test_summarizer.py` new tests, `test_token_budget.py`,
`test_model_capabilities.py`), chunking (8–12: `test_transcript_normalize.py`,
`test_summarizer_chunked.py`, `test_signals.py`), normalization (13–18:
`test_transcript_normalize.py`), claims (19–30: `test_claims.py`), state and
delivery (31–40: `test_signals.py`, `test_scraper.py`,
`test_research_state.py`, existing Telegram/analytics suites), analytics
(41–46: `test_research_analytics.py`).

## 15. End-to-end example (offline, model mocked)

```
RAW EXCERPT:
00:00:12 --> 00:00:15
Host: welcome back everyone today we talk nvidia
00:00:15 --> 00:00:19
today we talk nvidia and honestly I expect Nvidia to fall over the next
00:00:19 --> 00:00:23
over the next three months, but I remain bullish over five years
00:00:23 --> 00:00:27
Goldman expects the stock to reach $200 but that is their call not mine
00:00:27 --> 00:00:31
Host: could Apple fall 30 percent from here? I own Tesla by the way

NORMALIZED TEXT:
Host: welcome back everyone today we talk nvidia and honestly I expect Nvidia to fall over the next three months, but I remain bullish over five years Goldman expects the stock to reach $200 but that is their call not mine could Apple fall 30 percent from here? I own Tesla by the way

QUALITY FLAGS: {'caption_overlap_removed': 2, 'duplicate_cues_removed': 0, 'unintelligible_markers': 0, 'lines_joined': 4}
SEGMENT: {
 "segment_id": "5f3cb3c79a9f-s001",
 "video_id": "demo01",
 "sequence_number": 1,
 "start_seconds": 12,
 "end_seconds": 27,
 "start_character": 0,
 "end_character": 223,
 "speaker": "Host",
 "normalized_text": "Host: welcome back everyone today we talk nvidia and honestly I expect Nvidia to fall over the next three months, but I remain bullish over five years Goldman expects the stock to reach $200 but that is their call not mine ",
 "primary_category": "forecast",
 "secondary_tags": [
  "company_analysis",
  "introduction_or_outro",
  "price_target"
 ],
 "quality_flags": [],
 "excluded_from_headline": false,
 "exclusion_reason": null
}

VALIDATION RESULT: status=complete warnings=['claim 3: question reclassified from forecast', "claim 4: recommendation 'buy' not stated in evidence; set to none"]
{"claim_id": "clm_abf2e186f546b3dcc1ef", "segment_id": "5f3cb3c79a9f-s001", "attribution_type": "speaker_personal_view", "claim_type": "forecast", "is_forward_looking": true, "subject_mention": "Nvidia", "ticker": "NVDA", "ticker_source": "curated_mapping", "entity_resolution_status": "confirmed", "stance": "bearish", "recommendation_action": "none", "forecast_direction": "decrease", "target_value": null, "horizon_original": "over the next three months", "horizon_bucket": "short", "forecast_end_date": "2026-12-01", "certainty_level": "medium", "portfolio_disclosure": "not_stated", "evidence_text": "I expect Nvidia to fall over the next three months", "evidence_start_character": 62, "evidence_start_seconds": 12, "testable": true, "testability_issues": [], "review_required": false, "review_reasons": [], "coverage_status": "full"}
{"claim_id": "clm_16946afa67c8e1858536", "segment_id": "5f3cb3c79a9f-s001", "attribution_type": "speaker_personal_view", "claim_type": "stance", "is_forward_looking": true, "subject_mention": "Nvidia", "ticker": "NVDA", "ticker_source": "curated_mapping", "entity_resolution_status": "confirmed", "stance": "bullish", "recommendation_action": "none", "forecast_direction": null, "target_value": null, "horizon_original": "over five years", "horizon_bucket": "long", "forecast_end_date": "2031-09-01", "certainty_level": "medium", "portfolio_disclosure": "not_stated", "evidence_text": "I remain bullish over five years", "evidence_start_character": 118, "evidence_start_seconds": 12, "testable": false, "testability_issues": ["missing_metric", "missing_direction"], "review_required": false, "review_reasons": [], "coverage_status": "full"}
{"claim_id": "clm_4669b1c48af81a2108c2", "segment_id": "5f3cb3c79a9f-s001", "attribution_type": "speaker_quoting_third_party", "claim_type": "third_party_view", "is_forward_looking": true, "subject_mention": "the stock", "ticker": null, "ticker_source": "unresolved", "entity_resolution_status": "unresolved", "stance": "not_applicable", "recommendation_action": "none", "forecast_direction": null, "target_value": 200.0, "horizon_original": null, "horizon_bucket": "unspecified", "forecast_end_date": null, "certainty_level": "not_stated", "portfolio_disclosure": "not_stated", "evidence_text": "Goldman expects the stock to reach $200", "evidence_start_character": 151, "evidence_start_seconds": 12, "testable": false, "testability_issues": ["missing_horizon", "unresolved_entity"], "review_required": false, "review_reasons": [], "coverage_status": "full"}
{"claim_id": "clm_b47e2fe4a5c69afea043", "segment_id": "5f3cb3c79a9f-s002", "attribution_type": "interviewer_question", "claim_type": "question", "is_forward_looking": false, "subject_mention": "Apple", "ticker": "AAPL", "ticker_source": "curated_mapping", "entity_resolution_status": "confirmed", "stance": "not_applicable", "recommendation_action": "none", "forecast_direction": "decrease", "target_value": null, "horizon_original": null, "horizon_bucket": "unspecified", "forecast_end_date": null, "certainty_level": "not_stated", "portfolio_disclosure": "not_stated", "evidence_text": "could Apple fall 30 percent from here?", "evidence_start_character": 223, "evidence_start_seconds": 27, "testable": false, "testability_issues": [], "review_required": false, "review_reasons": [], "coverage_status": "full"}
{"claim_id": "clm_441e069643c80c31daa2", "segment_id": "5f3cb3c79a9f-s002", "attribution_type": "speaker_personal_view", "claim_type": "portfolio_disclosure", "is_forward_looking": false, "subject_mention": "Tesla", "ticker": "TSLA", "ticker_source": "curated_mapping", "entity_resolution_status": "confirmed", "stance": "neutral", "recommendation_action": "none", "forecast_direction": null, "target_value": null, "horizon_original": null, "horizon_bucket": "unspecified", "forecast_end_date": null, "certainty_level": "not_stated", "portfolio_disclosure": "owns_unspecified", "evidence_text": "I own Tesla by the way", "evidence_start_character": 262, "evidence_start_seconds": 27, "testable": false, "testability_issues": [], "review_required": false, "review_reasons": [], "coverage_status": "full"}

COMPATIBILITY SIGNAL (signals.jsonl row 'signals' field):
{"assets": [{"name": "Nvidia", "ticker": "NVDA", "type": "other", "stance": "bearish", "conviction": "medium", "action": "none", "catalysts": [], "price_target": null, "horizon": "short", "claim_ids": ["clm_abf2e186f546b3dcc1ef", "clm_16946afa67c8e1858536"], "reduced": "conflicting_horizons:short=bearish,long=bullish"}, {"name": "Tesla", "ticker": "TSLA", "type": "other", "stance": "neutral", "conviction": "unspecified", "action": "none", "catalysts": [], "price_target": null, "horizon": "unspecified", "claim_ids": ["clm_441e069643c80c31daa2"], "reduced": null}], "market_sentiment": "mixed", "topics": [], "derived_from": "claims"}

ANALYTICS ROWS (consensus, one view per source/asset/horizon bucket):
('NVDA', 'short') {'bullish': 0, 'bearish': 1, 'neutral': 0, 'sources': 1, 'net_stance': -1.0}
('NVDA', 'long') {'bullish': 1, 'bearish': 0, 'neutral': 0, 'sources': 1, 'net_stance': 1.0}
('TSLA', 'unspecified') {'bullish': 0, 'bearish': 0, 'neutral': 1, 'sources': 1, 'net_stance': None}
```

## 16. Limitations

- Segment categories are keyword heuristics; they gate exclusion (sponsor,
  disclaimer, intro) and never decide meaning. The LLM extractor does.
- Evidence timestamps are segment-level (source cues carry no per-word
  times); Supadata is fetched as plain text, so most segments have no
  seconds at all — `missing_timestamps` is reported in the quality header.
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
- Sector/exchange/benchmark ticker fields are carried but not resolved
  (null) — no curated source exists for them yet.
