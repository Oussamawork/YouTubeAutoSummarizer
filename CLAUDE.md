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
- `gemini_quota.py` — per-model daily free-tier accounting in
  `data/gemini_usage.json` (20 requests/day per model, resets midnight Pacific);
  also classifies a 429 as a per-day or per-minute limit, which decides whether
  a model is retired for the day or merely retried.
- `signals.py` — LLM extraction of structured market signals from summaries
  (opt-in via `MARKET_SIGNALS`; appends to `data/signals.jsonl`).
- `market_pulse.py` — weekly aggregation over `data/signals.jsonl` (top assets,
  consensus flips, new-on-radar) sent to Telegram by `weekly-pulse.yml`.
- `pulse_charts.py` — the pulse's companion PNG charts (consensus board, flip
  slope, weekly tone, bull-bear spread line, agreement-vs-attention map,
  target-upside ladder), styled for non-technical readers and sent as a Telegram
  photo album after the text pulse. Data prep is pure/testable; matplotlib
  imports lazily and every chart is best-effort — chart failures never block the
  text pulse. A chart with too little history to mean anything is either skipped
  (`MIN_SPREAD_WEEKS`) or labels itself as early days (`MATURE_SPREAD_WEEKS`).
- `channel_scorecard.py` — Friday per-channel accuracy scorecard: directional
  calls vs Stooq daily prices at 7/30-day horizons (`weekly-scorecard.yml`).
  Prices come from **Twelve Data** when `TWELVEDATA_API` is set (free tier: 800
  requests/day, 8/min — `_twelvedata_pace` respects the per-minute budget so a
  scorecard run doesn't turn into 429s). Internal symbols stay Stooq-shaped
  (`nvda.us`, `btcusd`) and are translated per provider by `twelvedata_symbol`.
  **Stooq is the keyless legacy path and is unusable server-side** (verified
  2026-08-17): the default UA gets 404, a browser UA gets a JavaScript challenge
  instead of CSV. That block silently disabled implied upside, track-record
  weighting, the scorecard and the price-target chart from the first scheduled
  run (2026-07-27) until Twelve Data replaced it. `market_pulse.fetch_latest_prices`
  logs one loud "no prices for any ticker" line when the source is down — the
  first thing to check when prices look wrong.
- `sendToTelegram.py` — Telegram delivery (HTML, with plain-text fallback); digest builder.
- `helpers.py` — channel file parsing (`<id|@handle> [digest] [max=N] [only=a,b]` per
  line; handles resolved at run time by `scraper.resolve_channel_handle`; `only=`
  is a whole-word title filter applied before any transcript fetch), dedup
  state (v2 schema + v1 migration), summary cleaning.
- `log.py` — colored logging helpers.

## Dedup state model
`seen_videos.json`: `channels` maps channel-id → watermark (`last_video_id`,
`last_published`); videos published after the watermark are candidates, oldest
first, capped at `MAX_VIDEOS_PER_RUN` (0 = no cap, the default). `pending` maps
video-id → retry record for
deferred videos (captions not up yet → retried until `NO_TRANSCRIPT_MAX_ATTEMPTS`
*and* `NO_TRANSCRIPT_MIN_HOURS` are both exceeded, and no more often than
`PENDING_RETRY_MIN_HOURS`; LLM quota exhausted). Deciding a video advances the
watermark; deferring does not.

## Design docs
- `docs/tdd-transcript-budget.md` — measured findings on transcript-credit
  efficiency (≈2.9 credits per delivered summary), the budget/pacing design and
  its critique, options considered, and the sequenced next steps. Read before
  changing `transcript.py` budget logic or channel caps.
- `docs/tdd-gemini-transcripts.md` — the August 2026 two-day outage, the real
  per-model free-tier limits (20 requests/day, not 1,500), measured cost of a
  Gemini video transcript, and why the transcript and summary model pools must
  stay disjoint. Read before changing either model list.

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
