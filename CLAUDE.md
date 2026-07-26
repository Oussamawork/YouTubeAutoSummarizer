# Developer guide (for Claude Code and humans)

YouTube auto-summarizer: fetches each channel's new videos, gets their transcripts,
summarizes them with an LLM, and posts the summaries to Telegram. Runs daily via
GitHub Actions (`.github/workflows/daily-summary.yml`); tests run on every PR
(`.github/workflows/ci.yml`).

## Module map
- `scraper.py` — entry point / orchestration; RSS feed (primary) + YouTube Data API
  (fallback); candidate selection against the dedup watermark; per-video outcomes
  (send / defer / give up); digest mode; `--video-url` on-demand path.
- `transcript.py` — transcript fetch (Supadata → youtube-transcript-api fallback);
  meters Supadata free-tier credits in `data/supadata_usage.json`, rotates across
  multiple keys, and paces a monthly budget over the days left in the month
  (over-budget videos defer silently via the `budget_exhausted` flag).
- `summarizer.py` — provider-agnostic LLM summarization (OpenAI-compatible API);
  also exposes `complete()` for generic calls over the same provider chain.
- `signals.py` — LLM extraction of structured market signals from summaries
  (opt-in via `MARKET_SIGNALS`; appends to `data/signals.jsonl`).
- `market_pulse.py` — weekly aggregation over `data/signals.jsonl` (top assets,
  consensus flips, new-on-radar) sent to Telegram by `weekly-pulse.yml`.
- `channel_scorecard.py` — Friday per-channel accuracy scorecard: directional
  calls vs Stooq daily prices at 7/30-day horizons (`weekly-scorecard.yml`).
- `sendToTelegram.py` — Telegram delivery (HTML, with plain-text fallback); digest builder.
- `helpers.py` — channel file parsing (`<id|@handle> [digest] [max=N]` per line; handles
  resolved at run time by `scraper.resolve_channel_handle`), dedup
  state (v2 schema + v1 migration), summary cleaning.
- `log.py` — colored logging helpers.

## Dedup state model
`seen_videos.json`: `channels` maps channel-id → watermark (`last_video_id`,
`last_published`); videos published after the watermark are candidates, oldest
first, capped at `MAX_VIDEOS_PER_RUN`. `pending` maps video-id → retry record for
deferred videos (captions not up yet → up to `NO_TRANSCRIPT_MAX_ATTEMPTS` runs;
LLM quota exhausted). Deciding a video advances the watermark; deferring does not.

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
