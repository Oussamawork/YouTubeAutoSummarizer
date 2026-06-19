# Developer guide (for Claude Code and humans)

YouTube auto-summarizer: fetches each channel's latest video, gets its transcript,
summarizes it with an LLM, and posts the summary to Telegram. Runs daily via GitHub
Actions (`.github/workflows/daily-summary.yml`).

## Module map
- `scraper.py` — entry point / orchestration; YouTube Data API; channel-entry resolution.
- `transcript.py` — transcript fetch (Supadata → youtube-transcript-api fallback).
- `summarizer.py` — provider-agnostic LLM summarization (OpenAI-compatible API).
- `sendToTelegram.py` — Telegram delivery (HTML, with plain-text fallback).
- `helpers.py` — channel-id file reading, dedup state, summary cleaning.
- `log.py` — colored logging helpers.

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
