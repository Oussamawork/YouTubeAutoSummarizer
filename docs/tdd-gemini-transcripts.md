# TDD — Gemini video transcripts and per-model quota

**Status:** implemented 2026-08-13. Numbers below are measured against the live
API with this project's key, not estimated.
**Scope:** what to do when Supadata has no credits, and how the free Gemini tier
is actually budgeted.

---

## 1. What happened

Supadata's credit pools were spent on **2026-08-11**. All three keys answered
`429 {"error":"limit-exceeded"}`, `youtube-transcript-api` is IP-blocked from
GitHub runners, so every video deferred with `budget_deferred` and **no summary
was delivered for two days**. Every scheduled run still exited *green* — a
deferral is a normal, successful outcome — so nothing surfaced the outage.

At the point of diagnosis: 37 videos in `pending`, all with `attempts: 0`
(budget deferrals deliberately don't consume a retry), and the last delivered
summary dated 2026-08-11.

## 2. Free-tier limits are per model, and much smaller than published

From AI Studio for this project (peak usage / limit, 90 days):

| Model | RPM | TPM | RPD |
| --- | --- | --- | --- |
| Gemini 3 Flash | **10 / 5** | 45.43K / 250K | **26 / 20** |
| Gemini 2.5 Flash | **7 / 5** | 139.65K / 250K | 16 / 20 |
| Gemini 3.7 Flash | 3 / 5 | 246.39K / 250K | 4 / 20 |
| Gemini 3.5 Flash | 1 / 5 | 7 / 250K | 2 / 20 |
| Gemini 3.6 Flash | 1 / 5 | 7 / 250K | 3 / 20 |

Two findings that drive the design:

1. **20 requests per day, per model** — not the 1,500 that third-party guides
   quote. The limit that matters is requests, not tokens.
2. **The pipeline was already over it**: 26/20 RPD and 10/5 RPM on its preferred
   model, while three other Flash models sat nearly unused. Summaries were being
   lost to quota with capacity to spare.

## 3. Measured cost of a Gemini transcript

A YouTube URL passed as `file_data`; Google fetches the video server-side, so
neither a Supadata credit nor an unblocked runner IP is involved.

| Metric | 20-minute video | 3-minute video |
| --- | --- | --- |
| Input tokens | 123,195 | — |
| Output tokens | 6,124 | — |
| Wall clock | ~125 s | ~21 s |
| Transcript | 24,175 chars (verbatim) | 1,732 chars |

- `mediaResolution: MEDIA_RESOLUTION_LOW` changed input tokens **not at all**
  (123,195 either way). There is no cheap knob.
- Two video calls in one minute consumed **246,390 of 250,000 TPM** on one
  model. Sequential processing (~2 min/video) keeps this safe; parallelising
  transcripts on a single model would not be.
- ~123k tokens/video vs ~9k for summarizing an existing transcript: **13x**. So
  Supadata stays first whenever it has credits.

## 4. Design

**Transcript sources, cheapest first** (`transcript.py`):
Supadata → Gemini video → `youtube-transcript-api`.

**Two disjoint model pools**, because quota is per model:

| Pool | Models | Budget | Set by |
| --- | --- | --- | --- |
| Transcripts | 3.6-flash, 3.5-flash | 40 req/day | `GEMINI_TRANSCRIPT_MODELS` |
| Summaries | 3.7-flash → 3-flash-preview → 2.5-flash | 60 req/day | `GEMINI_MODEL` + `GEMINI_FALLBACK_MODELS` |

They must never overlap: one day of 123k-token video calls would otherwise eat
the summary budget. `tests/test_summarizer.py` asserts the disjointness — it
caught `gemini-3-flash-preview` sitting in both lists during implementation.

**Exhaustion is deferral, not failure.** When every transcript model is capped,
`budget_exhausted` is returned, so the video defers *without* consuming a retry
attempt and is never written off. Within a run, a model that answered 429 is
skipped for the remaining videos.

**Silence is now audible.** A run that delivers nothing while every transcript
source is exhausted sends one Telegram message. This is the specific gap that
made the August outage invisible for two days.

## 5. Deliberately not built

- **A committed per-model daily counter** (the analogue of
  `data/supadata_usage.json`). Reactive 429 handling plus the within-run skip
  already avoids nearly all wasted calls; a persisted counter would add a state
  file, commit plumbing and Pacific-midnight reset logic to save at most one
  round trip per model per run. Revisit if quota waste shows up in the logs.
- **Alert de-duplication across runs.** The alert is stateless, so a multi-day
  outage produces one message per run (~8/day). Adding a "once per day" gate
  means new persisted state; the current behaviour is noisier but honest.

## 6. Operational notes

- `daily-summary.yml` timeout raised 20 → 45 minutes: a Gemini transcript takes
  ~2 minutes against Supadata's ~3 seconds.
- This is a private repo, so Actions minutes are metered. Runs that fall back to
  Gemini cost roughly 2 minutes per video instead of seconds.
- `SUPADATA_RESET_DAY` is still unset, so the local credit meter resets on the
  1st while the real pools reset on plan anniversaries (17th and 26th per the
  July dashboard). Set the repo variable to stop that drift.
