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

- ~~**A committed per-model daily counter**~~ — built after all, as
  `gemini_quota.py` + `data/gemini_usage.json`. The judgement above was wrong
  about the size of the prize: without a persisted count, every run rediscovers
  a spent model by spending a request on it, and with eight runs a day that is
  not "at most one round trip per model per run" but eight.
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

## 7. One status code, two limits (2026-08-16)

Three days of running the 3.7/3.6 summary chain surfaced a defect in how a 429
was read. From the 2026-08-15 16:31 run:

```
16:35:23 [WARN] Transient gemini-3.7-flash status 503 (attempt 1/3); retrying.
16:35:30 [WARN] Transient gemini-3.7-flash status 503 (attempt 2/3); retrying.
16:35:34 [WARN] gemini-3.7-flash rate-limited (429) after retries; treating as quota exhausted.
16:35:34 [INFO] Gemini gemini-3.7-flash marked spent for today (2026-08-15).
```

Two unrelated 503s consumed retry attempts 1 and 2, so the 429 landed on the
last attempt and was **never retried once** before the model was written off
until the Pacific reset. Reconstructing from the committed counter, 3.7 was at
13 of 20 requests: **7 requests were thrown away**, and every summary for the
rest of that day fell to 3.6.

The root cause is that the free tier enforces two different limits behind one
status code — 5 requests/minute and 20 requests/day — and only the response
body distinguishes them (`GenerateRequestsPerMinutePerProjectPerModel-FreeTier`
vs `...PerDay...`). That body was being discarded, so neither the code nor the
logs could tell a 60-second speed bump from a 24-hour outage. Four summary calls
inside one run can genuinely trip 5 RPM, which makes the per-minute case the
*likely* one, not the exotic one.

### Design

- **Rate limits get their own retry budget** (`LLM_MAX_RATE_LIMIT_RETRIES`),
  separate from the failure retries, so unrelated 5xx can never starve them. A
  rate-limit wait decrements `attempt` for the same reason an escalation does:
  waiting is not failing.
- **`gemini_quota.classify_429`** reads the quota name out of the body. Per-day
  → `mark_exhausted`, since no wait brings the budget back. Per-minute → back
  off and retry, honoring `RetryInfo.retryDelay` (sent in the body, not in a
  `Retry-After` header, so a header-only reader never saw it) capped at
  `LLM_RETRY_AFTER_CAP`.
- **Unknown quota → cost the run, not the day.** The provider is skipped for the
  remainder of the run and left usable by the next one. Worst case that wastes
  one request an hour later; the old behavior cost seven summaries.
- **The counter stopped lying.** `mark_exhausted` used to write the cap into the
  count, so `20` meant either "20 requests served" or "the API refused after 3"
  — leaving the file unable to answer how much budget a run actually used, which
  is the one question it is committed for. The decision now lives in a separate
  `spent` list and the count stays a true tally.
- `transcript.py` gets the same distinction, but no waiting: rotating to the
  next model is cheaper than sleeping out a per-minute window. A model
  rate-limited without a named day quota is remembered in
  `_RATE_LIMITED_THIS_RUN` so it is not re-asked for every video in the run.

## 8. Videos larger than the context window (2026-08-16)

The `gemini_http_400` in the 2026-08-15 run resolved to one video,
`ucrXJlTbB_w` ("COREWEAVE Q2 EARNINGS REPORT BREAKDOWN"):

```
[WARN] Gemini gemini-3.5-flash returned 400: "The input token count exceeds
       the maximum number of tokens allowed 1048576."
[WARN] Gemini gemini-3-flash-preview returned 400: "Request contains an invalid argument."
[WARN] Gemini gemini-2.5-flash returned 400: "The input token count exceeds ..."
[INFO] Transcript budget spent; deferring this video to a later run.
```

At the measured ~6.2k tokens per minute of video, 1,048,576 tokens is roughly
2.8 hours — this is a long earnings stream, not a malformed request.

Two things were wrong. All three models share that context window, so the
rotation spent three requests and ~90 seconds to be told the same thing, on
every retry of that video. And this is the video still sitting in `pending`
with `attempts: 0`: it defers on Supadata's `budget_exhausted` flag, which
deliberately does not consume a retry attempt.

**The deferral is correct and was left alone.** Supadata has no 1M-token
ceiling, so once its credits reset the video can still be transcribed. What was
wrong was only the wasted rotation: a size rejection now returns `too_large` and
stops after the first model, while any other 400 still rotates — the size limit
is model-independent, a bad request is not.

### The run summary was also miscounting

```
[WARN] Transcript failures by reason: gemini_ok=4, gemini_http_400=1
```

`gemini_ok` is a success. `scraper.py` filtered the breakdown against its own
hand-written copy of the success reasons, `("ok", "fallback_ok")`, and that copy
was never updated when the Gemini source was added — so every Gemini success was
counted and reported as a failure. The set now lives in `transcript.py` beside
the code that produces the reasons, and the scraper imports it, so it cannot
drift again.
