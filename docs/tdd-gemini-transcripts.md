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

## 9. A per-day 429 is a verdict, not a fact (2026-08-18)

Two videos on the evening of 2026-08-18 went out as

> ⏳ Summary deferred — the LLM provider quota/rate limit was reached.

with the run's own budget line saying why:

```
22:29:52 [INFO] Skipping gemini-3.7-flash (quota exhausted earlier this run).
22:29:52 [INFO] Skipping gemini-3.6-flash (quota exhausted earlier this run).
22:29:53 [INFO] Gemini daily budget: gemini-3.6-flash=2/20 left (capped by API),
                gemini-3.7-flash=17/20 left (capped by API)
```

The chain was out of budget with **17 of 3.7-flash's 20 free requests unspent**.
The write-off happened in the 07:41 UTC run — 00:41 Pacific, forty-one minutes
into the new quota day, after three recorded requests:

```
07:46:16 [WARN] gemini-3.7-flash rate-limited (429, day quota): [{
           "message": "You exceeded your current quota, please check your plan…
07:46:16 [INFO] Gemini gemini-3.7-flash marked spent for today (2026-08-18)
                after 3 recorded request(s).
```

The same shape recurs daily: 3.7-flash was written off after 8 recorded requests
on 08-16 and after **0** on 08-17. A limit that binds at 0, 3 and 8 requests is
not the 20-requests-per-day quota this project measured, so the code's inference
— "the API said per-day, therefore this model is finished until the Pacific
reset" — did not hold, and it was costing the *best* summary model most of every
day. 3.6-flash then absorbed the whole workload, hit its own real cap at 18, and
the day's last videos deferred.

Which limit Google actually meant is still unknown, and that is the second
finding: the log printed `body[:200]`, which stops inside the generic
`"You exceeded your current quota"` sentence, while the `quotaId` that names the
limit sits after it. Three days of logs cannot say whether the ceiling was
requests-per-day, tokens-per-day, or a stale count from before the reset.

### Design

- **The write-off is provisional.** `spent` grows from a list of model names to
  `{model: {"at", "used", "confirmations"}}`. `is_exhausted` honors a verdict for
  `GEMINI_SPENT_RECHECK_MINUTES` (45) and then lets the model be tried again. A
  rejected request costs no quota, so a re-probe is worth about a second of
  wall clock; believing a wrong verdict costs a day of the better model.
- **Repeats back off.** Each further verdict doubles the wait (capped at
  `GEMINI_SPENT_RECHECK_MAX_MINUTES`, 6h), so a model that genuinely is out for
  the day is left alone instead of being probed by every two-hourly run.
- **A served request clears the flag.** If the API answers a model it refused
  earlier, the verdict is stale by definition and `record()` drops it.
- **The counted cap stays final.** Requests we watched being served cannot be
  un-served: at 20 recorded, the model is done until the Pacific reset, and no
  waiting changes that. Only the API's *claim* is provisional.
- **The named quota is logged.** `gemini_quota.violation_summary` pulls the
  `quotaId`/`quotaValue` out of the body and logs them beside the classification,
  so the next occurrence says which ceiling was hit instead of leaving it to be
  reconstructed from committed counter files.
- **The skip line stopped lying.** "Skipping X (quota exhausted earlier this
  run)" was printed for both a limit hit minutes ago and a verdict inherited
  from a run hours earlier. The two are now distinguished, because the second
  reads as routine when it is the thing worth investigating.

### Not built

- **Raising `LLM_COMBINED_MAX_TOKENS` (3000).** Escalation does spend a second
  request, but it fired once in ten summaries in the 07:41 run — ~10% overhead
  against a chain that had 17 requests sitting idle. Not the binding constraint.
- **Widening the summary chain.** The model list is a quality decision
  (§ `summarizer.py`), and the budget it already has was not being spent.

## 10. 503s are billed: the storm that emptied the summary chain (2026-09-16 → 10-16)

From 2026-09-16 the gate log's `model_quota_deferred` outcomes went from 0–5 a
day to 40–114 a day while included videos stayed at 7–15, and by 2026-10-17
**63 stored transcripts had no claims**: 43 never got a research record and 20
waited on research quota. None of them reached the creators-vs-market tracker.

**What the API was doing.** Two first-runs-of-the-Pacific-day, where nothing
could have been spent before:

```
2026-10-08 08:36 UTC (01:36 PT)  sent=5, quota_deferred=20
  Gemini daily budget: gemini-3.6-flash=18/20 left (capped by API),
                       gemini-3.7-flash=17/20 left (capped by API)
2026-10-16 12:10 UTC (05:10 PT)
  gemini-3.7-flash rate-limited (429, day quota,
    GenerateRequestsPerDayPerProjectPerModel-FreeTier; limit=20)
  → marked spent "after 10 recorded request(s)"
  gemini-3.6-flash ... day quota ... → marked spent "after 2 recorded request(s)"
  (gemini-3-flash-preview=19/20 left, gemini-3.5-flash=19/20 left)
```

In the 10-16 run about seventeen requests to gemini-3.6-flash are visible, most
answered `503 UNAVAILABLE "This model is currently experiencing high demand"`
and retried up to three times per call, video after video; two were served. The
day quota was gone within that one run. The free tier **counts attempts it
answers with 503** against the 20 requests a day, while `gemini_quota.record`
counted only HTTP 200s — which is also the explanation § 9 could not give for a
model "written off after 0, 3 and 8 requests". On a calm day (2026-10-17 11:29
UTC) every request was served at the first attempt, 17 on 3.7-flash, and
nothing deferred for quota: capacity is fine until a 503 storm burns it.

**How a burnt quota became lost transcripts.**
- A call that saw only 503s returned `""`, so the summary path treated it as a
  failed summary: `manual_review`, a final outcome. 13 of the orphaned
  transcripts ended in `manual_review` (the outcome a failed, truncated or
  insufficient summary shares; the log does not say which), stored and never
  researched.
- A deferred video retries only while it is in its channel's ~15-entry RSS
  window; once it scrolls out, `_evict_orphaned_pending` drops it (FinVid posts
  several a day, so that takes days, not weeks). Research only ever started
  beside a delivered summary, so those transcripts were orphaned too.
- Research retries run only above a 4-request summary reserve, which a burnt
  quota never leaves.
- Only two models may write summaries (SUMMARY_MODELS), no Groq/LLM provider is
  configured, and GitHub ran the eight daily crons only 3–5 times a day.

### Design

- **Overload breaker** (`summarizer._call_provider`). A 5xx is retried at most
  `LLM_OVERLOAD_RETRIES` (2) times within a call, a model gets at most
  `LLM_OVERLOAD_MAX_FAILED_PER_RUN` (3) failed attempts per run, and then it
  is skipped for the rest of the run. Retry count does not change the cost per
  served request in a storm (≈ 1/(1−p) attempts either way); repeating the
  storm for every video of the run is what emptied the day, and that is capped.
- **An outage defers, never fails.** A call that ends on 5xx returns the quota
  sentinel, so the video waits for another model or the next run instead of
  `manual_review`.
- **Failed attempts are counted** (`gemini_quota.record_failed`, `failed` in
  `data/gemini_usage.json`). They spend the remaining figure in the budget line
  ("3/20 left (2 served + 15 failed 5xx attempt(s))") but not the served count,
  and never decide `is_exhausted`: the API's own verdict stays the authority.
- **No stored transcript is left behind.** `research_backfill --adopt-stored`
  gives every stored transcript without a research record a pending one;
  `--import-candidates` stores committed offline extractions
  (`data/research/candidates/`, validated exactly like a model response, run
  identity `offline:<source>`). The 63-transcript backlog of 2026-10-17 was
  cleared that way. The daily job runs `--adopt-stored --import-candidates
  --retry`.

### What would remove the constraint (owner decisions)

1. **Billing on the AI Studio project** (paid tier). It lifts the 20-request
   free-tier cap and the free tier's 503 deprioritisation. At the 2026-10
   volume (≈20–30 requests a day of 5–20k input and 3–4k output tokens, so
   roughly 7–8M input and 3M output tokens a month) a Flash model costs in the
   order of $5–15 a month; check Google's current price list. Set
   `GEMINI_REQUESTS_PER_DAY` to the new tier's limit afterwards.
2. **A third summary model** within the free tier: e.g. repo variables
   `GEMINI_FALLBACK_MODELS=gemini-3.6-flash,gemini-3-flash-preview` and the
   same roster in `SUMMARY_MODELS`, with gemini-3-flash-preview removed from
   `GEMINI_TRANSCRIPT_MODELS` so the pools stay disjoint (+20 requests a day;
   it idled at 19/20 on 2026-10-16). A quality trade-off — § `summarizer.py`.
3. **A second provider** (`GROQ_API_KEY`, or `LLM_API_KEY` + `LLM_BASE_URL` +
   `LLM_MODEL`) allowed in `SUMMARY_MODELS`. Its own limits (tokens per minute
   on free tiers) make it a fallback for short videos only.
