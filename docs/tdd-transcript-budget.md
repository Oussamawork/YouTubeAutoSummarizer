# TDD — Transcript budget and summarization efficiency

**Status:** Steps 1–2 implemented and **validated in production** (§2.5):
credits per delivered summary fell from 2.9 to **1.0**. Step 3 deferred as
nice-to-have. 17 open review findings tracked in §9.
**Date:** 2026-07-26
**Scope:** how transcripts are obtained and paid for, and what limits how many
videos the pipeline can summarize per day.

---

## 1. Context

Transcripts are the pipeline's only hard external cost. `transcript.py` fetches
them from Supadata (hosted, works from CI) and falls back to
`youtube-transcript-api`, which is **IP-blocked on GitHub Actions runners** and
therefore effectively unavailable in production. So: no Supadata credit, no
summary.

Supadata's free tier is a fixed monthly credit pool per key. The channel list
grew to 7 (from 4) on 2026-07-26, which made the ceiling bite.

---

## 2. Findings (measured, not estimated)

### 2.1 Credit efficiency is the dominant problem

| Metric | Value | Source |
| --- | --- | --- |
| Credits consumed (key 1) | **86 / 100** | Supadata dashboard, 2026-07-26 |
| Summaries delivered | **30** | rows in `data/signals.jsonl` |
| Videos deferred with no transcript | **47** | `pending` in `seen_videos.json` |
| **Credits per delivered summary** | **≈ 2.9** | 86 ÷ 30 |
| **Effective hit rate** | **≈ 35 %** | 30 ÷ 86 |

**~65 % of credits produce nothing.** At a 100 % hit rate the same 300-credit
budget would yield ~300 summaries instead of ~100. This dwarfs any rationing
gain and is the single highest-value thing to fix.

**We do not yet know why fetches fail.** `_fetch_supadata` logs the byte count
on success but not the failure shape (HTTP 200-with-empty-content vs 404 vs
captions-not-yet-generated). The 65 % figure is therefore a measured symptom
with an unmeasured cause. Leading hypotheses: very short videos / Shorts (both
backlogged channels post many), and captions not yet generated at fetch time.

### 2.2 Publishing rates are concentrated in two channels

| Channel | Videos/day | Pending backlog |
| --- | --- | --- |
| More Crypto Online | **9–12** (9 published on Jul 24 alone) | 22 |
| Parkev Tatevosian, CFA | **4–5** (observed 3, 4, 5) | 25 |
| Couch Investor, Joseph Carlson, HKCM, HKCM GLOBAL, Phantom | ~1 each | 0 |

All 47 backlogged videos belong to the two prolific channels; the other five
keep up with no backlog. Parkev's per-run cap was **3** against a ~4/day
publish rate, so his backlog grew every day and could never drain.

### 2.3 Credit pools reset on plan anniversaries, not calendar months

| Key | Credits left (Jul 26) | Resets |
| --- | --- | --- |
| `SUPADATA_API_KEY` | 14 | **Aug 17** |
| `SUPADATA_API_KEY_2` | 100 | **Aug 26** |
| `SUPADATA_API_KEY_3` | 100 | **Aug 26** |
| **Total available** | **214** | — |

A calendar-month assumption computed `300 ÷ 6 days left in July = 50 fetches/day`
and would have drained the pool in ~4 days. Cycle-aware pacing gives **9/day**
(214 ÷ 22 days to the earliest reset), which matches demand (~9/day: Parkev 4 +
five channels at ~1).

### 2.5 Result of Steps 1–2 (first run, 2026-07-26 22:56 UTC)

The gates and the diagnostics shipped together, so the first nightly run on
`86a6794` is the measurement.

```
Run summary: 7 channels | sent=10, unchanged=4, pending_evicted=47,
             title_filtered=2, too_short=16
```

| Metric | Before (§2.1) | This run | |
| --- | --- | --- | --- |
| Credits per delivered summary | 2.9 | **1.0** | 10 credits (`data/supadata_usage.json`) → 10 summaries |
| Hit rate | ~35 % | **100 %** | no `Transcript failures by reason` line was emitted at all |
| Shorts skipped before any credit | — | **16** | HKCM 4, HKCM GLOBAL 2, Phantom 3, More Crypto Online 7 |

**The hypothesis held.** Every fetch that ran, delivered — the waste really was
concentrated in very short videos. Effective capacity goes from ~100 summaries
per 300 credits to ~300, without buying anything.

Two things the run revealed that were previously invisible:

- **The HKCM channels are mostly Shorts.** All three had *only* short videos
  this run ("No new videos" after the gate), so their real long-form cadence is
  well below the ~1/day assumed in §2.2.
- **`SUPADATA_RESET_DAY` is still unset.** The usage file shows
  `"cycle": "2026-07-01"`, i.e. the calendar-month default, so the seeded 86
  credits were discarded at cycle rollover and the tracker now believes 290
  remain when ~204 actually do. The per-day ceiling (budget ÷ 28 = 10) is what
  kept the run in bounds — it spent exactly 10. Setting the variable to `17`
  restores accurate pacing; until then the ceiling is doing the work.

Also observed working as designed: one combined summarize+extract call returned
unusable JSON and fell back to the separate calls, keeping the summary
("Combined summary+signals call returned no JSON; using the separate calls").

### 2.4 Title filtering works

Applied to the 12 real More Crypto Online titles in `data/signals.jsonl`, the
`only=btc,bitcoin,eth,ethereum,sol,solana` filter keeps all 8 Bitcoin/Ethereum
videos and drops the 4 HBAR/XRP ones — a ~33 % cut with no false negatives.
Matching is whole-word: a substring test would match "eth" inside "whether" and
"sol" inside "solve".

---

## 3. What was built (merged)

| PR | Change |
| --- | --- |
| #23 | Credit metering, cycle pacing, multi-key rotation on 402/403, silent `budget_deferred` outcome, orphaned-`pending` eviction |
| #24 | Per-channel `only=` title filter, applied **before** any transcript fetch |
| #25 | Parkev first in the channel list with `max=6`; documented that file order = priority under the budget |
| #26 | Billing-cycle-aware pacing (`SUPADATA_RESET_DAY`), per-day ceiling of budget ÷ 28, usage seeded with the 86 credits already spent |
| #27 | This document |
| #28 | Per-outcome transcript failure reasons in the run summary; duration gate via a batched `videos.list` call (opt-in caption skipping); `tests/conftest.py` stubs the metadata lookup so tests stop reaching the network |

202 tests pass (`./scripts/check.sh`).

---

## 4. Critique of the current strategy

1. **It rations a resource instead of fixing why the resource evaporates.**
   Pacing, cycles, ceilings and key rotation all manage a 300-credit pool of
   which ~65 % is currently wasted. Efficiency work has a ~3× larger ceiling
   than budget work.
2. **The pacing is near-inert in the steady state.** Demand (~9/day) ≈ supply
   (~9.7/day). Pacing only binds while a backlog exists. It is justified now
   and cheap to keep, but it should not receive further investment.
3. **Diagnosis is missing.** We are optimising a failure rate whose cause has
   not been observed. That is the gap to close first.
4. **An earlier claim was wrong and is corrected here.** The 47 pending records
   were described as a credit leak from re-fetching every run. They are mostly
   *first* fetches that returned nothing; the waste is real, the mechanism was
   not. Orphan eviction is still correct (it stops unbounded state growth), but
   it is not a large credit saving.
5. **Kept because they are cheap and target the source:** title filtering,
   priority ordering, per-key rotation.

---

## 5. Options considered

| Option | Effect | Cost / risk |
| --- | --- | --- |
| **A. Failure diagnostics** | Turns the 65 % failure rate from a guess into a category breakdown | Trivial; one night of data |
| **B. Skip Shorts / caption-less videos** | Attacks the waste at source. `videos.list?part=contentDetails` returns `duration` **and** `caption` for **50 ids in 1 quota unit** (10 000 units/day available) | Low. `caption` is unreliable for auto-generated captions, so gate primarily on duration |
| **C. Gemini direct YouTube URL** | Removes Supadata **and** the caption dependency entirely: the model reads the video, so caption-less videos also become summarizable. Free in preview, [8 h of video/day](https://ai.google.dev/gemini-api/docs/video-understanding) vs our ~3–4 h | Medium. Needs the native Gemini endpoint (not the OpenAI-compatible path in `summarizer.py`); video tokens are heavy against free-tier limits; "preview" can change — keep Supadata as fallback |
| **D. Buy more credits / more free keys** | Linear capacity gain | Money, or more accounts to manage; leaves the 65 % waste untouched |
| **E. Do nothing further** | Pacing already prevents blackouts | Accepts ~100 summaries per 300 credits |

**Rejected:** D as a primary answer (paying 3× for waste), E (leaves the main
finding unaddressed).

---

## 6. Decision and next steps

Sequenced so each step's evidence informs the next.

### Step 1 — Instrument the failure path — ✅ IMPLEMENTED (PR #28)
- In `_fetch_supadata`, log the HTTP status and payload shape for every
  non-delivering response; count outcomes per category in the run summary.
- Record video duration alongside each `pending` record so Shorts can be
  identified retrospectively.
- **Acceptance:** one nightly run produces a breakdown such as
  `no_transcript: 200-empty=7, 404=1, short=5`.
- **Exit criterion:** if failures are *not* concentrated in Shorts/caption-less
  videos, skip Step 2 and go straight to Step 3.

### Step 2 — Gate on duration/captions before spending a credit — ✅ IMPLEMENTED (PR #28)
- Batch candidate video ids (≤50) into one `videos.list?part=contentDetails`
  call; skip videos under a configurable `min_duration` (default ~90 s) and
  optionally those reporting no captions.
- Per-channel override, consistent with the existing `only=` / `max=` options.
- **Acceptance:** credits per delivered summary drops measurably below 2.9;
  no drop in wanted summaries (verify against a run's title list).

### Step 3 — Pilot Gemini direct YouTube URL — 🔵 NICE TO HAVE (deferred)
- New code path in `summarizer.py` using the native endpoint with a
  `file_data` YouTube URL part; Supadata remains the fallback.
- Measure: summary quality vs the transcript path, tokens consumed, and whether
  the 8 h/day limit binds at our volume.
- **Acceptance:** equal-or-better summaries for one channel for a week with no
  quota incidents → widen; otherwise keep as an opportunistic fallback for
  videos whose transcripts fail.
- **Deferred by decision (2026-07-26):** Steps 1–2 attack the same waste at a
  fraction of the risk, and this path needs a second, non-OpenAI-compatible
  code path in `summarizer.py` against a preview API. Revisit if the failure
  breakdown shows the waste is *not* Shorts/caption-related, or if the
  transcript budget becomes binding again after Step 2's savings.

### Step 2b — Read the first diagnostics run — ✅ DONE (see §2.5)
The gate and the reason-logging shipped together, so the first nightly run after
PR #28 (2026-07-26, 22:00 UTC) is the first with data. Read its log for:
- `Run summary: … too_short=N, title_filtered=N` — how much the gates caught.
- `Transcript failures by reason: …` — the breakdown that decides what comes next.

Then act on it:

| If the breakdown is dominated by | Conclusion | Action |
| --- | --- | --- |
| `empty_content` on **short** videos | Shorts really are the waste | Raise `MIN_VIDEO_SECONDS` (e.g. 90 → 180) and re-measure |
| `empty_content` on **long** videos | Captions genuinely absent, not a length issue | Consider `SKIP_UNCAPTIONED=true` — but only after confirming the API's caption flag agrees with reality on a sample, since it reads `false` for auto-captioned videos |
| `http_4xx` / `job_incomplete` / `unreachable` | Not a caption problem at all | Step 2 was the wrong fix; promote **Step 3** (Gemini direct URL) or investigate the API integration |
| `budget_paced` / `no_credits` | The gates worked and the budget is simply binding | Add a key, trim caps, or accept the ceiling |

**Success measure for Steps 1–2:** credits per delivered summary below 2.9
(from `data/supadata_usage.json` count ÷ new rows in `data/signals.jsonl`),
with no wanted video dropped. **Met: 1.0, with 10/10 fetches delivering.**

Because the breakdown showed *no* failures at all, the "consider
`SKIP_UNCAPTIONED`" branch is moot for now — there is no residual caption-less
waste to remove. Leave it off. Re-open only if the failure line reappears.

### Step 4 — Re-evaluate the pacing layer
- Scheduled follow-up already exists for **2026-08-17** (trigger
  `trig_01GjK5iFCRnv6tfvyFE3wvuh`), when key 1 refills and the single-cycle
  model is least accurate (keys 2/3 do not refill until the 26th).
- If Steps 2–3 push demand comfortably below supply, consider retiring the
  ceiling/seeding complexity rather than extending it to per-key cycles.

---

## 7. Configuration required (owner)

| Setting | Value | Why |
| --- | --- | --- |
| `SUPADATA_API_KEY_2`, `SUPADATA_API_KEY_3` (secrets) | the two extra keys | Budget 100 → 300/cycle |
| `SUPADATA_RESET_DAY` (variable) | `17` | Earliest of the three reset days; without it pacing assumes the 1st |

---

## 8. Known limitations

- The budget model assumes a **single** billing cycle while the three keys reset
  on two different dates (17th, 26th). Using the earliest is conservative and
  accurate today; between Aug 17 and Aug 26 it may over-allow, ending in silent
  `budget_deferred` videos until keys 2/3 refill. Rotation on 402 is the real
  backstop. Revisit per Step 4.
- `youtube-transcript-api` cannot be relied on from CI (datacenter IPs are
  blocked), so Supadata has no free fallback today. Option C would change that.

---

## 9. Unrelated open findings (from the 2026-07-26 code review)

Tracked here so they are not lost; independent of the transcript budget.

Findings from three parallel reviews (correctness, robustness/ops, quality).
Each was demonstrated by its reviewer unless marked *suspected*. None is fixed
yet; the transcript-budget work took priority.

**Data-loss / product risk — fix first**

| # | Finding | Why it matters |
| --- | --- | --- |
| 1 | `data/signals.jsonl` stores full summary text and is committed to a **public** repo (`scraper.py`) | Publishes the premium product permanently into git history, undercutting the free/premium split. Fix: persist signals + `video_id` only, or move the corpus off the public repo |
| 2 | Telegram send failures are ignored — `send_telegram_message`'s return value is discarded and the watermark advances anyway; `_post` has no 429 handling | A rate-limit during a burst silently loses a summary while the run reports success. Fix: retry/backoff on 429, treat a failed send as `decided=False` |
| 3 | The state commit can be silently discarded — `git pull --rebase … \|\| true` swallows a conflict, then `git push` prints "Everything up-to-date" and exits 0 (`daily-summary.yml`) | Dedup state is lost → the next run re-sends every summary already delivered. Fix: drop `\|\| true`, abort on conflict, assert HEAD actually moved |

**Correctness of the weekly reports**

| # | Finding | Why it matters |
| --- | --- | --- |
| 4 | `market_pulse._asset_key` splits one asset in two when the ticker is sometimes null (`Tesla` vs `TSLA`) | Halves mention counts, suppresses consensus flips, and skews the scorecard-derived channel weights |
| 5 | Sub-dollar price targets render as `0` (`f"{avg:,.0f}"`) — real case: HBAR target 0.109 → "avg target 0 (-100 % implied)" | Visibly wrong output in the weekly pulse |
| 6 | `channel_scorecard.fetch_prices` retries only `RequestException`, not transient 5xx, and has **no test coverage** | A Stooq blip silently yields an empty scorecard that is indistinguishable from "no data yet", and quietly unweights the pulse |
| 7 | `channel_scorecard` `total_calls` sums across both horizons (reports 38 for 19 real calls) — *suspected* | Overstates sample size in the report header |

**Reliability / ops**

| # | Finding | Why it matters |
| --- | --- | --- |
| 8 | Missing secrets exit 0 — `main()` returns normally when required env vars are unset | A rotated or deleted secret produces a green run forever; `if: failure()` alerting never fires |
| 9 | No global run deadline: 3 providers × 3 attempts × 60 s ≈ 9 min per LLM call, and the combined call may be followed by the fallback | One pathological video can approach the workflow's 20-minute timeout |
| 10 | `channel_scorecard`/`market_pulse` fetch Stooq once per symbol with no shared cache, growing with the dataset | Unreachable Stooq × 15 symbols ≈ 765 s against a 600 s weekly job timeout |
| 11 | A partial multi-chunk Telegram send falls back to re-sending the whole message as plain text | Readers can see chunk 1 twice; the function still returns `True` |
| 12 | Deferred videos are selected oldest-first within the per-run cap | A stale backlog can crowd out current uploads. Partly mitigated by orphan eviction (#23); revisit if backlogs persist |

**Code quality (no user-visible impact today)**

| # | Finding |
| --- | --- |
| 13 | `signals.summarize_with_signals` accepts `channel_name` and never uses it — the combined path silently lost the channel grounding that `extract_signals` still applies. Either thread it into the prompt or drop the parameter and its call-site argument |
| 14 | The lenient JSON parse exists twice in `signals.py` (`_parse_signals` and the combined path); only one is covered by the preamble-tolerance test, so a fix to one won't reach the other |
| 15 | `summarizer.complete()` duplicates `summarize_transcript`'s provider-chain loop and has only happy-path tests, though all signal extraction flows through it — add quota-exhaustion and skip-exhausted-provider tests |
| 16 | `channel_scorecard` imports `_parse_date`/`_iter_assets` (private) from `market_pulse`, which lazily imports `channel_scorecard` inside two functions to dodge the cycle, both inside blanket `except Exception` — a rename would silently degrade the weekly report. Move the shared helpers into one module |
| 17 | Dead code: unreachable `return {}` in `fetch_prices`; the `end` date computed into `ranges` is never used; `helpers.save_to_json` uses bare `print`/`except Exception` where siblings use `log_error` |
