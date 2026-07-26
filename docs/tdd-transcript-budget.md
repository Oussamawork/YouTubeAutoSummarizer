# TDD — Transcript budget and summarization efficiency

**Status:** Steps 1–2 implemented (PR #28); Step 3 deferred as nice-to-have
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

191 tests pass (`./scripts/check.sh`).

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

| Severity | Finding |
| --- | --- |
| High | Telegram send failures are ignored — the watermark advances even when delivery fails, so a 429 during a burst silently loses a summary while the run reports success |
| High | The state commit can be silently discarded (`git pull --rebase … \|\| true` swallows a conflict, then `push` exits 0) → dedup state lost → mass re-sends |
| High | `data/signals.jsonl` stores full summary text and is committed to a **public** repo, publishing the premium product |
| Medium | `market_pulse` splits one asset into two entries when a ticker is sometimes null (`Tesla` vs `TSLA`), skewing counts, flips and channel weights |
| Medium | Sub-dollar price targets render as `0` (`f"{avg:,.0f}"`), e.g. HBAR 0.109 → "avg target 0 (-100 % implied)" |
| Medium | `channel_scorecard.fetch_prices` does not retry non-200 transient statuses and has no test coverage |
| Low | Missing secrets exit 0 (a green run forever); partial multi-chunk Telegram sends can duplicate content |
