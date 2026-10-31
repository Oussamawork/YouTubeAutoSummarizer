# TO-DO

What is still open after the creators-vs-market tracker, the ticker clean-up
and the Gemini quota fix (October 2026). Section 1 needs a decision from the
owner; everything after it can be done without one. Figures were measured on
2026-10-19 on a copy of the merged data with the offline extractions imported.
Older deferred items from the September audit are in
`docs/nice-to-have-for-later.md`.

## 1. Decisions for the owner

### 1.1 Gemini capacity (the cause of the 63-transcript backlog)

The free tier gives each model 20 requests a day and also counts the
"503 high demand" answers against them. On storm days, retries emptied both
summary models within one run (`docs/tdd-gemini-transcripts.md` § 10). The
code now limits the damage: at most 2 retries per call and 3 failed attempts
per model per run, a video waits instead of failing, failed attempts are
counted, and orphaned transcripts are adopted. It cannot add capacity.

| Option | Cost | What to set (repo variables) |
| --- | --- | --- |
| **A. Billing on the AI Studio project** (recommended) | about $5–15 a month at today's volume (check Google's current prices) | enable billing, then `GEMINI_REQUESTS_PER_DAY` = the paid tier's per-model limit |
| B. A third free summary model | free; summaries come from an older model on busy days | `GEMINI_FALLBACK_MODELS=gemini-3.6-flash,gemini-3-flash-preview`, `SUMMARY_MODELS=gemini-3.7-flash,gemini-3.6-flash,gemini-3-flash-preview`, `GEMINI_TRANSCRIPT_MODELS=gemini-3.5-flash,gemini-2.5-flash` (the pools must stay disjoint) |
| C. A second provider | its free tier limits tokens per minute, so short videos only | `GROQ_API_KEY` secret (or `LLM_API_KEY` + `LLM_BASE_URL` + `LLM_MODEL`), and add its model to `SUMMARY_MODELS` |

A removes both the cap and the free tier's 503 deprioritisation. B and C add
headroom, but a storm still burns free quota. This PR passes these variables
to the daily workflow; before it, setting them had no effect.

### 1.2 The offline extractions: keep, or redo with Gemini

Claims for 63 backlog videos were extracted in this session and committed to
`data/research/candidates/`. The daily job handled 20 of them with Gemini
before the merge, so the first run after it imports **42 videos (plus 1 with
no claims)**. They pass the same validation gate as a model response, but
they are denser: 15.4 claims per 10,000 transcript characters against 5.5
for Gemini (views: 10.6 against 2.9). 98 of the tracker's 415 calls start in
one of these videos, 51 of them TheStreet: its multi-guest shows yield one
call per guest pick.

- Keep them (recommended for now): coverage beats no coverage, and every
  rate prints its denominator.
- Or re-extract them with Gemini once capacity exists (1.1 A), so that one
  extractor covers the whole dataset. That needs a small `--reprocess` filter
  for `offline:` runs; today `--reprocess` only picks up version changes.

### 1.3 Scorecard rankings (`SCORECARD_RANKINGS`, off)

The strict scorecard stays experimental and unranked until the live
extraction benchmark has been run and reviewed. The benchmark needs 20+ real
videos and 200+ human-labelled claims: `python claims_eval.py
--export-transcripts` writes the labelling files, and `--live` with
`CLAIMS_EVAL_LIVE=1` runs it. The question is whether that labelling time is
worth it now. Recommendation: not yet. The strict layer has almost nothing
to score, because most stated horizons run into 2027–28; the mark-to-market
tracker answers the "creators vs the market" question in the meantime.

### 1.4 Prune 7 known-wrong cached price series

`btc.us`, `crwe.us`, `fr.us`, `gold.us`, `mogu.us`, `rci.us` and `sy.us` in
`data/prices.json` belong to the wrong companies (for example `sy.us` is
So-Young, not Sony). Nothing produces or needs them any more
(`python ticker_audit.py` → KNOWN-WRONG SYMBOLS). Deleting them is safe, but
it is a data deletion. Recommendation: delete.

### 1.5 Late Telegram summaries for videos that left the feed

A video deferred until it scrolls out of its channel's ~15-entry RSS window
is dropped from the retry queue and never gets a Telegram summary. Its stored
transcript does still reach the claims and the tracker (`--adopt-stored`).
Should such videos get a late summary? That costs summary requests, the
scarce resource in 1.1. Recommendation: no, unless capacity is added.

### 1.6 Non-US companies: which listing to price

These names have no priceable US line in the data: Samsung (12 claims),
SK Hynix (7, plus 4 spelled "SKH Heinix"), Tencent (4), Adidas (3) and BYD (1).

- Option 1: price the US ADR or OTC line where one exists, as is already done
  for SoftBank (SFTBY), Adyen (ADYEY) and BASF (BASFY): scored on the US quote,
  with no benchmark.
- Option 2: leave them unpriceable.

Recommendation: ADRs where liquid (check each with `python warm_prices.py
--find`). Samsung and SK Hynix wait for the US-listing check in section 2.

### 1.7 Daily schedule

Of the 8 scheduled daily runs, GitHub started only 2–6 a day between
2026-10-03 and 10-18, often hours late (for example 08:41, 18:11 and 23:41
for slots set at even hours). GitHub delays or drops scheduled runs at the
top of the hour under load. Moving the crons to an odd minute
(`17 12-22/2 * * *`, `17 2,7 * * *`) usually helps and shifts delivery by
minutes. If that is not enough, an external scheduler can call
`workflow_dispatch`. Recommendation: move the minute.

## 2. Ticker corrections that need the price key

Medium- and low-confidence corrections are applied only after the provider's
catalogue confirms them (CLAUDE.md conventions). This session had no
`TWELVEDATA_API` key. `python ticker_audit.py` prints each entry with its
evidence and the exact `warm_prices.py --find / --probe` step. After a check,
apply the result in `signals_data` / `instruments` with the evidence in a
comment, and remove the entry from `ticker_audit.NEEDS_VERIFICATION`.

| Spoken / recorded | Likely | Claims | Check |
| --- | --- | --- | --- |
| "the market", "stocks" (TheStreet, Couch Investor) | US stock market (SPY proxy) | 56 | map per channel; crypto channels stay placeholders |
| TTD, NBIS sectors | confirm GICS | 47 | update `instruments._CURATED` |
| Anthropic | pending IPO | 20 | after the listing: `--find 'Anthropic'` |
| Domino's | DPZ venue (XNAS?) | 12 | `--find "Domino's Pizza"` |
| Iron (FinVid) | IREN or Iron Mountain (IRM) | 11 | `--probe iren.us,irm.us` vs the stated 90–120 range |
| SK Hynix | its 2026 US listing | 7 | `--find 'SK Hynix'` |
| Cronos / Chronos (HKCM, Phantom) | CRO | 7 | `--probe crousd` vs both channels' levels |
| Target (Parkev) | TGT | 5 | title-scoped resolution |
| "Brand" (HKCM) | Brent oil | 4 | read HKCM 2026-10-14 |
| Adidas | ADDYY or unpriceable | 3 | see 1.6 |
| Voyager (Couch Investor) | VOYG | 3 | `--find 'Voyager Technologies'` |
| Progressive (TheStreet) | PGR | 3 | TheStreet-scoped alias |

Plus 17 smaller entries:

- Centuri (CTRI) and XP (Phantom: XPEV or XRP)
- NAR (NEAR), XLP (XRP) and Avex (AVAX)
- the CoinShares mining ETF (WGMI)
- MOGU (Moog, MOG.A) and Marathon Oil (MPC)
- Cipher Mining (CIFR), Coke (KO) and BICO (FICO)
- Sonic (S), HYPE and Circle (CRCL)
- the RCI/RECS and PSP/CDPI fund identities
- the KLAC/MRNA/APH/APP split checks

## 3. Names still without a ticker

After this PR, 138 names in 213 claims (76 of them directional views) have no
ticker. The ones worth a look:

- **Aurora** (HKCM, 5) and **Canopy** (HKCM, 1): the HKCM context is cannabis
  ("von den Cannabis Playern"), so these look like Aurora Cannabis (ACB) and
  Canopy Growth (CGC). Use an HKCM-scoped alias, because Aurora Innovation
  (AUR) exists. "Canopy" is also an ordinary word.
- **Siena** (TheStreet, 3): probably Ciena (CIEN), the optical networking
  stock. Verify first.
- **Lumen** (HKCM, 3): unclear. The stated entry ("4 6 Cent") does not fit
  Lumen Technologies.
- **Cadence** (FinVid, 1): Cadence Design (CDNS) or Cadence Bank (CADE).
- **AWX** (Phantom, 2), **Fermis** (FinVid, 1) and **Aerodrome Finance**
  (1): unknown or unpriced.
- **SKH Heinix**, **Hynix** and **Open AI**: fold them into SK Hynix and
  OpenAI together with 1.6.
- **Themes, not instruments** ("Mag Seven", "AI stocks", "DRAM", "equal
  weight S&P 500"): should any get a proxy (MAGS; RSP is already in the
  registry)?

Full worklist: `python ticker_audit.py --top 50`.

## 4. Code follow-ups (no decision needed)

1. **German-aware claim validation.** German buy, sell and hold wording is
   reset to `none`, and the validator reads no German position wording:
   "wir haben verkauft" / "short" / "nicht investiert" read as ownership
   unless the model itself said `no_position`, which is now always kept
   (`docs/tdd-full-transcript-claims.md` § 13.11). This is the suggested task
   "Make claim validation German-aware for actions".
2. **Gemini transcription retries 5xx without the breaker.** `transcript.py`
   retries 500/502/503/504 up to `GEMINI_TRANSCRIPT_MAX_RETRIES` (2) per video
   and never calls `gemini_quota.record_failed`. A storm can therefore still
   burn the transcript models' daily quota unseen. Mirror the summarizer's
   breaker.
3. **No negative price cache.** A symbol that returns no data is re-requested
   by every warm (the audit counts 96 needed symbols without a series). Real
   symbols fill on the next warm; junk ones never do. Remember a 404 for a
   few weeks.
4. **Testability is not recomputed at load.** `canonical_claims.reresolve`
   corrects entities but not `testable` / `testability_issues`, so a rule
   change there needs re-extraction or a recompute step. The 2026-10
   validator rules (number check, retrospectives, repeat key, price levels)
   reach stored claims the same way; the design doc § 13.11 counts what they
   would change (193 claims in 109 videos).
5. **Store the pulse's picks.** "Last week's picks" is rebuilt from claims
   each Monday. Persisting the list that was actually sent (for example in
   `data/pulse_picks.jsonl`) would let the one-week review grade exactly
   what readers saw.
6. **No production condition evaluator.** Nothing calls
   `research_state.record_condition_evaluation`, so conditional forecasts
   never become `condition_status=met` and never reach the strict scorecard.

## 5. After the merge: what to check

- **Next daily run.** The research step should log `Stored transcripts
  without research: 43 adopted` and `Offline candidates imported:
  {'complete': 42, 'already_done': 20, 'no_claims_found': 1}`. After that
  no stored transcript is left without research.
- **Friday 2026-10-23, 06:00 UTC.** First Friday price warm: the tracker's
  symbols, including this PR's new GEV / WDAY / FIX / APLD rows. Calls on
  those stay `no_prices` until then.
- **Friday 2026-10-23, 16:00 UTC.** The Friday report, led by creators vs
  the market and its two charts.
- **After the Friday warm.** `python ticker_audit.py` should no longer list
  the partial crypto candles stored by the 2026-10-18 warm, or the stale
  series (mcd.us, pg.us, cgc.us, ...).
- **Monday 2026-10-26, 08:00 UTC.** First pulse in the new layout.
- **Gemini usage.** `data/gemini_usage.json` now has a `failed` count per
  model. A storm shows up as "N served + M failed 5xx attempt(s)" in the
  budget line, and its videos defer instead of going to manual review.

## 6. Data notes

- Parkev's 2026-10-12 transcript has Planet Labs at "$1743" (meaning
  $17.43). The price-level audit flags it; nothing uses it as a target.
