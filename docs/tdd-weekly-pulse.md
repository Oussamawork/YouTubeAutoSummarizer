# TDD — Monday pulse: data reliability and creator-balanced analysis

**Status:** implemented 2026-10-06. Every number below was measured on the
committed research ledger (`data/research/`) on that date, not estimated.
**Scope:** what the weekly pulse (`market_pulse.py`, Mondays 08:00 UTC) reads,
how much of it it actually used, and why its conclusions were weaker than the
data allowed.

---

## 1. What the pulse reads

```
YouTube (9 channels) ─▶ gate (title / duration filters) ─▶ transcript ─▶ LLM summary
     + claims (combined call, or a queued separate extraction)
     ─▶ claims.jsonl ─▶ canonical_claims.load_canonical_claims ─▶ view claims
     ─▶ aggregate_views_by_asset ─▶ pulse text + charts
```

Three things decide how many opinions reach a Monday report: whether the
video was processed in time (model quota), whether its claims passed review
(the validator), and whether the asset resolved to one identity (aliases).
All three were leaking.

## 2. Findings

### 2.1 Coverage: a third to a half of each week was never read

Per publication week, latest gate outcome per video:

| Week of | included | model_quota_deferred | research `quota_deferred` |
| --- | --- | --- | --- |
| Sep 07 | 70 | 0 | 0 |
| Sep 14 | 71 | 0 | 0 |
| Sep 21 | 68 | 8 | 11 |
| Sep 28 | 58 | 15 | 12 |

Videos analyzed per pulse window fell 65 → 56 → 35 → 26. On Oct 6, 37 of the
window's 73 in-scope videos were unanalyzed: 25 still deferred for the summary
itself (one waited five days), 10 summarized but their claim extraction
deferred by the research reserve, 2 held. `gemini_usage.json` showed both
summary models answering "spent" after 0–1 local requests. Channels lower in
`channel_ids.txt` (TheStreet, Couch Investor) dropped out of whole weeks.
**The report said nothing about it** — "Based on 96 opinions in 26 videos from
6 creators" read the same as a complete week.

### 2.2 Validator false positives: 42% of opinions dropped

Since Sep 7, 1,183 claims were own-view, view-type, with a stance; **501
(42%)** carried `review_required` and never reached the pulse. By reason:

| Reason | Flags | False positive? |
| --- | --- | --- |
| `ticker_not_in_evidence` | 255 | 174 already resolved by the curated table from the spoken name; the model had merely written a ticker nobody said |
| `evidence_not_found` | 129 | 92 (71%) were verbatim sentences the model joined with "..." |
| `subject_not_in_evidence` | 92 | mostly genuine |
| `number_not_in_evidence:*` | ~130 | 21 were German notation ("5,84", "18.000") |
| `possible_third_party_view` | 9 | 6 were "according to **my** estimates / this metric" |

### 2.3 Identity: well-known names unresolved

Fortinet, McDonald's, Shopify, Qualcomm, Disney ("Disney" and "Disney stock"
as two assets), ServiceNow ("Service Now"), P&G, Pinterest, Costco and others
had no ticker: votes split from the ticker key, no prices, and the report
mixed "Fortinet" with "PANW".

### 2.4 Analysis: volume, not creators, set the conclusions

- **One channel made 50–57% of a week's views and 69% of its videos**
  (Parkev Tatevosian, ~4–5 videos a day). The takeaway ("Upbeat week: 17 of 26
  videos leaned bullish"), the tone chart ("65% of videos expect a rise, down
  from 80%"), the asset-class lean ("stocks 72% bullish, 25 creator calls")
  and the attention movers ("Fortinet 0→6 claims" — one deep-dive video) all
  measured that channel's output.
- **A 7-day window almost never puts two creators on one asset**: 47 of 52
  assets in the Sep 29 – Oct 5 window rested on one creator, so "Where
  creators agree" was two rows of "2 of 2".
- **"Consensus changed vs last week" needed 2+ creators in both weeks** and
  essentially never fired, while creators reversing their *own* calls (MU,
  BTC, TSLA, PANW) went unreported.
- **HKCM, HKCM GLOBAL and Phantom by HKCM counted as three independent
  creators** (S&P 500 "2 bearish" was the HKCM house twice).
- **A missing creator read as lost interest**: attention compared a
  backlog-shrunk week with a complete one.

## 3. Changes

**Reliability (all analytics, not only the pulse)**

- `claims.resolve_entity`: an unspoken model ticker is discarded; it asks for
  review only when the spoken name does not identify the asset on its own.
- `claims.locate_evidence`: an excerpt elided with "..." matches when every
  piece is verbatim, in order, each within `MAX_ELLIPSIS_GAP_CHARS` (1,500;
  measured median gap ~160, 90% under ~1,300).
- `claims.numbers_in` also reads German decimal commas and thousands dots;
  `_THIRD_PARTY_RE` no longer fires on "according to my/our/this/these/that".
- `claims.revalidate_review`, applied by `load_canonical_claims`: stored
  claims have the record-decidable flags the current rules no longer raise
  cleared (clear-only, recorded in `review_cleared`; testability unchanged).
  Stored `evidence_not_found` claims stay flagged — relocating them needs the
  transcript (see § 5).
- Curated aliases for the names in § 2.3; `" CORPORATION"`/`" INCORPORATED"`
  name suffixes; Nikkei/Nasdaq/Dow spelling folds; "support zone" and bare
  "stock" are not assets.

Result on the same ledger: flagged would-be views 501 → 324 (42% → 27%), and
new extractions also recover the ellipsis cases.

**Analysis (`market_pulse.py`)**

- **Consensus board**: each creator's latest view from the past 28 days
  (`CONSENSUS_LOOKBACK_DAYS`) for assets someone discussed *this week*; rows
  say how many views are fresh ("4 of 4 creators bullish (2 this week)").
- **Creator families**: `group=` in `channel_ids.txt` (HKCM's three channels)
  — one vote, one mood, one name.
- **Mood in creators**: `canonical_claims.creator_moods` — each creator's
  balance of per-asset votes under the 2/3 `lean`; the takeaway, the mood
  section, the asset-class line and the tone/spread charts count creators.
- **Changed their mind**: `canonical_claims.view_changes` — the same creator's
  latest directional view this week opposite to their previous one within the
  board window; differing named horizons are not a change.
- **Attention in creators**, counting only creators with views in both weeks.
- **Coverage**: `pulse_coverage` (gate outcomes + research state) — a
  "⚠️ Partial week" line under the takeaway below 75% coverage, and a footer
  naming the channels still queued.

Oct 5 pulse, same data, before → after: "Strongest agreement: NVDA (2 of 2
creators)" → "NVDA (4 of 4 creators)"; agreement rows 2 → 6 plus a
disagreement (META 3 vs 1); 6 creator reversals surfaced; mood "22 of 33
videos bullish" → "4 of 5 creators bullish"; coverage "45 of 77 videos
analyzed" stated up front.

## 4. Limits

- The board can carry a view up to four weeks old; the freshness count is
  the guard, not a decay weight.
- Creator mood weights every asset equally: a creator bullish on five small
  caps and bearish on the index reads bullish.
- Families are configuration: a new channel of an existing house must be
  given its `group=`.
- Coverage counts videos, not opinions; a waiting channel's views are simply
  absent, not estimated.

## 5. Next steps (not done here)

1. **Capacity** (the biggest lever): the free Gemini tier is the bottleneck
   behind § 2.1. More summary models in `GEMINI_FALLBACK_MODELS`, a paid key,
   or `max=` caps on the highest-volume channels would each restore coverage;
   the pulse now shows the gap but cannot close it.
2. **Relocate stored `evidence_not_found` claims** with the new ellipsis
   matcher (an offline backfill writing an overlay, like
   `condition_evaluations.jsonl`): ~92 more claims.
3. Remaining `subject_not_in_evidence` flags (92) are mostly real; a better
   extraction prompt (quote the sentence that names the asset) is the fix.
