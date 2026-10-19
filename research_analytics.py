"""
Analytics over validated canonical claims — never over summary prose.

Every report starts with a data-quality header that states the universe
honestly (discovered vs included vs transcribed vs fully processed), shows
every percentage with its denominator, and separates what headline analytics
include from what they exclude. Headline analytics use only claims that are
active (not superseded), not review-required, fully covered, evidence-located
and the source's own view (`claims.is_headline_claim`). Excluded counts are
shown, not hidden.

Identity is canonical: an asset is canonical_claims.asset_key (CHAINLINK and
LINK, ÖL and OIL, "Alphabet stock" and GOOGL are one key), a source is
canonical_claims.source_of, and a placeholder, basket or metric subject (THE
MARKET, PRICE, "wave two or B wave", "Google and Meta") has no key and is
left out of every per-asset view.

Consensus counts at most one CURRENT view per source per asset per horizon
bucket; short- and long-term views are never merged. A view change ("flip")
is the same creator's video-level call on the same asset (one vote per video,
creator_tracker.video_calls) reversing between two videos at least
MIN_FLIP_GAP_DAYS apart; same-video scenario talk never flips, and quicker
reversals are counted per creator (short_reversals), not listed. Conditional
claims are summarized compactly and never mixed into the track record. The
strict scorecard evaluates only matured, testable, evidence-backed PRICE
forecasts, with the entry/evaluation conventions documented on
`SCORECARD_RULES`; the mark-to-market "since the call" layer is
creator_tracker's, shown as a summary block (`build_report(tracker_record=)`).
"""
import argparse
import dataclasses
import functools
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from claims import is_headline_claim, carries_view, OWN_VIEW_ATTRIBUTIONS  # noqa: F401 - re-exported
from helpers import env_int
from log import log_info, log_warn
import canonical_claims
import claims as _claims
import creator_tracker
import instruments
import research_state
import scorecard_pricing as sp

BUCKETS = ("short", "medium", "long", "unspecified")
_DIRECTIONS = ("bullish", "bearish")
# Rank a source only once it has this many matured forecasts. An analytical
# rule, configurable — not a claim of statistical reliability.
MIN_SCORECARD_SAMPLE = env_int("MIN_SCORECARD_SAMPLE", 20)
MAX_PRICE_LAG_DAYS = sp.MAX_PRICE_LAG_DAYS
# A reversal between two videos is a changed view only when they are at least
# this many days apart. In the 30 days to 2026-10-16, 37 of the 64 claim-level
# "flips" the old rule listed were scenario talk inside ONE video ("a decent
# pullback in the next few weeks" / "if we break above 2,487 ... higher") and
# most of the rest came a day or two apart: chart updates, not changes of mind.
MIN_FLIP_GAP_DAYS = env_int("RESEARCH_FLIP_MIN_GAP_DAYS", 7)
MAX_FLIP_LINES = 10
# Conditional claims: example lines under the two summary lines.
CONDITIONAL_EXAMPLES = 6
# The strict scorecard scores price forecasts only: a revenue, margin, cash
# flow or rate forecast says nothing about where the share price goes (41 of
# the 83 forecasts waiting to mature in October 2026 were such forecasts, 20
# with "$600M-$2T targets" that would have been tested as share prices).
SCORED_FORECAST_METRICS = (None, "price")
# A claim on one of these asset types with no ticker at all is a market,
# macro or commodity subject (the Fed's rates, DRAM prices): not an
# instrument, which is different from a ticker the registry cannot resolve.
NOT_AN_INSTRUMENT_TYPES = frozenset({"macro", "commodity", "index", "sector", "currency", "bond"})
_NOT_AN_INSTRUMENT_KINDS = frozenset({"market", "basket", "metric"})
# Below this many scored forecasts a source's line is followed by one line
# per scored forecast: a rate over n=1 says less than the forecast itself.
CLAIM_LINES_BELOW_N = 5

SCORECARD_RULES = {
    "status": "experimental; source rankings disabled unless SCORECARD_RANKINGS=true",
    "publication_timestamp": "published_at from the video feed (UTC instant)",
    "entry_price": "next-close rule: the first session close STRICTLY AFTER the publication instant, "
                   "placed against the instrument's exchange session in the exchange's timezone; a "
                   "publication after the close, on a weekend or on a holiday takes the next trading "
                   "day's close; a publication without a time of day takes the next trading day's close "
                   f"(conservative); within {MAX_PRICE_LAG_DAYS} calendar days of that day",
    "crypto": "24/7 assets have no session: entry = the close of the UTC day the video was published "
              "(the first daily close after publication)",
    "evaluation_date": "forecast_end_date (claims.resolve_horizon); the first trading-day close on or "
                       f"after it, within {MAX_PRICE_LAG_DAYS} days",
    "forecast_metric": "price forecasts only: a forecast whose metric is anything but the price (or unstated) "
                       "— revenue, earnings, margins, cash flow, rates, market cap — is excluded as "
                       "non_price_metric and never scored as a share-price call",
    "instrument_metadata": "exchange, asset type, country, currency, sector and benchmark come from the "
                           "canonical instrument registry (instruments.py: curated table, else the "
                           "provider-verified ticker map); a claim's model-written sector is recorded as "
                           "speaker_sector and never selects a benchmark; an unresolved instrument is excluded "
                           "(unresolved_instrument), and a ticker-less macro, index, sector, commodity, currency "
                           "or bond subject is counted as not_an_instrument",
    "trading_calendar": "NYSE holiday rules for US listings (calendar_confidence=full); continuous for "
                        "crypto; weekday-only elsewhere (weekdays_only: local holidays are NOT modelled and "
                        "no bound on the resulting date error is claimed) — such claims are scored and "
                        "labelled but excluded from source rankings unless SCORECARD_RANK_WEEKDAY_CALENDARS=true",
    "price_targets": "a target is canonical_claims.price_target_of (a USD price level of a price_target or "
                     "price forecast; never an entry level, a revenue figure or a multiple), tested only when "
                     "it lies on the called side of the entry and within creator_tracker.PLAUSIBLE_TARGET_RATIO "
                     "(0.2x-5x) of it; reach/hit targets are tested by intraday_touch (daily high/low) when bars "
                     "are available, else by daily_close; when the video went up during the entry session only "
                     "that day's close counts (its high and low may have printed before the video); "
                     "horizon_close (the evaluation-date close) is recorded beside them; a target reached "
                     "inside the window is reached even if the horizon close moved away; with closes only, "
                     "intraday reach is recorded as unknown, never false",
    "adjusted_prices": "split-adjusted at minimum (provider adjust=splits); total-return-adjusted when "
                       "PRICE_ADJUSTMENT=all (dividends folded in); unadjusted or unknown series are "
                       "refused; provider, adjustment, corporate-action status, currency, requested and "
                       "resolved dates are recorded per scored claim",
    "benchmark": "resolved per claim from the INSTRUMENT's asset type, listing country and GICS sector "
                 "(scorecard_pricing.BENCHMARKS, BENCHMARKS_JSON override, per-instrument override or "
                 "none); null when none is defensible; sources scored under different benchmark methods "
                 "are not ranked against each other",
    "conditional_forecasts": "excluded from unconditional rankings; scored separately only once their "
                             "condition is recorded as met (condition_evaluations.jsonl)",
    "maturity": "a forecast is scored only once its evaluation date has passed",
}

# Kept for importers of the old names; the direction rule is canonical_claims.direction_of.
UP = canonical_claims.UP
DOWN = canonical_claims.DOWN


def pct(numerator, denominator):
    """'n/d (x%)' with the denominator always shown; 'n/0 (n/a)' when empty."""
    if not denominator:
        return f"{numerator}/0 (n/a)"
    return f"{numerator}/{denominator} ({100.0 * numerator / denominator:.0f}%)"


def _date(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except (TypeError, ValueError):
        return None


def _in_window(claim, start, end):
    d = _date(claim.get("published_at")) or _date(claim.get("extracted_at"))
    return d is not None and start < d <= end


def _source(claim):
    return canonical_claims.source_of(claim)


def _asset_key(claim):
    """canonical_claims.asset_key: the folded ticker, else the market key,
    else the spoken name; None for a placeholder, basket or metric subject."""
    return canonical_claims.asset_key(claim)


def _direction_of(claim):
    """bullish / bearish / None (canonical_claims.direction_of, with neutral
    and mixed views mapped to None)."""
    direction = canonical_claims.direction_of(claim)
    return direction if direction in _DIRECTIONS else None


def _bucket(claim):
    bucket = claim.get("horizon_bucket")
    return bucket if bucket in BUCKETS else "unspecified"


@functools.lru_cache(maxsize=1)
def _market_keys():
    """Every canonical market key (index, sector, commodity, macro theme)."""
    return frozenset(canonical_claims.MACRO_ALIASES.values()) | frozenset(creator_tracker.PROXIES)


def _resolved_subject(claim):
    """The claim's canonical asset key when its subject resolved to a ticker
    or to a known market, else None: a placeholder ("the market", "price"),
    a basket, a metric or a raw spoken name nobody could pin down
    ("Marcado Libé", "Kryptos") is never printed as if it were an asset."""
    key = _asset_key(claim)
    if key is None:
        return None
    if canonical_claims.resolved_ticker(claim):
        return key
    if claim.get("subject_kind") == "market" or key in _market_keys():
        return key
    return None


# --- Data-quality header ------------------------------------------------------


def quality_header(claims, gate_outcomes, state, runs, start, end):
    """Counts (with denominators) describing the universe behind the report."""
    gate = [g for g in gate_outcomes if (_date(g.get("decided_at")) or start) > start
            and (_date(g.get("decided_at")) or end) <= end]
    by_video = {}
    for g in gate:  # latest decision per video
        by_video[g.get("video_id")] = g
    outcomes = Counter(g.get("outcome") for g in by_video.values())
    discovered = len(by_video)
    included = outcomes.get("included", 0)
    videos = state.get("videos", {}) if state else {}
    in_window = {vid: e for vid, e in videos.items()
                 if (_date(e.get("published_at")) or _date(e.get("updated_at")) or start) > start
                 and (_date(e.get("published_at")) or _date(e.get("updated_at")) or end) <= end}
    transcribed = sum(1 for e in in_window.values() if e.get("transcript_hash"))
    status = Counter(e.get("research_status") for e in in_window.values())
    window_claims = [c for c in claims if _in_window(c, start, end)]
    headline = [c for c in window_claims if is_headline_claim(c)]
    sources = Counter(e.get("transcript_source") or "unknown" for e in in_window.values())
    return {
        "date_range": (start.isoformat(), end.isoformat()),
        "discovered_videos": discovered,
        "included_videos": included,
        "gate_outcomes": dict(outcomes),
        "transcribed_videos": transcribed,
        "research_videos": len(in_window),
        "fully_processed_videos": status.get("complete", 0) + status.get("no_claims_found", 0)
        + status.get("needs_review", 0),
        "partially_processed_videos": status.get("partial", 0),
        "awaiting_retry_videos": status.get("quota_deferred", 0) + status.get("failed_retryable", 0)
        + status.get("pending", 0),
        "claims": len(window_claims),
        "headline_claims": len(headline),
        "excluded_claims": len(window_claims) - len(headline),
        "forward_looking_claims": sum(1 for c in window_claims if c.get("is_forward_looking")),
        "testable_forecasts": sum(1 for c in window_claims if c.get("testable")),
        "review_required_claims": sum(1 for c in window_claims if c.get("review_required")),
        "unresolved_entities": sum(1 for c in window_claims
                                   if c.get("entity_resolution_status") in ("unresolved", "ambiguous")),
        "missing_timestamps": sum(1 for c in window_claims if c.get("evidence_start_seconds") is None),
        "legacy_claims": sum(1 for c in window_claims if c.get("schema_version") == "legacy"),
        "transcript_sources": dict(sources),
        "filters": "configured per-channel title filters (only=) and the minimum-duration gate; "
                   "a newly added channel contributes only its latest video",
    }


def format_quality_header(q):
    start, end = q["date_range"]
    lines = [
        f"📊 Research data quality — {start} → {end}",
        f"Among videos selected by the configured title and duration filters "
        f"during the reported period ({q['filters']}):",
        f"Discovered {q['discovered_videos']} · included {pct(q['included_videos'], q['discovered_videos'])}",
        f"Transcribed {pct(q['transcribed_videos'], q['research_videos'])} · fully processed "
        f"{pct(q['fully_processed_videos'], q['research_videos'])} · partial {q['partially_processed_videos']}"
        f" · awaiting retry {q['awaiting_retry_videos']}",
        f"Claims {q['claims']} · headline-eligible {pct(q['headline_claims'], q['claims'])} · excluded "
        f"{q['excluded_claims']} (review {q['review_required_claims']}, legacy {q['legacy_claims']})",
        f"Forward-looking {pct(q['forward_looking_claims'], q['claims'])} · testable forecasts "
        f"{pct(q['testable_forecasts'], q['forward_looking_claims'])}",
        f"Unresolved entities {pct(q['unresolved_entities'], q['claims'])} · claims without timestamps "
        f"{pct(q['missing_timestamps'], q['claims'])}",
    ]
    if q["transcript_sources"]:
        lines.append("Transcript sources: " + ", ".join(f"{k} {v}" for k, v in sorted(q["transcript_sources"].items())))
    if q["gate_outcomes"]:
        lines.append("Gate outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(q["gate_outcomes"].items())))
    return "\n".join(lines)


# --- Descriptive ------------------------------------------------------------


def descriptive(claims):
    """Counters over headline claims, with the distinct-unit views the report
    shows side by side (mentions vs claims vs videos vs sources). Per-asset
    counters use the canonical asset key; placeholder subjects have none."""
    out = {
        "claims_by_asset": Counter(), "videos_by_asset": defaultdict(set),
        "sources_by_asset": defaultdict(set), "segments_by_asset": defaultdict(set),
        "claims_by_channel": Counter(), "claims_by_month": Counter(),
        "claims_by_type": Counter(), "claims_by_sector": Counter(),
        "claims_by_certainty": Counter(), "claims_by_horizon": Counter(),
        "recommendations": Counter(), "stances": Counter(),
        "catalysts": Counter(), "risks": Counter(), "assumptions": Counter(),
    }
    for c in claims:
        key = _asset_key(c)
        if key:
            out["claims_by_asset"][key] += 1
            out["videos_by_asset"][key].add(c.get("video_id"))
            out["sources_by_asset"][key].add(_source(c))
            out["segments_by_asset"][key].add(c.get("segment_id"))
        out["claims_by_channel"][_source(c)] += 1
        d = _date(c.get("published_at"))
        if d:
            out["claims_by_month"][d.strftime("%Y-%m")] += 1
        out["claims_by_type"][c.get("claim_type")] += 1
        if c.get("sector"):
            out["claims_by_sector"][c["sector"]] += 1
        out["claims_by_certainty"][c.get("certainty_level")] += 1
        out["claims_by_horizon"][c.get("horizon_bucket") or "unspecified"] += 1
        if c.get("recommendation_action") not in (None, "none"):
            out["recommendations"][c["recommendation_action"]] += 1
        out["stances"][c.get("stance")] += 1
        for k in ("catalysts", "risks", "assumptions"):
            for item in c.get(k) or []:
                if isinstance(item, str):
                    out[k][item.strip().lower()] += 1
    return out


def net_stance(bull, bear):
    """(bull - bear) / (bull + bear), or None when there are no directional
    claims — never computed over a zero denominator."""
    total = bull + bear
    if not total:
        return None
    return (bull - bear) / total


# --- Consensus and flips ------------------------------------------------------


def latest_views(claims):
    """
    {asset: {bucket: {source: claim}}} — the LATEST directional or neutral
    view claim per source, per canonical asset key, per horizon bucket.
    Repeated claims inside one video therefore count once, a prolific channel
    is one source, and a placeholder subject (no key) never counts.
    """
    views = defaultdict(lambda: defaultdict(dict))
    for c in sorted(claims, key=lambda c: (c.get("published_at") or "", c.get("extracted_at") or "")):
        if not carries_view(c):
            continue  # disclosures, questions, third-party views never count
        key = _asset_key(c)
        if not key or canonical_claims.direction_of(c) is None:
            continue
        views[key][_bucket(c)][_source(c)] = c
    return views


def consensus(claims, start=None, end=None):
    """Per asset and horizon bucket: bullish/bearish/neutral SOURCE counts,
    independent-source count, date range, net stance (None when no
    directional sources)."""
    pool = [c for c in claims if start is None or _in_window(c, start, end)]
    out = {}
    for asset, buckets in latest_views(pool).items():
        for bucket, per_source in buckets.items():
            bull = [s for s, c in per_source.items() if _direction_of(c) == "bullish"]
            bear = [s for s, c in per_source.items() if _direction_of(c) == "bearish"]
            neutral = [s for s in per_source if s not in bull and s not in bear]
            dates = sorted(d for d in (_date(c.get("published_at")) for c in per_source.values()) if d)
            out[(asset, bucket)] = {
                "asset": asset, "horizon_bucket": bucket,
                "bullish_sources": sorted(bull), "bearish_sources": sorted(bear),
                "neutral_sources": sorted(neutral), "sources": len(per_source),
                "bullish": len(bull), "bearish": len(bear), "neutral": len(neutral),
                "net_stance": net_stance(len(bull), len(bear)),
                "date_range": (dates[0].isoformat(), dates[-1].isoformat()) if dates else None,
                "claim_ids": [c["claim_id"] for c in per_source.values()],
            }
    return out


def _reversals(claims):
    """
    [(previous VideoCall, new VideoCall, elapsed days or None)]: per
    (creator, asset), consecutive video-level calls
    (creator_tracker.video_calls: one lean per video over its directional,
    security-view, non-trigger claims; mixed and review-sensitive videos make
    no call) that point opposite ways.
    """
    vcalls, _ = creator_tracker.video_calls(list(claims or []))
    sequences = defaultdict(list)
    for vcall in vcalls:   # already in published order
        sequences[(vcall["creator"], vcall["asset"])].append(vcall)
    out = []
    for sequence in sequences.values():
        for prev, cur in zip(sequence, sequence[1:]):
            if prev["direction"] == cur["direction"]:
                continue
            gap = (cur["date"] - prev["date"]).days if prev["date"] and cur["date"] else None
            out.append((prev, cur, gap))
    return out


def _min_gap(min_gap_days):
    return MIN_FLIP_GAP_DAYS if min_gap_days is None else min_gap_days


def _union(claims, field):
    return {item for c in claims for item in (c.get(field) or []) if isinstance(item, str)}


def find_flips(claims, min_gap_days=None):
    """
    View changes: the same creator's video-level call on the same canonical
    asset reversing between two DIFFERENT videos at least MIN_FLIP_GAP_DAYS
    apart (see _reversals). Same-video scenario talk never flips; quicker
    reversals are counted by short_reversals instead. Each record keeps
    source, asset, horizon_bucket (the buckets the new video's claims name,
    joined, or "unspecified"), from, to, previous/new claim id (the first
    claim of each video), evidence and date, elapsed_days, the video ids, and
    changed_catalysts / changed_risks (new video's union minus the previous
    one's). Sorted by the new video's date.
    """
    gap_min = _min_gap(min_gap_days)
    claims = list(claims or [])
    by_id = {c.get("claim_id"): c for c in claims}
    flips = []
    for prev, cur, gap in _reversals(claims):
        if gap is None or gap < gap_min:
            continue
        old = [by_id[i] for i in prev["claim_ids"] if i in by_id]
        new = [by_id[i] for i in cur["claim_ids"] if i in by_id]
        named = [b for b in BUCKETS[:-1] if any(c.get("horizon_bucket") == b for c in new)]
        flips.append({
            "source": cur["creator"], "asset": cur["asset"], "horizon_bucket": "/".join(named) or "unspecified",
            "from": prev["direction"], "to": cur["direction"],
            "previous_claim_id": prev["claim_ids"][0] if prev["claim_ids"] else None,
            "new_claim_id": cur["claim_ids"][0] if cur["claim_ids"] else None,
            "previous_evidence": old[0].get("evidence_text") if old else None,
            "new_evidence": new[0].get("evidence_text") if new else None,
            "previous_date": prev["date"].isoformat(), "new_date": cur["date"].isoformat(),
            "elapsed_days": gap,
            "previous_video_id": prev["video_id"], "new_video_id": cur["video_id"],
            # Mostly guests' views in either video: on a panel channel a
            # reversal is often two different guests, not a change of mind.
            "guest_share": max(prev.get("guest_share") or 0.0, cur.get("guest_share") or 0.0),
            "changed_catalysts": sorted(_union(new, "catalysts") - _union(old, "catalysts")),
            "changed_risks": sorted(_union(new, "risks") - _union(old, "risks")),
        })
    flips.sort(key=lambda f: (f["new_date"], f["source"], f["asset"]))
    return flips


def short_reversals(claims, min_gap_days=None):
    """Counter {creator: n} of video-level reversals less than
    MIN_FLIP_GAP_DAYS apart: counted, never listed as view changes."""
    gap_min = _min_gap(min_gap_days)
    return Counter(cur["creator"] for _, cur, gap in _reversals(claims) if gap is not None and gap < gap_min)


def format_flip_lines(flips, short, min_gap_days=None, max_lines=MAX_FLIP_LINES):
    """The view-change block: heading, up to `max_lines` flips, and one line
    counting the short-term reversals per creator. [] when there is neither."""
    if not flips and not short:
        return []
    gap = _min_gap(min_gap_days)
    lines = [f"View changes (different videos ≥{gap} days apart; same-video scenarios ignored):"]
    for f in flips[:max_lines]:
        tag = f" [{f['horizon_bucket']}]" if f.get("horizon_bucket") else ""
        guests = "; guests' views" if (f.get("guest_share") or 0.0) > 0.5 else ""
        lines.append(f"• {f['source']} on {f['asset']}{tag}: {f['from']} → {f['to']} "
                     f"({f['previous_date']} → {f['new_date']}, {f['elapsed_days']}d{guests})")
    if len(flips) > max_lines:
        lines.append(f"…and {len(flips) - max_lines} more.")
    if not flips:
        lines.append("• none")
    if short:
        ordered = sorted(short.items(), key=lambda kv: (-kv[1], kv[0]))
        lines.append(f"Short-term reversals (<{gap} days apart, not counted as view changes): "
                     + " · ".join(f"{creator} {n}" for creator, n in ordered))
    return lines


# --- Scorecard --------------------------------------------------------------


def _symbol(claim):
    return canonical_claims.priceable_symbol(claim)


def _on_or_after(series, day):
    for offset in range(MAX_PRICE_LAG_DAYS + 1):
        d = day + timedelta(days=offset)
        if d in series:
            return d, series[d]
    return None, None


def _not_an_instrument(claim):
    """A ticker-less subject that is never one security: a macro, index,
    sector, commodity, currency or bond claim, or a market / basket / metric
    subject. A recorded ticker the registry cannot resolve is not this: it
    stays unresolved_instrument, a resolution gap to fix."""
    if claim.get("ticker") or canonical_claims.resolved_ticker(claim):
        return False
    return (claim.get("asset_type") in NOT_AN_INSTRUMENT_TYPES
            or claim.get("subject_kind") in _NOT_AN_INSTRUMENT_KINDS)


def _post_publication_range(series, entry):
    """
    The series a price target is tested on. When the video went up DURING the
    entry day's session (or, for crypto, during its UTC day), part of that
    day's high-low range printed before the video: only the day's close is
    known to come after it, so the entry day's high and low are replaced by
    its close. A series without bars, or an entry on a later day, is
    returned as it is.
    """
    day = _date(entry.get("resolved_trading_date"))
    if (entry.get("session_relation") != "during_session"
            or entry.get("requested_date") != entry.get("resolved_trading_date")
            or not series.highs or not series.lows or day not in series.closes):
        return series
    close = series.closes[day]
    return dataclasses.replace(series, highs={**series.highs, day: close}, lows={**series.lows, day: close})


def _plausible_target(target, entry_price, direction):
    """None when the level may be tested, else why not: outside
    creator_tracker.PLAUSIBLE_TARGET_RATIO of the entry (a misread revenue
    figure or market cap), or on the wrong side of it (a bullish "target"
    below the entry is reached the day it is made)."""
    low, high = creator_tracker.PLAUSIBLE_TARGET_RATIO
    if not entry_price or not (low <= target / entry_price <= high):
        return "implausible_level"
    if (target > entry_price) != (direction == "bullish"):
        return "wrong_side_of_entry"
    return None


def scorecard(claims, today, price_fetcher, min_sample=None, include_conditional=False):
    """
    Evaluate matured, testable, headline PRICE forecasts under
    scorecard_pricing: instrument metadata from the canonical registry (never
    from the claim's sector), exchange-aware next-close entry, split-adjusted
    (at least) series with recorded provenance, a per-claim benchmark or an
    honest null, price targets (canonical_claims.price_target_of within
    creator_tracker.PLAUSIBLE_TARGET_RATIO of the entry) by intraday touch /
    daily close / horizon close, and calendar confidence per claim. Each
    price series is fetched once per (symbol, start, end) window, so a second
    forecast or a benchmark on the same symbol gets its own window.

    Returns {source: {"n", "direction_hits", "target_hits", "target_n",
    "ranked_n", "ranked_direction_hits" (full-confidence calendars only),
    "raw_returns", "excess_returns", "mfe", "mae", "benchmark_methods",
    "target_test_methods", "calendar_confidence", "scored": [per-claim
    records]}} plus "_excluded" (why claims were not scored), "_min_sample"
    and "_rankings_enabled". Unconditional forecasts only, unless
    `include_conditional` — and then only those whose condition is
    recorded as met.
    """
    min_sample = MIN_SCORECARD_SAMPLE if min_sample is None else min_sample
    excluded = Counter()
    stats = defaultdict(lambda: {"n": 0, "direction_hits": 0, "target_hits": 0, "target_n": 0,
                                 "ranked_n": 0, "ranked_direction_hits": 0,
                                 "raw_returns": [], "excess_returns": [], "mfe": [], "mae": [],
                                 "benchmark_methods": Counter(), "target_test_methods": Counter(),
                                 "calendar_confidence": Counter(), "scored": []})
    series_cache = {}

    def series(symbol, start, end, exchange):
        key = (symbol, start, end)   # never by symbol alone: production fetchers slice to the window
        if key not in series_cache:
            series_cache[key] = sp.series_from_fetcher(price_fetcher, symbol, start, end, exchange)
        return series_cache[key]

    for c in claims:
        if not is_headline_claim(c):
            excluded["not_headline"] += 1
            continue
        if not c.get("is_forward_looking") or not c.get("testable"):
            excluded["not_testable"] += 1
            continue
        ttype = c.get("testability_type") or ("conditional_testable" if c.get("condition") else "unconditional_testable")
        if ttype == "conditional_testable":
            if not include_conditional:
                excluded["conditional"] += 1
                continue
            if c.get("condition_status") != "met":
                excluded[f"condition_{c.get('condition_status') or 'not_evaluated'}"] += 1
                continue
        if c.get("forecast_metric") not in SCORED_FORECAST_METRICS:
            excluded["non_price_metric"] += 1
            continue
        pub, end = _date(c.get("published_at")), _date(c.get("forecast_end_date"))
        direction = _direction_of(c)
        if not (pub and end and direction):
            excluded["missing_inputs"] += 1
            continue
        if end > today:
            excluded["not_matured"] += 1
            continue
        inst = instruments.resolve_instrument(c.get("ticker"), c.get("canonical_entity_name") or c.get("subject_mention"),
                                              c.get("asset_type"))
        if inst is None:
            excluded["not_an_instrument" if _not_an_instrument(c) else "unresolved_instrument"] += 1
            continue
        symbol = inst.symbol
        exchange = sp.exchange_for_instrument(inst)
        if exchange is None:
            excluded["unresolved_exchange"] += 1
            continue
        window_end = min(end + timedelta(days=MAX_PRICE_LAG_DAYS), today)
        ps = series(symbol, pub - timedelta(days=1), window_end, exchange)
        if not ps.scorable():
            excluded["unadjusted_or_unknown_prices" if ps.closes else "no_prices"] += 1
            continue
        entry = sp.entry_point(exchange, c.get("published_at"), ps.closes)
        exit_ = sp.evaluation_point(exchange, end, ps.closes)
        if entry["price"] is None or exit_["price"] is None:
            excluded["no_prices"] += 1
            continue
        entry_day, eval_day = _date(entry["resolved_trading_date"]), _date(exit_["resolved_trading_date"])
        if eval_day <= entry_day:
            excluded["evaluation_before_entry"] += 1
            continue
        ret = (exit_["price"] - entry["price"]) / entry["price"]
        bench_symbol, bench_method = instruments.benchmark_for(inst, exchange, symbol)
        excess, bench_return = None, None
        if bench_symbol:
            bs = series(bench_symbol, pub - timedelta(days=1), window_end, exchange)
            if bs.scorable():
                b_entry = sp.entry_point(exchange, c.get("published_at"), bs.closes)
                b_exit = sp.evaluation_point(exchange, end, bs.closes)
                if b_entry["price"] and b_exit["price"]:
                    bench_return = (b_exit["price"] - b_entry["price"]) / b_entry["price"]
                    excess = ret - bench_return
            if excess is None:
                bench_method = f"{bench_method}:no_benchmark_prices"
        window = [p for d, p in ps.closes.items() if entry_day <= d <= eval_day]
        mfe = (max(window) - entry["price"]) / entry["price"] if direction == "bullish" else (entry["price"] - min(window)) / entry["price"]
        mae = (entry["price"] - min(window)) / entry["price"] if direction == "bullish" else (max(window) - entry["price"]) / entry["price"]
        rankable = sp.rankable_calendar(exchange)
        bucket = stats[_source(c)]
        bucket["n"] += 1
        hit = 1 if (ret > 0) == (direction == "bullish") and ret != 0 else 0
        bucket["direction_hits"] += hit
        if rankable:
            bucket["ranked_n"] += 1
            bucket["ranked_direction_hits"] += hit
        bucket["raw_returns"].append(ret if direction == "bullish" else -ret)
        if excess is not None:
            bucket["excess_returns"].append(excess if direction == "bullish" else -excess)
        bucket["mfe"].append(mfe)
        bucket["mae"].append(mae)
        bucket["benchmark_methods"][bench_method if excess is not None else "none"] += 1
        bucket["calendar_confidence"][exchange.calendar_confidence] += 1
        target = canonical_claims.price_target_of(c)
        skipped = _plausible_target(target, entry["price"], direction) if target is not None else None
        target_record = {"target_value": None, "target_test_method": None, "target_reached": None,
                         "target_first_reached_date": None, "target_reached_within_window": None,
                         "horizon_close_target_met": None, "intraday_target_reached": None,
                         "daily_close_target_reached": None}
        if target is not None and skipped is None:
            bucket["target_n"] += 1
            target_record = sp.evaluate_target(_post_publication_range(ps, entry), float(target), direction,
                                               entry_day, eval_day)
            bucket["target_hits"] += 1 if target_record["target_reached"] else 0
            bucket["target_test_methods"][target_record["target_test_method"]] += 1
        bucket["scored"].append({
            "claim_id": c.get("claim_id"), "symbol": symbol, "direction": direction,
            "testability_type": ttype, "condition_status": c.get("condition_status"),
            **inst.metadata(), "speaker_sector": c.get("sector"), "speaker_asset_type": c.get("asset_type"),
            "exchange": exchange.code, "exchange_timezone": exchange.timezone,
            "calendar_confidence": exchange.calendar_confidence, "rankable": rankable,
            "calendar_note": None if exchange.calendar_confidence != "weekdays_only" else sp.WEEKDAY_CALENDAR_NOTE,
            "session_relation": entry["session_relation"], "entry_convention": entry["convention"],
            "entry_requested_date": entry["requested_date"], "entry_resolved_trading_date": entry["resolved_trading_date"],
            "entry_price": entry["price"], "evaluation_requested_date": exit_["requested_date"],
            "evaluation_resolved_trading_date": exit_["resolved_trading_date"], "evaluation_price": exit_["price"],
            "raw_return": ret, "benchmark_symbol": bench_symbol if excess is not None else None,
            "benchmark_method": bench_method, "benchmark_return": bench_return, "excess_return": excess,
            "direction_hit": bool(hit), "stated_target": target, "target_skipped": skipped,
            **target_record, **ps.provenance(),
        })
    out = {source: dict(v, benchmark_methods=dict(v["benchmark_methods"]),
                        target_test_methods=dict(v["target_test_methods"]),
                        calendar_confidence=dict(v["calendar_confidence"]))
           for source, v in stats.items()}
    out["_excluded"] = dict(excluded)
    out["_min_sample"] = min_sample
    out["_rankings_enabled"] = sp.SCORECARD_RANKINGS_ENABLED
    return out


def rankable_sources(sc, min_sample):
    """
    Sources that may be ranked against each other: rankings enabled, sample
    minimum met by forecasts on full-confidence calendars, and one shared
    benchmark method (a source scored against a sector ETF is not compared
    with one scored raw or against BTC).
    """
    if not sp.SCORECARD_RANKINGS_ENABLED:
        return []
    methods = {}
    for source, v in sc.items():
        # Only forecasts on a full-confidence calendar count toward the
        # sample (weekday-only calendars are scored but never ranked by
        # default — scorecard_pricing.rankable_calendar).
        if source.startswith("_") or v.get("ranked_n", v["n"]) < min_sample:
            continue
        used = {m for m in v.get("benchmark_methods", {}) if v["benchmark_methods"][m]}
        methods[source] = frozenset(used)
    if not methods:
        return []
    common = Counter(methods.values()).most_common(1)[0][0]
    return sorted(s for s, m in methods.items() if m == common)


# --- Conditional claims --------------------------------------------------------

# Statuses a recorded condition evaluation can carry, in listing order.
_EVALUATED_STATUSES = ("met", "partially_met", "not_met", "unknown")


def conditional_forecast_report(claims, today=None):
    """
    The separate conditional-forecast report: one row per conditional,
    non-legacy claim with its condition, whether it is objectively
    observable, the recorded condition outcome and the data source that
    would settle it. Never mixed into the unconditional scorecard. Each row
    also says whether it is the creator's own view (`own_view`,
    claims.carries_view), whether the condition is a trigger ("buy under
    $600") rather than a qualifier ("as long as support holds"; `trigger`,
    creator_tracker.is_trigger_condition), its canonical asset key (`asset`,
    None for a placeholder subject) and whether that subject resolved to a
    ticker or a known market (`resolved`).
    """
    rows = []
    for c in claims:
        if not c.get("condition") or c.get("schema_version") == "legacy":
            continue
        day = canonical_claims.claim_date(c)
        status = c.get("condition_status") or "not_evaluated"
        # Observability under the current rules (claims.condition_observability,
        # as entity resolution is re-run at load): "over the next 6 months" was
        # once read as a price level. A recorded evaluation keeps its source.
        observable, kind, source = _claims.condition_observability(c.get("condition"))
        if status in _EVALUATED_STATUSES and c.get("condition_data_source"):
            source = c.get("condition_data_source")
        rows.append({
            "source": _source(c), "asset": _asset_key(c), "condition": c.get("condition"),
            "observable": observable, "kind": kind,
            "status": status,
            "evaluation_date": c.get("condition_evaluation_date"), "data_source": source,
            "testability_type": c.get("testability_type"), "direction": _direction_of(c),
            "forecast_end_date": c.get("forecast_end_date"), "headline": is_headline_claim(c),
            "claim_id": c.get("claim_id"), "evidence": c.get("evidence_text"),
            "own_view": carries_view(c), "trigger": creator_tracker.is_trigger_condition(c),
            "resolved": _resolved_subject(c) is not None, "subject": c.get("subject_mention"),
            "date": day.isoformat() if day else None,
        })
    return sorted(rows, key=lambda r: (r["status"], r["source"], r["asset"] or ""))


def _newest_first(rows):
    return sorted(rows, key=lambda r: (r.get("date") or "", str(r.get("claim_id") or "")), reverse=True)


def _round_robin(rows):
    """Rows interleaved across creators (most rows first), each creator's
    newest first, so a handful of examples is not one creator's list."""
    groups = defaultdict(list)
    for r in _newest_first(rows):
        groups[r["source"]].append(r)
    order = sorted(groups, key=lambda s: (-len(groups[s]), s))
    out = []
    for i in range(max((len(g) for g in groups.values()), default=0)):
        out.extend(groups[s][i] for s in order if i < len(groups[s]))
    return out


def _condition_text(condition, limit=90):
    text = re.sub(r"^\s*if\s+", "", " ".join(str(condition or "").split()), flags=re.I)
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def format_conditional_report(rows, max_examples=CONDITIONAL_EXAMPLES):
    """
    The compact conditional block (at most max_examples + 3 lines):
      1. how many conditional claims are the creators' own views, per creator
         (top 6), and the condition statuses;
      2. how many hinge on a trigger vs a qualifier, and how many are
         observable (on a price level);
      then example lines — rows with a recorded evaluation first, whatever
      their own-view status, then observable own-view triggers interleaved
      across creators, newest first — and one line counting what was not
      listed. A row whose subject did not resolve to a ticker or a known
      market is counted, never printed.
    """
    if not rows:
        return ""
    own = [r for r in rows if r.get("own_view")]
    by_creator = Counter(r["source"] for r in own)
    by_status = Counter(r["status"] for r in rows)
    head = f"🔀 Conditional calls (scenarios, kept out of the track record): {len(own)} own-view of {len(rows)}"
    creators = " · ".join(f"{s} {n}" for s, n in sorted(by_creator.items(), key=lambda kv: (-kv[1], kv[0]))[:6])
    if creators:
        head += f" — {creators}"
    head += "; condition status " + ", ".join(
        f"{k} {v}" for k, v in sorted(by_status.items(), key=lambda kv: (-kv[1], kv[0])))
    lines = [head]
    if own:
        triggers = sum(1 for r in own if r.get("trigger"))
        observable = [r for r in own if r.get("observable")]
        on_price = sum(1 for r in observable if r.get("kind") == "price_level")
        lines.append(f"{triggers} hinge on a trigger (e.g. buy under $600), {len(own) - triggers} on a qualifier "
                     f"(as long as support holds); {len(observable)} are observable ({on_price} on a price level).")
    # Rows that could be listed: every evaluated row, and the creators' own
    # observable triggers. An own-view or evaluated row on an unresolved
    # subject is only counted.
    unresolved = sum(1 for r in rows if not r.get("resolved")
                     and (r.get("own_view") or r["status"] in _EVALUATED_STATUSES))
    listable = [r for r in rows if r.get("resolved") and (
        r["status"] in _EVALUATED_STATUSES or (r.get("own_view") and r.get("trigger") and r.get("observable")))]
    # Evaluated rows first (met, partially met, not met, unknown; newest
    # evaluation first), then the triggers interleaved across creators.
    evaluated = sorted(
        sorted((r for r in listable if r["status"] in _EVALUATED_STATUSES),
               key=lambda r: (r.get("evaluation_date") or "", r.get("date") or ""), reverse=True),
        key=lambda r: _EVALUATED_STATUSES.index(r["status"]))
    watched = [r for r in listable if r["status"] not in _EVALUATED_STATUSES]
    limit = max(0, max_examples or 0)
    shown, seen = [], set()
    for r in evaluated + _round_robin(watched):
        key = (r["source"], r["asset"], _condition_text(r["condition"]).lower())
        if key in seen:
            continue
        seen.add(key)
        if len(shown) < limit:
            shown.append(r)
    for r in shown:
        obs = "observable" if r["observable"] else "subjective"
        src = f" via {r['data_source']}" if r["data_source"] else ""
        lines.append(f"• {r['source']} on {r['asset']} [{r['direction'] or 'n/a'}] if "
                     f"\"{_condition_text(r['condition'])}\" — {obs}{src}; condition {r['status']}")
    more = len(seen) - len(shown)
    parts = ([f"{unresolved} on unresolved subjects"] if unresolved else []) + \
            ([f"{more} more observable trigger{'' if more == 1 else 's'}"] if more else [])
    if parts:
        lines.append("Not listed: " + " · ".join(parts) + ".")
    return "\n".join(lines)


# --- Scorecard lines ------------------------------------------------------------

# Why a forecast was not scored, in reader words, nearest to being scored
# first. condition_<status> reasons (include_conditional) sit with "conditional".
_EXCLUSION_WORDS = (
    ("evaluation_before_entry", "with the evaluation close not after the entry"),
    ("no_prices", "without prices for the window"),
    ("unadjusted_or_unknown_prices", "with prices of unknown adjustment"),
    ("unresolved_exchange", "on an exchange without session rules"),
    ("unresolved_instrument", "with a ticker not in the instrument registry"),
    ("not_an_instrument", "not an instrument (macro, index, sector or commodity subject)"),
    ("not_matured", "dated price forecast{s} not matured yet"),
    ("missing_inputs", "without a date or a direction"),
    ("non_price_metric", "business-metric forecast{s} (revenue, margins, cash flow, rates; not price calls)"),
    ("conditional", "conditional (scored only once the condition is met)"),
    ("not_testable", "not testable (no dated, measurable forecast)"),
    ("not_headline", "not the creator's own reviewed view"),
)
_EXCLUSION_RANK = {key: i for i, (key, _) in enumerate(_EXCLUSION_WORDS)}


def _exclusion_phrase(reason, n):
    words = dict(_EXCLUSION_WORDS).get(reason)
    if words is None and reason.startswith("condition_"):
        words = f"conditional, condition {reason[len('condition_'):].replace('_', ' ')}"
    words = (words or reason.replace("_", " ")).replace("{s}", "" if n == 1 else "s")
    return f"{n:,} {words}"


def _level(value):
    """A price level as read: 87,250 / 181.50 / 0.0866."""
    if not isinstance(value, (int, float)):
        return "?"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.2f}" if abs(value) >= 1 else f"{value:.4g}"


def format_exclusions(excluded):
    """'Not scored: 83 dated forecasts not matured yet · 1,214 not testable
    (...) · ...' — every reason with its count, nearest to scoring first."""
    if not excluded:
        return ""

    def rank(reason):
        if reason in _EXCLUSION_RANK:
            return _EXCLUSION_RANK[reason]
        if reason.startswith("condition_"):
            return _EXCLUSION_RANK["conditional"] - 0.5
        return len(_EXCLUSION_RANK)

    ordered = sorted(excluded.items(), key=lambda kv: (rank(kv[0]), kv[0]))
    return "Not scored: " + " · ".join(_exclusion_phrase(k, v) for k, v in ordered if v) + "."


def _claim_line(s):
    """One scored forecast, e.g. 'ETH ↓ 2026-09-17→2026-10-08: +1.1% vs
    btcusd +7.0% — wrong direction'."""
    arrow = "↑" if s.get("direction") == "bullish" else "↓"
    label = s.get("instrument_ticker") or s.get("symbol") or "?"
    line = (f"   {label} {arrow} {s.get('entry_resolved_trading_date')}→{s.get('evaluation_resolved_trading_date')}: "
            f"{100 * s.get('raw_return', 0.0):+.1f}%")
    if s.get("benchmark_return") is not None and s.get("benchmark_symbol"):
        line += f" vs {s['benchmark_symbol']} {100 * s['benchmark_return']:+.1f}%"
    if not s.get("raw_return"):
        verdict = "no move"
    else:
        verdict = "right direction" if (s["raw_return"] > 0) == (s.get("direction") == "bullish") else "wrong direction"
    line += f" — {verdict}"
    if s.get("target_test_method"):
        line += (f"; target {_level(s.get('target_value'))} "
                 f"{'reached' if s.get('target_reached') else 'not reached'} ({s['target_test_method']})")
    return line


def format_scorecard_lines(sc, excluded, min_sample, rankings_on, rankable):
    """The scorecard block: experimental and unranked unless rankings are
    enabled AND the sources share a benchmark method and the sample minimum.
    Each source line names its target and benchmark methods and the
    benchmark symbols; a source with fewer than CLAIM_LINES_BELOW_N scored
    forecasts lists each one; the exclusions close the block in words."""
    label = "ranked" if rankings_on and rankable else sp.SCORECARD_EXPERIMENTAL_LABEL
    lines = [f"Scorecard ({label}; matured, testable, evidence-backed unconditional price forecasts; "
             f"exchange-aware next-close entry, split-adjusted prices; sample minimum {min_sample}):"]
    sources = {k: v for k, v in (sc or {}).items() if not str(k).startswith("_")}
    if rankings_on and rankable:
        order = sorted(sources.items(), key=lambda kv: (
            kv[0] not in rankable,
            -(kv[1].get("ranked_direction_hits", kv[1]["direction_hits"]) / kv[1]["ranked_n"]
              if kv[1].get("ranked_n") else 0),
            kv[0]))
    else:
        order = sorted(sources.items(), key=lambda kv: kv[0])
    for source, v in order:
        med = _median(v["raw_returns"])
        mean = sum(v["raw_returns"]) / len(v["raw_returns"]) if v["raw_returns"] else 0.0
        methods = ", ".join(f"{k} {n}" for k, n in sorted(v.get("benchmark_methods", {}).items()))
        symbols = Counter(s.get("benchmark_symbol") for s in v.get("scored") or [] if s.get("benchmark_symbol"))
        versus = f" vs {', '.join(sym for sym, _ in sorted(symbols.items(), key=lambda kv: (-kv[1], kv[0])))}" \
            if symbols else ""
        excess = (f", excess vs benchmark mean {100 * sum(v['excess_returns']) / len(v['excess_returns']):+.1f}%"
                  if v["excess_returns"] else ", benchmark n/a")
        if rankings_on and rankable:
            note = "" if source in rankable else " (unranked: below sample minimum or different benchmark method)"
        else:
            note = " (unranked)"
        weekday = (v.get("calendar_confidence") or {}).get("weekdays_only", 0)
        calendar = f", {weekday} on weekday-only calendars (not ranked)" if weekday else ""
        target_methods = ", ".join(f"{k} {n}" for k, n in sorted((v.get("target_test_methods") or {}).items()))
        lines.append(
            f"• {source}: direction {pct(v['direction_hits'], v['n'])}, target "
            f"{pct(v['target_hits'], v['target_n'])}"
            + (f" ({target_methods})" if target_methods else "")
            + f", return in the called direction mean {100*mean:+.1f}% median "
            f"{100*(med or 0):+.1f}%{excess}, n={v['n']}{calendar}, benchmark {methods or 'none'}{versus}{note}"
        )
        if v["n"] < CLAIM_LINES_BELOW_N:
            lines += [_claim_line(s) for s in v.get("scored") or []]
    if not sources:
        lines.append("• no matured forecasts yet")
    if excluded:
        lines.append(format_exclusions(excluded))
    return "\n".join(lines)


def _median(values):
    if not values:
        return None
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


# --- Report -----------------------------------------------------------------


def build_report(claims, gate_outcomes, state, runs, start, end, today=None, price_fetcher=None,
                 max_assets=8, tracker_record=None):
    """The research report. `tracker_record` (creator_tracker.track_record)
    adds the creators-vs-market summary after the consensus block."""
    today = today or end
    lines = [format_quality_header(quality_header(claims, gate_outcomes, state, runs, start, end)), ""]
    window = [c for c in claims if _in_window(c, start, end)]
    headline = [c for c in window if is_headline_claim(c)]
    desc = descriptive(headline)
    if desc["claims_by_asset"]:
        lines.append("Top assets (headline claims · unique claims / segments / videos / sources):")
        for asset, n in desc["claims_by_asset"].most_common(max_assets):
            lines.append(
                f"• {asset} — {n} claim(s) / {len(desc['segments_by_asset'][asset])} segment(s) / "
                f"{len(desc['videos_by_asset'][asset])} video(s) / {len(desc['sources_by_asset'][asset])} source(s)"
            )
        lines.append("")
    if desc["stances"]:
        lines.append("Stance distribution: " + ", ".join(f"{k} {v}" for k, v in desc["stances"].most_common()))
    if desc["claims_by_horizon"]:
        lines.append("Horizons: " + ", ".join(f"{k} {v}" for k, v in desc["claims_by_horizon"].most_common()))
    if desc["claims_by_certainty"]:
        lines.append("Certainty (language strength, not probability): "
                     + ", ".join(f"{k} {v}" for k, v in desc["claims_by_certainty"].most_common()))
    if desc["recommendations"]:
        lines.append("Explicit recommendations: " + ", ".join(f"{k} {v}" for k, v in desc["recommendations"].most_common()))
    for label, key in (("Recurring catalysts", "catalysts"), ("Recurring risks", "risks")):
        top = [(k, v) for k, v in desc[key].most_common(5) if v > 1]
        if top:
            lines.append(f"{label}: " + ", ".join(f"{k} ×{v}" for k, v in top))
    cons = consensus(window)
    if cons:
        lines.append("")
        lines.append("Consensus (one current view per source, per asset, per horizon bucket):")
        ranked = sorted(cons.values(), key=lambda e: (-e["sources"], e["asset"], e["horizon_bucket"]))
        for e in ranked[:max_assets]:
            net = "n/a" if e["net_stance"] is None else f"{e['net_stance']:+.2f}"
            rng = f" {e['date_range'][0]}→{e['date_range'][1]}" if e["date_range"] else ""
            lines.append(
                f"• {e['asset']} [{e['horizon_bucket']}] — {e['bullish']}↑ {e['bearish']}↓ "
                f"{e['neutral']}· from {e['sources']} source(s), net {net}{rng}"
            )
    if tracker_record is not None:
        tracker = creator_tracker.format_summary_lines(tracker_record)
        if tracker:
            lines.append("")
            lines += tracker
    flip_lines = format_flip_lines(find_flips(window), short_reversals(window))
    if flip_lines:
        lines.append("")
        lines += flip_lines
    disclosures = canonical_claims.format_portfolio_disclosures(
        canonical_claims.portfolio_disclosures(window))
    if disclosures:
        lines.append("")
        lines.append(disclosures)
    conditional = format_conditional_report(conditional_forecast_report(window, today))
    if conditional:
        lines.append("")
        lines.append(conditional)
    if price_fetcher is not None:
        sc = scorecard(claims, today, price_fetcher)
        excluded, min_sample = sc.pop("_excluded"), sc.pop("_min_sample")
        rankings_on = sc.pop("_rankings_enabled")
        rankable = rankable_sources(dict(sc, _min_sample=min_sample), min_sample)
        lines.append("")
        lines.append(format_scorecard_lines(sc, excluded, min_sample, rankings_on, rankable))
    lines.append("")
    lines.append("⚠️ Extracted creator statements with evidence — research input, not investment advice.")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Research analytics over canonical claims")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-scorecard", action="store_true",
                        help="skip the strict scorecard and the creators-vs-market block (no prices read)")
    args = parser.parse_args(argv)
    today = datetime.now(timezone.utc).date()
    state = research_state.load_state()
    claims = canonical_claims.load_canonical_claims(state)
    fetcher, record = None, None
    if not args.no_scorecard:
        try:
            import channel_scorecard
            fetcher = channel_scorecard.fetch_price_series
        except Exception as e:  # pragma: no cover - import guard
            log_warn(f"Scorecard unavailable: {e}")
        # The mark-to-market block reads the committed price cache only.
        try:
            record = creator_tracker.track_record(claims, creator_tracker.default_as_of(today),
                                                  creator_tracker.cache_fetcher())
        except Exception as e:  # noqa: BLE001 - the block is optional; the report must still go out
            log_warn(f"Creators-vs-market block unavailable: {e}")
    report = build_report(claims, research_state.load_gate_outcomes(), state, research_state.load_runs(),
                          today - timedelta(days=args.days), today, today=today, price_fetcher=fetcher,
                          tracker_record=record)
    if args.dry_run or not os.getenv("TELEGRAM_TOKEN"):
        print(report)
        return 0
    from sendToTelegram import send_telegram_text
    ok = send_telegram_text(os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHANNEL_ID"), report)
    log_info("Research report sent." if ok else "Research report failed to send.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
