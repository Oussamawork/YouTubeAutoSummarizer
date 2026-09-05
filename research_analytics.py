"""
Analytics over validated canonical claims — never over summary prose.

Every report starts with a data-quality header that states the universe
honestly (discovered vs included vs transcribed vs fully processed), shows
every percentage with its denominator, and separates what headline analytics
include from what they exclude. Headline analytics use only claims that are
active (not superseded), not review-required, fully covered, evidence-located
and the source's own view (`claims.is_headline_claim`). Excluded counts are
shown, not hidden.

Consensus counts at most one CURRENT view per source per asset per horizon
bucket; short- and long-term views are never merged. A flip is the same
source changing directional view on the same asset in the same bucket. The
scorecard evaluates only matured, testable, evidence-backed forecasts, with
the entry/evaluation conventions documented on `SCORECARD_RULES`.
"""
import argparse
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from claims import is_headline_claim, carries_view, OWN_VIEW_ATTRIBUTIONS
from log import log_info, log_warn
import canonical_claims
import instruments
import research_state
import scorecard_pricing as sp

BUCKETS = ("short", "medium", "long", "unspecified")
# Rank a source only once it has this many matured forecasts. An analytical
# rule, configurable — not a claim of statistical reliability.
MIN_SCORECARD_SAMPLE = int(os.getenv("MIN_SCORECARD_SAMPLE") or 20)
MAX_PRICE_LAG_DAYS = sp.MAX_PRICE_LAG_DAYS

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
    "instrument_metadata": "exchange, asset type, country, currency, sector and benchmark come from the "
                           "canonical instrument registry (instruments.py: curated table, else the "
                           "provider-verified ticker map); a claim's model-written sector is recorded as "
                           "speaker_sector and never selects a benchmark; an unresolved instrument is excluded",
    "trading_calendar": "NYSE holiday rules for US listings (calendar_confidence=full); continuous for "
                        "crypto; weekday-only elsewhere (weekdays_only: local holidays are NOT modelled and "
                        "no bound on the resulting date error is claimed) — such claims are scored and "
                        "labelled but excluded from source rankings unless SCORECARD_RANK_WEEKDAY_CALENDARS=true",
    "price_targets": "reach/hit targets are tested by intraday_touch (daily high/low) when bars are "
                     "available, else by daily_close; horizon_close (the evaluation-date close) is recorded "
                     "beside them; a target reached inside the window is reached even if the horizon close "
                     "moved away; with closes only, intraday reach is recorded as unknown, never false",
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

UP = {"increase", "recover", "outperform"}
DOWN = {"decrease", "decline", "underperform"}


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
    return claim.get("channel_name") or claim.get("channel_id") or "unknown"


def _asset_key(claim):
    return claim.get("ticker") or " ".join((claim.get("subject_mention") or "").split()).upper() or None


def _direction_of(claim):
    """bullish / bearish / None for a claim, from stance first and then the
    forecast direction."""
    if claim.get("stance") in ("bullish", "bearish"):
        return claim["stance"]
    d = claim.get("forecast_direction")
    if d in UP:
        return "bullish"
    if d in DOWN:
        return "bearish"
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
    shows side by side (mentions vs claims vs videos vs sources)."""
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
    {asset: {bucket: {source: claim}}} — the LATEST directional headline
    claim per source, per asset, per horizon bucket. Repeated claims inside
    one video therefore count once, and a prolific channel is one source.
    """
    views = defaultdict(lambda: defaultdict(dict))
    for c in sorted(claims, key=lambda c: (c.get("published_at") or "", c.get("extracted_at") or "")):
        key = _asset_key(c)
        direction = _direction_of(c)
        if not key or not carries_view(c):
            continue  # disclosures, questions, third-party views never count
        bucket = c.get("horizon_bucket") or "unspecified"
        stance = direction or ("neutral" if c.get("stance") in ("neutral", "mixed") else None)
        if stance is None:
            continue
        views[key][bucket][_source(c)] = c
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


def find_flips(claims):
    """Same source, same asset, comparable horizon bucket, opposing
    directional stance. Both claims and both pieces of evidence are kept."""
    history = defaultdict(list)
    for c in sorted(claims, key=lambda c: (c.get("published_at") or "", c.get("extracted_at") or "")):
        if not carries_view(c):
            continue
        direction = _direction_of(c)
        key = _asset_key(c)
        if not direction or not key:
            continue
        history[(_source(c), key, c.get("horizon_bucket") or "unspecified")].append(c)
    flips = []
    for (source, asset, bucket), seq in history.items():
        for prev, cur in zip(seq, seq[1:]):
            if {_direction_of(prev), _direction_of(cur)} == {"bullish", "bearish"}:
                d0, d1 = _date(prev.get("published_at")), _date(cur.get("published_at"))
                flips.append({
                    "source": source, "asset": asset, "horizon_bucket": bucket,
                    "from": _direction_of(prev), "to": _direction_of(cur),
                    "previous_claim_id": prev["claim_id"], "new_claim_id": cur["claim_id"],
                    "previous_evidence": prev.get("evidence_text"), "new_evidence": cur.get("evidence_text"),
                    "previous_date": d0.isoformat() if d0 else None,
                    "new_date": d1.isoformat() if d1 else None,
                    "elapsed_days": (d1 - d0).days if d0 and d1 else None,
                    "changed_catalysts": sorted(set(cur.get("catalysts") or []) - set(prev.get("catalysts") or [])),
                    "changed_risks": sorted(set(cur.get("risks") or []) - set(prev.get("risks") or [])),
                })
    return flips


# --- Scorecard --------------------------------------------------------------


def _symbol(claim):
    return canonical_claims.priceable_symbol(claim)


def _on_or_after(series, day):
    for offset in range(MAX_PRICE_LAG_DAYS + 1):
        d = day + timedelta(days=offset)
        if d in series:
            return d, series[d]
    return None, None


def scorecard(claims, today, price_fetcher, min_sample=None, include_conditional=False):
    """
    Evaluate matured, testable, headline forecasts under scorecard_pricing:
    instrument metadata from the canonical registry (never from the claim's
    sector), exchange-aware next-close entry, split-adjusted (at least)
    series with recorded provenance, a per-claim benchmark or an honest
    null, price targets by intraday touch / daily close / horizon close,
    and calendar confidence per claim.

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
        if symbol not in series_cache:
            series_cache[symbol] = sp.series_from_fetcher(price_fetcher, symbol, start, end, exchange)
        return series_cache[symbol]

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
            excluded["unresolved_instrument"] += 1
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
        target = c.get("target_value")
        target_record = {"target_test_method": None, "target_reached": None, "target_first_reached_date": None,
                         "target_reached_within_window": None, "horizon_close_target_met": None,
                         "intraday_target_reached": None, "daily_close_target_reached": None}
        if c.get("target_kind") == "absolute_value" and target:
            bucket["target_n"] += 1
            target_record = sp.evaluate_target(ps, float(target), direction, entry_day, eval_day)
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


def conditional_forecast_report(claims, today=None):
    """
    The separate conditional-forecast report: every conditional forecast
    with its condition, whether it is objectively observable, the recorded
    condition outcome and the data source that would settle it. Never
    mixed into the unconditional scorecard.
    """
    rows = []
    for c in claims:
        if not c.get("condition") or c.get("schema_version") == "legacy":
            continue
        rows.append({
            "source": _source(c), "asset": _asset_key(c), "condition": c.get("condition"),
            "observable": c.get("condition_observable"), "kind": c.get("condition_kind"),
            "status": c.get("condition_status") or "not_evaluated",
            "evaluation_date": c.get("condition_evaluation_date"), "data_source": c.get("condition_data_source"),
            "testability_type": c.get("testability_type"), "direction": _direction_of(c),
            "forecast_end_date": c.get("forecast_end_date"), "headline": is_headline_claim(c),
            "claim_id": c.get("claim_id"), "evidence": c.get("evidence_text"),
        })
    return sorted(rows, key=lambda r: (r["status"], r["source"], r["asset"] or ""))


def format_conditional_report(rows):
    if not rows:
        return ""
    by_status = Counter(r["status"] for r in rows)
    lines = ["🔀 Conditional forecasts (kept out of unconditional rankings): "
             + ", ".join(f"{k} {v}" for k, v in sorted(by_status.items()))]
    for r in rows[:15]:
        obs = "observable" if r["observable"] else "subjective"
        src = f" via {r['data_source']}" if r["data_source"] else ""
        lines.append(f"• {r['source']} on {r['asset'] or 'unresolved'} [{r['direction'] or 'n/a'}] if "
                     f"\"{r['condition']}\" — {obs}{src}; condition {r['status']}")
    if len(rows) > 15:
        lines.append(f"…and {len(rows) - 15} more.")
    return "\n".join(lines)


def format_scorecard_lines(sc, excluded, min_sample, rankings_on, rankable):
    """The scorecard block: experimental and unranked unless rankings are
    enabled AND the sources share a benchmark method and the sample minimum."""
    label = "ranked" if rankings_on and rankable else sp.SCORECARD_EXPERIMENTAL_LABEL
    lines = [f"Scorecard ({label}; matured, testable, evidence-backed unconditional forecasts; "
             f"exchange-aware next-close entry, split-adjusted prices; sample minimum {min_sample}):"]
    if rankings_on and rankable:
        order = sorted(sc.items(), key=lambda kv: (
            kv[0] not in rankable,
            -(kv[1].get("ranked_direction_hits", kv[1]["direction_hits"]) / kv[1]["ranked_n"]
              if kv[1].get("ranked_n") else 0),
            kv[0]))
    else:
        order = sorted(sc.items(), key=lambda kv: kv[0])
    for source, v in order:
        med = _median(v["raw_returns"])
        mean = sum(v["raw_returns"]) / len(v["raw_returns"]) if v["raw_returns"] else 0.0
        methods = ", ".join(f"{k} {n}" for k, n in sorted(v.get("benchmark_methods", {}).items()))
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
            + f", return mean {100*mean:+.1f}% median "
            f"{100*(med or 0):+.1f}%{excess}, n={v['n']}{calendar}, benchmark {methods or 'none'}{note}"
        )
    if not sc:
        lines.append("• no matured forecasts yet")
    if excluded:
        lines.append("Excluded from scoring: " + ", ".join(f"{k} {v}" for k, v in sorted(excluded.items())))
    return "\n".join(lines)


def _median(values):
    if not values:
        return None
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


# --- Report -----------------------------------------------------------------


def build_report(claims, gate_outcomes, state, runs, start, end, today=None, price_fetcher=None,
                 max_assets=8):
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
    flips = find_flips(window)
    if flips:
        lines.append("")
        lines.append("Stance changes (same source, asset and horizon; a changed view is not a wrong one):")
        for f in flips[:10]:
            lines.append(f"• {f['source']} on {f['asset']} [{f['horizon_bucket']}]: {f['from']} → {f['to']} "
                         f"({f['previous_date']} → {f['new_date']}, {f['elapsed_days']}d)")
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


def main():
    parser = argparse.ArgumentParser(description="Research analytics over canonical claims")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-scorecard", action="store_true")
    args = parser.parse_args()
    today = datetime.now(timezone.utc).date()
    state = research_state.load_state()
    claims = canonical_claims.load_canonical_claims(state)
    fetcher = None
    if not args.no_scorecard:
        try:
            import channel_scorecard
            fetcher = channel_scorecard.fetch_prices
        except Exception as e:  # pragma: no cover - import guard
            log_warn(f"Scorecard unavailable: {e}")
    report = build_report(claims, research_state.load_gate_outcomes(), state, research_state.load_runs(),
                          today - timedelta(days=args.days), today, today=today, price_fetcher=fetcher)
    if args.dry_run or not os.getenv("TELEGRAM_TOKEN"):
        print(report)
        return 0
    from sendToTelegram import send_telegram_text
    ok = send_telegram_text(os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHANNEL_ID"), report)
    log_info("Research report sent." if ok else "Research report failed to send.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
