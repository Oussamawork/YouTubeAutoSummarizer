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

from claims import is_headline_claim, OWN_VIEW_ATTRIBUTIONS
from log import log_info, log_warn
import research_state

BUCKETS = ("short", "medium", "long", "unspecified")
# Rank a source only once it has this many matured forecasts. An analytical
# rule, configurable — not a claim of statistical reliability.
MIN_SCORECARD_SAMPLE = int(os.getenv("MIN_SCORECARD_SAMPLE") or 20)
MAX_PRICE_LAG_DAYS = 5
BENCHMARK_SYMBOL = "spy.us"

SCORECARD_RULES = {
    "publication_timestamp": "published_at from the video feed (UTC)",
    "entry_price": "first daily close ON or AFTER the publication date (next-close rule), "
                   f"within {MAX_PRICE_LAG_DAYS} calendar days",
    "evaluation_date": "forecast_end_date (resolved by claims.resolve_horizon); evaluated at the "
                       f"first close on or after it, within {MAX_PRICE_LAG_DAYS} days",
    "timezone": "UTC dates; the price provider's close is its venue's session close",
    "trading_calendar": "weekends/holidays roll forward to the next available close",
    "adjusted_prices": "provider daily closes as served (Twelve Data time_series, unadjusted)",
    "benchmark": f"{BENCHMARK_SYMBOL} over the same entry/evaluation dates when available",
    "conditional_forecasts": "excluded (testability issue conditional_outcome_not_observable)",
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
        if not key or not is_headline_claim(c):
            continue
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
        if not is_headline_claim(c):
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
    ticker = (claim.get("ticker") or "").lower()
    if not ticker:
        return None
    if claim.get("asset_type") == "crypto":
        return f"{ticker}usd"
    if claim.get("asset_type") in ("stock", "etf", None):
        return f"{ticker}.us"
    return None


def _on_or_after(series, day):
    for offset in range(MAX_PRICE_LAG_DAYS + 1):
        d = day + timedelta(days=offset)
        if d in series:
            return d, series[d]
    return None, None


def scorecard(claims, today, price_fetcher, min_sample=None):
    """
    Evaluate matured, testable, headline forecasts. Returns
    {source: {"n": int, "direction_hits": int, "target_hits": int,
              "target_n": int, "raw_returns": [...], "excess_returns": [...],
              "mfe": [...], "mae": [...]}} plus a "_excluded" Counter saying
    why claims were not scored. Sources under `min_sample` are reported but
    not ranked (see build_report).
    """
    min_sample = MIN_SCORECARD_SAMPLE if min_sample is None else min_sample
    excluded = Counter()
    stats = defaultdict(lambda: {"n": 0, "direction_hits": 0, "target_hits": 0, "target_n": 0,
                                 "raw_returns": [], "excess_returns": [], "mfe": [], "mae": []})
    series_cache = {}

    def series(symbol, start, end):
        if symbol not in series_cache:
            series_cache[symbol] = price_fetcher(symbol, start, end) or {}
        return series_cache[symbol]

    for c in claims:
        if not is_headline_claim(c):
            excluded["not_headline"] += 1
            continue
        if not c.get("is_forward_looking") or not c.get("testable"):
            excluded["not_testable"] += 1
            continue
        pub, end = _date(c.get("published_at")), _date(c.get("forecast_end_date"))
        symbol = _symbol(c)
        direction = _direction_of(c)
        if not (pub and end and symbol and direction):
            excluded["missing_inputs"] += 1
            continue
        if end > today:
            excluded["not_matured"] += 1
            continue
        prices = series(symbol, pub, min(end + timedelta(days=MAX_PRICE_LAG_DAYS), today))
        entry_day, entry = _on_or_after(prices, pub)
        eval_day, exit_ = _on_or_after(prices, end)
        if not entry or not exit_:
            excluded["no_prices"] += 1
            continue
        ret = (exit_ - entry) / entry
        bench = series(BENCHMARK_SYMBOL, pub, min(end + timedelta(days=MAX_PRICE_LAG_DAYS), today))
        b_entry, b_exit = _on_or_after(bench, pub)[1], _on_or_after(bench, end)[1]
        excess = ret - ((b_exit - b_entry) / b_entry) if b_entry and b_exit else None
        window = [p for d, p in prices.items() if entry_day <= d <= eval_day]
        mfe = (max(window) - entry) / entry if direction == "bullish" else (entry - min(window)) / entry
        mae = (entry - min(window)) / entry if direction == "bullish" else (max(window) - entry) / entry
        bucket = stats[_source(c)]
        bucket["n"] += 1
        bucket["direction_hits"] += 1 if (ret > 0) == (direction == "bullish") and ret != 0 else 0
        bucket["raw_returns"].append(ret if direction == "bullish" else -ret)
        if excess is not None:
            bucket["excess_returns"].append(excess if direction == "bullish" else -excess)
        bucket["mfe"].append(mfe)
        bucket["mae"].append(mae)
        target = c.get("target_value")
        if c.get("target_kind") == "absolute_value" and target:
            bucket["target_n"] += 1
            reached = max(window) >= target if direction == "bullish" else min(window) <= target
            bucket["target_hits"] += 1 if reached else 0
    out = {source: dict(v) for source, v in stats.items()}
    out["_excluded"] = dict(excluded)
    out["_min_sample"] = min_sample
    return out


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
    if price_fetcher is not None:
        sc = scorecard(claims, today, price_fetcher)
        excluded, min_sample = sc.pop("_excluded"), sc.pop("_min_sample")
        lines.append("")
        lines.append(f"Scorecard (matured, testable, evidence-backed forecasts; ranked from {min_sample}):")
        ranked = sorted(sc.items(), key=lambda kv: (kv[1]["n"] < min_sample,
                                                    -(kv[1]["direction_hits"] / kv[1]["n"] if kv[1]["n"] else 0)))
        for source, v in ranked:
            med = _median(v["raw_returns"])
            mean = sum(v["raw_returns"]) / len(v["raw_returns"]) if v["raw_returns"] else 0.0
            note = "" if v["n"] >= min_sample else " (unranked: below sample minimum)"
            lines.append(
                f"• {source}: direction {pct(v['direction_hits'], v['n'])}, target "
                f"{pct(v['target_hits'], v['target_n'])}, return mean {100*mean:+.1f}% median "
                f"{100*(med or 0):+.1f}%, n={v['n']}{note}"
            )
        if not sc:
            lines.append("• no matured forecasts yet")
        if excluded:
            lines.append("Excluded from scoring: " + ", ".join(f"{k} {v}" for k, v in sorted(excluded.items())))
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
    claims = research_state.load_active_claims(state)
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
