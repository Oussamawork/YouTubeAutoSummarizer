import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

import canonical_claims
import channel_scorecard as cs
import pulse_charts
import scorecard_pricing as sp
from helpers import env_flag
from log import log_info, log_warn, log_error
from sendToTelegram import send_telegram_text, send_telegram_photo_album
# The dataset layer lives in signals_data; the names are re-exported here so
# `market_pulse.aggregate_assets` etc. keep working for existing callers.
from signals_data import (  # noqa: F401  (re-exports)
    SIGNALS_FILE, STANCE_SCORE, NET_THRESHOLD, CONVICTION_WEIGHTS,
    ASSET_ALIASES, TICKER_ALIASES, UNPRICEABLE_TICKERS, LEARNED_TICKERS_FILE,
    load_signals, _parse_date, _in_window, _normalized_name, learned_tickers,
    reset_learned_tickers, canonical_ticker, _asset_key, _iter_assets,
    aggregate_assets, _directional_mentions, net_stance, _direction, _overall_tone,
)

# Weekly market pulse (Mondays, weekly-pulse.yml): one Telegram report — top
# assets with net stance PER HORIZON BUCKET, consensus flips vs the prior
# window, assets newly on the radar — plus the companion charts.
#
# PRODUCTION DATA SOURCE: canonical claims, read through
# canonical_claims.load_canonical_claims (active runs only, no legacy rows,
# no repeats) and aggregated per (asset, horizon bucket) by
# canonical_claims.aggregate_views. A short-term bearish and a long-term
# bullish view on the same asset are two rows, never one averaged stance.
# data/signals.jsonl is a backward-compatible view: the legacy pulse over it
# (build_pulse / aggregate_assets) is kept for existing callers and runs
# only when PULSE_DATA_SOURCE=legacy is set explicitly.
# The output is research over creator opinions, not investment advice.

load_dotenv('.env')

PULSE_DATA_SOURCE = (os.getenv("PULSE_DATA_SOURCE") or "canonical").strip().lower()

# An asset is "new on the radar" when it appears in the current window but in
# none of the records this many days before the window started.
NEW_ASSET_LOOKBACK_DAYS = 30
MAX_ASSETS_IN_REPORT = 8

DISCLAIMER = "⚠️ Aggregated creator opinions — research input, not investment advice."

# Accuracy weighting: once a channel has at least this many scorecard-evaluated
# calls, its stances are weighted by track record (0.5 + hit rate → 0.5..1.5)
# instead of counting 1.0 like everyone else. Proven channels move the
# consensus more; proven-wrong channels move it less.
MIN_TRACK_CALLS = 5


def channel_weights_from_track_record(records, today, price_fetcher=None):
    """
    Weight per channel from its scorecard track record: 0.5 + 7-day hit rate,
    only once the channel has MIN_TRACK_CALLS evaluated calls. Channels without
    enough history weigh the default 1.0. Best-effort: returns ({}, {}) on any
    failure so the pulse still goes out unweighted.

    Returns (weights, details) — details maps channel -> (hits, total) for the
    report footer.
    """
    try:
        fetcher = price_fetcher or cs.fetch_prices
        stats = cs.evaluate(records, today, price_fetcher=fetcher)
        weights, details = {}, {}
        for channel, horizons in stats.items():
            bucket = horizons.get(7, {"hits": 0, "total": 0})
            if bucket["total"] >= MIN_TRACK_CALLS:
                weights[channel] = 0.5 + bucket["hits"] / bucket["total"]
                details[channel] = (bucket["hits"], bucket["total"])
        return weights, details
    except Exception as e:
        log_warn(f"Track-record weighting unavailable this week: {e}")
        return {}, {}


def find_flips(current, previous):
    """
    Assets whose directional consensus flipped between the previous and current
    windows (bullish -> bearish or vice versa). A move to/from "mixed" is not a
    flip — only a real sign change is worth an alert.
    """
    flips = []
    for key, entry in current.items():
        prev = previous.get(key)
        if not prev:
            continue
        cur_dir = _direction(net_stance(entry))
        prev_dir = _direction(net_stance(prev))
        if {cur_dir, prev_dir} == {"bullish", "bearish"}:
            flips.append({
                "label": entry["label"],
                "from": prev_dir, "to": cur_dir,
                "mentions": entry["mentions"],
            })
    return sorted(flips, key=lambda f: -f["mentions"])


def find_new_assets(current, older_records):
    """Asset keys present in the current window but absent from the lookback
    records that preceded it."""
    seen_before = {_asset_key(a) for _, a in _iter_assets(older_records)}
    return {key: entry for key, entry in current.items() if key not in seen_before}


def _format_asset_line(entry, latest_price=None):
    score = net_stance(entry)
    stance_part = f"net {_direction(score)} ({entry['bull']}↑/{entry['bear']}↓"
    if entry["neutral"]:
        # Breadth, separated from direction: neutral mentions say how widely
        # the asset is on the radar, not what anyone thinks of it.
        stance_part += f", {entry['neutral']} neutral"
    stance_part += ")"
    parts = [
        f"• {entry['label']} — {stance_part}",
        f"{entry['mentions']} mention{'s' if entry['mentions'] != 1 else ''} "
        f"across {len(entry['channels'])} channel{'s' if len(entry['channels']) != 1 else ''}",
    ]
    if entry["actions"]:
        actions = " ".join(f"{a}×{n}" for a, n in entry["actions"].most_common())
        parts.append(actions)
    if entry["targets"]:
        avg = sum(entry["targets"]) / len(entry["targets"])
        target_part = f"avg target {avg:,.0f}"
        if latest_price:
            implied = (avg - latest_price) / latest_price
            target_part += f" ({implied:+.0%} implied)"
        parts.append(target_part)
    return " · ".join(parts)


def _latest_price(series):
    """Most recent close from a {date: close} series, or None."""
    if not series:
        return None
    return series[max(series)]


def fetch_latest_prices(entries, price_fetcher=None, today=None):
    """
    Best-effort latest close per asset key for the entries that carry price
    targets (that's the only place the pulse uses live prices). Any failure
    just means no implied-upside annotation.
    """
    try:
        fetcher = price_fetcher or cs.fetch_prices
        today = today or datetime.now(timezone.utc).date()
        prices = {}
        attempted = 0
        for key, entry in entries.items():
            if not entry["targets"] or not entry.get("ticker"):
                continue
            symbol = cs.symbol_for({"ticker": entry["ticker"], "type": entry["type"]})
            if not symbol:
                continue
            # Price targets are quoted in dollars. A foreign listing's close is
            # in its own currency, so an implied move against it would be
            # arithmetic on two different units — skip rather than mislead.
            if not cs.quotes_in_usd(symbol):
                continue
            attempted += 1
            series = fetcher(symbol, today - timedelta(days=10), today)
            latest = _latest_price(series)
            if latest:
                prices[key] = latest
        # A single unpriceable ticker is routine; every one failing means the
        # price source itself is down or blocking us. That distinction was
        # invisible while each failure only logged its own per-symbol warning,
        # so it gets one loud line — it silently disables implied upside,
        # track-record weighting and the target chart.
        if attempted and not prices:
            log_warn(
                f"No prices returned for any of the {attempted} tickers tried — "
                "the price source looks unreachable or is blocking this runner. "
                "Implied-upside annotations and the price-target chart are off "
                "until it recovers."
            )
        return prices
    except Exception as e:
        log_warn(f"Latest-price lookup unavailable this week: {e}")
        return {}


def build_pulse(current_records, previous_records, older_records, start, end,
                channel_weights=None, weight_details=None, latest_prices=None):
    """
    Render the plain-text weekly pulse. Returns "" when the current window has
    no analyzable records (caller skips the send). `channel_weights` biases the
    net-stance consensus by track record; `latest_prices` (asset key -> close)
    annotates price targets with the implied move.
    """
    if not current_records:
        return ""

    current = aggregate_assets(current_records, channel_weights)
    previous = aggregate_assets(previous_records, channel_weights)
    latest_prices = latest_prices or {}

    channels = {r.get("channel_name") for r in current_records if r.get("channel_name")}
    lines = [
        f"📈 Weekly Market Pulse — {start.isoformat()} → {end.isoformat()}",
        f"Videos analyzed: {len(current_records)} · Channels: {len(channels)}",
    ]

    tone = _overall_tone(current_records)
    if tone:
        tone_line = " · ".join(f"{k} {v}" for k, v in tone.most_common())
        lines.append(f"Overall tone: {tone_line}")

    if current:
        lines.append("")
        lines.append("Top assets:")
        # Direction leads the ranking: an asset three channels have real calls
        # on outranks a megacap that ten videos name-dropped neutrally. Total
        # mentions still break ties and stay visible in each line as context.
        ranked = sorted(
            current.items(),
            key=lambda kv: (-_directional_mentions(kv[1]), -abs(net_stance(kv[1])),
                            -kv[1]["mentions"], kv[1]["label"]),
        )
        for key, entry in ranked[:MAX_ASSETS_IN_REPORT]:
            lines.append(_format_asset_line(entry, latest_prices.get(key)))
        if len(ranked) > MAX_ASSETS_IN_REPORT:
            lines.append(f"…and {len(ranked) - MAX_ASSETS_IN_REPORT} more.")

    flips = find_flips(current, previous)
    if flips:
        lines.append("")
        lines.append("🔄 Consensus flips vs prior week:")
        for flip in flips:
            lines.append(f"• {flip['label']}: {flip['from']} → {flip['to']}")

    new_assets = find_new_assets(current, older_records)
    if new_assets:
        lines.append("")
        lines.append(f"🆕 New on the radar (past {NEW_ASSET_LOOKBACK_DAYS} days):")
        for entry in sorted(new_assets.values(), key=lambda e: -e["mentions"])[:5]:
            lines.append(f"• {entry['label']} — {_direction(net_stance(entry))}")

    if weight_details:
        lines.append("")
        weighted = " · ".join(
            f"{channel} {channel_weights[channel]:.2f} ({hits}/{total})"
            for channel, (hits, total) in sorted(weight_details.items())
        )
        lines.append(f"⚖️ Consensus weighted by 7d track record: {weighted}")

    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def _canonical_weights(claims, today, price_fetcher):
    """
    Track-record weights from the canonical scorecard, only when source
    rankings are enabled (they are a ranking): 0.5 + direction hit rate for
    sources with MIN_TRACK_CALLS scored forecasts. Otherwise every source
    weighs 1.0. Best-effort.
    """
    if not sp.SCORECARD_RANKINGS_ENABLED:
        return {}, {}
    try:
        import research_analytics as ra
        sc = ra.scorecard(claims, today, price_fetcher or cs.fetch_price_series)
        weights, details = {}, {}
        for source, v in sc.items():
            # Only claims on a full-confidence calendar count toward a
            # ranking-style weight (scorecard_pricing.rankable_calendar).
            if source.startswith("_") or v["ranked_n"] < MIN_TRACK_CALLS:
                continue
            weights[source] = 0.5 + v["ranked_direction_hits"] / v["ranked_n"]
            details[source] = (v["ranked_direction_hits"], v["ranked_n"])
        return weights, details
    except Exception as e:
        log_warn(f"Track-record weighting unavailable this week: {e}")
        return {}, {}


def canonical_tone_weeks(claims, today, num_weeks=pulse_charts.SPREAD_WEEKS):
    """Weekly tone rows (pulse_charts.tone_weeks shape) from each video's own
    view claims — canonical_claims.video_tone — never from a model's overall
    sentiment field."""
    tone = canonical_claims.video_tone(claims)
    dates = canonical_claims.video_dates(claims)
    if not dates:
        return []
    earliest = min(dates.values())
    weeks, end = [], today
    for weeks_ago in range(num_weeks):
        start = end - timedelta(days=7)
        bucket = [vid for vid, d in dates.items() if start < d <= end and vid in tone]
        if bucket:
            counts = {k: sum(1 for vid in bucket if tone[vid] == k) for k in ("bullish", "bearish", "neutral", "mixed")}
            first = start + timedelta(days=1)
            weeks.append({
                "label": f"{first.strftime('%b %-d')} - {end.strftime('%b %-d')}",
                "short_label": end.strftime("%b %-d"), "weeks_ago": weeks_ago, "n": len(bucket),
                **counts, "partial": earliest > start + timedelta(days=1),
            })
        end = start
    weeks.reverse()
    return weeks


def _canonical_pulse_inputs(days, today, price_fetcher, claims=None):
    """The production inputs: canonical claims sliced into the current,
    previous and lookback windows and aggregated per (asset, horizon)."""
    claims = canonical_claims.load_canonical_claims() if claims is None else claims
    window_start = today - timedelta(days=days)
    prev_start = window_start - timedelta(days=days)
    lookback_start = window_start - timedelta(days=NEW_ASSET_LOOKBACK_DAYS)
    current = [c for c in claims if canonical_claims.in_window(c, window_start, today)]
    previous = [c for c in claims if canonical_claims.in_window(c, prev_start, window_start)]
    older = [c for c in claims if canonical_claims.in_window(c, lookback_start, window_start)]
    weights, details, latest_prices = {}, {}, {}
    current_views = canonical_claims.aggregate_views(current)
    if current_views:
        latest_prices = fetch_latest_prices(current_views, price_fetcher=price_fetcher, today=today)
        weights, details = _canonical_weights(claims, today, price_fetcher)
    view_claims = canonical_claims.view_claims(current)
    return {
        "source": "canonical", "claims": claims, "window_start": window_start,
        "current": current, "previous": previous, "older": older,
        "current_views": canonical_claims.aggregate_views(current, weights),
        "previous_views": canonical_claims.aggregate_views(previous, weights),
        "older_keys": set(canonical_claims.aggregate_views(older)),
        "videos": len({c.get("video_id") for c in view_claims}),
        "channels": len({canonical_claims.source_of(c) for c in view_claims}),
        "tone": canonical_tone_weeks(claims, today),
        "weights": weights, "weight_details": details, "latest_prices": latest_prices,
    }


def build_canonical_pulse(inputs, today):
    """Render the plain-text pulse from canonical inputs. "" when the window
    holds no view claims."""
    current, previous = inputs["current_views"], inputs["previous_views"]
    if not inputs["videos"]:
        return ""
    lines = [
        f"📈 Weekly Market Pulse — {inputs['window_start'].isoformat()} → {today.isoformat()}",
        f"Videos with views: {inputs['videos']} · Sources: {inputs['channels']} · "
        "data: canonical claims (one current view per source, per asset, per horizon)",
    ]
    week = [w for w in inputs["tone"] if w["weeks_ago"] == 0]
    if week:
        w = week[0]
        lines.append("Overall tone: " + " · ".join(f"{k} {w[k]}" for k in ("bullish", "bearish", "mixed", "neutral") if w[k]))
    if current:
        lines.append("")
        lines.append("Top assets (per horizon bucket):")
        ranked = sorted(
            current.items(),
            key=lambda kv: (-_directional_mentions(kv[1]), -abs(net_stance(kv[1])),
                            -kv[1]["mentions"], kv[1]["label"]),
        )
        for key, entry in ranked[:MAX_ASSETS_IN_REPORT]:
            lines.append(_format_asset_line(entry, inputs["latest_prices"].get(key)))
        if len(ranked) > MAX_ASSETS_IN_REPORT:
            lines.append(f"…and {len(ranked) - MAX_ASSETS_IN_REPORT} more.")
    flips = find_flips(current, previous)
    if flips:
        lines.append("")
        lines.append("🔄 Consensus flips vs prior week (same asset and horizon):")
        for flip in flips:
            lines.append(f"• {flip['label']}: {flip['from']} → {flip['to']}")
    new_assets = {k: e for k, e in current.items() if k not in inputs["older_keys"]}
    if new_assets:
        lines.append("")
        lines.append(f"🆕 New on the radar (past {NEW_ASSET_LOOKBACK_DAYS} days):")
        for entry in sorted(new_assets.values(), key=lambda e: -e["mentions"])[:5]:
            lines.append(f"• {entry['label']} — {_direction(net_stance(entry))}")
    disclosures = canonical_claims.format_portfolio_disclosures(
        canonical_claims.portfolio_disclosures(inputs["current"]))
    if disclosures:
        lines.append("")
        lines.append(disclosures)
    if inputs["weight_details"]:
        lines.append("")
        weighted = " · ".join(
            f"{channel} {inputs['weights'][channel]:.2f} ({hits}/{total})"
            for channel, (hits, total) in sorted(inputs["weight_details"].items())
        )
        lines.append(f"⚖️ Consensus weighted by scorecard track record: {weighted}")
    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def _pulse_inputs(days, today, path, price_fetcher):
    """Load the dataset, slice the windows, and compute the best-effort extras
    (track-record weights, latest prices) that both the text pulse and the
    charts consume — shared so the price lookups run once per pulse, not once
    per output. Canonical claims unless PULSE_DATA_SOURCE=legacy."""
    if PULSE_DATA_SOURCE != "legacy":
        return _canonical_pulse_inputs(days, today, price_fetcher)
    records = load_signals(path)
    window_start = today - timedelta(days=days)
    prev_start = window_start - timedelta(days=days)
    lookback_start = window_start - timedelta(days=NEW_ASSET_LOOKBACK_DAYS)

    current = [r for r in records if _in_window(r, window_start, today)]
    previous = [r for r in records if _in_window(r, prev_start, window_start)]
    older = [r for r in records if _in_window(r, lookback_start, window_start)]

    weights, details, latest_prices = {}, {}, {}
    if current:
        # Latest prices first, deliberately: the price source enforces a
        # per-run request budget, and this is what readers actually see (the
        # implied-move annotations and the price-target chart). Track-record
        # weighting is a refinement that already degrades to unweighted, so it
        # spends whatever budget is left rather than competing for it.
        latest_prices = fetch_latest_prices(
            aggregate_assets(current), price_fetcher=price_fetcher, today=today
        )
        weights, details = channel_weights_from_track_record(
            records, today, price_fetcher=price_fetcher
        )
    return {
        "source": "legacy", "records": records, "window_start": window_start,
        "current": current, "previous": previous, "older": older,
        "weights": weights, "weight_details": details,
        "latest_prices": latest_prices,
    }


def generate_pulse(days=7, today=None, path=SIGNALS_FILE, price_fetcher=None,
                   inputs=None):
    """Build the pulse text for the trailing `days` window. Track-record
    weighting and implied-upside annotations are best-effort and switch on
    automatically once enough scorecard history exists. `inputs` accepts a
    precomputed _pulse_inputs() result so main() can share it with the
    charts."""
    today = today or datetime.now(timezone.utc).date()
    inputs = inputs or _pulse_inputs(days, today, path, price_fetcher)
    if inputs.get("source") == "canonical":
        return build_canonical_pulse(inputs, today)
    if not inputs["current"]:
        return ""
    return build_pulse(
        inputs["current"], inputs["previous"], inputs["older"],
        inputs["window_start"], today,
        channel_weights=inputs["weights"], weight_details=inputs["weight_details"],
        latest_prices=inputs["latest_prices"],
    )


def generate_charts(days=7, today=None, path=SIGNALS_FILE, price_fetcher=None,
                    inputs=None, out_dir="charts"):
    """Render the pulse's companion charts as PNGs; returns their paths in
    album order. Entirely best-effort: any failure (matplotlib missing, bad
    data, rendering error) logs a warning and returns [] so the text pulse is
    never blocked by its illustrations."""
    try:
        today = today or datetime.now(timezone.utc).date()
        inputs = inputs or _pulse_inputs(days, today, path, price_fetcher)
        if inputs.get("source") == "canonical":
            if not inputs["videos"]:
                return []
            current, previous = inputs["current_views"], inputs["previous_views"]
            data = pulse_charts.build_chart_data(
                [], current, previous, inputs["window_start"], today,
                tone=inputs["tone"], videos=inputs["videos"], channels=inputs["channels"],
            )
        else:
            if not inputs["current"]:
                return []
            current = aggregate_assets(inputs["current"], inputs["weights"])
            previous = aggregate_assets(inputs["previous"], inputs["weights"])
            data = pulse_charts.build_chart_data(
                inputs["records"], current, previous, inputs["window_start"], today
            )
        data["upside"] = pulse_charts.upside_rows(current, inputs["latest_prices"])
        return pulse_charts.render_charts(data, out_dir)
    except Exception as e:
        log_warn(f"Pulse charts unavailable this week: {e}")
        return []


def research_quality_section(days, today):
    """
    The data-quality header over the canonical claims dataset (coverage,
    exclusions, denominators), so the pulse never implies more was analyzed
    than was. Best-effort: "" when the research ledger is empty or anything
    fails, so the legacy pulse is never blocked by its companion.
    """
    try:
        import research_analytics
        import research_state
        state = research_state.load_state()
        claims = canonical_claims.load_canonical_claims(state)
        if not state.get("videos") and not claims:
            return ""
        header = research_analytics.quality_header(
            claims, research_state.load_gate_outcomes(), state, research_state.load_runs(),
            today - timedelta(days=days), today,
        )
        return research_analytics.format_quality_header(header)
    except Exception as e:
        log_warn(f"Research quality section unavailable this week: {e}")
        return ""


CHART_ALBUM_CAPTION = (
    "This week in charts - how to read each one is written on the image."
)


def main():
    parser = argparse.ArgumentParser(description="Weekly market pulse from canonical claims")
    parser.add_argument("--days", type=int, default=7, help="Window size in days (default 7)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the pulse and write charts locally instead of sending them")
    parser.add_argument("--no-charts", action="store_true",
                        help="Send the text pulse only, without the chart album")
    parser.add_argument("--charts-dir", default="charts",
                        help="Directory the chart PNGs are written to (default: charts/)")
    args = parser.parse_args()

    today = datetime.now(timezone.utc).date()
    inputs = _pulse_inputs(args.days, today, SIGNALS_FILE, None)
    pulse = generate_pulse(days=args.days, today=today, inputs=inputs)
    if not pulse:
        log_info(f"No {inputs.get('source', 'signal')} records with views in the window; "
                 "skipping the pulse this week.")
        return 0

    chart_paths = []
    if not args.no_charts:
        chart_paths = generate_charts(days=args.days, today=today, inputs=inputs,
                                      out_dir=args.charts_dir)

    if args.dry_run:
        print(pulse)
        quality = research_quality_section(args.days, today)
        if quality:
            print("\n" + quality)
        if chart_paths:
            log_info(f"Charts written: {', '.join(chart_paths)}")
        return 0

    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHANNEL_ID")
    if not token or not chat_id:
        log_error("TELEGRAM_TOKEN and TELEGRAM_CHANNEL_ID must be set to send the pulse.")
        return 1

    if not send_telegram_text(token, chat_id, pulse):
        return 1
    log_info("Weekly market pulse sent.")
    quality = research_quality_section(args.days, today)
    if quality and not send_telegram_text(token, chat_id, quality):
        log_warn("Research quality section failed to send; the pulse went out without it.")
    # The charts illustrate the pulse; failing to send them shouldn't fail the
    # run once the text is out.
    if chart_paths and not send_telegram_photo_album(
        token, chat_id, chart_paths, caption=CHART_ALBUM_CAPTION
    ):
        log_warn("Chart album failed to send; the text pulse went out without it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
