import argparse
import html as _html
import os
from collections import defaultdict
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

import canonical_claims
import channel_scorecard as cs
import pulse_charts
import scorecard_pricing as sp
from helpers import env_flag
from log import log_info, log_warn, log_error
from sendToTelegram import send_telegram_html, send_telegram_text, send_telegram_photo_album
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

DISCLAIMER = "⚠️ Creator opinions, not investment advice."

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
    targets = pulse_charts.plausible_targets(entry["targets"], latest_price)
    if targets:
        avg = sum(targets) / len(targets)
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
    previous and lookback windows and aggregated per asset
    (canonical_claims.aggregate_views_by_asset: one vote per creator)."""
    claims = canonical_claims.load_canonical_claims() if claims is None else claims
    window_start = today - timedelta(days=days)
    prev_start = window_start - timedelta(days=days)
    lookback_start = window_start - timedelta(days=NEW_ASSET_LOOKBACK_DAYS)
    current = [c for c in claims if canonical_claims.in_window(c, window_start, today)]
    previous = [c for c in claims if canonical_claims.in_window(c, prev_start, window_start)]
    older = [c for c in claims if canonical_claims.in_window(c, lookback_start, window_start)]
    weights, details, latest_prices = {}, {}, {}
    current_views = canonical_claims.aggregate_views_by_asset(current)
    if current_views:
        latest_prices = fetch_latest_prices(current_views, price_fetcher=price_fetcher, today=today)
        weights, details = _canonical_weights(claims, today, price_fetcher)
        if weights:
            current_views = canonical_claims.aggregate_views_by_asset(current, weights)
    view_claims = [c for c in canonical_claims.view_claims(current) if not c.get("review_required")]
    return {
        "source": "canonical", "claims": claims, "window_start": window_start,
        "current": current, "previous": previous, "older": older,
        "current_views": current_views,
        "previous_views": canonical_claims.aggregate_views_by_asset(previous, weights),
        "older_keys": set(canonical_claims.aggregate_views_by_asset(older)),
        "view_claims": len(view_claims),
        "horizon_claims": sum(1 for c in view_claims if c.get("horizon_bucket") in canonical_claims.HORIZON_LABELS),
        "videos": len({c.get("video_id") for c in view_claims}),
        "channels": len({canonical_claims.source_of(c) for c in view_claims}),
        "tone": canonical_tone_weeks(claims, today),
        "weights": weights, "weight_details": details, "latest_prices": latest_prices,
    }


# Reader-facing thresholds. With ~8 creators, an asset one creator talked
# about is that creator's opinion, not a consensus: 90% of assets in a
# typical week rest on a single creator, so every ranked section asks for
# at least two creators with a view.
MIN_CREATORS = 2
MIN_DISAGREE_CREATORS = 3
MAX_AGREEMENT_ROWS = 6
MAX_DISAGREEMENT_ROWS = 3
MAX_DISCUSSED = 5
MAX_MOVERS = 3
MAX_PULSE_DISCLOSURE_SOURCES = 4
MOVER_MIN_CLAIMS = 3        # attention shift needs ≥3 more / fewer claims…
MOVER_MIN_CREATORS = 1      # …and a creator more / fewer, so one talkative video can't move it
MIN_LEAN_VOTES = 5          # asset-class and creator lean lines need this many votes
EMOJI = {"bullish": "🟢", "bearish": "🔴", "mixed": "⚖️"}


def _voters(entry):
    """Creators who took a side (a split creator counts: they had a view)."""
    return sum(1 for v in entry["votes"].values() if v != "neutral")


def _consensus(entry):
    """bullish / bearish / mixed over the creators who took a side, by the
    same 2/3 rule a creator's own vote uses. A split creator counts in the
    denominator: one bullish and one split creator is not agreement."""
    voters = _voters(entry)
    if voters and 3 * entry["bull"] >= 2 * voters:
        return "bullish"
    if voters and 3 * entry["bear"] >= 2 * voters:
        return "bearish"
    return "mixed"


def _plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def _money(value):
    return f"${value:,.2f}" if value < 20 else f"${value:,.0f}"


def _agreement_line(entry, price, fmt):
    side = _consensus(entry)
    count = entry["bull"] if side == "bullish" else entry["bear"]
    parts = [f"{EMOJI[side]} {fmt.b(entry['label'])} — {count} of {_plural(_voters(entry), 'creator')} {side}"]
    if entry["horizons"]:
        named = [canonical_claims.HORIZON_LABELS[b] for b in ("short", "medium", "long") if entry["horizons"].get(b)]
        parts.append("horizon: " + " & ".join(named))
    if entry["actions"]:
        parts.append(", ".join(f"{n} say{'s' if n == 1 else ''} {a}" for a, n in entry["actions"].most_common()))
    # A target is shown only against a known price: without one there is no
    # check that the number is a share price at all (a dominance percentage
    # once read as "target $50").
    targets = pulse_charts.plausible_targets(entry["targets"], price) if price else []
    if targets:
        avg = sum(targets) / len(targets)
        parts.append(f"target {_money(avg)} ({(avg - price) / price:+.0%} vs {_money(price)}), "
                     f"{_plural(len(targets), 'creator')}")
    return " · ".join(parts)


def _disagreement_line(entry, fmt):
    bulls = sorted(s for s, v in entry["votes"].items() if v == "bullish")
    bears = sorted(s for s, v in entry["votes"].items() if v == "bearish")
    minority, side = (bulls, "bullish") if len(bulls) < len(bears) else (bears, "bearish")
    return (f"{EMOJI['mixed']} {fmt.b(entry['label'])} — {len(bulls)} bullish vs {len(bears)} bearish "
            f"({side}: {fmt.esc(', '.join(minority))})")


def _attention_movers(current, previous):
    up, down = [], []
    for key in set(current) | set(previous):
        cur, prev = current.get(key), previous.get(key)
        claims_now, claims_before = (cur or {}).get("mentions", 0), (prev or {}).get("mentions", 0)
        creators_now = len((cur or {}).get("channels", ())) 
        creators_before = len((prev or {}).get("channels", ()))
        label = (cur or prev)["label"]
        delta = claims_now - claims_before
        if delta >= MOVER_MIN_CLAIMS and creators_now - creators_before >= MOVER_MIN_CREATORS:
            up.append((delta, label, claims_before, claims_now))
        elif -delta >= MOVER_MIN_CLAIMS and creators_before - creators_now >= MOVER_MIN_CREATORS:
            down.append((-delta, label, claims_before, claims_now))
    order = lambda rows: sorted(rows, key=lambda r: (-r[0], r[1]))[:MAX_MOVERS]
    return order(up), order(down)


def _lean_share(entries):
    """(bullish share, votes) over creator votes in `entries`."""
    bull = sum(e["bull"] for e in entries)
    bear = sum(e["bear"] for e in entries)
    return (bull / (bull + bear) if bull + bear else None), bull + bear


class _Fmt:
    """Telegram HTML or plain text from one set of section builders."""

    def __init__(self, html):
        self.html = html

    def esc(self, text):
        return _html.escape(str(text), quote=False) if self.html else str(text)

    def b(self, text):
        return f"<b>{self.esc(text)}</b>" if self.html else str(text)

    def i(self, text):
        return f"<i>{self.esc(text)}</i>" if self.html else str(text)


def build_canonical_pulse(inputs, today, html=False):
    """Render the pulse from canonical inputs, as plain text or (html=True)
    Telegram HTML. "" when the window holds no view claims.

    Layout, most useful first: a one-sentence takeaway built from the
    numbers below it (a fixed template, never an LLM call), where creators
    agree, where they disagree, what drew attention, changes vs last week,
    the mood, what creators say they own, and one footer line on coverage.
    """
    fmt = _Fmt(html)
    current, previous = inputs["current_views"], inputs["previous_views"]
    if not inputs["videos"]:
        return ""
    prices = inputs["latest_prices"]

    ranked = sorted(current.items(), key=lambda kv: (-_voters(kv[1]), -abs(net_stance(kv[1])),
                                                     -kv[1]["mentions"], kv[1]["label"]))
    eligible = [(k, e) for k, e in ranked if _voters(e) >= MIN_CREATORS]
    agree = [(k, e) for k, e in eligible if _consensus(e) in ("bullish", "bearish")]
    disagree = [(k, e) for k, e in eligible
                if _consensus(e) == "mixed" and e["bull"] and e["bear"] and _voters(e) >= MIN_DISAGREE_CREATORS]
    shown = {k for k, _ in agree[:MAX_AGREEMENT_ROWS]} | {k for k, _ in disagree[:MAX_DISAGREEMENT_ROWS]}

    week = next((w for w in inputs["tone"] if w["weeks_ago"] == 0), None)
    lines = [f"📈 {fmt.b('Weekly Market Pulse')} · {fmt.esc(pulse_charts.window_label(inputs['window_start'], today))}"]

    # Takeaway.
    takeaway = []
    if week and week["n"]:
        bull_share = week["bullish"] / week["n"]
        mood = ("Upbeat week" if bull_share >= 0.6 else
                "Cautious week" if week["bearish"] >= week["bullish"] else "Mixed week")
        takeaway.append(f"{mood}: {week['bullish']} of {week['n']} videos leaned bullish.")
    if agree:
        top = agree[0][1]
        side = _consensus(top)
        takeaway.append(f"Strongest agreement: {top['label']} "
                        f"({top['bull'] if side == 'bullish' else top['bear']} of {_voters(top)} creators {side}).")
        bear = next((e for _, e in agree if _consensus(e) == "bearish"), None)
        if bear is not None and bear is not top:
            takeaway.append(f"Clearest bear call: {bear['label']}.")
    if disagree:
        takeaway.append(f"Split on {disagree[0][1]['label']}.")
    if takeaway:
        lines.append(fmt.esc(" ".join(takeaway)))

    if agree:
        lines += ["", fmt.b("Where creators agree")]
        lines += [_agreement_line(e, prices.get(k), fmt) for k, e in agree[:MAX_AGREEMENT_ROWS]]
    if disagree:
        lines += ["", fmt.b("Where they disagree")]
        lines += [_disagreement_line(e, fmt) for _, e in disagree[:MAX_DISAGREEMENT_ROWS]]

    discussed = sorted(current.values(), key=lambda e: (-len(e["channels"]), -e["videos"], -e["mentions"], e["label"]))
    if discussed:
        lines += ["", fmt.b("Most discussed")]
        lines.append(" · ".join(
            f"{fmt.esc(e['label'])} ({_plural(len(e['channels']), 'creator')}, {e['mentions']} claims)"
            for e in discussed[:MAX_DISCUSSED]))

    up, down = _attention_movers(current, previous)
    if up or down:
        lines += ["", fmt.b("Attention vs last week")]
        if up:
            lines.append("⬆️ " + " · ".join(f"{fmt.esc(l)} ({a}→{b} claims)" for _, l, a, b in up))
        if down:
            lines.append("⬇️ " + " · ".join(f"{fmt.esc(l)} ({a}→{b} claims)" for _, l, a, b in down))

    flips = []
    for key, entry in current.items():
        prev = previous.get(key)
        if not prev or _voters(entry) < MIN_CREATORS or _voters(prev) < MIN_CREATORS:
            continue
        before, after = _consensus(prev), _consensus(entry)
        if {before, after} == {"bullish", "bearish"}:
            flips.append((entry["label"], before, after))
    if flips:
        lines += ["", fmt.b("Consensus changed vs last week")]
        lines += [f"🔄 {fmt.b(label)}: {before} → {after}" for label, before, after in sorted(flips)]

    new = [e for k, e in ranked if k not in inputs["older_keys"] and k not in shown and _voters(e) >= MIN_CREATORS]
    if new:
        lines += ["", fmt.b(f"New on the radar (not discussed in the prior {NEW_ASSET_LOOKBACK_DAYS} days)")]
        lines.append(" · ".join(f"{EMOJI.get(_consensus(e), '')} {fmt.esc(e['label'])} ({_plural(_voters(e), 'creator')})"
                                for e in new[:5]))

    mood_lines = []
    if week and week["n"]:
        parts = [f"{week[k]} {k}" for k in ("bullish", "mixed", "bearish") if week[k]]
        if week["neutral"]:
            parts.append(f"{week['neutral']} without a clear view")
        mood_lines.append(f"Videos: {' · '.join(parts)}")
    classes = []
    for name, types in (("stocks", ("stock", "etf")), ("crypto", ("crypto",))):
        share, votes = _lean_share([e for e in current.values() if e["type"] in types])
        if share is not None and votes >= MIN_LEAN_VOTES:
            classes.append(f"{name} {share:.0%} bullish ({votes} creator calls)")
    if classes:
        mood_lines.append("By asset class: " + " · ".join(classes))
    per_creator = defaultdict(lambda: [0, 0])
    for e in current.values():
        for source, vote in e["votes"].items():
            if vote in ("bullish", "bearish"):
                per_creator[source][vote == "bearish"] += 1
    leaners = [(b / (b + r), s, b, r) for s, (b, r) in per_creator.items() if b + r >= MIN_LEAN_VOTES]
    if len(leaners) >= 2:
        leaners.sort()
        low, high = leaners[0], leaners[-1]
        mood_lines.append(f"Most bullish creator: {fmt.esc(high[1])} ({high[2]}↑ {high[3]}↓) · "
                          f"most cautious: {fmt.esc(low[1])} ({low[2]}↑ {low[3]}↓)")
    if mood_lines:
        lines += ["", fmt.b("Mood")] + mood_lines

    groups = canonical_claims.disclosure_groups(canonical_claims.portfolio_disclosures(inputs["current"]))
    if groups:
        lines += ["", fmt.b("What creators say they own")]
        for source, parts in groups[:MAX_PULSE_DISCLOSURE_SOURCES]:
            lines.append(f"• {fmt.esc(source)} — " + fmt.esc("; ".join(f"{p} {names}" for p, names in parts)))
        if len(groups) > MAX_PULSE_DISCLOSURE_SOURCES:
            lines.append(f"+{len(groups) - MAX_PULSE_DISCLOSURE_SOURCES} more creators")

    if inputs["weight_details"]:
        weighted = " · ".join(
            f"{channel} {inputs['weights'][channel]:.2f} ({hits}/{total})"
            for channel, (hits, total) in sorted(inputs["weight_details"].items())
        )
        lines += ["", fmt.esc(f"⚖️ Weighted by scorecard track record: {weighted}")]

    horizon_pct = inputs["horizon_claims"] / inputs["view_claims"] if inputs["view_claims"] else 0
    lines += ["", fmt.i(
        f"Based on {inputs['view_claims']} opinions in {_plural(inputs['videos'], 'video')} from "
        f"{_plural(inputs['channels'], 'creator')}; {horizon_pct:.0%} name a time horizon. "
        f"Assets need {MIN_CREATORS}+ creators with a view to be ranked."),
        fmt.esc(DISCLAIMER)]
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
                   inputs=None, html=False):
    """Build the pulse text for the trailing `days` window. Track-record
    weighting and implied-upside annotations are best-effort and switch on
    automatically once enough scorecard history exists. `inputs` accepts a
    precomputed _pulse_inputs() result so main() can share it with the
    charts."""
    today = today or datetime.now(timezone.utc).date()
    inputs = inputs or _pulse_inputs(days, today, path, price_fetcher)
    if inputs.get("source") == "canonical":
        return build_canonical_pulse(inputs, today, html=html)
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

    if inputs.get("source") == "canonical":
        html_pulse = generate_pulse(days=args.days, today=today, inputs=inputs, html=True)
        sent = send_telegram_html(token, chat_id, html_pulse, pulse, what="Weekly market pulse")
    else:
        sent = send_telegram_text(token, chat_id, pulse)
    if not sent:
        return 1
    log_info("Weekly market pulse sent.")
    # The full data-quality header is maintainer detail (gate outcomes,
    # headline-eligible counts): it goes to the workflow log, and the pulse
    # carries a one-line coverage footer instead.
    quality = research_quality_section(args.days, today)
    if quality:
        print(quality)
    # The charts illustrate the pulse; failing to send them shouldn't fail the
    # run once the text is out.
    if chart_paths and not send_telegram_photo_album(
        token, chat_id, chart_paths, caption=CHART_ALBUM_CAPTION
    ):
        log_warn("Chart album failed to send; the text pulse went out without it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
