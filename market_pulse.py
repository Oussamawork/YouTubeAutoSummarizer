import argparse
import html as _html
import os
from collections import Counter, defaultdict
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

import canonical_claims
import channel_scorecard as cs
import creator_tracker
import instruments
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

# Weekly market pulse (Mondays 08:00 UTC, weekly-pulse.yml): one Telegram
# report plus its companion charts. The canonical layout, most useful first:
# a templated takeaway; what the market did (S&P 500, Nasdaq-100, Bitcoin,
# Ether: last week and four weeks); where creators agree, each row with the
# move since the creators' first call against the S&P 500 (crypto: Bitcoin);
# one pooled creators-vs-market line; where they disagree; last week's picks
# one week on; consensus changes; new on the radar; the mood; what creators
# say they own; and a footer naming the price date and the coverage.
#
# PRODUCTION DATA SOURCE: canonical claims, read through
# canonical_claims.load_canonical_claims (active runs only, no legacy rows,
# no repeats) and aggregated per ASSET, one vote per creator, by
# canonical_claims.aggregate_views_by_asset. Every price is a daily close
# from the committed cache (data/prices.json, warmed Sundays and Fridays) at
# ONE as-of — the last completed US session (creator_tracker.default_as_of:
# the Friday close for the Monday run) — read through creator_tracker, so the
# pulse never makes a live price request and never shows a partial weekend
# crypto bar as "now"; when the cache is behind, the Market block says so.
# Wherever two creators must agree, two channels of one house
# (creator_tracker.creator_houses: HKCM and Phantom by HKCM) count once.
# data/signals.jsonl is a backward-compatible view: the legacy pulse over it
# (build_pulse / aggregate_assets) is kept for existing callers and runs
# only when PULSE_DATA_SOURCE=legacy is set explicitly.
# The output is research over creator opinions, not investment advice. The
# pulse never ranks creators or prints per-creator results: those are the
# Friday scorecard's, labelled mark-to-market, not a verdict.

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


def _closes_of(answer):
    """{date: close} from a price fetcher's answer: a plain map (the legacy
    fetchers) or the closes of a scorecard_pricing.PriceSeries (the
    creator_tracker fetchers)."""
    closes = getattr(answer, "closes", answer)
    return closes if isinstance(closes, dict) else {}


def _cache_closes(symbol, start, end):
    """The pulse's default price source: closes from the committed cache
    only (creator_tracker.cache_fetcher), never a live request. The Sunday
    warm covers what the pulse prices up to the last completed session."""
    return creator_tracker.cache_fetcher()(symbol, start, end).closes


def fetch_latest_prices(entries, price_fetcher=None, today=None, as_of=None):
    """
    Best-effort close per asset key for the entries that carry price targets
    (the only place the pulse prints a price level): the last close in
    [as_of - 10 days, as_of]. `as_of` is the last completed US session —
    creator_tracker.default_as_of(today) unless the caller passes the
    tracker's effective as-of — so a Monday run reads the Friday close the
    Sunday warm cached, and a partial weekend crypto bar is never shown as
    "now". The default fetcher reads that cache only: the pulse never makes
    a live request. The symbol is canonical_claims.priceable_symbol (the
    instrument registry first). Any failure just means no target annotation.
    """
    try:
        fetcher = price_fetcher or _cache_closes
        as_of = as_of or creator_tracker.default_as_of(today or datetime.now(timezone.utc).date())
        prices = {}
        attempted = 0
        for key, entry in entries.items():
            if not entry["targets"] or not entry.get("ticker"):
                continue
            symbol = canonical_claims.priceable_symbol(entry)
            if not symbol:
                continue
            # Price targets are quoted in dollars. A foreign listing's close is
            # in its own currency, so an implied move against it would be
            # arithmetic on two different units — skip rather than mislead.
            if not cs.quotes_in_usd(symbol):
                continue
            attempted += 1
            latest = _latest_price(_closes_of(fetcher(symbol, as_of - timedelta(days=10), as_of)))
            if latest:
                prices[key] = latest
        # A single unpriceable ticker is routine; every one failing means the
        # price data itself is missing (the warm job did not run, or the
        # source behind it is down). That distinction was invisible while each
        # failure only logged its own per-symbol warning, so it gets one loud
        # line — it silently disables the target annotations and chart.
        if attempted and not prices:
            log_warn(
                f"No prices returned for any of the {attempted} tickers tried (closes to {as_of}) — "
                "the price cache looks unwarmed or the price source behind it is unreachable. "
                "Target annotations and the price-target chart are off until it recovers."
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


def _display_labels(*views):
    """Name every tickered entry the way the reports do —
    instruments.display_label: "SpaceX (SPCX)", "Planet Labs (PL)", while a
    megacap stays "NVDA" — in this one place, so the text, the charts and
    last week's picks agree. Without display_label the labels stay tickers."""
    label_of = getattr(instruments, "display_label", None)
    if label_of is None:
        return
    for entries in views:
        for entry in entries.values():
            if entry.get("ticker"):
                entry["label"] = label_of(entry["ticker"]) or entry["label"]


def _canonical_pulse_inputs(days, today, price_fetcher, claims=None):
    """The production inputs: canonical claims sliced into the current,
    previous and lookback windows and aggregated per asset
    (canonical_claims.aggregate_views_by_asset: one vote per creator), plus
    the market side, every price at ONE as-of. `track` is the creator
    tracker's mark-to-market record at the last completed US session
    (creator_tracker.track_record at default_as_of(today): the Friday close
    for the Monday run); `as_of` is its effective date, the last S&P 500
    close the price data holds; `market` its market context; `picks` last
    week's agree rows, as sent, one week on (last_week_picks). A
    price_fetcher of None reads the committed cache only, never live."""
    claims = canonical_claims.load_canonical_claims() if claims is None else claims
    window_start = today - timedelta(days=days)
    prev_start = window_start - timedelta(days=days)
    lookback_start = window_start - timedelta(days=NEW_ASSET_LOOKBACK_DAYS)
    current = [c for c in claims if canonical_claims.in_window(c, window_start, today)]
    previous = [c for c in claims if canonical_claims.in_window(c, prev_start, window_start)]
    older = [c for c in claims if canonical_claims.in_window(c, lookback_start, window_start)]
    # Only what was published by `today` can be a call a reader saw (this
    # matters when an earlier Monday is rebuilt; in production it is all).
    known = [c for c in claims if (canonical_claims.claim_date(c) or today) <= today]
    record = creator_tracker.track_record(known, creator_tracker.default_as_of(today), price_fetcher)
    as_of = record["as_of"]
    houses = creator_tracker.creator_houses({canonical_claims.source_of(c) for c in claims})
    weights, details, latest_prices = {}, {}, {}
    current_views = canonical_claims.aggregate_views_by_asset(current)
    if current_views:
        latest_prices = fetch_latest_prices(current_views, price_fetcher=price_fetcher, today=today, as_of=as_of)
        weights, details = _canonical_weights(claims, today, price_fetcher)
        if weights:
            current_views = canonical_claims.aggregate_views_by_asset(current, weights)
    for key, entry in current_views.items():
        entry["targets"] = _row_targets(entry, latest_prices.get(key))
    previous_views = canonical_claims.aggregate_views_by_asset(previous, weights)
    _display_labels(current_views, previous_views)
    view_claims = [c for c in canonical_claims.view_claims(current) if not c.get("review_required")]
    return {
        "source": "canonical", "claims": claims, "window_start": window_start,
        "current": current, "previous": previous, "older": older,
        "current_views": current_views, "previous_views": previous_views,
        "older_keys": set(canonical_claims.aggregate_views_by_asset(older)),
        "view_claims": len(view_claims),
        "videos": len({c.get("video_id") for c in view_claims}),
        "channels": len({canonical_claims.source_of(c) for c in view_claims}),
        "tone": canonical_tone_weeks(claims, today),
        "weights": weights, "weight_details": details, "latest_prices": latest_prices,
        "as_of": as_of, "requested_as_of": record["requested_as_of"], "track": record, "market": record["market"],
        "houses": houses,
        "picks": last_week_picks(known, today, days, as_of, price_fetcher, houses=houses),
    }


def _row_targets(entry, price):
    """
    The price targets the pulse shows for an asset, the text and the upside
    chart alike: one per creator (their latest in the window,
    canonical_claims.aggregate_views_by_asset's target_by), each on its
    creator's own side of the close (a bullish creator's above it, a
    bearish one's below it, so a bearish fair value never sits on a bullish
    row), and — when the asset has a bullish or bearish consensus — only the
    consensus side's. Plausibility against the close is left to
    pulse_charts.plausible_targets, which both apply.
    """
    by = entry.get("target_by")
    if by is None:                       # an entry without attribution keeps its targets
        return list(entry.get("targets") or [])
    side = _consensus(entry)
    out = []
    for creator, target in sorted(by.items()):
        vote = entry["votes"].get(creator)
        if side in ("bullish", "bearish") and vote != side:
            continue
        if price and ((vote == "bullish" and target <= price) or (vote == "bearish" and target >= price)):
            continue
        out.append(target)
    return out


# Reader-facing thresholds. With ~8 creators, an asset one creator talked
# about is that creator's opinion, not a consensus: 90% of assets in a
# typical week rest on a single creator, so every ranked section asks for
# at least two creators with a view.
MIN_CREATORS = 2
MIN_DISAGREE_CREATORS = 3
MAX_AGREEMENT_ROWS = 6
MAX_DISAGREEMENT_ROWS = 3
MAX_PULSE_DISCLOSURE_SOURCES = 4
MIN_LEAN_VOTES = 5          # asset-class and creator lean lines need this many votes
# An action ("2 say buy") or an average target decorates an agree row only
# when this many creators on the row's side named it: one creator's number
# is their opinion, and a dissenter's level never sits on the majority's row.
MIN_SHARED_CREATORS = 2
# The pooled creators-vs-market line needs this many marked calls.
MIN_POOLED_CALLS = 10
# A market series whose last close is this many days older than the as-of
# is left out rather than printed as this week's move.
STALE_CONTEXT_DAYS = 3
# weekly-pulse.yml runs Mondays at 08:00 UTC ('0 8 * * 1'): last week's picks
# are rebuilt from the claims extracted by then.
PULSE_RUN_HOUR_UTC = 8
EMOJI = {"bullish": "🟢", "bearish": "🔴", "mixed": "⚖️"}
_MINUS = "\u2212"   # a typographic minus, as wide as the plus sign
_BENCHMARK_SHORT = {"S&P 500": "S&P"}
_REPORT_LABEL = creator_tracker.REPORT_LABEL[:1].upper() + creator_tracker.REPORT_LABEL[1:]


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


def _agreement_share(entry):
    """The larger side's share of the creators who took a side (a split
    creator counts in the denominator): 4 of 4 is 1.0, 3 of 4 is 0.75."""
    return max(entry["bull"], entry["bear"]) / max(1, _voters(entry))


def _rank(views):
    """[(asset key, entry)] most creators with a view first, then the
    largest share of them agreeing (4 of 4 before 3 of 4 with one split),
    then the clearest net stance, then the most opinions, then by label."""
    return sorted(views.items(), key=lambda kv: (-_voters(kv[1]), -_agreement_share(kv[1]),
                                                 -abs(net_stance(kv[1])), -kv[1]["mentions"], kv[1]["label"]))


def _houses(creators, houses):
    """Distinct creator houses among `creators` (creator_tracker.creator_houses:
    "Phantom by HKCM" is HKCM's); without a map every creator is its own."""
    return len({(houses or {}).get(c, c) for c in creators})


def _voter_houses(entry, houses=None):
    """Houses among the creators who took a side on the asset."""
    return _houses([c for c, v in entry["votes"].items() if v != "neutral"], houses)


def _side_houses(entry, houses=None):
    """Houses among the creators on the asset's consensus side."""
    side = _consensus(entry)
    return _houses([c for c, v in entry["votes"].items() if v == side], houses)


def agree_rows(views, limit=MAX_AGREEMENT_ROWS, houses=None):
    """The "Where creators agree" rows for per-asset views
    (canonical_claims.aggregate_views_by_asset): [(asset key, entry)] with a
    2/3 consensus backed by creators from MIN_CREATORS or more houses (two
    channels of one house, `houses` from creator_tracker.creator_houses, are
    one opinion), in the pulse's ranking, at most `limit` — one of which is
    kept for the strongest bearish agreement when there is one, so a
    bullish week never hides it. Pure: the pulse renders this week's rows
    with it and last_week_picks rebuilds last week's with it."""
    rows = [(k, e) for k, e in _rank(views)
            if _consensus(e) in ("bullish", "bearish") and _side_houses(e, houses) >= MIN_CREATORS]
    top = rows[:max(0, limit)]
    if len(top) > 1 and not any(_consensus(e) == "bearish" for _, e in top):
        bear = next(((k, e) for k, e in rows[len(top):] if _consensus(e) == "bearish"), None)
        if bear:
            top = top[:-1] + [bear]
    return top


def _extracted_at(claim):
    """The claim's extraction instant (UTC), or None."""
    try:
        moment = datetime.fromisoformat(str(claim.get("extracted_at")).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _seen_by(claim, cutoff):
    """True when the claim was extracted by `cutoff`; a claim without an
    extraction time cannot be shown to have been seen."""
    moment = _extracted_at(claim)
    return moment is not None and moment <= cutoff


def _pick_benchmark(priced):
    """The market a pick is held against: none for an instrument that IS the
    benchmark (SPY, VOO, IVV, BTC), Bitcoin for other crypto, else the S&P
    500 (SPY) — creator_tracker's rule."""
    if priced["symbol"] in creator_tracker.SELF_BENCHMARKED:
        return None
    if priced["asset_class"] == "crypto":
        return creator_tracker.CRYPTO_MARKET_SYMBOL
    return creator_tracker.MARKET_SYMBOL


def last_week_picks(claims, today, days, as_of, price_fetcher=None, houses=None):
    """
    Last week's "Where creators agree" rows as that run sent them, held one
    week. The rows are rebuilt with agree_rows from the claims last week's
    run could see: published in its window and extracted by
    PULSE_RUN_HOUR_UTC that day. A tenth to a third of a week's claims are
    extracted later, so a list rebuilt from today's claims would grade rows
    no reader saw (claims without an extraction time cannot be shown to have
    been seen and are left out). A run on another weekday reads
    "last week's run" as today - 7 days.

    Each pick is priced with creator_tracker.instrument_for (a market proxy
    ETF, else the instrument registry; never guessed) and
    creator_tracker.price_move, from the first close on or after that run's
    day to `as_of`, and held against the market over the same closes: the
    S&P 500 (SPY), Bitcoin for other crypto, nothing for SPY or BTC
    themselves. Icons: ✅ moved the called way and beat the benchmark (or
    has none), ➖ moved the called way but lagged it, ❌ moved against the
    call. A theme with no instrument, or with no prices, is listed as not
    priced.

    Returns {run, start, end, rows, unpriced, right, avg, spy}: rows carry
    asset, label, direction, symbol, ret, dir_ret, benchmark, bench_ret,
    right, beat and icon; avg is the equal-weight move IN THE CALLED
    DIRECTION (a bearish pick that fell counts as a gain), spy the S&P 500's
    own move over [start, end] (None when unpriced) — reported beside it,
    never as if both were measured the same way. `houses`
    (creator_tracker.creator_houses) as in agree_rows; read from the claims
    when not given.
    """
    run = today - timedelta(days=7)
    cutoff = datetime(run.year, run.month, run.day, PULSE_RUN_HOUR_UTC, tzinfo=timezone.utc)
    window_start = run - timedelta(days=days)
    sent = [c for c in claims if canonical_claims.in_window(c, window_start, run) and _seen_by(c, cutoff)]
    views = canonical_claims.aggregate_views_by_asset(sent)
    _display_labels(views)
    if houses is None:
        houses = creator_tracker.creator_houses({canonical_claims.source_of(c) for c in claims})
    out = {"run": run, "start": run, "end": as_of, "rows": [], "unpriced": [], "right": 0, "avg": None,
           "spy": None}
    for key, entry in agree_rows(views, houses=houses):
        side = _consensus(entry)
        priced = creator_tracker.instrument_for(key, entry.get("ticker"), None, entry.get("type"))
        move = creator_tracker.price_move(priced["symbol"], run, as_of, price_fetcher) if priced else None
        if move is None:
            out["unpriced"].append(entry["label"])
            continue
        sign = 1 if side == "bullish" else -1
        bench = _pick_benchmark(priced)
        bench_move = creator_tracker.price_move(bench, move["start_date"], move["end_date"], price_fetcher) \
            if bench else None
        dir_ret = sign * move["ret"]
        right = dir_ret > 0
        beat = None if bench_move is None else dir_ret - sign * bench_move["ret"] > 0
        out["rows"].append({
            "asset": key, "label": entry["label"], "direction": side, "symbol": priced["symbol"],
            "ret": move["ret"], "dir_ret": dir_ret, "benchmark": bench,
            "bench_ret": bench_move["ret"] if bench_move else None, "right": right, "beat": beat,
            "icon": "✅" if right and beat is not False else "➖" if right else "❌",
        })
    if out["rows"]:
        out["right"] = sum(r["right"] for r in out["rows"])
        out["avg"] = sum(r["dir_ret"] for r in out["rows"]) / len(out["rows"])
        spy = creator_tracker.price_move(creator_tracker.MARKET_SYMBOL, run, as_of, price_fetcher)
        if spy is not None:
            out.update(spy=spy["ret"], start=spy["start_date"], end=spy["end_date"])
    return out


def _plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def _money(value):
    return f"${value:,.2f}" if value < 20 else f"${value:,.0f}"


def _pct(change, digits=1):
    """A signed percentage with a typographic minus: +1.2%, −2.2%. A move
    that rounds to zero gets one more decimal (+0.02%), so a pick's ✅ or ❌
    beside it can be read off the number."""
    if change and round(abs(change) * 100, digits) == 0:
        digits += 1
    return f"{change:+.{digits}%}".replace("-", _MINUS)


def _mon_d(day):
    return f"{day:%b} {day.day}"


def _dow_mon_d(day):
    return f"{day:%a %b} {day.day}"


def _moved(change):
    """'rose 1.2%' / 'fell 2.2%' / 'was flat'."""
    if abs(change) < 0.0005:
        return "was flat"
    return f"{'rose' if change > 0 else 'fell'} {abs(change):.1%}"


def _backed_calls(record, key, side, backers):
    """How many of `backers` hold a current tracker call on the asset in the
    row's direction, priced or not."""
    summary = ((record or {}).get("assets") or {}).get(key) or {}
    return sum(1 for c, r in (summary.get("current") or {}).items() if c in backers and r["direction"] == side)


def _target_part(entry, side, price):
    """'avg target $X (+y% vs $close)' over the row's targets (_row_targets:
    one per consensus-side creator, their latest in the window, on the row's
    side of the close), each plausible against it
    (pulse_charts.plausible_targets) — the same numbers the upside chart
    plots — and only when MIN_SHARED_CREATORS creators named one. A
    dissenter's level never reaches the row."""
    if not price:
        return None
    levels = pulse_charts.plausible_targets(entry.get("targets") or [], price)
    levels = [t for t in levels if (t > price if side == "bullish" else t < price)]
    if len(levels) < MIN_SHARED_CREATORS:
        return None
    avg = sum(levels) / len(levels)
    return f"avg target {_money(avg)} ({_pct((avg - price) / price, 0)} vs {_money(price)})"


def _agreement_line(key, entry, price, record, fmt):
    """(row, priced) for one agree row: the consensus count, then the move
    since the consensus side's first call (creator_tracker.asset_outcome:
    each backer's current call, the earliest priced one leading, against the
    S&P 500 — crypto against Bitcoin — over the same closes) or "new call,
    no close yet", then shared actions and targets. `priced` says whether a
    move or pending fragment was added."""
    side = _consensus(entry)
    count = entry["bull"] if side == "bullish" else entry["bear"]
    parts = [f"{count} of {_plural(_voters(entry), 'creator')} {side}"]
    backers = sorted(s for s, v in entry["votes"].items() if v == side)
    outcome = creator_tracker.asset_outcome(record, key, side, backers) if record else None
    priced = False
    if outcome and outcome["creators"] and outcome["move"] is not None:
        move = f"since {_mon_d(outcome['since'])}: {_pct(outcome['move'])}"
        if outcome["bench_move"] is not None:
            bench = _BENCHMARK_SHORT.get(outcome["benchmark_label"], outcome["benchmark_label"])
            move += f" ({bench} {_pct(outcome['bench_move'])})"
        parts.append(move)
        priced = True
    elif outcome and outcome["pending"] and len(outcome["pending"]) == _backed_calls(record, key, side, backers):
        parts.append("new call, no close yet")      # every backer's call is too new for a close
        priced = True
    shared = [(a, n) for a, n in entry["actions"].most_common() if n >= MIN_SHARED_CREATORS]
    if shared:
        parts.append(", ".join(f"{n} say {a}" for a, n in shared))
    target = _target_part(entry, side, price)
    if target:
        parts.append(target)
    return f"{EMOJI[side]} {fmt.b(entry['label'])}" + fmt.esc(" — " + " · ".join(parts)), priced


def _disagreement_line(entry, fmt):
    bulls = sorted(s for s, v in entry["votes"].items() if v == "bullish")
    bears = sorted(s for s, v in entry["votes"].items() if v == "bearish")
    minority, side = (bulls, "bullish") if len(bulls) < len(bears) else (bears, "bearish")
    return (f"{EMOJI['mixed']} {fmt.b(entry['label'])} — {len(bulls)} bullish vs {len(bears)} bearish "
            f"({side}: {fmt.esc('; '.join(minority))})")


def _lean_share(entries):
    """(bullish share, votes) over creator votes in `entries`."""
    bull = sum(e["bull"] for e in entries)
    bear = sum(e["bear"] for e in entries)
    return (bull / (bull + bear) if bull + bear else None), bull + bear


def _creator_leans(views):
    """{creator: lean}: canonical_claims.lean over each creator's directional
    votes across the window's assets. The takeaway counts creators, not
    videos: one prolific creator made half of some weeks' videos and most of
    their bearish ones."""
    votes = defaultdict(Counter)
    for entry in views.values():
        for source, vote in entry["votes"].items():
            votes[source][vote] += 1
    return {s: canonical_claims.lean(v["bullish"], v["bearish"]) for s, v in votes.items()}


def _market_row(market, symbol):
    return next((r for r in market if r["symbol"] == symbol), None)


def _fresh_market(inputs):
    """The tracker's market-context rows fit to print: those whose last close
    is within STALE_CONTEXT_DAYS of the as-of (an older series would print a
    stale week as this one)."""
    as_of = inputs.get("as_of")
    return [r for r in inputs.get("market") or []
            if as_of is None or (as_of - r["as_of"]).days <= STALE_CONTEXT_DAYS]


def _takeaway(current, agree, market, picks):
    """The fixed-template opening: the creator-level lean, the strongest
    agreement, what the market did in the week to the last close (named, so
    a stale cache can never pass for "last week") and how last week's picks
    went. Never an LLM call."""
    sentences = []
    leans = _creator_leans(current)
    if leans:
        bull = sum(v == "bullish" for v in leans.values())
        bear = sum(v == "bearish" for v in leans.values())
        if bull or bear:
            sentences.append(f"{bull} of {_plural(len(leans), 'creator')} leaned bullish, "
                             f"{bear or 'none'} bearish.")
        else:
            sentences.append(f"None of the {_plural(len(leans), 'creator')} leaned bullish or bearish.")
    if agree:
        top = agree[0][1]
        side = _consensus(top)
        # "on Micron (MU)" rather than "Micron (MU) (4 of 4 ...)": a display
        # label can carry its own parentheses.
        sentences.append(f"Strongest agreement: {top['bull'] if side == 'bullish' else top['bear']} of "
                         f"{_voters(top)} creators {side} on {top['label']}.")
    moves, week_end = [], None
    for symbol, name in ((creator_tracker.MARKET_SYMBOL, "the S&P 500"),
                         (creator_tracker.CRYPTO_MARKET_SYMBOL, "Bitcoin")):
        row = _market_row(market, symbol)
        if row and row["week"] is not None:
            moves.append(f"{name} {_moved(row['week'])}")
            week_end = week_end or row["as_of"]
    if moves:
        sentences.append(f"In the week to {_dow_mon_d(week_end)}, {' and '.join(moves)}.")
    if picks.get("rows"):
        sentences.append(f"Last week's picks: {picks['right']} of {len(picks['rows'])} went the creators' way.")
    return " ".join(sentences)


def _stale_prices_line(inputs):
    """A warning when the newest close the pulse prices at is older than the
    last completed session (the warm job did not run): the reader must not
    take an old week for this one. None when the prices are current."""
    as_of, requested = inputs.get("as_of"), inputs.get("requested_as_of")
    if not as_of or not requested or as_of >= requested:
        return None
    return (f"⚠️ Prices are stale: the newest close in the data is {_dow_mon_d(as_of)}, not "
            f"{_dow_mon_d(requested)}; every move here ends there.")


def _market_lines(market, fmt, stale=None):
    """'Market · Fri Oct 2 → Fri Oct 9 close', then the week's and the four
    weeks' moves of the market-context symbols that have data (a symbol
    without data is left out, never printed as n/a), and the stale-price
    warning when there is one."""
    week = [r for r in market if r["week"] is not None]
    four = [r for r in market if r["four_weeks"] is not None]
    if not (week or four):
        return ["", fmt.esc(stale)] if stale else []
    anchor = _market_row(market, creator_tracker.MARKET_SYMBOL) or market[0]
    head = fmt.b("Market")
    if anchor["week_from"]:
        head += fmt.esc(f" · {_dow_mon_d(anchor['week_from'])} → {_dow_mon_d(anchor['as_of'])} close")
    lines = ["", head]
    if week:
        lines.append(fmt.esc(" · ".join(f"{r['label']} {_pct(r['week'])}" for r in week)))
    if four:
        lines.append(fmt.esc("4 weeks: " + " · ".join(f"{r['label']} {_pct(r['four_weeks'])}" for r in four)))
    if stale:
        lines.append(fmt.esc(stale))
    return lines


def _creators_vs_market_lines(record, fmt):
    """One pooled, mark-to-market line over every call the tracker could
    mark (creator_tracker.track_record's pooled block), with its
    denominators. Never per creator and never a ranking: that is the Friday
    scorecard's, behind the same label."""
    pooled = (record or {}).get("pooled") or {}
    calls = pooled.get("calls") or 0
    if calls < MIN_POOLED_CALLS or not record.get("since") or not record.get("market_data"):
        return []
    text = (f"Calls since {_mon_d(record['since'])}, marked to {_dow_mon_d(record['as_of'])}: {calls} calls · "
            f"{pooled['right']}/{calls} moving the called way")
    if pooled["bench_n"]:
        text += (f" · {pooled['beat']}/{pooled['bench_n']} beat the S&P 500 in the called direction over the "
                 "same days (crypto vs Bitcoin)")
    text += f". {_REPORT_LABEL} — per-creator detail in Friday's scorecard."
    return ["", fmt.b("Creators vs the market"), fmt.esc(text)]


def _picks_lines(picks, fmt):
    """'Last week's picks, one week on' (last_week_picks); nothing when no
    pick could be priced."""
    rows = picks.get("rows") or []
    if not rows:
        return []
    head = fmt.b("Last week's picks, one week on") + fmt.esc(
        f" · {_dow_mon_d(picks['start'])} → {_dow_mon_d(picks['end'])}")
    line = " · ".join(f"{r['icon']} {r['label']} {_pct(r['ret'])}"
                      f"{' (bearish)' if r['direction'] == 'bearish' else ''}" for r in rows)
    # The average is in the called direction; the S&P 500's move is its own,
    # so it stands beside it rather than as a like-for-like comparison.
    summary = (f"{picks['right']} of {len(rows)} went the creators' way · together {_pct(picks['avg'])} "
               "in the called direction")
    if picks.get("spy") is not None:
        summary += f" · S&P 500 {_pct(picks['spy'])}"
    if picks.get("unpriced"):
        summary += " · not priced: " + ", ".join(picks["unpriced"])
    legend = "✅ went their way and beat the market · ➖ went their way, lagged the market · ❌ went against them"
    return ["", head, fmt.esc(line), fmt.esc(summary), fmt.esc(legend)]


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
    Telegram HTML (every dynamic string escaped). "" when the window holds
    no view claims.

    Layout, most useful first, every sentence a fixed template over the
    numbers below it (never an LLM call): the takeaway; what the market did
    last week and over four weeks, at one as-of close; where creators agree,
    each row with the move since the creators' first call against the S&P
    500 (crypto: Bitcoin); one pooled creators-vs-market line; where they
    disagree; last week's picks one week on; changes vs last week; what is
    new on the radar; the mood; what creators say they own; and a footer
    naming the price date and the coverage. Per-creator results and rankings
    never appear: they belong to the Friday scorecard.
    """
    fmt = _Fmt(html)
    current, previous = inputs["current_views"], inputs["previous_views"]
    if not inputs["videos"]:
        return ""
    prices = inputs["latest_prices"]
    record = inputs.get("track")
    market = _fresh_market(inputs)
    picks = inputs.get("picks") or {}

    houses = inputs.get("houses")
    ranked = _rank(current)
    agree = agree_rows(current, houses=houses)
    disagree = [(k, e) for k, e in ranked if _voter_houses(e, houses) >= max(MIN_CREATORS, MIN_DISAGREE_CREATORS)
                and _consensus(e) == "mixed" and e["bull"] and e["bear"]][:MAX_DISAGREEMENT_ROWS]
    shown = {k for k, _ in agree} | {k for k, _ in disagree}

    week = next((w for w in inputs["tone"] if w["weeks_ago"] == 0), None)
    lines = [f"📈 {fmt.b('Weekly Market Pulse')} · {fmt.esc(pulse_charts.window_label(inputs['window_start'], today))}"]
    takeaway = _takeaway(current, agree, market, picks)
    if takeaway:
        lines.append(fmt.esc(takeaway))
    lines += _market_lines(market, fmt, _stale_prices_line(inputs))

    if agree:
        rows = [_agreement_line(k, e, prices.get(k), record, fmt) for k, e in agree]
        head = fmt.b("Where creators agree")
        if any(priced for _, priced in rows):
            head += fmt.esc(" · move since the first call (vs S&P 500, crypto vs Bitcoin)")
        lines += ["", head] + [row for row, _ in rows]
    lines += _creators_vs_market_lines(record, fmt)
    if disagree:
        lines += ["", fmt.b("Where they disagree")]
        lines += [_disagreement_line(e, fmt) for _, e in disagree]
    lines += _picks_lines(picks, fmt)

    flips = []
    for key, entry in current.items():
        prev = previous.get(key)
        if not prev or _voter_houses(entry, houses) < MIN_CREATORS or _voter_houses(prev, houses) < MIN_CREATORS:
            continue
        before, after = _consensus(prev), _consensus(entry)
        if {before, after} == {"bullish", "bearish"}:
            flips.append((entry["label"], before, after))
    if flips:
        lines += ["", fmt.b("Consensus changed vs last week")]
        lines += [f"🔄 {fmt.b(label)}: {before} → {after}" for label, before, after in sorted(flips)]

    new = [e for k, e in ranked
           if k not in inputs["older_keys"] and k not in shown and _voter_houses(e, houses) >= MIN_CREATORS]
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
            lines.append(f"+{_plural(len(groups) - MAX_PULSE_DISCLOSURE_SOURCES, 'more creator')}")

    if inputs["weight_details"]:
        weighted = " · ".join(
            f"{channel} {inputs['weights'][channel]:.2f} ({hits}/{total})"
            for channel, (hits, total) in sorted(inputs["weight_details"].items())
        )
        lines += ["", fmt.esc(f"⚖️ Weighted by scorecard track record: {weighted}")]

    footer = []
    if record and record.get("market_data") and inputs.get("as_of"):
        footer.append(f"Prices: daily closes to {_dow_mon_d(inputs['as_of'])}.")
    footer.append(f"Based on {_plural(inputs['view_claims'], 'opinion')} in {_plural(inputs['videos'], 'video')} "
                  f"from {_plural(inputs['channels'], 'creator')}; an asset needs {MIN_CREATORS}+ creators "
                  "to be listed.")
    lines += ["", fmt.i(" ".join(footer)), fmt.esc(DISCLAIMER)]
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
                min_creators=MIN_CREATORS, houses=inputs.get("houses"),
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
