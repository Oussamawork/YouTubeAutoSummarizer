import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from log import log_info, log_warn, log_error
from sendToTelegram import send_telegram_text, send_telegram_photo_album

# Weekly market pulse: aggregates the per-video signals accumulated in
# data/signals.jsonl (see signals.py) into one Telegram report — top mentioned
# assets with net stance, consensus flips vs the prior window, and assets newly
# on the radar. Runs from its own weekly workflow; read-only over the dataset.
# The output is research over creator opinions, not investment advice.

load_dotenv('.env')

SIGNALS_FILE = "data/signals.jsonl"
STANCE_SCORE = {"bullish": 1, "bearish": -1, "neutral": 0}
# An asset is "new on the radar" when it appears in the current window but in
# none of the records this many days before the window started.
NEW_ASSET_LOOKBACK_DAYS = 30
MAX_ASSETS_IN_REPORT = 8
# Net-stance thresholds: mean score above/below this counts as a directional
# consensus; in between reads as mixed.
NET_THRESHOLD = 0.15
# A directional vote is weighted by how strongly the speaker stated the view.
# "unspecified" is honest absence (the extractor is forbidden from guessing a
# conviction), so it weighs the same as a stated-weak one — a bare lean should
# not move consensus like a table-pounding call with a target does.
CONVICTION_WEIGHTS = {"high": 1.5, "medium": 1.0, "low": 0.75, "unspecified": 0.75}

DISCLAIMER = "⚠️ Aggregated creator opinions — research input, not investment advice."

# Accuracy weighting: once a channel has at least this many scorecard-evaluated
# calls, its stances are weighted by track record (0.5 + hit rate → 0.5..1.5)
# instead of counting 1.0 like everyone else. Proven channels move the
# consensus more; proven-wrong channels move it less.
MIN_TRACK_CALLS = 5


def load_signals(path=SIGNALS_FILE):
    """Read signal records from the JSONL file; malformed lines are skipped
    with a warning. Returns [] when the file is missing (dataset not started)."""
    if not os.path.exists(path):
        log_warn(f"No signals file at {path}; nothing to aggregate yet.")
        return []
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    log_warn(f"Skipping malformed JSONL line {lineno} in {path}.")
                    continue
                if isinstance(record, dict):
                    records.append(record)
    except OSError as e:
        log_error(f"Could not read {path}: {e}")
        return []
    return records


def _parse_date(value):
    try:
        return datetime.strptime((value or "").strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def _in_window(record, start, end):
    """True when the record's date is in (start, end] — half-open window."""
    d = _parse_date(record.get("date"))
    return d is not None and start < d <= end


# Canonical name -> ticker for assets the extractor records without one.
# The extraction prompt deliberately forbids the model from supplying tickers
# it wasn't given (that rule stopped ticker hallucination), so speakers who say
# "Chevron" but never "CVX" produce ticker-less records. Without this map the
# same asset aggregates under two keys (BTC vs BITCOIN — splitting mention
# counts, diluting consensus, breaking flip detection) and every ticker-less
# directional call is invisible to the scorecard. A curated table in code is
# deterministic and auditable in a way a model guess never is.
# Keys are the normalized (upper, single-spaced) names observed in the dataset;
# private companies (SpaceX, OpenAI, ...) are deliberately absent — they have
# no ticker, and aggregating them by name is the correct behavior.
ASSET_ALIASES = {
    # crypto
    "BITCOIN": "BTC", "ETHEREUM": "ETH", "SOLANA": "SOL",
    # megacaps and frequently discussed stocks
    "NVIDIA": "NVDA", "MICROSOFT": "MSFT", "APPLE": "AAPL", "AMAZON": "AMZN",
    "META": "META", "META PLATFORMS": "META", "ALPHABET": "GOOGL",
    "GOOGLE": "GOOGL", "TESLA": "TSLA", "NETFLIX": "NFLX",
    # semis / hardware
    "MICRON": "MU", "INTEL": "INTC", "AMD": "AMD", "ASML": "ASML",
    "BROADCOM": "AVGO", "MARVELL": "MRVL", "TSMC": "TSM",
    "TAIWAN SEMICONDUCTOR": "TSM", "TAIWAN SEMICONDUCTOR MANUFACTURING": "TSM",
    "WESTERN DIGITAL": "WDC", "SEAGATE": "STX", "COHERENT": "COHR",
    "NEBIUS": "NBIS", "NEBUS": "NBIS",  # incl. the transcript's misspelling
    # software / fintech
    "PALANTIR": "PLTR", "SALESFORCE": "CRM", "IBM": "IBM", "ADOBE": "ADBE",
    "SNOWFLAKE": "SNOW", "SERVICENOW": "NOW", "ATLASSIAN": "TEAM",
    "ZSCALER": "ZS", "THE TRADE DESK": "TTD", "TRADE DESK": "TTD",
    "CROWDSTRIKE": "CRWD", "PALO ALTO": "PANW", "PALO ALTO NETWORKS": "PANW",
    "SOFI": "SOFI", "PAYPAL": "PYPL", "BLOCK": "XYZ", "REDDIT": "RDDT",
    "ZETA": "ZETA", "AXON": "AXON", "MERCADO LIBRE": "MELI",
    "MERCADOLIBRE": "MELI", "UBER": "UBER", "NU HOLDINGS": "NU",
    "ALIBABA": "BABA", "ORACLE": "ORCL", "SHERWIN WILLIAMS": "SHW",
    # energy
    "OCCIDENTAL PETROLEUM": "OXY", "CHEVRON": "CVX", "EXXON MOBIL": "XOM",
    "EXXONMOBIL": "XOM", "TOTAL ENERGIES": "TTE", "TOTALENERGIES": "TTE",
    # other observed
    "WALMART": "WMT", "HOME DEPOT": "HD", "UNDER ARMOUR": "UAA",
    "UNITED RENTALS": "URI", "BAE SYSTEMS": "BAESY", "SOFTBANK": "SFTBY",
    "ADYEN": "ADYEY",
    "TAIWAN SEMICONDUCTOR MANUFACTURING COMPANY": "TSM",
}

# Recorded-ticker variants folded to one canonical symbol: dual share classes
# and renames that speakers use interchangeably would otherwise still split an
# asset across two keys even when a ticker WAS recorded.
TICKER_ALIASES = {
    "GOOG": "GOOGL",   # Alphabet share classes
    "UA": "UAA",       # Under Armour share classes
    "SQ": "XYZ",       # Block's 2025 ticker change
    # Speakers say the company name and the extractor records it in the ticker
    # field. Left alone these split one asset across two buckets — the dataset
    # carried both AAPL and APPLE, diluting mention counts and consensus in
    # every chart — and they price as nothing, since no exchange lists "APPLE".
    "APPLE": "AAPL", "GOOGLE": "GOOGL", "ALPHABET": "GOOGL", "NVIDIA": "NVDA",
    "AMAZON": "AMZN", "MICROSOFT": "MSFT", "TESLA": "TSLA", "NETFLIX": "NFLX",
    "SALESFORCE": "CRM", "WESTERN DIGITAL": "WDC", "MASTEC": "MTZ",
    "NCINO": "NCNO", "BLOCK": "XYZ", "PALANTIR": "PLTR", "BROADCOM": "AVGO",
    # Tickers the transcript got wrong outright.
    "NEBL": "NBIS",    # Nebius is NBIS
    "RUBY": "RBRK",    # "Rubric" mis-transcribed; the company is Rubrik
    "PAS": "PAAS",     # Pan American Silver
}

# Recorded tickers that no US listing can price: private companies, and
# foreign or unlisted names the speakers discuss by local ticker. Kept
# explicit so they are skipped up front instead of spending a price request
# per run to rediscover a 404. They still aggregate by name in the pulse —
# only the price lookup is suppressed.
UNPRICEABLE_TICKERS = {
    "OPENAI", "STRIPE", "BYTEDANCE", "ANTHROPIC", "WAYMO", "ANDURIL",  # private
    "SPACEX", "SPCX",                           # private; see below
    "CXMT", "YMTC",                             # unlisted Chinese memory makers
    "SK HYNIX", "BASF", "VOW", "P911", "ADYEN",  # non-US listings
}
# SPACEX was aliased to SPCX on the evidence that "spcx.us returns closes".
# That is not evidence of the right instrument: SpaceX is private and has no
# listed equity, and the same note recorded that symbol search surfaced only
# Thai DRs and 3x leveraged ETPs. The series bears that out — 12 sessions from
# 108.27 to 146.15 with ±10% days, the signature of a leveraged ETP — so three
# real calls were being scored, with amplified returns, against an unrelated
# security. A ticker that prices *something* is not a ticker that prices the
# asset; when the instrument cannot be verified, no price is the honest answer.


def _normalized_name(asset):
    return " ".join((asset.get("name") or "").split()).upper()


LEARNED_TICKERS_FILE = "data/ticker_map.json"
_LEARNED = None


def learned_tickers(path=LEARNED_TICKERS_FILE):
    """
    Tickers resolved by ticker_resolver and committed to data/ticker_map.json.
    Read as plain data (not by importing the resolver) so the pulse has no
    dependency on the LLM or price provider. Loaded once per run.
    """
    global _LEARNED
    if _LEARNED is None:
        _LEARNED = {}
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict) and payload.get("version") == 1:
                    for key, entry in (payload.get("tickers") or {}).items():
                        if isinstance(entry, dict) and entry.get("ticker"):
                            _LEARNED[key.strip().upper()] = entry["ticker"].strip().upper()
        except (OSError, ValueError) as e:
            log_warn(f"Could not read the learned ticker map: {e}")
    return _LEARNED


def reset_learned_tickers(value=None):
    """Test seam for the process-wide learned map."""
    global _LEARNED
    _LEARNED = value
    return _LEARNED


def canonical_ticker(asset):
    """
    The asset's ticker, taking the recorded one first and falling back to the
    curated alias table, then to the learned map. None when none of them
    knows one.

    Order is deliberate: the curated table is hand-checked and wins, so a
    learned entry can never quietly override a decision someone made on
    purpose. Learned entries only fill gaps the table doesn't cover.
    """
    ticker = (asset.get("ticker") or "").strip().upper()
    name = _normalized_name(asset)
    if not ticker:
        ticker = ASSET_ALIASES.get(name)
    if ticker and ticker in TICKER_ALIASES:
        return TICKER_ALIASES[ticker]
    learned = learned_tickers()
    for key in filter(None, (ticker, name)):
        if key in learned:
            return learned[key]
    return ticker or None


def _asset_key(asset):
    return canonical_ticker(asset) or _normalized_name(asset)


def _iter_assets(records):
    """Yield (record, asset) for every valid asset entry in the records."""
    for record in records:
        signals = record.get("signals")
        if not isinstance(signals, dict):
            continue
        for asset in signals.get("assets", []):
            if isinstance(asset, dict) and _asset_key(asset):
                yield record, asset


def aggregate_assets(records, channel_weights=None):
    """
    Fold records into per-asset stats:
    {key: {label, type, mentions, channels, bull, bear, neutral,
           bull_w, bear_w, neutral_w, actions, targets}}
    Raw counts drive the display; the *_w sums (each stance counted at its
    channel's weight, default 1.0) drive the net-stance consensus.
    """
    channel_weights = channel_weights or {}
    stats = {}
    for record, asset in _iter_assets(records):
        key = _asset_key(asset)
        entry = stats.setdefault(key, {
            # Canonical label/ticker, not the first-seen raw one: a record
            # carrying GOOG must not name the GOOGL bucket.
            "label": canonical_ticker(asset) or asset.get("name") or key,
            "ticker": canonical_ticker(asset),
            "type": asset.get("type"),
            "mentions": 0, "channels": set(),
            "bull": 0, "bear": 0, "neutral": 0,
            "bull_w": 0.0, "bear_w": 0.0, "neutral_w": 0.0,
            "actions": Counter(), "targets": [],
        })
        entry["mentions"] += 1
        if record.get("channel_name"):
            entry["channels"].add(record["channel_name"])
        # A vote's weight composes the channel's track record with how strongly
        # the speaker stated the view — a proven channel's high-conviction call
        # moves consensus most; a hedged lean from anyone moves it least.
        weight = channel_weights.get(record.get("channel_name"), 1.0)
        weight *= CONVICTION_WEIGHTS.get(asset.get("conviction"), 0.75)
        stance = asset.get("stance")
        if stance == "bullish":
            entry["bull"] += 1
            entry["bull_w"] += weight
        elif stance == "bearish":
            entry["bear"] += 1
            entry["bear_w"] += weight
        else:
            entry["neutral"] += 1
            entry["neutral_w"] += weight
        action = asset.get("action")
        if action and action != "none":
            entry["actions"][action] += 1
        target = asset.get("price_target")
        if isinstance(target, (int, float)) and not isinstance(target, bool):
            entry["targets"].append(target)
    return stats


def _directional_mentions(entry):
    """How many mentions actually took a side. Attention ranks by this."""
    return entry["bull"] + entry["bear"]


def net_stance(entry):
    """
    Weighted mean stance score in [-1, 1] over the DIRECTIONAL votes only.

    Neutral mentions are breadth, not opinion: an asset name-dropped neutrally
    in five videos and called bullish in three is a 3-0 bullish consensus with
    wide radar coverage — not a "mixed" one. Counting neutrals in the
    denominator conflated "widely mentioned" with "no consensus" (the analyst-
    consensus convention is the same: abstentions don't dilute the rating).
    No directional votes at all reads as 0.0 -> mixed.
    """
    total = entry["bull_w"] + entry["bear_w"]
    if not total:
        return 0.0
    return (entry["bull_w"] - entry["bear_w"]) / total


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
        # Imported lazily: channel_scorecard imports helpers from this module,
        # so a top-level import would be circular.
        import channel_scorecard as cs
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


def _direction(score):
    if score > NET_THRESHOLD:
        return "bullish"
    if score < -NET_THRESHOLD:
        return "bearish"
    return "mixed"


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


def _overall_tone(records):
    tone = Counter()
    for record in records:
        signals = record.get("signals")
        if isinstance(signals, dict) and signals.get("market_sentiment"):
            tone[signals["market_sentiment"]] += 1
    return tone


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
        import channel_scorecard as cs
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


def _pulse_inputs(days, today, path, price_fetcher):
    """Load the dataset, slice the windows, and compute the best-effort extras
    (track-record weights, latest prices) that both the text pulse and the
    charts consume — shared so the price lookups run once per pulse, not once
    per output."""
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
        "records": records, "window_start": window_start,
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
        from pulse_charts import build_chart_data, render_charts, upside_rows

        today = today or datetime.now(timezone.utc).date()
        inputs = inputs or _pulse_inputs(days, today, path, price_fetcher)
        if not inputs["current"]:
            return []
        current = aggregate_assets(inputs["current"], inputs["weights"])
        previous = aggregate_assets(inputs["previous"], inputs["weights"])
        data = build_chart_data(
            inputs["records"], current, previous, inputs["window_start"], today
        )
        data["upside"] = upside_rows(current, inputs["latest_prices"])
        return render_charts(data, out_dir)
    except Exception as e:
        log_warn(f"Pulse charts unavailable this week: {e}")
        return []


CHART_ALBUM_CAPTION = (
    "This week in charts - how to read each one is written on the image."
)


def main():
    parser = argparse.ArgumentParser(description="Weekly market pulse from data/signals.jsonl")
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
        log_info("No signal records in the window; skipping the pulse this week.")
        return 0

    chart_paths = []
    if not args.no_charts:
        chart_paths = generate_charts(days=args.days, today=today, inputs=inputs,
                                      out_dir=args.charts_dir)

    if args.dry_run:
        print(pulse)
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
    # The charts illustrate the pulse; failing to send them shouldn't fail the
    # run once the text is out.
    if chart_paths and not send_telegram_photo_album(
        token, chat_id, chart_paths, caption=CHART_ALBUM_CAPTION
    ):
        log_warn("Chart album failed to send; the text pulse went out without it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
