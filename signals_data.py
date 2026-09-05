"""The signals dataset: loading `data/signals.jsonl`, asset identity, date
windows and per-asset aggregation.

This is the leaf every analytics module reads from — the weekly pulse, its
charts, the channel scorecard and the price-cache warmer. It imports nothing
from any of them, which is what lets those modules import each other freely:
before this split `market_pulse` was both the data layer and a report, so
`channel_scorecard` had to import it at the top while it imported
`channel_scorecard` lazily inside functions, and `ticker_resolver` compared an
exception's class *name* because it could not import the class.
"""
import json
import os
from collections import Counter
from datetime import datetime

from log import log_warn, log_error

SIGNALS_FILE = "data/signals.jsonl"
STANCE_SCORE = {"bullish": 1, "bearish": -1, "neutral": 0}
# Net-stance thresholds: mean score above/below this counts as a directional
# consensus; in between reads as mixed.
NET_THRESHOLD = 0.15
# A directional vote is weighted by how strongly the speaker stated the view.
# "unspecified" is honest absence (the extractor is forbidden from guessing a
# conviction), so it weighs the same as a stated-weak one — a bare lean should
# not move consensus like a table-pounding call with a target does.
CONVICTION_WEIGHTS = {"high": 1.5, "medium": 1.0, "low": 0.75, "unspecified": 0.75}


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
    "SPACEX": "SPCX",  # verified live 2026-08-17: spcx.us returns closes,
                       # while symbol search only surfaces Thai DRs and 3x
                       # leveraged ETPs, which are the wrong instrument
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
    "CXMT", "YMTC",                             # unlisted Chinese memory makers
    "SK HYNIX", "BASF", "VOW", "P911", "ADYEN",  # non-US listings
}


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
    _LEARNED_ENTRIES.clear()
    return _LEARNED


_LEARNED_ENTRIES = {}


def learned_ticker_entries(path=LEARNED_TICKERS_FILE):
    """
    The learned map's full records — {NAME: {"ticker", "exchange", "company",
    "via", "checked"}} — for consumers that need the verified listing venue
    (instruments.py). Empty when the process-wide map has been replaced by a
    test seam, so a test never reads the committed file by accident.
    """
    if _LEARNED is not None and not _LEARNED:
        return {}
    if not _LEARNED_ENTRIES:
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict) and payload.get("version") == 1:
                    for key, entry in (payload.get("tickers") or {}).items():
                        if isinstance(entry, dict) and entry.get("ticker"):
                            _LEARNED_ENTRIES[key.strip().upper()] = dict(entry)
        except (OSError, ValueError) as e:
            log_warn(f"Could not read the learned ticker map: {e}")
    return _LEARNED_ENTRIES


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
        ticker = canonical_ticker(asset)
        entry = stats.setdefault(key, {
            # Canonical label/ticker, not the first-seen raw one: a record
            # carrying GOOG must not name the GOOGL bucket.
            "label": ticker or asset.get("name") or key,
            "ticker": ticker,
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


def _direction(score):
    if score > NET_THRESHOLD:
        return "bullish"
    if score < -NET_THRESHOLD:
        return "bearish"
    return "mixed"


def _overall_tone(records):
    tone = Counter()
    for record in records:
        signals = record.get("signals")
        if isinstance(signals, dict) and signals.get("market_sentiment"):
            tone[signals["market_sentiment"]] += 1
    return tone
