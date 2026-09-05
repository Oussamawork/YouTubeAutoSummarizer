"""Learn the real ticker for an asset the transcript named badly.

Speakers say "Rubric" for Rubrik, "Nebl" for Nebius, or just "SK Hynix" with
no ticker at all. The curated table in market_pulse covers the cases someone
noticed; this resolves the rest, and writes what it learns to
`data/ticker_map.json` so it is paid for once.

The design point, and the reason this is not the thing the codebase already
rejected (see market_pulse.ASSET_ALIASES): **the model proposes, the price
provider's catalogue disposes.** An LLM is good at "Rubric is probably
Rubrik" and bad at knowing whether RBRK exists; the provider's symbol search
is authoritative on the second and silent on the first. So a suggestion is
only accepted when the provider lists it AND the listing's company name
actually looks like the asset we asked about — otherwise a plausible-sounding
ticker that happens to belong to an unrelated company would sail through.

Every entry records how it was resolved, so the map stays auditable: a human
can read data/ticker_map.json and see which entries came from a search hit and
which from a model suggestion.
"""
import json
import os
import re
from difflib import SequenceMatcher
from datetime import datetime, timezone

import channel_scorecard as cs
import summarizer
from helpers import write_json_atomic
from log import log_info, log_warn

# How alike two words must be to count as the same name. 0.8 sits between the
# worst real mis-hearing observed (rubric/rubrik at 0.83) and the closest
# false pair (rubrik/ruby at 0.60).
NAME_SIMILARITY = 0.8

MAP_FILE = "data/ticker_map.json"
MAP_VERSION = 1
# Words that carry no identity, so they must not be what makes a listing's
# name "match" the asset we asked about.
_STOPWORDS = {
    "inc", "inc.", "incorporated", "corp", "corp.", "corporation", "company",
    "co", "co.", "ltd", "ltd.", "limited", "plc", "sa", "nv", "ag", "the",
    "holdings", "holding", "group", "technologies", "technology", "systems",
    "international", "class", "common", "stock", "shares", "adr", "sponsored",
    "and", "se", "kgaa", "ab", "oyj", "spa", "asa", "nyse", "nasdaq",
}

# A country in the listing's name that the speaker never said marks a
# different legal entity: "McDonald's Holdings Company (Japan)" trades on OTC
# as MDNDF and is not McDonald's Corporation. Measured 2026-08-17 — a symbol
# search for "McDonald's" returns the Japan entity and never MCD at all, so
# exchange ranking has nothing to promote and this is the only thing that
# stops the wrong security being adopted.
GEO_QUALIFIERS = {
    "japan", "japanese", "china", "chinese", "india", "indian", "korea",
    "korean", "brazil", "brazilian", "mexico", "mexican", "europe", "european",
    "germany", "german", "france", "french", "italy", "italian", "spain",
    "spanish", "britain", "british", "australia", "australian", "canada",
    "canadian", "africa", "african", "asia", "asian", "russia", "russian",
    "taiwan", "thailand", "indonesia", "malaysia", "singapore", "philippines",
    "vietnam", "turkey", "poland", "netherlands", "dutch", "swiss",
}


def _tokens(text):
    return {t for t in re.split(r"[^a-z0-9]+", (text or "").lower())
            if t and t not in _STOPWORDS}


def names_match(asset_name, listing_name):
    """
    True when a listing plausibly belongs to the asset. Requires a shared
    identity word — "Rubrik" vs "Rubrik Inc" passes, "Rubrik" vs "Ruby Tuesday"
    does not. This is the guard that stops a hallucinated-but-real ticker being
    accepted for the wrong company.

    A listing that adds a country the asset never mentioned is rejected even
    when the rest matches: the shared name is the franchise, not the company.
    """
    asset, listing = _tokens(asset_name), _tokens(listing_name)
    if not asset or not listing:
        return False
    if (listing & GEO_QUALIFIERS) - asset:
        return False
    if asset & listing:
        return True
    # Transcripts mangle spelling ("Rubric"/"Rubrik", "Nebius"/"Nebus"), and a
    # mis-hearing is usually a substitution mid-word, which a shared-prefix
    # test misses. Sequence similarity catches those while still keeping
    # genuinely different names apart: rubric/rubrik scores 0.83 and
    # nebus/nebius 0.91, where rubrik/ruby only reaches 0.60.
    for a in asset:
        for b in listing:
            if min(len(a), len(b)) >= 4 and SequenceMatcher(None, a, b).ratio() >= NAME_SIMILARITY:
                return True
    return False


def load(path=MAP_FILE):
    """Learned ticker map as {recorded_key: entry}; {} when absent or unreadable."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError) as e:
        log_warn(f"Could not read the ticker map at {path}: {e}")
        return {}
    if not isinstance(payload, dict) or payload.get("version") != MAP_VERSION:
        return {}
    entries = payload.get("tickers")
    return entries if isinstance(entries, dict) else {}


def save(entries, path=MAP_FILE):
    """Write the learned map. Returns True on success; never raises."""
    payload = {"version": MAP_VERSION, "tickers": dict(sorted(entries.items()))}
    if not write_json_atomic(path, payload, indent=1):
        return False
    resolved = sum(1 for e in entries.values() if e.get("ticker"))
    log_info(f"Ticker map saved: {resolved} resolved, {len(entries) - resolved} unlisted.")
    return True


# Words that mark a listing as a fund/product rather than the company itself.
# A call on Anthropic scored against "Harbor Anthropic AI Lab Ecosystem ETF",
# or on semiconductors against "ProShares Ultra Semiconductors" (2x leveraged),
# would be measuring a different instrument — and in the leveraged case would
# double every move the scorecard reads.
FUND_MARKERS = {
    "etf", "etn", "etp", "fund", "trust", "index", "ecosystem", "ultra",
    "leveraged", "inverse", "2x", "3x", "ishares", "proshares", "vaneck",
    "spdr", "invesco", "harbor", "coinshares", "direxion", "wisdomtree",
    "bitwise", "grayscale", "amplify", "roundhill", "defiance",
}

# Sector, theme and geography labels the extractor files as "stock". They are
# not companies, and any listing whose name shares the word is a coincidence:
# "Aerospace" matched Honeywell Aerospace, "Software" matched Unity Software.
GENERIC_LABELS = {
    "aerospace", "software", "hardware", "semiconductors", "semis", "altcoins",
    "crypto", "miners", "mining", "gold", "silver", "financials", "banks",
    "healthcare", "biotech", "energy", "oil", "utilities", "industrials",
    "materials", "retail", "consumer", "technology", "tech", "defense",
    "china", "europe", "japan", "india", "emerging", "markets", "market",
    "stocks", "equities", "indices", "bonds", "treasuries", "commodities",
    "quantum", "robotics", "cloud", "cybersecurity", "infrastructure",
}


def is_generic_label(asset_name):
    """True when the 'asset' is a sector or theme, not a company."""
    tokens = _tokens(asset_name)
    return bool(tokens) and tokens <= GENERIC_LABELS


def _is_fund(listing_name):
    return bool(_tokens(listing_name) & FUND_MARKERS)


def _usable_listing(row, recorded_ticker=None):
    """
    A row we can actually price and that is genuinely the company: US-listed,
    quoted in dollars, and not a fund tracking it.

    The fund exclusion is skipped only when the speaker named that exact
    ticker — someone saying "GDX" means the ETF, but someone saying "gold
    miners" does not.
    """
    if not (row.get("country") == "United States"
            and (row.get("currency") or "").upper() == "USD"
            and (row.get("symbol") or "").isalpha()):
        return False
    instrument = (row.get("instrument_type") or "").lower()
    if "warrant" in instrument or "right" in instrument:
        return False
    if _is_fund(row.get("instrument_name")) or "etf" in instrument or "fund" in instrument:
        return (recorded_ticker or "").upper() == (row.get("symbol") or "").upper()
    return True


# Exchanges that carry a company's primary US listing. An OTC row for a
# foreign franchise entity (McDonald's Holdings Japan) is a real listing and
# the wrong one, so primary venues are preferred when both are offered.
PRIMARY_EXCHANGES = ("NASDAQ", "NYSE", "NYSE ARCA", "AMEX", "BATS")


def _rank(asset_name, row):
    """Sort key preferring the closest name on the most primary exchange."""
    name_score = SequenceMatcher(None, (asset_name or "").lower(),
                                 (row.get("instrument_name") or "").lower()).ratio()
    exchange = (row.get("exchange") or "").upper()
    primary = 0 if exchange in PRIMARY_EXCHANGES else 1
    return (primary, -round(name_score, 3), len(row.get("symbol") or ""))


def search_candidates(query, api_key, searcher=None, recorded_ticker=None,
                      asset_name=None):
    """Usable listings for a query, best match first."""
    if searcher is None:
        searcher = cs.search_symbols
    rows = [row for row in (searcher(query, api_key) or [])
            if _usable_listing(row, recorded_ticker)]
    return sorted(rows, key=lambda row: _rank(asset_name or query, row))


LLM_SYSTEM = (
    "You map company names mentioned in finance videos to their US stock ticker. "
    "Transcripts mis-spell names, so correct obvious mis-hearings. "
    "Reply with JSON only: {\"candidates\": [\"TICKER\", ...], \"company\": \"Full name\"}. "
    "List at most 3 tickers, most likely first, US listings (including ADRs) only. "
    "Use an empty list if the company is private, unlisted, or you are unsure — "
    "a wrong ticker is far worse than none."
)


def llm_candidates(asset_name, recorded_ticker=None, completer=None):
    """
    Ticker guesses from the LLM. These are *hypotheses*: nothing here reaches
    the dataset until the provider confirms the ticker exists and belongs to
    this company. Returns [] on any failure or quota exhaustion.
    """
    if completer is None:
        completer = summarizer.complete
    prompt = f"Company as heard in the video: {asset_name!r}."
    if recorded_ticker:
        prompt += f" The transcript recorded the ticker as {recorded_ticker!r}, which may be wrong."
    try:
        raw = completer(LLM_SYSTEM, prompt, json_mode=True)
    except Exception as e:  # a resolver must never break the caller
        log_warn(f"Ticker suggestion failed for {asset_name!r}: {e}")
        return []
    if not raw or raw == "QUOTA_EXHAUSTED":
        return []
    try:
        payload = json.loads(raw)
    except ValueError:
        log_warn(f"Ticker suggestion for {asset_name!r} was not JSON.")
        return []
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(candidates, list):
        return []
    return [c.strip().upper() for c in candidates
            if isinstance(c, str) and c.strip()][:3]


def resolve(asset_name, recorded_ticker, api_key, searcher=None, completer=None):
    """
    Work out the real US ticker for one asset, or None when there isn't one.

    Order matters: the provider's own search is tried first, because when it
    answers, no model is involved at all. The LLM is the fallback for names
    the catalogue cannot match on spelling — and its answer is then put back
    through the same verification.

    Returns an entry dict recording the outcome and how it was reached.
    """
    if is_generic_label(asset_name):
        return {"ticker": None, "company": None, "exchange": None,
                "via": "not-a-company",
                "checked": datetime.now(timezone.utc).date().isoformat()}

    stamp = datetime.now(timezone.utc).date().isoformat()
    # "unchecked" is deliberately not "unresolved": a lookup that never
    # completed must not be cached as a verdict, or one rate-limited run would
    # permanently mark a real company as having no listing.

    def entry(ticker, listing, via):
        return {"ticker": ticker,
                "company": (listing or {}).get("instrument_name"),
                "exchange": (listing or {}).get("exchange"),
                "via": via, "checked": stamp}

    # 1. Ask the catalogue directly, by name and by the recorded ticker.
    try:
        for query in filter(None, (asset_name, recorded_ticker)):
            for row in search_candidates(query, api_key, searcher=searcher,
                                         recorded_ticker=recorded_ticker,
                                         asset_name=asset_name):
                symbol = (row.get("symbol") or "").upper()
                # An exact ticker match is itself the identity check: the
                # speaker named this symbol and the catalogue confirms it
                # exists. Company-name similarity does not apply — nobody
                # expects "GDX" to look like "VanEck Gold Miners ETF".
                if symbol and symbol == (recorded_ticker or "").upper():
                    return entry(symbol, row, "search")
                if names_match(asset_name, row.get("instrument_name")):
                    return entry(symbol, row, "search")
    except cs.SearchUnavailable:
        return entry(None, None, "unchecked")

    # 2. Let the model propose, then verify each proposal against the
    #    catalogue — both that the ticker exists and that it is this company.
    try:
        candidates = llm_candidates(asset_name, recorded_ticker, completer=completer)
    except Exception:
        candidates = []
    for candidate in candidates:
        try:
            rows = search_candidates(candidate, api_key, searcher=searcher,
                                     recorded_ticker=recorded_ticker,
                                     asset_name=asset_name)
        except cs.SearchUnavailable:
            return entry(None, None, "unchecked")
        for row in rows:
            if row.get("symbol", "").upper() != candidate:
                continue
            if names_match(asset_name, row.get("instrument_name")):
                log_info(f"Resolved {asset_name!r} -> {candidate} "
                         f"({row.get('instrument_name')}) via suggestion + catalogue.")
                return entry(candidate, row, "llm+verified")
            log_warn(f"Rejected {candidate} for {asset_name!r}: the listing is "
                     f"{row.get('instrument_name')!r}, a different company.")

    return entry(None, None, "unresolved")
