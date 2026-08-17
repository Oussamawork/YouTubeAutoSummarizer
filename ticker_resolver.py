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
    """
    asset, listing = _tokens(asset_name), _tokens(listing_name)
    if not asset or not listing:
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
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": MAP_VERSION, "tickers": dict(sorted(entries.items()))},
                      f, indent=1, sort_keys=True)
            f.write("\n")
    except OSError as e:
        log_warn(f"Could not write the ticker map to {path}: {e}")
        return False
    resolved = sum(1 for e in entries.values() if e.get("ticker"))
    log_info(f"Ticker map saved: {resolved} resolved, {len(entries) - resolved} unlisted.")
    return True


def _usable_listing(row):
    """A row we can actually price: US-listed and quoted in dollars. The free
    provider plan carries US markets only, and a dollar close is the one that
    can be compared against the dollar price targets speakers give."""
    return (row.get("country") == "United States"
            and (row.get("currency") or "").upper() == "USD"
            and (row.get("symbol") or "").isalpha())


def search_candidates(query, api_key, searcher=None):
    """US/USD listings the provider returns for a name or ticker."""
    if searcher is None:
        import channel_scorecard as cs
        searcher = cs.search_symbols
    return [row for row in (searcher(query, api_key) or []) if _usable_listing(row)]


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
        import summarizer
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
            for row in search_candidates(query, api_key, searcher=searcher):
                if names_match(asset_name, row.get("instrument_name")):
                    return entry(row["symbol"].upper(), row, "search")
    except Exception as e:
        if type(e).__name__ == "SearchUnavailable":
            return entry(None, None, "unchecked")
        raise

    # 2. Let the model propose, then verify each proposal against the
    #    catalogue — both that the ticker exists and that it is this company.
    try:
        candidates = llm_candidates(asset_name, recorded_ticker, completer=completer)
    except Exception:
        candidates = []
    for candidate in candidates:
        try:
            rows = search_candidates(candidate, api_key, searcher=searcher)
        except Exception as e:
            if type(e).__name__ == "SearchUnavailable":
                return entry(None, None, "unchecked")
            raise
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
