"""
Creators vs the market: every creator's calls marked to market.

The strict scorecard (research_analytics.scorecard, the Friday report's
verdict layer) judges only matured, dated, testable forecasts, and the
horizons creators state run to 2027-28: it scores one claim. This module is
the separate MARK-TO-MARKET layer the reports lead with instead: what each
creator called, what the asset has done since, and what the market did over
the same days. Every output is labelled "mark-to-market, not a verdict",
prints every rate with its denominator, orders creators by how many calls
they made (never by results) and never ranks them, whatever
SCORECARD_RANKINGS says.

Pipeline (each rule in full in TRACKER_RULES):

  canonical claims (canonical_claims.load_canonical_claims)
   -> view claims that are directional views on the SECURITY, carry no
      trigger condition and resolve to an asset key; for the track record,
      only views on something tradable (a resolved ticker, a market with a
      PROXIES fund, or an instrument-typed name) — a theme, a sector without
      a fund or a macro view is not a call
   -> one vote per creator x asset x video (canonical_claims.lean; a mixed
      video, or one whose vote flips when its review-required own views are
      counted, makes no call); spellings of one fund ("the S&P 500", "US
      stocks") are one asset
   -> episodes per creator x asset: a same-direction video reiterates the
      open call, an opposite one closes it as reversed, an ended horizon
      closes it as matured
   -> instrument: PROXIES for index / commodity / sector keys, else
      instruments.resolve_instrument on the call's RESOLVED ticker; an
      unresolved instrument is excluded and counted, never guessed (a spoken
      name is never read as a ticker)
   -> entry = the next close after publication, one effective as-of close
      for every asset class, the market over IDENTICAL dates as benchmark
   -> per-creator, pooled and per-asset aggregates (two channels of one
      house count once wherever two creators must agree), market context,
      and plain-text Telegram messages

Prices come through an injected fetcher. The default (cache_fetcher) reads
the committed cache, data/prices.json, and never the network: the reports
have no live price budget. warm_prices covers needed_price_ranges, which
returns exactly the requests track_record makes (one per symbol) because
both run the same private planner, so the warm job and the tracker cannot
drift apart.

channel_scorecard, market_pulse, research_analytics, pulse_charts and
warm_prices import this module, so it imports none of them, nor anything
that reaches the network: `import creator_tracker` never loads `requests`.
The public API is __all__; every other name is private.
"""
import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone

import canonical_claims
import instruments
import price_cache
import scorecard_pricing as sp
from helpers import env_int, write_json_atomic
from log import describe_error, log_info, log_warn

__all__ = [
    "LOOKBACK_DAYS", "MIN_CALL_AGE_DAYS", "MIN_CREATOR_SAMPLE", "MIN_CREATOR_CALLS_SHOWN",
    "STALE_MARK_DAYS", "MARKET_LOOKBACK_DAYS", "CONTEXT_DAYS", "WINDOW_DAYS", "MAX_MESSAGE_CHARS",
    "MARKET_SYMBOL", "CRYPTO_MARKET_SYMBOL", "SELF_BENCHMARKED", "CONTEXT_SYMBOLS", "PROXIES",
    "PLAUSIBLE_TARGET_RATIO", "SECURITY_VIEW_TYPES", "NON_PRICE_METRICS", "PRICE_METRICS",
    "QUALIFIER_CONDITION", "REPORT_LABEL", "TRACKER_RULES",
    "NOT_AN_INSTRUMENT_TYPES", "CREATOR_HOUSES",
    "default_as_of", "is_security_view", "is_trigger_condition", "video_calls", "instrument_for",
    "creator_houses", "cache_fetcher", "static_fetcher", "track_record", "market_context", "price_move",
    "asset_outcome", "unpriced_assets", "needed_price_ranges", "format_market_lines", "format_report",
    "format_summary_lines", "main",
]

# --- Constants -------------------------------------------------------------------

# Claims published after as_of - LOOKBACK_DAYS are used (later ones too: a
# video from after the as-of opens a call that is counted as too_recent).
# Half a year: every open call's symbol is refetched on every warm run (one
# request each), so the lookback is what bounds the warm job's cost — at
# about a dozen new symbols a week a year-long book outgrows the 90-minute
# warm (docs/tdd-full-transcript-claims.md § 11).
LOOKBACK_DAYS = env_int("TRACKER_LOOKBACK_DAYS", 180)
# An OPEN call is aggregated once its mark is this many days after its entry.
MIN_CALL_AGE_DAYS = env_int("TRACKER_MIN_CALL_AGE_DAYS", 7)
# Creators with fewer scored calls are tagged "(small sample)".
MIN_CREATOR_SAMPLE = env_int("TRACKER_MIN_CREATOR_SAMPLE", 10)
# Creators with fewer scored calls get no line of their own in the report.
MIN_CREATOR_CALLS_SHOWN = 3
STALE_MARK_DAYS = 5          # an open call's mark this much older than the as-of is stale
MARKET_LOOKBACK_DAYS = 14    # how far back the as-of may fall to the last S&P 500 close
CONTEXT_DAYS = 35            # market-context history: four weeks back plus slack
WINDOW_DAYS = 28             # the same-length check: every old-enough call 28 days on
MAX_MESSAGE_CHARS = 3900     # Telegram's 4,096 minus headroom

MARKET_SYMBOL = "spy.us"
CRYPTO_MARKET_SYMBOL = "btcusd"
# Instruments that ARE their benchmark: compared with nothing (raw return only).
SELF_BENCHMARKED = frozenset({"spy.us", "voo.us", "ivv.us", "btcusd"})
CONTEXT_SYMBOLS = (("spy.us", "S&P 500"), ("qqq.us", "Nasdaq-100"), ("btcusd", "Bitcoin"), ("ethusd", "Ether"))

# Canonical market keys (canonical_claims.asset_key of an un-tickered
# subject, claims.MARKET_SUBJECTS) priced through a liquid US ETF. Never a
# target: an ETF's price is not the index level a speaker names. Keys not
# here (DAX, NIKKEI 225, DXY, BITCOIN / USDT DOMINANCE, COPPER, FED,
# INFLATION, US TREASURIES, BONDS, ALTCOINS) are never priced.
PROXIES = {
    "S&P 500": ("spy.us", "S&P 500 (SPY)"),
    "US STOCK MARKET": ("spy.us", "US stocks (SPY)"),
    "NASDAQ": ("qqq.us", "Nasdaq-100 (QQQ)"),
    "DOW": ("dia.us", "Dow (DIA)"),
    "RUSSELL 2000": ("iwm.us", "Russell 2000 (IWM)"),
    "SMALL CAPS": ("iwm.us", "Small caps (IWM)"),
    "MID CAPS": ("mdy.us", "Mid caps (MDY)"),
    "GOLD": ("gld.us", "Gold (GLD)"),
    "SILVER": ("slv.us", "Silver (SLV)"),
    "OIL": ("uso.us", "Oil (USO)"),
    "GOLD MINERS": ("gdx.us", "Gold miners (GDX)"),
    "TECHNOLOGY": ("xlk.us", "Tech sector (XLK)"),
    "SEMICONDUCTORS": ("smh.us", "Semiconductors (SMH)"),
    "SOFTWARE": ("igv.us", "Software (IGV)"),
    "FINANCIALS": ("xlf.us", "Financials (XLF)"),
    "ENERGY": ("xle.us", "Energy sector (XLE)"),
    "CONSUMER DISCRETIONARY": ("xly.us", "Consumer discretionary (XLY)"),
    "CYBERSECURITY": ("cibr.us", "Cybersecurity (CIBR)"),
    "HEALTHCARE": ("xlv.us", "Health care (XLV)"),
    "INDUSTRIALS": ("xli.us", "Industrials (XLI)"),
    "CONSUMER STAPLES": ("xlp.us", "Consumer staples (XLP)"),
    "UTILITIES": ("xlu.us", "Utilities (XLU)"),
    "MATERIALS": ("xlb.us", "Materials (XLB)"),
    "BANKS": ("kbe.us", "Banks (KBE)"),
}

# A target outside 0.2x-5x of the entry is a misread number (a revenue
# figure, a market cap), not a price level.
PLAUSIBLE_TARGET_RATIO = (0.2, 5.0)

# What makes a claim a view on the SECURITY. Without this filter "Amazon's
# free cash flow is negative this year" (a bearish earnings forecast, said in
# a video about Nvidia) made a creator bearish on AMZN, who then "reversed"
# four times in eight days.
SECURITY_VIEW_TYPES = frozenset({"recommendation", "price_target", "valuation_view", "stance"})
NON_PRICE_METRICS = frozenset({"revenue", "earnings", "margin", "rate"})
PRICE_METRICS = frozenset({None, "price", "market_cap"})
# A view the track record cannot hold against a price: without a resolved
# ticker or a PROXIES fund, a subject of these types (or a market, basket or
# metric subject) is a theme, a strategy, a sector without a fund, macro or
# an unlisted company. Such views are not calls (claim exclusion
# not_an_instrument), so "no ticker" counts only real ticker gaps.
NOT_AN_INSTRUMENT_TYPES = frozenset({"macro", "commodity", "index", "sector", "currency", "bond", "other",
                                     "private_company"})


def _env_houses():
    """TRACKER_CREATOR_HOUSES: a JSON object {"channel": "house"}; {} when
    unset or unreadable (logged)."""
    raw = (os.getenv("TRACKER_CREATOR_HOUSES") or "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except ValueError:
        log_warn("TRACKER_CREATOR_HOUSES is not a JSON object; channel houses are read from the names only.")
        return {}
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items() if str(k).strip() and str(v).strip()}


# Channels of one house — a channel and its spin-offs ("Phantom by HKCM" and
# "HKCM GLOBAL" are HKCM's) — count ONCE wherever two creators must agree:
# two presenters of one house agreeing is one opinion, not a consensus.
# creator_houses reads the house from the names (a channel whose name
# contains another channel's whole name belongs to that channel's house);
# TRACKER_CREATOR_HOUSES ({"channel": "house"}, JSON) overrides or adds.
# Per-creator lines stay per channel.
CREATOR_HOUSES = _env_houses()

# A condition that only QUALIFIES a view ("as long as $76,220 support holds",
# "for investors with a high risk tolerance", "unless something major
# happens") is a call with an invalidation level or a suitability caveat,
# not a trigger ("I'd buy at 600", "on a pullback").
QUALIFIER_CONDITION = re.compile(
    r"risk tolerance|\bas long as\b|\bunless\b|for (?:long[- ]term |patient )?investors|"
    r"if you (?:have|are|can)|\bif they (?:can )?execute|over the long (?:run|term)|\bsolange\b", re.I)

REPORT_LABEL = "mark-to-market, not a verdict"

TRACKER_RULES = {
    "unit": "One call = one creator (channel) x asset x change of view. Inputs are canonical view claims "
            "(canonical_claims.view_claims) that are directional (direction_of bullish/bearish), a view on the "
            "security (is_security_view: a recommendation, price target, valuation view or stance, or a "
            "forecast/opinion about the price or market cap; never a revenue/earnings/margin/rate forecast, a "
            "risk or a catalyst), not trigger-conditional (is_trigger_condition; a qualifier such as 'as long as "
            "support holds' is not a trigger), that resolve to an asset key and that name something tradable: "
            "a resolved ticker, a PROXIES market, or a name not typed as a theme, sector, macro, bond, index, "
            "commodity, currency, unlisted company or 'other' (else not_an_instrument). Market keys priced "
            "through one fund are one asset (S&P 500 = US STOCK MARKET; RUSSELL 2000 = SMALL CAPS). Hosts and "
            "guests of one channel vote together; channels whose calls are mostly guests' are tagged. Wherever "
            "two creators must agree, two channels of one house (creator_houses) count once.",
    "video_direction": "Each video casts one vote per asset: canonical_claims.lean over its qualifying claims "
                       "(2/3 rule). A mixed video makes no call (video_mixed), and the vote must not change when "
                       "the same video's review-required own views (same filters, review flag ignored) are added "
                       "(else video_not_review_robust).",
    "dedup": "Episodes per (creator, asset) in published order: a same-direction video is a reiteration (entry "
             "unchanged, videos + 1); an opposite-direction video closes the call as reversed and opens a new "
             "one; a video published after the opening video's stated horizon (the earliest horizon among its "
             "claims in the call's direction) opens a new call.",
    "entry": "scorecard_pricing.entry_point on the instrument's exchange: the first close strictly after the "
             "publication instant (NYSE sessions and holidays; crypto: the close of the UTC day it was "
             "published). A call without one is too_recent when published on or after the as-of close, else "
             "no_entry_close.",
    "mark": "Open: the last close on or before the as-of. Reversed: the reversing video's entry close, when it "
            "printed on or before the as-of (else the call is still open at the as-of). Matured (the opening "
            "video's horizon ended on or before the as-of): scorecard_pricing.evaluation_point at the horizon "
            "end. When both apply, the earlier event closes the call.",
    "as_of": "Requested as-of = default_as_of(today), the last completed NYSE session. Effective as-of = the "
             "last spy.us close on or before it (within MARKET_LOOKBACK_DAYS); every symbol, crypto included, "
             "is marked at its close on or before that date, so all asset classes share one date. Prices more "
             "than a week behind the requested as-of are flagged as stale. With no S&P 500 close within "
             "MARKET_LOOKBACK_DAYS nothing is marked (every priced call is stale_mark) and the report says the "
             "prices are stale instead: a book of only the reversed calls would be a biased sample.",
    "benchmark": "The market over IDENTICAL entry and mark dates: SPY for US stocks, ETFs and index/sector/"
                 "commodity proxies; BTC for other crypto; nothing for SPY, VOO, IVV and BTC themselves (raw "
                 "return only). A missing benchmark close leaves the comparison empty and the call still counts.",
    "metrics": "ret = mark / entry - 1; dir_ret = ret for bullish, -ret for bearish; right = dir_ret > 0. "
               "excess = ret - benchmark ret; dir_excess is signed like dir_ret; beat = dir_excess > 0: the call "
               "beat the market IN ITS DIRECTION — a bullish call rose more (or fell less) than the market, a "
               "bearish call fell more (or rose less), so a bearish call can beat the market while its asset "
               "rose. The w28 fields repeat this at the evaluation close 28 days after the entry, for calls "
               "marked at least that late (the same-length check).",
    "min_age": "An open call is aggregated once its mark is MIN_CALL_AGE_DAYS after its entry "
               "(younger_than_min_age otherwise; it keeps its prices and returns), and a call reversed sooner "
               "than that is scenario talk, not a call that ran its course (reversed_within_min_age; the research "
               "report's view changes need RESEARCH_FLIP_MIN_GAP_DAYS between videos for the same reason). An open "
               "call whose mark is more than STALE_MARK_DAYS older than the as-of is stale_mark. Matured calls "
               "count whatever their length: the creator named that horizon.",
    "targets": "The latest target stated before the mark: walking the call's videos from the newest, the first "
               "video whose own next close (scorecard_pricing.entry_point) printed before the mark, and in it the "
               "first canonical_claims.price_target_of level in the call's direction that is beyond both the "
               "entry and that statement close, on the call's side, and within PLAUSIBLE_TARGET_RATIO of the "
               "entry (later levels in a video are its bull or extension cases); never for proxies. "
               "progress = (mark - entry) / (target - entry); reached = scorecard_pricing.evaluate_target from "
               "the day AFTER the statement close to the mark (intraday touch where bars exist, else closes), so "
               "neither a level the price had passed before it was stated nor an entry-day high that may have "
               "printed before the video ever counts.",
    "exclusions": "Every call is scored or carries exactly one reason: no_ticker, unresolved_instrument (an "
                  "instrument is never guessed), no_prices, too_recent, no_entry_close (also when the series "
                  "starts after the entry session: the entry is never shifted to a later close), "
                  "no_post_call_close, stale_mark, younger_than_min_age, reversed_within_min_age. Aggregates use "
                  "scored calls only; creators are ordered by their number of scored calls, never by results, "
                  "and never ranked.",
    "prices": "Read through an injected fetcher; the default, cache_fetcher, reads data/prices.json only. One "
              "request per symbol from the day before its first call through the requested as-of — or, when "
              "every call on it is closed (reversed or matured), through its last closing event plus a few "
              "days, so closed calls stop costing warm requests; needed_price_ranges returns exactly those "
              "requests for warm_prices. A series of unknown adjustment is refused (no_prices).",
}

_DIRECTIONS = ("bullish", "bearish")
_ARROW = {"bullish": "▲", "bearish": "▼"}
_BENCHMARK_LABELS = {MARKET_SYMBOL: "S&P 500", CRYPTO_MARKET_SYMBOL: "Bitcoin"}
_CLAIM_EXCLUSIONS = ("not_directional", "not_a_security_view", "trigger_condition", "no_asset",
                     "not_an_instrument", "video_mixed", "video_not_review_robust")
_REASON_WORDS = {
    "no_ticker": "no ticker",
    "unresolved_instrument": "ticker not in the instrument registry",
    "no_prices": "no cached prices",
    "too_recent": "made after the last close",
    "no_entry_close": "no entry close",
    "no_post_call_close": "no close since the call yet",
    "stale_mark": "stale prices",
    "younger_than_min_age": f"open less than {MIN_CALL_AGE_DAYS} days",
    "reversed_within_min_age": f"reversed within {MIN_CALL_AGE_DAYS} days",
}
_CALL_ROW_KEYS = (
    "creator", "asset", "label", "ticker", "symbol", "asset_class", "resolution",
    "direction", "call_date", "published_at", "video_id", "title", "videos", "claim_ids", "guest_share", "status",
    "entry_date", "entry_price", "mark_date", "mark_price", "days", "ret", "dir_ret", "right",
    "benchmark", "benchmark_label", "bench_ret", "excess", "dir_excess", "beat",
    "w28_dir_ret", "w28_dir_excess",
    "target", "target_date", "target_progress", "target_reached", "target_reached_date", "target_method",
    "scored", "reason",
)
_DEFINITIONS = ("▲ bullish ▼ bearish · right = moved the called way · beat = beat the S&P 500 in the called "
                "direction over the same days: a ▲ call rose more (or fell less), a ▼ call fell more (or rose "
                "less) (crypto: vs Bitcoin)")
# A closed call's prices are needed up to its closing close: the reversing
# video's next close or the horizon's evaluation close, at most a weekend,
# a holiday and scorecard_pricing.MAX_PRICE_LAG_DAYS after the event.
_CLOSE_SLACK_DAYS = 10
_MAX_AGREED_ROWS = 8
_MAX_SPLIT_ROWS = 6
_MAX_TARGET_ROWS = 5
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


# --- Small helpers -------------------------------------------------------------------


def _utc_today():
    return datetime.now(timezone.utc).date()


def _as_date(value):
    """A date from a date, datetime or ISO string; None otherwise."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


def _clean(values):
    """{date: positive finite float} from a fetcher's map; anything else is dropped."""
    out = {}
    if not isinstance(values, dict):
        return out
    for key, value in values.items():
        day = _as_date(key)
        if day is None or isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0:
            out[day] = number
    return out


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def _publication_key(published_at, day, video_id):
    """Sort key: the publication instant (UTC), then the video id."""
    instant, _ = sp.parse_publication(published_at)
    if instant is None:
        instant = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc) if day else _EPOCH
    return instant, video_id or ""


def _display_name(key):
    """Readable label for an un-tickered asset key ("BLOOM ENERGY" -> "Bloom Energy")."""
    if not key or any(ch.isdigit() or ch == "&" for ch in key):
        return key
    return " ".join(w if len(w) <= 2 else "-".join(p.capitalize() for p in w.split("-")) for w in key.split())


def _name_words(name):
    return tuple(re.findall(r"\w+", (name or "").casefold()))


def _contains_words(words, part):
    n = len(part)
    return any(words[i:i + n] == part for i in range(len(words) - n + 1))


def creator_houses(creators):
    """
    {creator: house} for creator (channel) names: a channel belongs to the
    house of the shortest OTHER given channel whose whole name it contains as
    words ("Phantom by HKCM" and "HKCM GLOBAL" -> "HKCM"), else it is its own
    house. CREATOR_HOUSES (TRACKER_CREATOR_HOUSES) overrides either way. Pass
    every channel name known (the record's, the claims'), since a house is
    only seen when its root channel is among them.
    """
    names = sorted({str(c) for c in (creators or ()) if c})
    words = {n: _name_words(n) for n in names}
    out = {}
    for name in names:
        if name in CREATOR_HOUSES:
            out[name] = CREATOR_HOUSES[name]
            continue
        mine = words[name]
        parents = [other for other in names
                   if other != name and words[other] and len(words[other]) < len(mine)
                   and len("".join(words[other])) >= 3 and _contains_words(mine, words[other])]
        root = min(parents, key=lambda p: (len(words[p]), p)) if parents else name
        out[name] = CREATOR_HOUSES.get(root, root)
    return out


def _house_count(creators, houses):
    """How many distinct houses `creators` come from (a creator missing from
    `houses` is its own)."""
    return len({(houses or {}).get(c, c) for c in creators})


# --- Calendar ---------------------------------------------------------------------------


def default_as_of(today):
    """
    The last NYSE trading day strictly before `today`: the newest close a run
    can rely on. The Friday 16:00 UTC job runs mid-session (so Thursday), the
    Monday job gets Friday, and the Tuesday after Labor Day gets the Friday
    before it.
    """
    today = _as_date(today) or _utc_today()
    exchange = sp.EXCHANGES["us"]
    day = today - timedelta(days=1)
    for _ in range(15):
        if sp.is_trading_day(exchange, day):
            return day
        day -= timedelta(days=1)
    return today - timedelta(days=1)


# --- Which claims make calls ---------------------------------------------------------------


def is_security_view(claim):
    """True when the claim is a view on the security itself (see SECURITY_VIEW_TYPES)."""
    kind, metric = claim.get("claim_type"), claim.get("forecast_metric")
    if kind in SECURITY_VIEW_TYPES:
        return metric not in NON_PRICE_METRICS
    if kind in ("forecast", "opinion"):
        return metric in PRICE_METRICS
    return False


def is_trigger_condition(claim):
    """True when the claim's view waits on a trigger ("if it drops to $600");
    a qualifier ("as long as support holds") is not one."""
    condition = claim.get("condition")
    if not condition or not str(condition).strip():
        return False
    return not QUALIFIER_CONDITION.search(str(condition))


# One price identity per fund: market keys priced through the same ETF are one
# asset ("the S&P 500" and "the US stock market" are both SPY), the first key
# in PROXIES naming it.
_PROXY_KEY = {symbol: key for key, (symbol, _) in reversed(list(PROXIES.items()))}


def _call_asset(claim):
    """The asset a claim votes on: canonical_claims.asset_key, with market
    keys of one fund folded to one (US STOCK MARKET -> S&P 500)."""
    key = canonical_claims.asset_key(claim)
    return _PROXY_KEY[PROXIES[key][0]] if key in PROXIES else key


# A plural grouping key ("AI STOCKS", "GROWTH STOCKS", "EACH OF THESE
# COMPANIES") names a group, not one security.
_GROUP_SUFFIXES = (" STOCKS", " AKTIEN", " COMPANIES", " NAMES")


def _is_instrument(claim, asset):
    """True when the claim names something tradable: a PROXIES market, a
    resolved ticker, or a name of an instrument type (see
    NOT_AN_INSTRUMENT_TYPES); a market, basket or metric subject, or a
    plural group ("AI stocks"), without a ticker never is."""
    if asset in PROXIES or canonical_claims.resolved_ticker(claim):
        return True
    if claim.get("subject_kind") in ("market", "basket", "metric") or asset.endswith(_GROUP_SUFFIXES):
        return False
    return claim.get("asset_type") not in NOT_AN_INSTRUMENT_TYPES


def _qualifies(claim):
    return (canonical_claims.direction_of(claim) in _DIRECTIONS and is_security_view(claim)
            and not is_trigger_condition(claim) and canonical_claims.asset_key(claim) is not None)


def _evidence_position(claim):
    position = claim.get("evidence_start_character")
    return (not isinstance(position, (int, float)), position if isinstance(position, (int, float)) else 0,
            str(claim.get("claim_id") or ""))


def _video_call(creator, asset, video_id, direction, rows):
    published = next((c.get("published_at") for c in rows if c.get("published_at")), None)
    day = next((d for d in (canonical_claims.claim_date(c) for c in rows) if d), None)
    # The horizon is the call's own: a dissenting minority claim's one-week
    # horizon must not mature a long-term call early.
    ends = [d for d in (_as_date(c.get("forecast_end_date")) for c in rows
                        if canonical_claims.direction_of(c) == direction) if d]
    guests = sum(c.get("attribution_type") == "guest_personal_view" for c in rows)
    return {
        "creator": creator, "asset": asset, "video_id": video_id, "published_at": published, "date": day,
        "title": next((c.get("video_title") for c in rows if c.get("video_title")), None),
        "direction": direction, "claim_ids": [c.get("claim_id") for c in rows],
        "horizon_end": min(ends) if ends else None, "guest_share": guests / len(rows),
    }


def _video_groups(claims, instruments_only=False):
    """([(VideoCall, its qualifying claims)] in published order, exclusion
    Counter). With `instruments_only` (the track record), a claim on a theme,
    macro or a market without a fund is excluded as not_an_instrument."""
    claims = list(claims or [])
    excluded = Counter(dict.fromkeys(_CLAIM_EXCLUSIONS, 0))
    groups = defaultdict(list)
    for c in canonical_claims.view_claims(claims):
        asset = _call_asset(c)
        if canonical_claims.direction_of(c) not in _DIRECTIONS:
            excluded["not_directional"] += 1
        elif not is_security_view(c):
            excluded["not_a_security_view"] += 1
        elif is_trigger_condition(c):
            excluded["trigger_condition"] += 1
        elif asset is None:
            excluded["no_asset"] += 1
        elif instruments_only and not _is_instrument(c, asset):
            excluded["not_an_instrument"] += 1
        else:
            groups[(canonical_claims.source_of(c), asset, c.get("video_id"))].append(c)
    # The review shadow: a video's review-required own views under the same
    # filters, review flag ignored. A call must survive them: one creator's
    # SpaceX video read "bearish" only because its two bullish targets were
    # review-flagged (the subject was named a sentence earlier).
    shadow = defaultdict(Counter)
    for c in claims:
        if not c.get("review_required") or not _qualifies(c):
            continue
        if not canonical_claims.view_claims([dict(c, review_required=False)]):
            continue
        key = (canonical_claims.source_of(c), _call_asset(c), c.get("video_id"))
        if key in groups:
            shadow[key][canonical_claims.direction_of(c)] += 1
    out = []
    for key, rows in groups.items():
        counts = Counter(canonical_claims.direction_of(c) for c in rows)
        vote = canonical_claims.lean(counts["bullish"], counts["bearish"])
        if vote not in _DIRECTIONS:
            excluded["video_mixed"] += 1
            continue
        extra = shadow.get(key)
        if extra and canonical_claims.lean(counts["bullish"] + extra["bullish"],
                                           counts["bearish"] + extra["bearish"]) != vote:
            excluded["video_not_review_robust"] += 1
            continue
        rows.sort(key=_evidence_position)
        out.append((_video_call(*key, vote, rows), rows))
    out.sort(key=lambda pair: _publication_key(pair[0]["published_at"], pair[0]["date"], pair[0]["video_id"]))
    return out, excluded


def video_calls(claims):
    """
    (video calls, exclusion Counter): one VideoCall per (creator, asset,
    video) whose qualifying claims lean one way, sorted by publication.
    A VideoCall is a dict: creator, asset, video_id, published_at, date,
    title, direction, claim_ids, horizon_end, guest_share. The Counter keys
    are not_directional, not_a_security_view, trigger_condition, no_asset,
    not_an_instrument (always 0 here: themes and markets without a fund make
    video calls; only the track record leaves them out), video_mixed and
    video_not_review_robust.
    """
    groups, excluded = _video_groups(claims)
    return [vcall for vcall, _ in groups], excluded


def _episodes(groups):
    """(calls, reiterations) from published-order video calls, per (creator, asset)."""
    sequences = defaultdict(list)
    for vcall, rows in groups:
        sequences[(vcall["creator"], vcall["asset"])].append((vcall, rows))
    calls, reiterations = [], 0
    for sequence in sequences.values():
        current = None
        for vcall, rows in sequence:
            if current is not None:
                end = current["opening"]["horizon_end"]
                if end is not None and vcall["date"] is not None and vcall["date"] > end:
                    calls.append(current)               # the stated horizon ended first
                    current = None
                elif vcall["direction"] == current["direction"]:
                    current["videos"].append(vcall)     # a reiteration: entry unchanged
                    current["parts"].append(rows)
                    current["claims"].extend(rows)
                    reiterations += 1
                    continue
                else:
                    current["closed_by"] = vcall        # a reversal closes the call
                    calls.append(current)
                    current = None
            current = {"creator": vcall["creator"], "asset": vcall["asset"], "direction": vcall["direction"],
                       "opening": vcall, "videos": [vcall], "parts": [rows], "claims": list(rows),
                       "closed_by": None}
        calls.append(current)
    return calls, reiterations


# --- Instruments -----------------------------------------------------------------------------


def instrument_for(asset, ticker=None, name=None, asset_type=None):
    """
    The Priced instrument for an asset, or None. A Priced is a dict: symbol,
    label, ticker, asset_class (stock|etf|crypto), exchange
    (scorecard_pricing.Exchange), source (registry|provider_catalogue|proxy),
    instrument (instruments.Instrument or None). A canonical market key is
    priced through its PROXIES ETF; anything else only through its RESOLVED
    `ticker` and instruments.resolve_instrument — the curated registry or
    the provider-verified learned map. Without a ticker there is no
    instrument: the asset key of an un-tickered claim is a spoken name, and
    a name that happens to spell a registry ticker ("GM" the resolver left
    ambiguous, "Low", "Team") is not that company. A symbol is never built
    from a claim's asset_type, and a cached series alone never makes a
    ticker priceable.
    """
    if asset in PROXIES:
        symbol, label = PROXIES[asset]
        return {"symbol": symbol, "label": label, "ticker": None, "asset_class": "etf",
                "exchange": sp.EXCHANGES["us"], "source": "proxy", "instrument": None}
    if not ticker:
        return None
    inst = instruments.resolve_instrument(ticker, name, asset_type)
    if inst is None:
        return None
    exchange = sp.exchange_for_instrument(inst)
    if exchange is None:
        return None
    display = getattr(inst, "name", None)
    return {"symbol": inst.symbol, "label": f"{display} ({inst.ticker})" if display else inst.ticker,
            "ticker": inst.ticker, "asset_class": inst.asset_type, "exchange": exchange,
            "source": "registry" if inst.source == "curated" else inst.source, "instrument": inst}


def _resolve_call(call):
    """Attach the call's instrument, or the reason it has none."""
    with_ticker = next((c for c in call["claims"] if canonical_claims.resolved_ticker(c)), None)
    source = with_ticker or call["claims"][0]
    ticker = canonical_claims.resolved_ticker(with_ticker) if with_ticker else None
    call["ticker"] = ticker
    call["priced"] = instrument_for(call["asset"], ticker,
                                    source.get("canonical_entity_name") or source.get("subject_mention"),
                                    source.get("asset_type"))
    call["unresolved"] = None
    if call["priced"] is None:
        call["unresolved"] = "unresolved_instrument" if ticker or call["asset"] in PROXIES else "no_ticker"


def _benchmark_symbol(priced):
    if priced is None or priced["symbol"] in SELF_BENCHMARKED:
        return None
    return CRYPTO_MARKET_SYMBOL if priced["asset_class"] == "crypto" else MARKET_SYMBOL


# --- The planner (shared by track_record and needed_price_ranges) ------------------------------


def _plan(claims, as_of, lookback_days=None):
    """
    The calls and the price requests track_record(claims, as_of) makes: one
    (symbol, start, as_of) per symbol. needed_price_ranges returns the same
    list, so the warm job and the tracker cannot drift.
    """
    lookback = LOOKBACK_DAYS if lookback_days is None else lookback_days
    floor = as_of - timedelta(days=lookback)
    used = [c for c in (claims or []) if (d := canonical_claims.claim_date(c)) is not None and d > floor]
    groups, excluded = _video_groups(used, instruments_only=True)
    calls, reiterations = _episodes(groups)
    for call in calls:
        _resolve_call(call)
    calls.sort(key=lambda call: (call["opening"]["date"] or date.min, call["creator"], call["asset"],
                                 _publication_key(call["opening"]["published_at"], call["opening"]["date"],
                                                  call["opening"]["video_id"])))
    since = min((call["opening"]["date"] for call in calls if call["opening"]["date"]), default=None)
    return {"calls": calls, "video_calls": len(groups), "claim_exclusions": excluded,
            "reiterations": reiterations, "since": since, "requests": _requests(calls, as_of, since)}


def _context_start(as_of, since):
    start = as_of - timedelta(days=CONTEXT_DAYS)
    return min(start, since - timedelta(days=1)) if since is not None else start


def _needed_until(call, as_of):
    """The last date a call's own prices are read: the as-of for a call
    still open at it, else its first closing event (the reversing video, the
    horizon's end) plus _CLOSE_SLACK_DAYS — its closing close is no later."""
    closer, end = call["closed_by"], call["opening"]["horizon_end"]
    events = [d for d in ((closer or {}).get("date"), end) if d is not None and d <= as_of]
    return min(as_of, min(events) + timedelta(days=_CLOSE_SLACK_DAYS)) if events else as_of


def _requests(calls, as_of, since):
    ranges = {}

    def want(symbol, start, end):
        if not symbol or start > as_of:
            return
        if symbol in ranges:
            known_start, known_end = ranges[symbol]
            ranges[symbol] = (min(start, known_start), max(end, known_end))
        else:
            ranges[symbol] = (start, end)

    for call in calls:
        priced, day = call["priced"], call["opening"]["date"]
        if priced is None or day is None or day > as_of:
            continue                    # nothing to price, or published after the as-of
        start, end = day - timedelta(days=1), _needed_until(call, as_of)
        want(priced["symbol"], start, end)
        want(_benchmark_symbol(priced), start, end)
    context = _context_start(as_of, since)
    for symbol, _ in CONTEXT_SYMBOLS:
        want(symbol, context, as_of)    # the market context always runs to the as-of
    return sorted((symbol, start, end) for symbol, (start, end) in ranges.items())


def needed_price_ranges(claims, today):
    """
    [(symbol, start, end)] the tracker will read for a run on `today`
    (as_of = default_as_of(today)): exactly the requests track_record makes,
    one per symbol, ending at that as-of — or earlier for a symbol whose
    calls have all closed (see TRACKER_RULES['prices']). warm_prices covers
    these so the reports never fetch prices live.
    """
    return list(_plan(claims, default_as_of(today))["requests"])


# --- Price fetchers ------------------------------------------------------------------------------


def cache_fetcher(cache=None):
    """
    A price fetcher over the committed daily-close cache (price_cache.active()
    at call time, or the given {symbol: entry} dict). It never reaches the
    network: a symbol or range the cache lacks comes back with fewer (or no)
    closes. Returns scorecard_pricing.PriceSeries sliced to [start, end],
    with the entry's adjustment and its daily highs/lows when cached. The
    fetcher carries .price_provenance and .requests ((symbol, start, end)
    per call).
    """
    def fetch(symbol, start, end):
        fetch.requests.append((symbol, start, end))
        source = price_cache.active() if cache is None else cache
        entry = (source or {}).get(symbol) or {}
        adjustment = entry.get("adjustment") or "split_adjusted"
        return sp.PriceSeries(
            symbol=symbol, closes=price_cache.slice_range(entry, start, end), provider="price_cache",
            adjustment=adjustment, corporate_action_status=adjustment,
            requested_start=start, requested_end=end,
            highs=price_cache.slice_range(entry, start, end, "highs") or None,
            lows=price_cache.slice_range(entry, start, end, "lows") or None)

    fetch.price_provenance = {"provider": "price_cache", "adjustment": "split_adjusted",
                              "corporate_action_status": "split_adjusted"}
    fetch.requests = []
    return fetch


def static_fetcher(series, highs=None, lows=None, adjustment="split_adjusted"):
    """
    A price fetcher over in-memory {symbol: {date: close}} data (and optional
    {symbol: {date: high/low}}), for tests and offline use. Same contract as
    cache_fetcher, .requests included.
    """
    def cut(table, symbol, start, end):
        out = {}
        for key, value in ((table or {}).get(symbol) or {}).items():
            day = _as_date(key)
            if day is not None and start <= day <= end:
                out[day] = value
        return out

    def fetch(symbol, start, end):
        fetch.requests.append((symbol, start, end))
        return sp.PriceSeries(
            symbol=symbol, closes=cut(series, symbol, start, end), provider="static", adjustment=adjustment,
            corporate_action_status=adjustment, requested_start=start, requested_end=end,
            highs=cut(highs, symbol, start, end) or None, lows=cut(lows, symbol, start, end) or None)

    fetch.price_provenance = {"provider": "static", "adjustment": adjustment, "corporate_action_status": adjustment}
    fetch.requests = []
    return fetch


def _load(fetcher, symbol, start, end, problems):
    """
    One request: {closes, highs, lows, adjustment} or None. Any fetcher
    answer goes through scorecard_pricing.series_from_fetcher, so a plain
    {date: close} map is accepted, but one of unknown adjustment (no
    .price_provenance) is refused rather than assumed. A failing fetcher is
    "no prices", never an exception; problems are logged once per run.
    """
    try:
        series = sp.series_from_fetcher(fetcher, symbol, start, end)
    except Exception as e:  # noqa: BLE001 - a fetcher failure must never sink a report
        problems.setdefault("failed", []).append(f"{symbol}: {describe_error(e)}")
        return None
    closes = _clean(series.closes)
    if not closes:
        return None
    if not series.scorable():
        problems.setdefault("unknown_adjustment", []).append(symbol)
        return None
    return {"closes": closes, "highs": _clean(series.highs), "lows": _clean(series.lows),
            "adjustment": series.adjustment}


def _log_problems(problems):
    refused = problems.get("unknown_adjustment")
    if refused:
        log_warn(f"Creator tracker: refused {len(refused)} price series of unknown adjustment "
                 f"({', '.join(refused[:5])}{', ...' if len(refused) > 5 else ''}); counted as no prices. "
                 "A fetcher must return scorecard_pricing.PriceSeries or carry .price_provenance.")
    failed = problems.get("failed")
    if failed:
        log_warn(f"Creator tracker: the price fetcher failed for {len(failed)} symbol(s) "
                 f"({'; '.join(failed[:3])}{'; ...' if len(failed) > 3 else ''}); counted as no prices.")


def _capped(data, as_of):
    """The closes of a loaded series on or before `as_of`."""
    if not data:
        return {}
    return {d: v for d, v in data["closes"].items() if d <= as_of}


def _effective_as_of(spy, requested):
    """(the last spy.us close on or before `requested` within
    MARKET_LOOKBACK_DAYS, True), else (requested, False)."""
    floor = requested - timedelta(days=MARKET_LOOKBACK_DAYS)
    days = [d for d in (spy or {}).get("closes", {}) if floor <= d <= requested]
    return (max(days), True) if days else (requested, False)


def _move(symbol, closes, start, end):
    days = [d for d in closes if start <= d <= end]
    if not days:
        return None
    first, last = min(days), max(days)
    if first >= last:
        return None
    return {"symbol": symbol, "start_date": first, "start_price": closes[first], "end_date": last,
            "end_price": closes[last], "ret": closes[last] / closes[first] - 1}


# --- Pricing one call ------------------------------------------------------------------------------


def _structural_status(call, as_of):
    """open / reversed / matured as of `as_of`, for a call that has no entry close."""
    events = []
    closer, end = call["closed_by"], call["opening"]["horizon_end"]
    if closer is not None and closer["date"] is not None and closer["date"] <= as_of:
        events.append((closer["date"], "reversed"))
    if end is not None and end <= as_of:
        events.append((end, "matured"))
    return min(events)[1] if events else "open"


def _mark(call, exchange, closes, as_of):
    """(status, mark date, mark price); the date is None for a matured call
    whose series has no close at the horizon end."""
    candidates, matured_without_close = [], False
    closer = call["closed_by"]
    if closer is not None:
        point = sp.entry_point(exchange, closer["published_at"], closes)
        if point["price"] is not None:
            candidates.append((_as_date(point["resolved_trading_date"]), point["price"], "reversed"))
    end = call["opening"]["horizon_end"]
    if end is not None and end <= as_of:
        point = sp.evaluation_point(exchange, end, closes)
        if point["price"] is not None:
            candidates.append((_as_date(point["resolved_trading_date"]), point["price"], "matured"))
        else:
            matured_without_close = True
    if candidates:
        day, price, status = min(candidates, key=lambda c: c[0])   # the earlier event closes the call
        return status, day, price
    if matured_without_close:
        return "matured", None, None
    last = max(closes)
    return "open", last, closes[last]


def _target_for(call, exchange, closes, entry_price, mark_date):
    """
    (level, statement close date) of the call's latest USD price target
    stated before the mark, or (None, None). Walking the call's videos from
    the newest, a video counts only when its own next close (the entry rule
    applied to its publication) printed before the mark; its first level in
    the call's direction wins when it lies beyond BOTH the entry and that
    statement close on the call's side — a level the price had already
    passed when it was said is not a target — and within
    PLAUSIBLE_TARGET_RATIO of the entry. Within a video the first such level
    wins: speakers state their target, then the bull case or the "extension
    target" (AVGO "$638 based on my assumptions ... the bull case is closer
    to $1,000"; BTC "on its way to the 88,600 to 92,200 area ... an
    extension target at 98,000").
    """
    low, high = PLAUSIBLE_TARGET_RATIO
    direction = call["direction"]
    for vcall, rows in zip(reversed(call["videos"]), reversed(call["parts"])):
        stated = sp.entry_point(exchange, vcall["published_at"], closes)
        day = _as_date(stated["resolved_trading_date"])
        if day is None or day >= mark_date:
            continue                     # said after the mark, or no close to place it against
        reference = closes[day]
        for claim in rows:
            if canonical_claims.direction_of(claim) != direction:
                continue
            level = canonical_claims.price_target_of(claim)
            if level is None:
                continue
            if direction == "bullish":
                on_side = level > max(entry_price, reference)
            else:
                on_side = level < min(entry_price, reference)
            if on_side and low <= level / entry_price <= high:
                return level, day
    return None, None


def _track_target(row, call, data, exchange, closes, entry_price, mark_date, mark_price):
    target, stated = _target_for(call, exchange, closes, entry_price, mark_date)
    if target is None:
        return
    first = stated + timedelta(days=1)     # never the statement session's own high or low

    def window(values):
        return {d: v for d, v in (values or {}).items() if first <= d <= mark_date}

    series = sp.PriceSeries(symbol=row["symbol"], closes=window(data["closes"]), adjustment=data["adjustment"],
                            highs=window(data["highs"]) or None, lows=window(data["lows"]) or None)
    result = sp.evaluate_target(series, target, call["direction"], first, mark_date)
    row.update(target=target, target_date=stated,
               target_progress=(mark_price - entry_price) / (target - entry_price),
               target_reached=result["target_reached"],
               target_reached_date=_as_date(result["target_first_reached_date"]),
               target_method=result["target_test_method"])


class _EverySession(dict):
    """A closes map with a close on every day: entry_point over it names the
    session the entry rule wants, whatever a series holds."""

    def __contains__(self, day):
        return True

    def __getitem__(self, day):
        return 1.0


def _wanted_session(exchange, published_at):
    """The session whose close the entry rule picks for a publication, or None."""
    return _as_date(sp.entry_point(exchange, published_at, _EverySession())["resolved_trading_date"])


def _price(row, call, priced, series, closes, as_of):
    exchange, opening = priced["exchange"], call["opening"]
    entry = sp.entry_point(exchange, opening["published_at"], closes)
    if entry["price"] is None:
        row["reason"] = "too_recent" if opening["date"] >= as_of else "no_entry_close"
        return
    entry_date, entry_price = _as_date(entry["resolved_trading_date"]), entry["price"]
    wanted = _wanted_session(exchange, opening["published_at"])
    if wanted is not None and entry_date > wanted and min(closes) > wanted:
        # The series starts after the session the rule wants: entering at
        # its first close would be a guess (a cache gap the warm fills).
        row["reason"] = "no_entry_close"
        return
    row.update(entry_date=entry_date, entry_price=entry_price)
    status, mark_date, mark_price = _mark(call, exchange, closes, as_of)
    row["status"] = status
    if mark_date is None:
        row["reason"] = "stale_mark"          # matured, but the series stops before the horizon close
        return
    if status == "open" and (as_of - mark_date).days > STALE_MARK_DAYS:
        row.update(mark_date=mark_date, mark_price=mark_price, reason="stale_mark")
        return
    if mark_date <= entry_date:
        row["reason"] = "no_post_call_close"
        return
    sign = 1 if call["direction"] == "bullish" else -1
    ret = mark_price / entry_price - 1
    row.update(mark_date=mark_date, mark_price=mark_price, days=(mark_date - entry_date).days,
               ret=ret, dir_ret=sign * ret, right=sign * ret > 0)
    bench = _benchmark_symbol(priced)
    bench_closes = _capped(series.get(bench), as_of) if bench else {}
    if bench:
        row.update(benchmark=bench, benchmark_label=_BENCHMARK_LABELS.get(bench))
        start, end = bench_closes.get(entry_date), bench_closes.get(mark_date)
        if start and end:
            excess = ret - (end / start - 1)
            row.update(bench_ret=end / start - 1, excess=excess, dir_excess=sign * excess, beat=sign * excess > 0)
    window_day = entry_date + timedelta(days=WINDOW_DAYS)
    if window_day <= mark_date:
        point = sp.evaluation_point(exchange, window_day, closes)
        day = _as_date(point["resolved_trading_date"]) if point["price"] is not None else None
        if day is not None and day <= mark_date:
            window_ret = point["price"] / entry_price - 1
            row["w28_dir_ret"] = sign * window_ret
            start, end = bench_closes.get(entry_date), bench_closes.get(day)
            if start and end:
                row["w28_dir_excess"] = sign * (window_ret - (end / start - 1))
    if priced["source"] != "proxy":
        _track_target(row, call, series[priced["symbol"]], exchange, closes, entry_price, mark_date, mark_price)
    if status in ("open", "reversed") and row["days"] < MIN_CALL_AGE_DAYS:
        # Keeps its prices and returns, not aggregated: an open call is too
        # young to read, and one reversed this soon was scenario talk.
        row["reason"] = "younger_than_min_age" if status == "open" else "reversed_within_min_age"
    else:
        row["scored"] = True


def _call_row(call, series, as_of, requested):
    opening, priced, claims = call["opening"], call["priced"], call["claims"]
    row = dict.fromkeys(_CALL_ROW_KEYS)
    row.update(
        creator=call["creator"], asset=call["asset"],
        label=priced["label"] if priced else (call["ticker"] or _display_name(call["asset"])),
        ticker=priced["ticker"] if priced else call["ticker"],
        symbol=priced["symbol"] if priced else None,
        asset_class=priced["asset_class"] if priced else None,
        resolution=priced["source"] if priced else None,
        direction=call["direction"], call_date=opening["date"], published_at=opening["published_at"],
        video_id=opening["video_id"], title=opening["title"], videos=len(call["videos"]),
        claim_ids=[c.get("claim_id") for c in claims],
        guest_share=sum(c.get("attribution_type") == "guest_personal_view" for c in claims) / len(claims),
        status=_structural_status(call, as_of), scored=False)
    if priced is None:
        row["reason"] = call["unresolved"]
    elif opening["date"] > requested:
        row["reason"] = "too_recent"             # published after the as-of: never requested
    else:
        closes = _capped(series.get(priced["symbol"]), as_of)
        if not closes:
            row["reason"] = "no_prices"
        else:
            _price(row, call, priced, series, closes, as_of)
    return row


# --- Aggregates -----------------------------------------------------------------------------------


def _creator_summaries(rows):
    """{creator: CreatorSummary} over scored calls, most scored calls first, then by name."""
    every, scored = defaultdict(list), defaultdict(list)
    for r in rows:
        every[r["creator"]].append(r)
        if r["scored"]:
            scored[r["creator"]].append(r)
    out = {}
    for creator in sorted(scored, key=lambda c: (-len(scored[c]), c)):
        calls = scored[creator]
        bench = [r for r in calls if r["dir_excess"] is not None]
        w28 = [r for r in calls if r["w28_dir_ret"] is not None]
        w28_bench = [r for r in calls if r["w28_dir_excess"] is not None]
        out[creator] = {
            "calls": len(calls),
            "bull": sum(r["direction"] == "bullish" for r in calls),
            "bear": sum(r["direction"] == "bearish" for r in calls),
            "right": sum(bool(r["right"]) for r in calls),
            "bench_n": len(bench), "beat": sum(bool(r["beat"]) for r in bench),
            "median_dir_ret": _median(r["dir_ret"] for r in calls),
            "median_dir_excess": _median(r["dir_excess"] for r in bench),
            "mean_dir_excess": _mean(r["dir_excess"] for r in bench),
            "median_days": _median(r["days"] for r in calls),
            "w28_n": len(w28), "w28_right": sum(r["w28_dir_ret"] > 0 for r in w28),
            "w28_bench_n": len(w28_bench), "w28_beat": sum(r["w28_dir_excess"] > 0 for r in w28_bench),
            "best": max(calls, key=lambda r: r["dir_ret"]),
            "worst": min(calls, key=lambda r: r["dir_ret"]),
            "small_sample": len(calls) < MIN_CREATOR_SAMPLE,
            "mostly_guests": (_mean(r["guest_share"] for r in every[creator]) or 0.0) > 0.5,
            "unscored": len(every[creator]) - len(calls),
        }
    return out


def _pooled(scored, market_move):
    bench = [r for r in scored if r["dir_excess"] is not None]
    w28 = [r for r in scored if r["w28_dir_ret"] is not None]
    w28_bench = [r for r in scored if r["w28_dir_excess"] is not None]
    return {
        "calls": len(scored), "right": sum(bool(r["right"]) for r in scored),
        "bench_n": len(bench), "beat": sum(bool(r["beat"]) for r in bench),
        "w28_n": len(w28), "w28_right": sum(r["w28_dir_ret"] > 0 for r in w28),
        "w28_bench_n": len(w28_bench), "w28_beat": sum(r["w28_dir_excess"] > 0 for r in w28_bench),
        "market_return": market_move["ret"] if market_move else None,
    }


def _asset_summaries(rows, houses=None):
    """{asset: AssetSummary} for every asset with a call; assets with the
    most creators holding a scored current call first. `houses` is the
    number of distinct creator houses (creator_houses) among the creators
    with a scored current call on the consensus side (both sides when
    split): two channels of one house count once."""
    by_asset = defaultdict(list)
    for r in rows:
        by_asset[r["asset"]].append(r)
    out = []
    for asset, calls in by_asset.items():
        ordered = sorted(calls, key=lambda r: _publication_key(r["published_at"], r["call_date"], r["video_id"]))
        latest = {}
        for r in ordered:
            latest[r["creator"]] = r
        current = dict(sorted(latest.items()))
        bulls = sorted(c for c, r in current.items() if r["scored"] and r["direction"] == "bullish")
        bears = sorted(c for c, r in current.items() if r["scored"] and r["direction"] == "bearish")
        consensus = "split" if bulls and bears else "bullish" if bulls else "bearish" if bears else None
        lead = None
        if consensus in _DIRECTIONS:
            lead = min((current[c] for c in (bulls or bears)),
                       key=lambda r: (r["entry_date"], r["call_date"], r["creator"]))
        named = next((r for r in reversed(ordered) if r["symbol"]), ordered[-1])
        out.append({
            "asset": asset, "label": named["label"], "symbol": named["symbol"], "current": current,
            "bulls": bulls, "bears": bears, "consensus": consensus,
            "houses": _house_count(bulls + bears, houses),
            "since": lead["entry_date"] if lead else None, "move": lead["ret"] if lead else None,
            "benchmark_label": lead["benchmark_label"] if lead else None,
            "bench_move": lead["bench_ret"] if lead else None,
            "dir_excess": lead["dir_excess"] if lead else None,
            "right": lead["right"] if lead else None, "beat": lead["beat"] if lead else None,
        })
    out.sort(key=lambda s: (-(len(s["bulls"]) + len(s["bears"])), s["asset"]))
    return {s["asset"]: s for s in out}


def _change(closes, start, end):
    if start is None or end is None or start >= end or start not in closes or end not in closes:
        return None
    return closes[end] / closes[start] - 1


def _market_rows(series, as_of, since):
    """MarketRows for CONTEXT_SYMBOLS from loaded series, every one capped at
    `as_of` — and at the newest S&P 500 close when that is older, so a
    crypto row never runs past the stocks' last date (or onto a weekend's
    in-progress candle)."""
    spy = _capped(series.get(MARKET_SYMBOL), as_of)
    if spy:
        as_of = min(as_of, max(spy))
    rows = []
    for symbol, label in CONTEXT_SYMBOLS:
        closes = _capped(series.get(symbol), as_of)
        if not closes:
            continue
        last = max(closes)

        def on_or_before(day):
            return max((d for d in closes if d <= day), default=None)

        week_from = on_or_before(last - timedelta(days=7))
        four_from = on_or_before(last - timedelta(days=28))
        since_date = min((d for d in closes if d >= since), default=None) if since is not None else None
        if since_date is not None and since_date >= last:
            since_date = None
        rows.append({"symbol": symbol, "label": label, "as_of": last, "close": closes[last],
                     "week": _change(closes, week_from, last), "week_from": week_from,
                     "four_weeks": _change(closes, four_from, last),
                     "since": _change(closes, since_date, last), "since_date": since_date})
    return rows


# --- Public entry points --------------------------------------------------------------------------


def track_record(claims, as_of, price_fetcher=None, lookback_days=None):
    """
    The full mark-to-market record for `claims` at `as_of` (normally
    default_as_of(today)): a dict with requested_as_of, as_of (effective),
    market_data, since, calls, excluded, claim_exclusions, stats, creators,
    pooled, assets and market (see TRACKER_RULES and the module docstring).
    price_fetcher=None reads the committed cache only (cache_fetcher). Each
    planned symbol is requested exactly once; nothing else is fetched.
    """
    requested = _as_date(as_of) or default_as_of(_utc_today())
    fetcher = price_fetcher or cache_fetcher()
    plan = _plan(claims, requested, lookback_days)
    problems = {}
    series = {symbol: _load(fetcher, symbol, start, end, problems) for symbol, start, end in plan["requests"]}
    _log_problems(problems)
    effective, market_data = _effective_as_of(series.get(MARKET_SYMBOL), requested)
    rows = [_call_row(call, series, effective, requested) for call in plan["calls"]]
    if not market_data:
        # No S&P 500 close near the as-of: open calls are already stale, and
        # the reversed calls left would be a biased book (reversals follow
        # adverse moves). Nothing is marked until the prices are warmed.
        for r in rows:
            if r["scored"]:
                r.update(scored=False, reason="stale_mark")
    scored = [r for r in rows if r["scored"]]
    reasons = Counter(r["reason"] for r in rows if not r["scored"])
    since = plan["since"]
    market_move = (_move(MARKET_SYMBOL, _capped(series.get(MARKET_SYMBOL), effective), since, effective)
                   if since is not None else None)
    return {
        "requested_as_of": requested, "as_of": effective, "market_data": market_data, "since": since,
        "calls": rows,
        "excluded": dict(sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))),
        "claim_exclusions": plan["claim_exclusions"],
        "stats": {"video_calls": plan["video_calls"], "calls": len(rows), "reiterations": plan["reiterations"],
                  "reversals": sum(r["status"] == "reversed" for r in rows),
                  "matured": sum(r["status"] == "matured" for r in rows), "scored": len(scored)},
        "creators": _creator_summaries(rows),
        "pooled": _pooled(scored, market_move),
        "assets": _asset_summaries(rows, creator_houses({r["creator"] for r in rows})),
        "market": _market_rows(series, effective, since),
    }


def market_context(as_of, price_fetcher=None, since=None):
    """
    [MarketRow] for CONTEXT_SYMBOLS that have data: symbol, label, as_of
    (the last close used), close, week (from the last close on or before 7
    days earlier), week_from, four_weeks (28 days), since (from the first
    close on or after `since`) and since_date. Every symbol is capped at the
    effective as-of (the last spy.us close on or before `as_of`), so crypto
    and stocks share one date.
    """
    requested = _as_date(as_of) or default_as_of(_utc_today())
    since = _as_date(since)
    fetcher = price_fetcher or cache_fetcher()
    start, problems = _context_start(requested, since), {}
    series = {symbol: _load(fetcher, symbol, start, requested, problems) for symbol, _ in CONTEXT_SYMBOLS}
    _log_problems(problems)
    effective, _ = _effective_as_of(series.get(MARKET_SYMBOL), requested)
    return _market_rows(series, effective, since)


def price_move(symbol, start, end, price_fetcher=None):
    """The move from the first close on or after `start` to the last close
    on or before `end`: {symbol, start_date, start_price, end_date,
    end_price, ret}, or None without two such closes."""
    start, end = _as_date(start), _as_date(end)
    if not symbol or start is None or end is None or start > end:
        return None
    problems = {}
    data = _load(price_fetcher or cache_fetcher(), symbol, start, end, problems)
    _log_problems(problems)
    return _move(symbol, data["closes"], start, end) if data else None


def asset_outcome(record, asset, direction=None, creators=None):
    """
    How the current calls on one asset have done: asset, label, symbol,
    direction, creators (with a priced call, younger ones included),
    pending (no entry close yet), the earliest priced call's since,
    entry_price, move, benchmark_label, bench_move, dir_excess, right and
    beat, and targets [{creator, target, progress, reached}]. Filtered to
    `creators` and `direction`; without a direction the calls must all
    share one. None when the asset is unknown or nothing is left.
    """
    summary = ((record or {}).get("assets") or {}).get(asset)
    if not summary:
        return None
    if isinstance(creators, str):
        creators = [creators]
    wanted = None if creators is None else set(creators)
    calls = [r for c, r in summary["current"].items() if wanted is None or c in wanted]
    if direction is not None:
        calls = [r for r in calls if r["direction"] == direction]
    if not calls:
        return None
    directions = {r["direction"] for r in calls}
    if direction is None:
        if len(directions) != 1:
            return None
        direction = directions.pop()
    # A stale_mark call keeps no move to report (with no current S&P 500
    # close, track_record unmarks every call).
    priced = [r for r in calls if r["ret"] is not None and r["reason"] != "stale_mark"]
    first = min(priced, key=lambda r: (r["entry_date"], r["call_date"], r["creator"])) if priced else None

    def lead(key):
        return first[key] if first else None

    return {
        "asset": asset, "label": summary["label"], "symbol": summary["symbol"], "direction": direction,
        "creators": sorted(r["creator"] for r in priced),
        "pending": sorted(r["creator"] for r in calls
                          if r["entry_price"] is None and r["reason"] in ("too_recent", "no_entry_close")),
        "since": lead("entry_date"), "entry_price": lead("entry_price"), "move": lead("ret"),
        "benchmark_label": lead("benchmark_label"), "bench_move": lead("bench_ret"),
        "dir_excess": lead("dir_excess"), "right": lead("right"), "beat": lead("beat"),
        "targets": [{"creator": r["creator"], "target": r["target"], "progress": r["target_progress"],
                     "reached": r["target_reached"]} for r in priced if r["target"] is not None],
    }


def unpriced_assets(record, limit=12):
    """[(asset, calls)] the calls with no instrument (no_ticker,
    unresolved_instrument), most calls first: the maintainer's ticker
    worklist for the workflow log, never the report."""
    counts = Counter(r["asset"] for r in (record or {}).get("calls") or []
                     if r["reason"] in ("no_ticker", "unresolved_instrument"))
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max(0, limit)]


# --- Formatting -------------------------------------------------------------------------------------


def _mon_d(day):
    return f"{day:%b} {day.day}" if day else "?"


def _dow_mon_d(day):
    return f"{day:%a %b} {day.day}" if day else "?"


def _number(value):
    if value is None:
        return "?"
    if abs(value) >= 1000 or (abs(value) >= 1 and value == int(value)):
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.4g}"


def _plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def _header(record):
    since = record.get("since")
    calls = f"calls since {_mon_d(since)}, " if since else ""
    return (f"📈 Creators vs the market — {calls}marked to {_dow_mon_d(record.get('as_of'))} close "
            f"({REPORT_LABEL})")


def _stale_warning(record):
    requested, as_of = record.get("requested_as_of"), record.get("as_of")
    if not record.get("market_data"):
        return (f"⚠️ The prices are stale: no S&P 500 close in the {MARKET_LOOKBACK_DAYS} days to "
                f"{_dow_mon_d(requested)}; every mark is the newest close its series has.")
    if requested and as_of and as_of < requested - timedelta(days=7):
        return (f"⚠️ The prices are stale: the newest S&P 500 close is {_dow_mon_d(as_of)}, more than a week "
                f"before {_dow_mon_d(requested)}.")
    return None


def _newest_market_close(record):
    row = next((r for r in record.get("market") or [] if r.get("symbol") == MARKET_SYMBOL), None)
    return row["as_of"] if row else None


def _stale_notice(record):
    """The one line a report carries when its calls could not be marked
    because the prices are stale (no S&P 500 close within
    MARKET_LOOKBACK_DAYS), or None."""
    if not record or record.get("market_data") or not record.get("calls"):
        return None
    newest = _newest_market_close(record)
    since = f"the newest S&P 500 close in the data is {_dow_mon_d(newest)}" if newest else \
        "there is no recent S&P 500 close in the data"
    return (f"📈 Creators vs the market — not marked this week: the prices are stale ({since}, more than "
            f"{MARKET_LOOKBACK_DAYS} days before {_dow_mon_d(record.get('requested_as_of'))}). "
            f"{_plural(record['stats']['calls'], 'call')} wait for fresh closes.")


def _creator_line(creator, s):
    tags = (" (mostly guests)" if s["mostly_guests"] else "") + (" (small sample)" if s["small_sample"] else "")
    line = (f"• {creator}{tags} — {_plural(s['calls'], 'call')} ({s['bull']}▲ {s['bear']}▼) · "
            f"right {s['right']}/{s['calls']}")
    if s["bench_n"]:
        line += f" · beat {s['beat']}/{s['bench_n']}"
        if s["median_dir_excess"] is not None:
            line += f" · median vs market {100 * s['median_dir_excess']:+.1f}%"
    return line


def _call_brief(r):
    return f"{r['label']} {_ARROW[r['direction']]} {_mon_d(r['call_date'])} {r['ret']:+.0%}"


def _creator_block(record):
    shown, few = [], []
    for creator, s in (record.get("creators") or {}).items():
        (shown if s["calls"] >= MIN_CREATOR_CALLS_SHOWN else few).append((creator, s))
    lines = []
    for creator, s in shown:
        lines.append(_creator_line(creator, s))
        if s["calls"] >= MIN_CREATOR_SAMPLE:
            lines.append(f"   best {_call_brief(s['best'])} · worst {_call_brief(s['worst'])}")
    if few:
        lines.append("Too few calls to judge yet: " + " · ".join(f"{c} {s['calls']}" for c, s in few))
    return lines


def _agreement_block(record):
    assets = list((record.get("assets") or {}).values())
    # Two channels of one house (HKCM and Phantom by HKCM) are one opinion.
    agreed = [s for s in assets if s["consensus"] in _DIRECTIONS
              and s.get("houses", len(s["bulls"] or s["bears"])) >= 2]
    agreed.sort(key=lambda s: (-len(s["bulls"] or s["bears"]), s["label"]))
    split = [s for s in assets if s["consensus"] == "split"]
    split.sort(key=lambda s: (-(len(s["bulls"]) + len(s["bears"])), s["label"]))
    lines = []
    if agreed:
        lines.append("🤝 Where creators agreed (2+ creators from different houses on the same side; move since the "
                     "first of their calls vs the market over the same days):")
        for s in agreed[:_MAX_AGREED_ROWS]:
            row = (f"• {s['label']} {_ARROW[s['consensus']]} {len(s['bulls'] or s['bears'])} creators — "
                   f"{s['move']:+.1%} since {_mon_d(s['since'])}")
            if s["bench_move"] is not None:
                row += f" vs {s['benchmark_label']} {s['bench_move']:+.1%}"
            lines.append(row + (" ✓" if s["right"] else " ✗"))
    if split:
        lines += ([""] if lines else []) + ["⚔️ Where they disagreed:"]
        for s in split[:_MAX_SPLIT_ROWS]:
            sides = []
            for creator in s["bulls"] + s["bears"]:
                r = s["current"][creator]
                sides.append(f"{_ARROW[r['direction']]} {creator} {_mon_d(r['call_date'])} {r['ret']:+.0%} "
                             f"{'✓' if r['right'] else '✗'}")
            lines.append(f"• {s['label']}: " + " · ".join(sides))
    return lines


def _target_block(record):
    tracked = [r for r in record.get("calls") or []
               if r["scored"] and r["status"] == "open" and r["target"] is not None]
    if not tracked:
        return []
    reached = sum(bool(r["target_reached"]) for r in tracked)
    lines = [f"🎯 Price targets on open calls ({len(tracked)} tracked, {reached} reached):"]
    for r in sorted(tracked, key=lambda r: (-r["target_progress"], r["call_date"], r["creator"]))[:_MAX_TARGET_ROWS]:
        # A target restated in a later video is dated there: the reach test
        # starts after that statement, never before it.
        stated = r.get("target_date")
        when = f"call {_mon_d(r['call_date'])}, target stated {_mon_d(stated)}" \
            if stated and r.get("entry_date") and stated > r["entry_date"] else _mon_d(r["call_date"])
        row = (f"• {r['label']} {_ARROW[r['direction']]} {_number(r['target'])} by {r['creator']} "
               f"({when}): {_number(r['entry_price'])} → {_number(r['mark_price'])}, "
               f"{r['target_progress']:.0%} of the way")
        if r["target_reached"]:
            row += f", reached {_mon_d(r['target_reached_date'])}"
        lines.append(row)
    return lines


def _coverage_line(record):
    stats = record.get("stats") or {}
    line = (f"Coverage: {stats.get('calls', 0)} calls on tradable assets (one per creator, asset and change of "
            f"view) → {stats.get('scored', 0)} marked to market.")
    parts = [f"{_REASON_WORDS.get(k, k.replace('_', ' '))} {v}" for k, v in (record.get("excluded") or {}).items()]
    line += (" Not marked: " + ", ".join(parts) + "." if parts else "")
    themes = (record.get("claim_exclusions") or {}).get("not_an_instrument") or 0
    if themes:
        line += (f" Views on themes, macro and markets without a fund ({_plural(themes, 'opinion')}) are not "
                 "calls.")
    return line


def _split_line(line):
    return [line[i:i + MAX_MESSAGE_CHARS] for i in range(0, len(line), MAX_MESSAGE_CHARS)] or [""]


def _pack(lines):
    """Messages of at most MAX_MESSAGE_CHARS, split only at line boundaries
    (a single line longer than that is cut, which no report line is)."""
    messages, current, size = [], [], 0
    for line in lines:
        for piece in _split_line(line):
            extra = len(piece) + (1 if current else 0)
            if current and size + extra > MAX_MESSAGE_CHARS:
                messages.append("\n".join(current))
                current, size, extra = [], 0, len(piece)
            current.append(piece)
            size += extra
    if current:
        messages.append("\n".join(current))
    return [m.strip("\n") for m in messages if m.strip()]


def format_market_lines(market, since=None):
    """One '• {label}: week +x% · 4 weeks +y% · since {Mon D} +z%' line per
    MarketRow (missing parts omitted; a row older than the newest one says
    which close it ends at)."""
    rows = [r for r in (market or []) if r and r.get("as_of")]
    if not rows:
        return []
    newest = max(r["as_of"] for r in rows)
    since = _as_date(since)
    lines = []
    for r in rows:
        parts = []
        if r.get("week") is not None:
            parts.append(f"week {r['week']:+.1%}")
        if r.get("four_weeks") is not None:
            parts.append(f"4 weeks {r['four_weeks']:+.1%}")
        if r.get("since") is not None:
            # The period's start date reads best ("since Sep 5" for a Saturday
            # start measured from Tuesday's close), unless the series itself
            # only starts later: then its own first close is named.
            first = r.get("since_date")
            day = since if since and first and (first - since).days <= 4 else first or since
            parts.append(f"since {_mon_d(day)} {r['since']:+.1%}")
        if not parts:
            continue
        symbol = r.get("symbol") or ""
        label = f"{r['label']} ({symbol[:-3].upper()})" if symbol.endswith(".us") else r["label"]
        if r["as_of"] < newest:
            label += f" (to {_mon_d(r['as_of'])})"
        lines.append(f"• {label}: " + " · ".join(parts))
    return lines


def format_report(record):
    """
    The Telegram report (plain text, '• ' bullets), [] when no call is
    scored. Message 1: header with the effective as-of and REPORT_LABEL, a
    stale-price warning when due, the market, the definitions, one line per
    creator (ordered by number of calls; best/worst for samples of
    MIN_CREATOR_SAMPLE or more), the creators with too few calls, the
    same-length check and the base-rate caveat. Message 2: where creators
    agreed and disagreed, open price targets, coverage and the disclaimer.
    Every message is at most MAX_MESSAGE_CHARS, split at line boundaries.
    When the prices are stale (no S&P 500 close within
    MARKET_LOOKBACK_DAYS) the report is that one line instead, so readers
    learn why the section is missing.
    """
    notice = _stale_notice(record)
    if notice:
        return [notice]
    if not record or not (record.get("stats") or {}).get("scored"):
        return []
    since, pooled = record.get("since"), record["pooled"]
    first = [_header(record)]
    warning = _stale_warning(record)
    if warning:
        first.append(warning)
    market = format_market_lines(record.get("market"), since)
    if market:
        first += ["", "Market:"] + market
    first += ["", _DEFINITIONS] + _creator_block(record)
    tail = []
    if pooled["w28_n"]:
        check = (f"Same-length check — the {pooled['w28_n']} calls at least 4 weeks old, 4 weeks after the call: "
                 f"right {pooled['w28_right']}/{pooled['w28_n']}")
        if pooled["w28_bench_n"]:
            check += f", beat {pooled['w28_beat']}/{pooled['w28_bench_n']}"
        tail.append(check + ".")
    if pooled["market_return"] is not None and pooled["market_return"] > 0:
        tail.append(f"S&P 500 {pooled['market_return']:+.1%} since {_mon_d(since)}: in a rising market most "
                    "bullish calls look right — read beat.")
    if tail:
        first += [""] + tail
    second = _agreement_block(record)
    targets = _target_block(record)
    if targets:
        second += ([""] if second else []) + targets
    second += ([""] if second else []) + [_coverage_line(record), "Research input, not investment advice."]
    return _pack(first) + _pack(second)


def format_summary_lines(record, max_creators=6):
    """The compact block for the research report: the header, a pooled line,
    the definitions, up to `max_creators` creator lines, a line naming every
    creator left out (with their calls), and the base-rate caveat in a
    rising market; [] when no call is scored (the stale notice when the
    prices are stale)."""
    notice = _stale_notice(record)
    if notice:
        return [notice]
    if not record or not (record.get("stats") or {}).get("scored"):
        return []
    pooled = record["pooled"]
    lines = [_header(record)]
    warning = _stale_warning(record)
    if warning:
        lines.append(warning)
    line = f"{_plural(pooled['calls'], 'call')} · right {pooled['right']}/{pooled['calls']}"
    if pooled["bench_n"]:
        line += f" · beat {pooled['beat']}/{pooled['bench_n']}"
    if pooled["market_return"] is not None:
        line += f" · S&P 500 {pooled['market_return']:+.1%} over the period"
    lines += [line, _DEFINITIONS]
    creators = list((record.get("creators") or {}).items())
    shown = [(c, s) for c, s in creators if s["calls"] >= MIN_CREATOR_CALLS_SHOWN][:max(0, max_creators)]
    lines += [_creator_line(c, s) for c, s in shown]
    named = {c for c, _ in shown}
    left = [(c, s) for c, s in creators if c not in named]
    if left:
        lines.append(f"+{_plural(len(left), 'more creator')} (" + ", ".join(f"{c} {s['calls']}" for c, s in left)
                     + ") — see Friday's report")
    if pooled["market_return"] is not None and pooled["market_return"] > 0:
        lines.append(f"In a rising market (S&P 500 {pooled['market_return']:+.1%} since "
                     f"{_mon_d(record.get('since'))}) most bullish calls look right — read beat.")
    return lines


# --- CLI ----------------------------------------------------------------------------------------------


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def main(argv=None):
    """python creator_tracker.py [--as-of YYYY-MM-DD] [--json PATH]: the
    report from the committed claims and price cache (no key, no network)."""
    parser = argparse.ArgumentParser(
        description="Creators vs the market: every creator's calls marked to market from the committed "
                    "claims and price cache (no API key, no network).")
    parser.add_argument("--as-of", type=date.fromisoformat, metavar="YYYY-MM-DD",
                        help="requested as-of date (default: the last completed NYSE session)")
    parser.add_argument("--json", metavar="PATH", help="also write the full record as JSON (dates as ISO strings)")
    args = parser.parse_args(argv)
    requested = args.as_of or default_as_of(_utc_today())
    record = track_record(canonical_claims.load_canonical_claims(), requested)
    stats = record["stats"]
    log_info(f"Creator tracker: {stats['calls']} calls from {stats['video_calls']} video calls, "
             f"{stats['scored']} marked to market at {record['as_of']} (requested {requested}).")
    worklist = unpriced_assets(record)
    if worklist:
        log_info("Calls with no instrument (the ticker worklist): "
                 + ", ".join(f"{asset} {n}" for asset, n in worklist))
    if args.json and not write_json_atomic(args.json, _jsonable(record)):
        return 1
    messages = format_report(record)
    if not messages:
        log_info("No call can be marked to market yet.")
        return 0
    print("\n\n".join(messages))
    return 0


if __name__ == "__main__":
    sys.exit(main())
