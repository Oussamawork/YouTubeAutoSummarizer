"""
Price rules for the forecast scorecard: exchanges, calendars, the
publication-time entry rule, price-series provenance and benchmarks.

A scorecard is only as honest as its entry price. This module makes the
three decisions the old "first close on or after the publication date" rule
got wrong, and documents each:

PUBLICATION-TIME RULE (`entry_point`)
  A price that printed BEFORE the video was published can never be the
  entry: the speaker may have seen it. The instrument's exchange, its
  timezone and its trading calendar decide whether the publication landed
  before, during or after a session, and the entry is the first session
  CLOSE strictly after the publication instant ("next-close" convention):
    before the open / during the session on a trading day -> that day's close
    after the close, a weekend, a holiday                  -> next trading day's close
  The evaluation price uses the same rule from the forecast end date. Daily
  data carries no intraday timestamps, so a publication whose instant cannot
  be placed against the session (missing time of day) takes the conservative
  next-trading-day close. 24/7 assets (crypto) have no session: their daily
  close is the UTC day boundary, so the entry is the close of the UTC day
  the video was published on — the first daily close after publication.

PRICE SERIES PROVENANCE (`PriceSeries`)
  Every series records its provider, adjustment type (split-adjusted at
  minimum; total-return-adjusted where the provider supports it — Twelve
  Data's `adjust=all` folds dividends in too, `adjust=splits` is its default
  and folds only splits), corporate-action status, currency, the requested
  dates and the trading dates they resolved to. Unadjusted series are
  refused for scoring (a 10:1 split would read as a 90% crash).

BENCHMARKS (`resolve_benchmark`)
  No universal SPY. The benchmark comes from asset type, geography (the
  exchange's country) and sector through a configurable table
  (BENCHMARKS, overridable with the BENCHMARKS_JSON env var). The asset
  type, country and sector are the INSTRUMENT's (instruments.py, curated
  or provider-verified), never a claim's model-written sector. When no
  defensible benchmark exists the comparison is null — raw performance
  only, nothing fabricated — and sources scored under different benchmark
  methods are never ranked against each other.

CALENDAR CONFIDENCE
  Only NYSE closures are modelled by rule (calendar_confidence=full);
  crypto is continuous. Every other exchange is weekday-only
  (weekdays_only): its local holidays are NOT known here, so a publication
  or evaluation date on one resolves to whatever close the provider next
  returns — possibly days later, possibly a carried-forward quote — and no
  bound on that error is claimed. Such claims are scored and labelled but
  excluded from source rankings unless SCORECARD_RANK_WEEKDAY_CALENDARS
  is set.

PRICE TARGETS (`evaluate_target`)
  Reach/hit targets are judged by intraday_touch (daily high/low) when
  bars are available and by daily_close otherwise; the horizon close is
  recorded separately and never turns a reached target into a miss.
"""
import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from helpers import env_flag

# Rankings stay off until the methodology below is the one in production
# for every scored claim AND the live extraction evaluation has been run and
# reviewed by hand (docs/tdd-full-transcript-claims.md § 13.6). The scorecard
# itself is reported as experimental.
SCORECARD_RANKINGS_ENABLED = env_flag("SCORECARD_RANKINGS", default=False)
SCORECARD_EXPERIMENTAL_LABEL = "EXPERIMENTAL — unranked"
# Claims whose exchange calendar is only weekday-based are scored (with the
# calendar confidence recorded) but kept OUT of source rankings unless this
# is set: a local holiday there is invisible, so the entry may resolve days
# late or to a stale close and the comparison is not like-for-like.
RANK_WEEKDAY_CALENDARS = env_flag("SCORECARD_RANK_WEEKDAY_CALENDARS", default=False)
RANKABLE_CALENDAR_CONFIDENCE = {"full", "continuous"}
WEEKDAY_CALENDAR_NOTE = ("weekday-only calendar: local holidays are not modelled; the entry or evaluation "
                         "may have resolved to the first close the provider returned after a closure of "
                         "unknown length, or to a carried-forward quote")
# What the provider is asked to adjust for. "splits" (Twelve Data's default)
# is the minimum this scorecard accepts; "all" adds dividends (a total-return
# proxy). Recorded on every series so a reader knows which they are looking at.
PRICE_ADJUSTMENT = (os.getenv("PRICE_ADJUSTMENT") or "splits").strip().lower()
ADJUSTMENT_LABELS = {"splits": "split_adjusted", "all": "total_return_adjusted",
                     "dividends": "dividend_adjusted", "none": "unadjusted"}
ACCEPTABLE_ADJUSTMENTS = {"split_adjusted", "total_return_adjusted", "dividend_adjusted"}
MAX_PRICE_LAG_DAYS = 5


# --- Exchanges and calendars --------------------------------------------------


@dataclass(frozen=True)
class Exchange:
    code: str
    timezone: str
    open: time
    close: time
    country: str
    currency: str
    calendar: str            # "nyse" (full holiday rules) | "weekdays" | "continuous"
    calendar_confidence: str  # "full" | "weekdays_only" | "continuous"

    def tz(self):
        return ZoneInfo(self.timezone)


EXCHANGES = {
    "us": Exchange("XNYS", "America/New_York", time(9, 30), time(16, 0), "US", "USD", "nyse", "full"),
    "lse": Exchange("XLON", "Europe/London", time(8, 0), time(16, 30), "GB", "GBP", "weekdays", "weekdays_only"),
    "xetra": Exchange("XETR", "Europe/Berlin", time(9, 0), time(17, 30), "DE", "EUR", "weekdays", "weekdays_only"),
    "ams": Exchange("XAMS", "Europe/Amsterdam", time(9, 0), time(17, 30), "NL", "EUR", "weekdays", "weekdays_only"),
    "tse": Exchange("XTKS", "Asia/Tokyo", time(9, 0), time(15, 30), "JP", "JPY", "weekdays", "weekdays_only"),
    "hkex": Exchange("XHKG", "Asia/Hong_Kong", time(9, 30), time(16, 0), "HK", "HKD", "weekdays", "weekdays_only"),
    "krx": Exchange("XKRX", "Asia/Seoul", time(9, 0), time(15, 30), "KR", "KRW", "weekdays", "weekdays_only"),
    "sse": Exchange("XSHG", "Asia/Shanghai", time(9, 30), time(15, 0), "CN", "CNY", "weekdays", "weekdays_only"),
    "szse": Exchange("XSHE", "Asia/Shanghai", time(9, 30), time(15, 0), "CN", "CNY", "weekdays", "weekdays_only"),
}
CRYPTO = Exchange("CRYPTO", "UTC", time(0, 0), time(23, 59, 59), "GLOBAL", "USD", "continuous", "continuous")


def resolve_exchange(symbol, asset_type=None):
    """Exchange for an internal symbol ("nvda.us", "btcusd"), or None when
    the venue is unknown — an unresolved exchange excludes the claim."""
    symbol = (symbol or "").strip().lower()
    if not symbol:
        return None
    if "." in symbol:
        return EXCHANGES.get(symbol.rsplit(".", 1)[-1])
    if symbol.endswith("usd") and (asset_type in (None, "crypto")):
        return CRYPTO
    return None


def _nth_weekday(year, month, weekday, n):
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year, month, weekday):
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _easter(year):
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _observed(d):
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def nyse_holidays(year):
    """NYSE full-day closures by rule (New Year's, MLK, Presidents', Good
    Friday, Memorial, Juneteenth, Independence, Labor, Thanksgiving,
    Christmas), weekend observances applied."""
    days = {
        _observed(date(year, 1, 1)), _nth_weekday(year, 1, 0, 3), _nth_weekday(year, 2, 0, 3),
        _easter(year) - timedelta(days=2), _last_weekday(year, 5, 0), _observed(date(year, 6, 19)),
        _observed(date(year, 7, 4)), _nth_weekday(year, 9, 0, 1), _nth_weekday(year, 11, 3, 4),
        _observed(date(year, 12, 25)),
    }
    # A Saturday New Year's Day is not observed on the Friday before (NYSE
    # rule 7.2), so the observed shift only applies when it moves forward.
    if date(year, 1, 1).weekday() == 5:
        days.discard(date(year - 1, 12, 31))
    return days


def is_trading_day(exchange, day):
    if exchange.calendar == "continuous":
        return True
    if day.weekday() >= 5:
        return False
    if exchange.calendar == "nyse":
        return day not in nyse_holidays(day.year)
    return True


def next_trading_day(exchange, day, inclusive=False):
    d = day if inclusive else day + timedelta(days=1)
    for _ in range(15):
        if is_trading_day(exchange, d):
            return d
        d += timedelta(days=1)
    return None


def parse_publication(value):
    """(datetime in UTC, had_time_of_day). Date-only strings carry no time."""
    if value is None:
        return None, False
    text = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None, False
    had_time = "T" in text or " " in text
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc), had_time


def session_relation(exchange, published_utc, had_time=True):
    """
    ("before_open" | "during_session" | "after_close" | "non_trading_day" |
    "time_unknown", local trading date) for a publication instant.
    """
    local = published_utc.astimezone(exchange.tz())
    day = local.date()
    if exchange.calendar == "continuous":
        return "during_session", day
    if not is_trading_day(exchange, day):
        return "non_trading_day", day
    if not had_time:
        return "time_unknown", day
    t = local.time()
    if t < exchange.open:
        return "before_open", day
    if t >= exchange.close:
        return "after_close", day
    return "during_session", day


def entry_point(exchange, published_at, closes, max_lag=MAX_PRICE_LAG_DAYS):
    """
    The entry price under the next-close rule. Returns a dict:
      requested_date, resolved_trading_date, price, session_relation,
      convention, reason (when no entry could be chosen).
    `closes` is {date: close}. A close dated the publication day is used
    only when the publication landed before that session's close.
    """
    published_utc, had_time = parse_publication(published_at)
    out = {"requested_date": None, "resolved_trading_date": None, "price": None,
           "session_relation": None, "convention": "next_close", "reason": None}
    if published_utc is None:
        out["reason"] = "unparseable_publication_time"
        return out
    if had_time:
        relation, local_day = session_relation(exchange, published_utc, had_time)
    else:
        # A bare date is a calendar day, not an instant: it is not shifted
        # through the exchange timezone, and its session cannot be placed.
        local_day = published_utc.date()
        relation = "during_session" if exchange.calendar == "continuous" else \
            "non_trading_day" if not is_trading_day(exchange, local_day) else "time_unknown"
    out["requested_date"] = local_day.isoformat()
    out["session_relation"] = relation
    if relation in ("before_open", "during_session"):
        first = local_day
    else:
        # after the close, a non-trading day, or an unplaceable time:
        # conservatively the NEXT trading day's close.
        first = next_trading_day(exchange, local_day)
        if relation == "time_unknown":
            out["convention"] = "next_trading_day_close (publication time unknown)"
    if first is None:
        out["reason"] = "no_trading_day_found"
        return out
    for offset in range(max_lag + 1):
        d = first + timedelta(days=offset)
        if d in closes and is_trading_day(exchange, d):
            out["resolved_trading_date"], out["price"] = d.isoformat(), closes[d]
            return out
    out["reason"] = "no_close_within_lag"
    return out


def evaluation_point(exchange, day, closes, max_lag=MAX_PRICE_LAG_DAYS):
    """First close on or after the forecast end date (a trading day)."""
    first = next_trading_day(exchange, day, inclusive=True)
    out = {"requested_date": day.isoformat(), "resolved_trading_date": None, "price": None, "reason": None}
    if first is None:
        out["reason"] = "no_trading_day_found"
        return out
    for offset in range(max_lag + 1):
        d = first + timedelta(days=offset)
        if d in closes and is_trading_day(exchange, d):
            out["resolved_trading_date"], out["price"] = d.isoformat(), closes[d]
            return out
    out["reason"] = "no_close_within_lag"
    return out


# --- Price series provenance ---------------------------------------------------


@dataclass
class PriceSeries:
    symbol: str
    closes: dict
    provider: str = "unknown"
    adjustment: str = "unknown"          # split_adjusted | total_return_adjusted | unadjusted | unknown
    corporate_action_status: str = "unknown"
    currency: str = None
    requested_start: date = None
    requested_end: date = None
    highs: dict = None                   # {date: high}, when the provider returned bars
    lows: dict = None
    notes: list = field(default_factory=list)

    @property
    def granularity(self):
        """daily_ohlc when every close in the series has a high and a low
        beside it, else daily_close (intraday touches are then unknown)."""
        if self.highs and self.lows and all(d in self.highs and d in self.lows for d in self.closes):
            return "daily_ohlc"
        return "daily_close"

    def provenance(self):
        return {
            "price_provider": self.provider, "adjustment_type": self.adjustment,
            "corporate_action_status": self.corporate_action_status, "currency": self.currency,
            "requested_start": self.requested_start.isoformat() if self.requested_start else None,
            "requested_end": self.requested_end.isoformat() if self.requested_end else None,
            "price_data_granularity": self.granularity,
        }

    def scorable(self):
        return self.adjustment in ACCEPTABLE_ADJUSTMENTS and bool(self.closes)


def adjustment_label(adjust_param=None):
    return ADJUSTMENT_LABELS.get((adjust_param or PRICE_ADJUSTMENT).lower(), "unknown")


def series_from_fetcher(fetcher, symbol, start, end, exchange=None):
    """
    Wrap a price fetcher's answer as a PriceSeries. A fetcher returning a
    PriceSeries is passed through; a plain {date: close} dict (the legacy
    fetcher shape) is labelled with the provider's configured adjustment
    only when that provider is Twelve Data via channel_scorecard, else
    "unknown" — which is refused for scoring rather than assumed.
    """
    result = fetcher(symbol, start, end)
    if isinstance(result, PriceSeries):
        return result
    closes = result or {}
    provider, adjustment, status = "unknown", "unknown", "unknown"
    if getattr(fetcher, "price_provenance", None):
        prov = fetcher.price_provenance
        provider, adjustment = prov.get("provider", provider), prov.get("adjustment", adjustment)
        status = prov.get("corporate_action_status", status)
    # A closes dict that carries bars beside it (channel_scorecard.Bars)
    # gives the intraday range; a plain dict gives closes only.
    highs, lows = getattr(result, "highs", None), getattr(result, "lows", None)
    return PriceSeries(symbol=symbol, closes=dict(closes), provider=provider, adjustment=adjustment,
                       corporate_action_status=status,
                       currency=exchange.currency if exchange else None,
                       requested_start=start, requested_end=end,
                       highs=dict(highs) if highs else None, lows=dict(lows) if lows else None)


# --- Price-target success ---------------------------------------------------------
#
# "Reach $200 by year end" can be judged three ways, and they disagree
# whenever the price touches the target and moves away again:
#   intraday_touch   the daily HIGH (bullish) / LOW (bearish) reached the
#                    target on any session in the window — needs bars
#   daily_close      a session CLOSE reached the target on any day in the window
#   horizon_close    the close on the evaluation date is at/beyond the target
# The recorded method is intraday_touch when bars are available and
# daily_close otherwise; horizon_close is always recorded beside it. A
# target reached inside the window is reached, whatever the horizon close
# did afterwards. With closes only, intraday reach is UNKNOWN (None), never
# False.

TARGET_TEST_METHODS = ("intraday_touch", "daily_close", "horizon_close")


def evaluate_target(series, target, direction, entry_day, eval_day):
    """
    The price-target record for one claim over [entry_day, eval_day]:
    target_test_method, target_reached, target_first_reached_date,
    target_reached_within_window, horizon_close_target_met,
    intraday_target_reached, intraday_first_reached_date,
    daily_close_target_reached, daily_close_first_reached_date,
    price_data_granularity. Direction is "bullish" (reach at or above) or
    "bearish" (reach at or below).
    """
    up = direction == "bullish"

    def hit(value):
        return value is not None and (value >= target if up else value <= target)

    days = [d for d in sorted(series.closes) if entry_day <= d <= eval_day]
    close_first = next((d for d in days if hit(series.closes[d])), None)
    ohlc = series.granularity == "daily_ohlc"
    intraday_first = None
    if ohlc:
        extremes = series.highs if up else series.lows
        intraday_first = next((d for d in days if hit(extremes.get(d))), None)
    horizon_met = hit(series.closes.get(eval_day)) if eval_day in series.closes else None
    method = "intraday_touch" if ohlc else "daily_close"
    first = intraday_first if ohlc else close_first
    return {
        "target_value": target, "target_test_method": method,
        "target_reached": first is not None,
        "target_first_reached_date": first.isoformat() if first else None,
        "target_reached_within_window": first is not None,
        "horizon_close_target_met": horizon_met,
        "intraday_target_reached": (intraday_first is not None) if ohlc else None,
        "intraday_first_reached_date": intraday_first.isoformat() if intraday_first else None,
        "daily_close_target_reached": close_first is not None,
        "daily_close_first_reached_date": close_first.isoformat() if close_first else None,
        "price_data_granularity": series.granularity,
        "window": [entry_day.isoformat(), eval_day.isoformat()],
    }


# --- Benchmarks -----------------------------------------------------------------

# (asset_type, country, sector) -> internal benchmark symbol. Most specific
# first; None means "no defensible benchmark". Sector names are the ones the
# extraction prompt lets the model state; anything else falls through to the
# country-level index.
BENCHMARKS = {
    ("stock", "US", "technology"): "xlk.us",
    ("stock", "US", "financials"): "xlf.us",
    ("stock", "US", "energy"): "xle.us",
    ("stock", "US", "healthcare"): "xlv.us",
    ("stock", "US", "consumer discretionary"): "xly.us",
    ("stock", "US", "consumer staples"): "xlp.us",
    ("stock", "US", "industrials"): "xli.us",
    ("stock", "US", "utilities"): "xlu.us",
    ("stock", "US", "materials"): "xlb.us",
    ("stock", "US", "real estate"): "xlre.us",
    ("stock", "US", "communication services"): "xlc.us",
    ("stock", "US", None): "spy.us",
    ("etf", "US", None): "spy.us",
    ("crypto", "GLOBAL", None): "btcusd",
}


def _benchmark_table():
    table = dict(BENCHMARKS)
    raw = os.getenv("BENCHMARKS_JSON")
    if raw:
        try:
            for key, value in json.loads(raw).items():
                parts = [p.strip() or None for p in key.split("|")]
                while len(parts) < 3:
                    parts.append(None)
                table[tuple(parts[:3])] = value
        except (ValueError, AttributeError):
            pass
    return table


def resolve_benchmark(asset_type, exchange, sector=None, symbol=None):
    """
    (benchmark symbol, method) — method names the rule used ("sector",
    "country", "asset_class") or explains the null ("unresolved_exchange",
    "no_benchmark_for_asset_type", "self_benchmark"). Never invents one.
    """
    if exchange is None:
        return None, "unresolved_exchange"
    table = _benchmark_table()
    country = exchange.country
    sector_key = (sector or "").strip().lower() or None
    for key, method in (((asset_type, country, sector_key), "sector"),
                        ((asset_type, country, None), "country"),
                        ((asset_type, "GLOBAL", None), "asset_class")):
        if key[2] is None and method == "sector":
            continue
        bench = table.get(key)
        if bench:
            if symbol and bench == symbol:
                return None, "self_benchmark"
            return bench, method
    return None, "no_benchmark_for_asset_type"


def exchange_for_instrument(instrument):
    """The Exchange an Instrument trades on, or None when its venue is not
    one this module has session rules for."""
    if instrument is None:
        return None
    if instrument.exchange_key == "crypto":
        return CRYPTO
    return EXCHANGES.get(instrument.exchange_key)


def exchange_for_claim(claim, symbol):
    return resolve_exchange(symbol, claim.get("asset_type"))


def rankable_calendar(exchange):
    """Whether claims on this exchange may enter source rankings."""
    return exchange is not None and (RANK_WEEKDAY_CALENDARS
                                     or exchange.calendar_confidence in RANKABLE_CALENDAR_CONFIDENCE)
