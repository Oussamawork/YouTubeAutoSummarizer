"""The signals dataset: loading `data/signals.jsonl`, asset identity, date
windows and per-asset aggregation.

This is the leaf every analytics module reads from — the weekly pulse, its
charts, the channel scorecard and the price-cache warmer. It imports nothing
from any of them, which is what lets those modules import each other freely:
before this split `market_pulse` was both the data layer and a report, so
`channel_scorecard` had to import it at the top while it imported
`channel_scorecard` lazily inside functions, and `ticker_resolver` compared an
exception's class *name* because it could not import the class. The one
upward reference is a call-time read of instruments.REGISTRY in
canonical_ticker (instruments imports this module, so it is deferred).
"""
import json
import os
import re
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
# Keys are the normalized (upper, single-spaced) names observed in the dataset,
# the transcripts' misspellings included ("ADLESIAN" is HKCM's spoken
# Atlassian). Private companies (OpenAI, Anthropic, ...) are deliberately
# absent — they have no ticker, and aggregating them by name is the correct
# behavior. SpaceX is no longer one of them: it has been listed as SPCX since
# June 2026 (the evidence is at TICKER_ALIASES["SPACEX"]).
#
# Only certain or high-confidence mappings belong here, and never a common
# word, a channel's private shorthand or a market subject. Couch Investor's
# "New" (Nu Holdings) and "Core" (CoreWeave) and HKCM's "XP" (XPeng) are
# channel-scoped spellings; "GOLD", "OIL", "S&P 500" and "NASDAQ" are market
# subjects, not securities; "TARGET", "NICE", "GRAB", "GM", "MERCK" and "COKE"
# are words or ambiguous names (NICE and GRAB still resolve from a title or
# spoken ticker). Every key is also matched in free text
# (transcript_normalize._ALIAS_RE), and every value needs an
# instruments.REGISTRY row or an UNPRICEABLE_TICKERS entry:
# tests/test_ticker_corrections.py pins the deny list and the registry rule.
ASSET_ALIASES = {
    # crypto
    "BITCOIN": "BTC", "ETHEREUM": "ETH", "SOLANA": "SOL", "CHAINLINK": "LINK",
    "RIPPLE": "XRP", "CARDANO": "ADA", "DOGECOIN": "DOGE",
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
    "DUTCH BROS": "BROS", "BOOKING HOLDINGS": "BKNG", "BOOKING": "BKNG",
    "DOORDASH": "DASH", "NIKE": "NKE", "MASTERCARD": "MA", "VISA": "V",
    "ROYAL CARIBBEAN": "RCL", "INTERNATIONAL PAPER": "IP", "UPWORK": "UPWK",
    "ROCKET LAB": "RKLB", "COREWEAVE": "CRWV", "PFIZER": "PFE",
    "APPLIED MATERIALS": "AMAT", "VERTIV": "VRT", "HIMS & HERS": "HIMS",
    "NOVO NORDISK": "NVO", "STARBUCKS": "SBUX", "PEPSI": "PEP", "PEPSICO": "PEP",
    "UNITEDHEALTH": "UNH", "UNITED HEALTH": "UNH", "UNITEDHEALTH GROUP": "UNH",
    "MACY'S": "M", "MACYS": "M", "PELOTON": "PTON", "PLANET LABS": "PL",
    "LULULEMON": "LULU", "LULU": "LULU", "ELI LILLY": "LLY", "LILLY": "LLY", "CAVA": "CAVA",
    "CAVA GROUP": "CAVA", "NIBIUS": "NBIS", "MARCADO LIBRE": "MELI", "CARNIVAL": "CCL",
    "NORWEGIAN": "NCLH", "NORWEGIAN CRUISE LINE": "NCLH", "DIEBOLD": "DBD", "COINBASE": "COIN", "STRATEGY": "MSTR", "MICROSTRATEGY": "MSTR",
    # --- 2026-10 audit of the canonical claims nothing resolved. Certain or
    # high-confidence mappings only; where a speaker stated a price it was
    # checked against the cached close ("current market price is 262" vs
    # mcd.us 255.69, "176" vs ftnt.us 180.95, "$1,700" vs meli.us 1,696.56).
    "SPACEX": "SPCX", "SPACE X": "SPCX",   # listed since June 2026, see TICKER_ALIASES
    # technology
    "FORTINET": "FTNT", "QUALCOMM": "QCOM", "RUBRIK": "RBRK", "RUBRIC": "RBRK", "SERVICE NOW": "NOW",
    "IONQ": "IONQ", "SHOPIFY": "SHOP", "FICO": "FICO", "FAIR ISAAC": "FICO", "DELL": "DELL",
    "TEXAS INSTRUMENTS": "TXN", "HEWLETT PACKARD ENTERPRISE": "HPE", "OKTA": "OKTA", "SENTINELONE": "S",
    "MONGODB": "MDB", "LUMENTUM": "LITE", "APPLOVIN": "APP", "FIGMA": "FIG", "AMPHENOL": "APH",
    "KLA": "KLAC", "TSM": "TSM", "APPLIED OPTO ELECTRONICS": "AAOI", "APPLIED OPTOELECTRONICS": "AAOI",
    # communication services
    "GOOGLE ALPHABET": "GOOGL", "PINTEREST": "PINS", "DISNEY": "DIS", "WALT DISNEY": "DIS", "AT&T": "T",
    "VERIZON": "VZ", "T-MOBILE": "TMUS", "TMOBILE US": "TMUS", "TAKE-TWO": "TTWO",
    "TAKE-TWO INTERACTIVE": "TTWO", "TKO": "TKO",
    # consumer
    "MCDONALD'S": "MCD", "MCDONALDS": "MCD", "COCA-COLA": "KO", "COCA COLA": "KO", "PEPSI-COLA": "PEP",
    "PROCTER & GAMBLE": "PG", "COSTCO": "COST", "HERSHEY": "HSY", "SMUCKER": "SJM", "J.M. SMUCKER": "SJM",
    "GENERAL MILLS": "GIS", "GENERAL MILLS INCORPORATED": "GIS", "TJX": "TJX", "TJX COMPANIES": "TJX",
    "DICK'S SPORTING GOODS": "DKS", "DECKERS": "DECK", "GARMIN": "GRMN", "TEXAS ROADHOUSE": "TXRH",
    "POOL CORPORATION": "POOL", "DRAFTKINGS": "DKNG", "SONY": "SONY", "SONY CORPORATION": "SONY",
    "SONY GROUP": "SONY", "FORD": "F", "GENERAL MOTORS": "GM", "LUCID": "LCID", "RIVIAN": "RIVN",
    "XPENG": "XPEV", "NIO": "NIO",
    # financials (the big banks are in TICKER_ALIASES: speakers quote them as
    # sources, so their names must not count as free-text asset mentions)
    "UPSTART": "UPST", "ROBINHOOD": "HOOD", "CITI": "C", "NU HOLDING": "NU", "GRAB HOLDINGS": "GRAB",
    # health care
    "OSCAR HEALTH": "OSCR", "OCUGEN": "OCGN", "ILLUMINA": "ILMN", "INTUITIVE SURGICAL": "ISRG",
    "CRISPR THERAPEUTICS": "CRSP", "CANOPY GROWTH": "CGC", "NOVO": "NVO",
    # cannabis (Phantom by HKCM 2026-09: "Curaleaf, Aurora Cannabis, aber auch Green Thumb Industries")
    "AURORA CANNABIS": "ACB", "CURALEAF": "CURLF", "CURALEAF HOLDINGS": "CURLF",
    "GREEN THUMB INDUSTRIES": "GTBIF",
    # industrials, utilities, energy, materials, real estate
    "BLOOM ENERGY": "BE", "WASTE MANAGEMENT": "WM", "JOBY": "JOBY", "JOBY AVIATION": "JOBY", "BOEING": "BA",
    "RAYTHEON": "RTX", "CATERPILLAR": "CAT", "EATON": "ETN", "QUANTA SERVICES": "PWR", "COPART": "CPRT",
    "PACCAR": "PCAR", "LYFT": "LYFT", "FERGUSON ENTERPRISES": "FERG", "TUTOR PERINI": "TPC",
    "VOYAGER TECHNOLOGIES": "VOYG", "AST SPACE MOBILE": "ASTS", "SERVE ROBOTICS": "SERV", "FIVERR": "FVRR",
    "CEG": "CEG", "CONSTELLATION ENERGY": "CEG", "AMERICAN ELECTRIC POWER": "AEP", "PHILLIPS 66": "PSX",
    "VALERO": "VLO", "FREEPORT-MCMORAN": "FCX", "DIGITAL REALTY": "DLR",
    # A bare recorded "DOW" is the index (NOT_A_TICKER); the company name keeps
    # the chemicals stock (NYSE: DOW) priceable.
    "DOW INC": "DOW", "DOW INC.": "DOW", "DOW CHEMICAL": "DOW",
    # Barrick Gold became Barrick Mining and moved from NYSE: GOLD to NYSE: B
    # in May 2025; whatever trades as GOLD now is neither Barrick nor the metal.
    "BARRICK": "B", "BARRICK GOLD": "B", "BARRICK GOLD CORPORATION": "B", "BARRICK MINING": "B",
    "BARRICK MINING CORPORATION": "B",
    "FRESNILLO": "FRESNILLO", "FRESNILLO PLC": "FRESNILLO",   # London (FRES); unpriceable, not NYSE: FR
    # bitcoin miners and AI data centres
    "MARA": "MARA", "MARA HOLDINGS": "MARA", "RIOT PLATFORMS": "RIOT", "CLEANSPARK": "CLSK",
    "TERAWULF": "WULF",
    # transcription variants, each read from its evidence
    "MARCADO LIBÉ": "MELI", "MARCADO LIBE": "MELI", "MARCADO LIBRA": "MELI",   # "back under $1,700 per share"
    "ELI LIY": "LLY", "LILY": "LLY", "NOVOR NORDISK": "NVO", "NOVO NORDLESS": "NVO",
    "NCINO": "NCNO",    # "nCino": its TICKER_ALIASES key is ticker-shaped, so only read from a ticker field
    "RSP": "RSP",       # "the broader market, RSP as an ETF": Invesco S&P 500 Equal Weight
    "ADLESIAN": "TEAM", "ATLESION": "TEAM",   # "Adlesian aus dem Softwarebereich"
    "PALUNTEER": "PLTR", "PALANTIA": "PLTR", "THE TRADEDESK": "TTD", "TRADEDESK": "TTD",
    "VERTOV": "VRT",                          # "caterpillar, Vertov and Eaton"
    "MARLL": "MRVL",                          # "a connectivity company ... custom silicon"
    "FREQUENT MAC MORAN": "FCX",              # "predominantly copper ... gold"
    "TEROWOLF": "WULF", "TUDOR PERINI": "TPC", "CROWD STRIKE": "CRWD", "WESTERN DIG": "WDC",
    # crypto: the bare symbols are names too ("XRP", "ETH"), and Polygon's
    # token migrated from MATIC to POL in 2024
    "XRP": "XRP", "ETH": "ETH", "BTC": "BTC", "SOL": "SOL", "POLYGON": "POL", "VTHO": "VTHO",
    "VETHOR": "VTHO", "PENDLE": "PENDLE", "PENDEL": "PENDLE", "AVAX": "AVAX", "AVALANCHE": "AVAX",
    "SUI": "SUI", "SUIE": "SUI", "TAO": "TAO", "BITTENSOR": "TAO", "VECHAIN": "VET", "POLKADOT": "DOT",
}

# Trailing words speakers attach to a company name ("Amazon stock", "Nvidia
# shares", "Canopy Growth-Aktie", "Sony Corporation") that are not part of it:
# stripped before the alias lookup so the asset folds into its ticker instead
# of aggregating under a second key.
_NAME_SUFFIXES = (" STOCK", " STOCKS", " SHARES", " SHARE", " AKTIE", " AKTIEN", "-AKTIE", "-AKTIEN",
                  " INC", " INC.", " CORP", " CORP.", " CORPORATION", " PLC")
NAME_SUFFIXES = _NAME_SUFFIXES   # public: claims.py strips the same words

# Recorded-ticker variants folded to one canonical symbol: dual share classes
# and renames that speakers use interchangeably would otherwise still split an
# asset across two keys even when a ticker WAS recorded.
TICKER_ALIASES = {
    "GOOG": "GOOGL",   # Alphabet share classes
    "UA": "UAA",       # Under Armour share classes
    "SQ": "XYZ",       # Block's 2025 ticker change
    "MATIC": "POL",    # Polygon's token migrated from MATIC to POL in 2024
    # Speakers say the company name and the extractor records it in the ticker
    # field. Left alone these split one asset across two buckets — the dataset
    # carried both AAPL and APPLE, diluting mention counts and consensus in
    # every chart — and they price as nothing, since no exchange lists "APPLE".
    "APPLE": "AAPL", "GOOGLE": "GOOGL", "ALPHABET": "GOOGL", "NVIDIA": "NVDA",
    "AMAZON": "AMZN", "MICROSOFT": "MSFT", "TESLA": "TSLA", "NETFLIX": "NFLX",
    "SALESFORCE": "CRM", "WESTERN DIGITAL": "WDC", "MASTEC": "MTZ",
    "NCINO": "NCNO", "BLOCK": "XYZ", "PALANTIR": "PLTR", "BROADCOM": "AVGO",
    "XPENG": "XPEV",   # XPeng's NYSE ADR; nothing is listed as XPENG
    # Banks resolve as a claim's subject ("I always like Citi ... and Goldman
    # has always been the superior") but are kept out of ASSET_ALIASES, whose
    # keys also count as free-text mentions: in "Goldman expects the stock to
    # reach $200" the bank is the source, and "the stock" is whatever the
    # speaker named before. Name-shaped keys here are read as names too.
    "GOLDMAN": "GS", "GOLDMAN SACHS": "GS", "JP MORGAN": "JPM", "JPMORGAN": "JPM",
    "BANK OF AMERICA": "BAC", "CITIGROUP": "C",
    # Tickers the transcript got wrong outright.
    # SpaceX has traded as SPCX since its June 2026 IPO. "spcx.us returns
    # closes" only proved that a series exists; the data says it is SpaceX:
    #   - creators type the ticker in their titles: Parkev 2026-08-04, 08-07
    #     and 09-14 ("... | SPCX Stock Analysis"), and HKCM 2026-10-09 says
    #     "SpaceX mit dem Ticker SPCX";
    #   - stated prices sit within ~1% of the spcx.us closes: "IPO Levels ...
    #     ungefähr 147 USD" (Phantom 09-06) vs 147.95 on 09-04, "$150 per
    #     share" (Parkev 09-14) vs 151.21 / 148.15 on 09-11 / 09-14, and 469
    #     USD at +172% (Phantom 10-06) implies 172.4 vs 171.92 on 10-06;
    #   - the venue is Nasdaq (FinVid 09-15), as the XNAS registry row says.
    # Symbol search only surfaces Thai DRs and 3x leveraged ETPs, which are
    # the wrong instrument.
    "SPACEX": "SPCX",
    "NEBL": "NBIS",    # Nebius is NBIS
    "RUBY": "RBRK",    # "Rubric" mis-transcribed; the company is Rubrik
    "PAS": "PAAS",     # Pan American Silver
    "KLA": "KLAC",     # KLA Corp trades as KLAC
    "I-REN": "IREN",   # Iris Energy, renamed IREN
}

# Recorded tickers that no US listing can price: private companies, and
# foreign or unlisted names the speakers discuss by local ticker. Kept
# explicit so they are skipped up front instead of spending a price request
# per run to rediscover a 404. They still aggregate by name in the pulse —
# only the price lookup is suppressed. Only entries canonical_ticker can
# actually return belong here: ADYEN and BASF were dropped because the alias
# table and the learned map turn them into the ADRs ADYEY / BASFY first.
UNPRICEABLE_TICKERS = {
    "OPENAI", "STRIPE", "BYTEDANCE", "ANTHROPIC", "WAYMO", "ANDURIL",  # private
    "CXMT", "YMTC",                             # unlisted Chinese memory makers
    "SK HYNIX", "VOW", "P911",                  # non-US listings
    # SK Hynix's 2026 US listing and Anthropic's pending IPO are on the
    # ticker audit's needs-verification list: neither ticker is stated in the
    # data, so both stay unpriceable until a catalogue lookup confirms one.
    "FRESNILLO",                                # London-listed (FRES)
    # NYSE "GOLD" is neither Barrick (NYSE: B since May 2025) nor the metal;
    # the metal is priced only through a labelled proxy (GLD), never gold.us.
    "GOLD",
    "IRNT",                                     # IronNet, delisted in 2023
    # Korean, Taiwanese and Shanghai codes: every one became a junk ".us"
    # symbol that each cache warm retried.
    "000660", "000660.KS", "005930", "005930.KS", "2454.TW", "688041.SS",
    "ETHBTC",                                   # a ratio, not an instrument
    # Legacy rows' type rule built these into the wrong company (ticker_audit
    # KNOWN_WRONG_SYMBOLS): TheStreet's "Mogu ... space propulsion" is Moog
    # (MOG.A, on the needs-verification list), not MOGU Inc.; "RCI" was the
    # Columbia Research Enhanced Core ETF, not Rogers Communications.
    "MOGU", "RCI",
}

# Curated names that are also ordinary words ("our strategy", "outrun the
# avalanche", "das Pendel", "more upstart type companies", "a lucid look"):
# in a title, inside a subject or in free transcript text they name the
# asset only when written in capitals or followed by "stock" / "Aktie".
# Read by claims (titles, subjects) and transcript_normalize (free text).
COMMON_WORD_NAMES = frozenset({"STRATEGY", "AVALANCHE", "BLOCK", "BOOKING", "CARNIVAL", "NORWEGIAN", "COHERENT",
                               "LILY", "PENDEL", "TKO", "UPSTART", "LUCID", "POLYGON", "TAO"})

# Name keys of ASSET_ALIASES that are ALSO another company's listed ticker:
# "Ford" is Ford Motor (F), but FORD in a ticker field is Forward Industries
# (Nasdaq: FORD). A recorded or spoken ticker is never folded through these.
LISTED_TICKER_KEYS = frozenset({"FORD"})

# Index and benchmark names that titles and speakers put in the ticker field
# ("NASDAQ kämpft mit dem Allzeithoch!"). None is a security — Nasdaq Inc is
# NDAQ, and nasdaq.us or spx.us price nothing or the wrong thing — so a
# recorded value in this set is dropped and the name decides: an index never
# becomes a ticker, and the subject aggregates under its name.
NOT_A_TICKER = frozenset({
    "NASDAQ", "NDX", "SPX", "S&P 500", "S&P", "SP500", "DAX", "DXY", "NIKKEI", "DOW", "DJI", "DJIA", "VIX",
})

# A TICKER_ALIASES key shaped like a ticker (GOOG, RUBY, NEBL, PAS) is only
# ever read from the ticker field: as a spoken name it means something else.
_TICKER_SHAPED = re.compile(r"^[A-Z]{1,5}$")

# A company's ticker before a rename, now held by another issuer (so no global
# TICKER_ALIASES fold is safe): a record naming the company under its old
# ticker means the company. Keyed by the current ticker.
FORMER_TICKERS = {
    "B": frozenset({"GOLD"}),   # Barrick Gold -> Barrick Mining, NYSE: GOLD -> B in May 2025
}


def _normalized_name(asset):
    return " ".join((asset.get("name") or "").split()).upper()


def _name_alias(name):
    """The curated ticker for an exact normalized name: ASSET_ALIASES, then
    the name-shaped TICKER_ALIASES keys (SPACEX, MASTEC, I-REN), the same
    two tables claims.resolve_entity reads."""
    if name in ASSET_ALIASES:
        return ASSET_ALIASES[name]
    if name in TICKER_ALIASES and not _TICKER_SHAPED.match(name):
        return TICKER_ALIASES[name]
    return None


def _alias_for_name(name):
    """Curated ticker for a spoken name, tolerating "<name> stock"."""
    if not name:
        return None
    found = _name_alias(name)
    if found:
        return found
    for suffix in _NAME_SUFFIXES:
        if name.endswith(suffix):
            found = _name_alias(name[: -len(suffix)])
            if found:
                return found
    return None


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


def _fold_recorded(ticker):
    """A recorded ticker folded through the curated tables: a company name in
    the ticker field ("SOLANA") through ASSET_ALIASES, then share classes,
    renames and mis-transcriptions through TICKER_ALIASES."""
    if ticker and ticker not in TICKER_ALIASES and ticker in ASSET_ALIASES and ticker not in LISTED_TICKER_KEYS:
        # No listing prices a name (Twelve Data 404s SOLANA/USD) and it splits
        # the asset from its real ticker. Every alias whose key is already a
        # ticker maps to itself, so real tickers pass through unchanged.
        ticker = ASSET_ALIASES[ticker]
    return TICKER_ALIASES.get(ticker, ticker)


def _curated_decision(ticker):
    """True when the curated tables already decided what `ticker` is: an
    UNPRICEABLE_TICKERS entry or an instruments.REGISTRY row. The registry is
    read lazily — instruments imports this module, and by the time a ticker
    is resolved both are loaded."""
    if ticker in UNPRICEABLE_TICKERS:
        return True
    import instruments
    return ticker in instruments.REGISTRY


def canonical_ticker(asset):
    """
    The asset's ticker after the curated tables, or None when nothing knows
    one. `asset` is {"ticker", "name"}; any other key (e.g. "channel") is
    ignored. The rules, in order:

    1. An index name in the ticker field (NOT_A_TICKER: NASDAQ, SPX, DAX, ...)
       is dropped and the name decides; an index never becomes a ticker.
    2. A curated name alias (ASSET_ALIASES after suffix stripping, plus the
       name-shaped TICKER_ALIASES keys) outranks a recorded ticker that
       contradicts it. Such records name another company: "Sony Corporation"
       recorded as SY (So-Young, ~$2.6) is SONY, "CoreWeave" recorded as CRWE
       (Crown Equity, a penny stock) is CRWV. Agreement is judged after
       folding the recorded ticker (Google/GOOG -> GOOGL; Nvidia/NVDA is
       unchanged). A recorded ticker that is itself a curated decision — a
       registered instrument or an UNPRICEABLE entry — is kept, unless it is
       the named company's former ticker ("Barrick Mining Corporation" under
       its pre-2025 GOLD is B).
    3. Otherwise the recorded ticker, folded through ASSET_ALIASES (a company
       name in the ticker field, "SOLANA" -> SOL) and TICKER_ALIASES (share
       classes, renames, mis-transcriptions).
    4. The learned map only fills gaps: it is consulted only when neither
       curated table decided, so a learned entry can never quietly override
       a decision someone made on purpose.
    """
    recorded = (asset.get("ticker") or "").strip().upper()
    if recorded in NOT_A_TICKER:
        recorded = ""
    name = _normalized_name(asset)
    folded = _fold_recorded(recorded)
    curated = _alias_for_name(name)
    if curated:
        curated = TICKER_ALIASES.get(curated, curated)
        if (not folded or folded == curated or not _curated_decision(folded)
                or folded in FORMER_TICKERS.get(curated, ())):
            return curated
        return folded
    if recorded in ASSET_ALIASES or recorded in TICKER_ALIASES:
        return folded
    learned = learned_tickers()
    for key in (recorded, name):
        if key and key not in NOT_A_TICKER and key in learned:
            return learned[key]
    return recorded or None


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
