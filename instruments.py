"""
Canonical instrument metadata: the deterministic source for everything the
scorecard needs to know about a tradable thing.

A claim's `sector` and `asset_type` come from the model (or the speaker)
and are kept for transcript analysis, but they never choose a benchmark or
a trading calendar: an LLM that calls Tesla "technology" would silently
score it against the wrong index. Every scorecard decision starts from the
registry below instead:

    instrument_id   "<MIC>:<ticker>", e.g. XNAS:NVDA, CRYPTO:BTC
    ticker          the canonical ticker (signals_data alias tables applied)
    symbol          the internal, provider-independent price symbol
                    ("nvda.us", "btcusd")
    exchange_key    key into scorecard_pricing.EXCHANGES ("us") or "crypto"
    mic             ISO 10383 market identifier of the primary listing
    asset_type      stock | etf | crypto
    country         listing country (drives the country-level benchmark)
    currency        trading currency of the listing
    sector          GICS sector, lower-case, as the benchmark table spells it
                    — None when not curated (falls to the country benchmark)
    benchmark       "auto" (resolve through scorecard_pricing.BENCHMARKS),
                    an explicit internal symbol, or None = no defensible
                    benchmark (raw performance only)
    source          "curated" (hand-checked table below) or
                    "provider_catalogue" (ticker_resolver's verified map,
                    which carries the listing exchange but no sector)
    name            a display name, set only where the bare ticker is unclear
                    (T, V, PL, SPCX, ...); display_label() renders it

The registry covers every ticker the alias tables in signals_data can
produce (or the ticker is in signals_data.UNPRICEABLE_TICKERS): a test
enforces that promise, so a new alias without its instrument fails CI instead
of silently dropping out of the scorecard as unresolved_instrument.
Anything not in the curated table and not in the verified learned map is
UNRESOLVED: the scorecard excludes the claim rather than guessing an
exchange or a benchmark. A learned entry is found by its verified name or,
failing that, by its ticker. Sectors below follow the GICS classification
as of 2026-09 for the primary US listing (ADRs/OTC listings are scored on
their US quote; their home-market index is not a defensible benchmark for a
USD quote, so those carry benchmark=None).
"""
from dataclasses import dataclass

from signals_data import UNPRICEABLE_TICKERS, canonical_ticker, learned_ticker_entries

# GICS sector spellings used by scorecard_pricing.BENCHMARKS.
IT, FIN, ENE, HC, CD, CS, IND, UTIL, MAT, RE, COMM = (
    "technology", "financials", "energy", "healthcare", "consumer discretionary", "consumer staples",
    "industrials", "utilities", "materials", "real estate", "communication services")


@dataclass(frozen=True)
class Instrument:
    instrument_id: str
    ticker: str
    symbol: str
    exchange_key: str
    mic: str
    asset_type: str
    country: str
    currency: str
    sector: str = None
    benchmark: str = "auto"
    source: str = "curated"
    name: str = None

    def metadata(self):
        return {"instrument_id": self.instrument_id, "instrument_ticker": self.ticker,
                "instrument_symbol": self.symbol, "instrument_mic": self.mic,
                "instrument_asset_type": self.asset_type, "instrument_country": self.country,
                "instrument_currency": self.currency, "instrument_sector": self.sector,
                "instrument_benchmark_rule": self.benchmark, "instrument_source": self.source}


def _us(ticker, mic, sector, benchmark="auto", name=None):
    return Instrument(f"{mic}:{ticker}", ticker, f"{ticker.lower()}.us", "us", mic, "stock", "US", "USD",
                      sector, benchmark, name=name)


def _etf(ticker, mic, sector=None, name=None):
    return Instrument(f"{mic}:{ticker}", ticker, f"{ticker.lower()}.us", "us", mic, "etf", "US", "USD", sector, "auto",
                      name=name)


def _crypto(ticker, name=None):
    return Instrument(f"CRYPTO:{ticker}", ticker, f"{ticker.lower()}usd", "crypto", "CRYPTO", "crypto",
                      "GLOBAL", "USD", None, "auto", name=name)


# Curated registry: every ticker the alias tables in signals_data can
# produce. OTC ADRs carry benchmark=None (see the module docstring). A name
# is given only where the bare ticker is unclear: every one- or two-letter
# ticker, and the few others a reader would not place (SPCX, NOW, TEAM, ...);
# the megacaps stay bare.
_CURATED = [
    # crypto (24/7, UTC-day close)
    _crypto("BTC"), _crypto("ETH"), _crypto("SOL"), _crypto("XRP"), _crypto("LINK", name="Chainlink"),
    _crypto("ADA"), _crypto("DOGE"), _crypto("POL", name="Polygon"), _crypto("AVAX"), _crypto("SUI", name="Sui"),
    _crypto("TAO", name="Bittensor"), _crypto("VET", name="VeChain"), _crypto("VTHO", name="VeThor"),
    _crypto("PENDLE"), _crypto("DOT"),
    # megacaps
    _us("NVDA", "XNAS", IT), _us("MSFT", "XNAS", IT), _us("AAPL", "XNAS", IT), _us("AMZN", "XNAS", CD),
    _us("META", "XNAS", COMM), _us("GOOGL", "XNAS", COMM), _us("TSLA", "XNAS", CD), _us("NFLX", "XNAS", COMM),
    # semis / hardware
    _us("MU", "XNAS", IT, name="Micron"), _us("INTC", "XNAS", IT), _us("AMD", "XNAS", IT), _us("ASML", "XNAS", IT),
    _us("AVGO", "XNAS", IT), _us("MRVL", "XNAS", IT), _us("TSM", "XNYS", IT, name="TSMC"), _us("WDC", "XNAS", IT),
    _us("STX", "XNAS", IT), _us("COHR", "XNYS", IT), _us("NBIS", "XNAS", IT, name="Nebius"),
    # software / fintech / internet
    _us("PLTR", "XNAS", IT), _us("CRM", "XNYS", IT), _us("IBM", "XNYS", IT), _us("ADBE", "XNAS", IT),
    _us("SNOW", "XNYS", IT), _us("NOW", "XNYS", IT, name="ServiceNow"), _us("TEAM", "XNAS", IT, name="Atlassian"),
    _us("ZS", "XNAS", IT, name="Zscaler"),
    _us("TTD", "XNAS", COMM), _us("CRWD", "XNAS", IT), _us("PANW", "XNAS", IT), _us("SOFI", "XNAS", FIN),
    _us("PYPL", "XNAS", FIN), _us("XYZ", "XNYS", FIN, name="Block"), _us("RDDT", "XNYS", COMM),
    _us("ZETA", "XNYS", None),
    _us("AXON", "XNAS", IND), _us("MELI", "XNAS", CD), _us("UBER", "XNYS", IND),
    _us("NU", "XNYS", FIN, name="Nu Holdings"),
    _us("BABA", "XNYS", CD), _us("ORCL", "XNYS", IT), _us("SHW", "XNYS", MAT), _us("RBRK", "XNYS", IT),
    _us("NCNO", "XNAS", IT), _us("MTZ", "XNYS", IND),
    # energy
    _us("OXY", "XNYS", ENE), _us("CVX", "XNYS", ENE), _us("XOM", "XNYS", ENE), _us("TTE", "XNYS", ENE),
    # other observed (Walmart moved its listing from NYSE to Nasdaq on 2025-12-09)
    _us("WMT", "XNAS", CS), _us("HD", "XNYS", CD, name="Home Depot"), _us("UAA", "XNYS", CD),
    _us("URI", "XNYS", IND), _us("PAAS", "XNYS", MAT), _us("SPCX", "XNAS", None, name="SpaceX"),
    # --- 2026-10: every alias output and every ticker with 3+ canonical
    # claims. sector None = no curated GICS sector, so the country benchmark
    # applies. DPZ's MIC (XNAS) is on the ticker audit's needs-verification
    # list; NYSE and Nasdaq share one exchange here, so the symbol, calendar
    # and benchmark do not depend on it — only the instrument id does.
    # technology
    _us("AMAT", "XNAS", IT), _us("DBD", "XNYS", IT), _us("MDB", "XNAS", IT), _us("FICO", "XNYS", IT),
    _us("NICE", "XNAS", IT), _us("NVTS", "XNAS", IT), _us("QCOM", "XNAS", IT), _us("PATH", "XNYS", IT),
    _us("FTNT", "XNAS", IT), _us("IONQ", "XNYS", IT), _us("RGTI", "XNAS", IT), _us("SHOP", "XNAS", IT),
    _us("TXN", "XNAS", IT), _us("DELL", "XNYS", IT), _us("HPE", "XNYS", IT), _us("OKTA", "XNAS", IT),
    _us("S", "XNYS", IT, name="SentinelOne"), _us("APH", "XNYS", IT), _us("FIG", "XNYS", IT),
    _us("LITE", "XNAS", IT), _us("APP", "XNAS", IT), _us("AAOI", "XNAS", IT), _us("KLAC", "XNAS", IT),
    _us("SNDK", "XNAS", IT), _us("CRWV", "XNAS", None, name="CoreWeave"), _us("UPWK", "XNAS", None),
    _us("GRAB", "XNAS", None),
    # communication services
    _us("T", "XNYS", COMM, name="AT&T"), _us("VZ", "XNYS", COMM, name="Verizon"), _us("TKO", "XNYS", COMM),
    _us("PINS", "XNYS", COMM), _us("DIS", "XNYS", COMM, name="Disney"), _us("TTWO", "XNAS", COMM),
    _us("TMUS", "XNAS", COMM),
    # consumer discretionary
    _us("BKNG", "XNAS", CD, name="Booking"), _us("BROS", "XNYS", CD), _us("CAVA", "XNYS", CD),
    _us("CCL", "XNYS", CD), _us("DASH", "XNAS", CD), _us("LULU", "XNAS", CD), _us("M", "XNYS", CD, name="Macy's"),
    _us("NCLH", "XNYS", CD), _us("NKE", "XNYS", CD), _us("PTON", "XNAS", CD), _us("RCL", "XNYS", CD),
    _us("SBUX", "XNAS", CD), _us("NIO", "XNYS", CD), _us("RIVN", "XNAS", CD), _us("LCID", "XNAS", CD),
    _us("LOW", "XNYS", CD), _us("DKNG", "XNAS", CD), _us("GM", "XNYS", CD, name="General Motors"),
    _us("CMG", "XNYS", CD), _us("XPEV", "XNYS", CD), _us("DECK", "XNYS", CD), _us("MCD", "XNYS", CD),
    _us("F", "XNYS", CD, name="Ford"), _us("TJX", "XNYS", CD), _us("DKS", "XNYS", CD), _us("GRMN", "XNYS", CD),
    _us("TXRH", "XNAS", CD), _us("POOL", "XNAS", CD), _us("SONY", "XNYS", CD),
    _us("DPZ", "XNAS", CD, name="Domino's"),
    # consumer staples
    _us("PEP", "XNAS", CS, name="PepsiCo"), _us("CELH", "XNAS", CS), _us("KO", "XNYS", CS, name="Coca-Cola"),
    _us("COST", "XNAS", CS), _us("CLX", "XNYS", CS), _us("SJM", "XNYS", CS), _us("HSY", "XNYS", CS),
    _us("GIS", "XNYS", CS), _us("PG", "XNYS", CS, name="Procter & Gamble"),
    # financials
    _us("COIN", "XNAS", FIN), _us("MA", "XNYS", FIN, name="Mastercard"), _us("V", "XNYS", FIN, name="Visa"),
    _us("UPST", "XNAS", FIN), _us("HOOD", "XNAS", FIN, name="Robinhood"), _us("C", "XNYS", FIN, name="Citigroup"),
    _us("BAC", "XNYS", FIN), _us("GS", "XNYS", FIN, name="Goldman Sachs"), _us("JPM", "XNYS", FIN),
    _us("MSTR", "XNAS", None, name="Strategy"),
    # health care
    _us("HIMS", "XNYS", HC), _us("LLY", "XNYS", HC), _us("NVO", "XNYS", HC), _us("PFE", "XNYS", HC),
    _us("UNH", "XNYS", HC), _us("MRK", "XNYS", HC), _us("CGC", "XNAS", HC), _us("OCGN", "XNAS", HC), _us("ILMN", "XNAS", HC),
    _us("OSCR", "XNYS", HC), _us("ISRG", "XNAS", HC), _us("CRSP", "XNAS", HC),
    _us("ACB", "XNAS", HC, name="Aurora Cannabis"),
    # industrials
    _us("RKLB", "XNAS", IND), _us("VRT", "XNYS", IND), _us("MMM", "XNYS", IND), _us("LMT", "XNYS", IND),
    _us("ACHR", "XNYS", IND), _us("BE", "XNYS", IND, name="Bloom Energy"),
    _us("WM", "XNYS", IND, name="Waste Management"), _us("JOBY", "XNYS", IND), _us("FERG", "XNYS", IND),
    _us("VOYG", "XNYS", IND), _us("RTX", "XNYS", IND), _us("PCAR", "XNAS", IND), _us("BA", "XNYS", IND, name="Boeing"),
    _us("LYFT", "XNAS", IND), _us("PWR", "XNYS", IND), _us("CPRT", "XNAS", IND), _us("TPC", "XNYS", IND),
    _us("CAT", "XNYS", IND), _us("ETN", "XNYS", IND),
    # utilities, energy, materials, real estate
    _us("CEG", "XNAS", UTIL), _us("AEP", "XNAS", UTIL), _us("PSX", "XNYS", ENE), _us("VLO", "XNYS", ENE),
    _us("IP", "XNYS", MAT, name="International Paper"), _us("FCX", "XNYS", MAT),
    _us("DOW", "XNYS", MAT, name="Dow Inc."),
    _us("B", "XNYS", MAT, name="Barrick"), _us("DLR", "XNYS", RE),
    # no curated sector (country benchmark)
    _us("PL", "XNYS", None, name="Planet Labs"), _us("MARA", "XNAS", None), _us("FVRR", "XNYS", None),
    _us("RIOT", "XNAS", None), _us("ASTS", "XNAS", None), _us("SERV", "XNAS", None), _us("WULF", "XNAS", None),
    _us("CLSK", "XNAS", None), _us("IREN", "XNAS", None),
    # named in the 2026-10-17 backlog extraction
    _us("DDOG", "XNAS", IT), _us("INTU", "XNAS", IT), _us("LRCX", "XNAS", IT), _us("NOK", "XNYS", IT),
    _us("JNJ", "XNYS", HC), _us("TDOC", "XNYS", HC), _us("DUK", "XNYS", UTIL), _us("MCO", "XNYS", FIN),
    _us("DLO", "XNAS", FIN, name="dLocal"), _us("ONON", "XNYS", CD, name="On Holding"),
    _us("DUOL", "XNAS", None, name="Duolingo"),
    _us("GEV", "XNYS", IND, name="GE Vernova"), _us("WDAY", "XNAS", IT), _us("FIX", "XNYS", IND, name="Comfort Systems"),
    _us("APLD", "XNAS", None, name="Applied Digital"),
    # broad and sector ETFs (a claim about SPY scores raw: SPY vs SPY is no benchmark)
    _etf("SPY", "ARCX"), _etf("VOO", "ARCX"), _etf("QQQ", "XNAS"), _etf("IWM", "ARCX"), _etf("DIA", "ARCX"),
    _etf("XLK", "ARCX", IT), _etf("XLF", "ARCX", FIN), _etf("XLE", "ARCX", ENE), _etf("XLV", "ARCX", HC),
    _etf("XLY", "ARCX", CD), _etf("XLP", "ARCX", CS), _etf("XLI", "ARCX", IND), _etf("XLU", "ARCX", UTIL),
    _etf("XLB", "ARCX", MAT), _etf("XLRE", "ARCX", RE), _etf("XLC", "ARCX", COMM),
    _etf("RSP", "ARCX", name="Equal-weight S&P 500"),   # Invesco S&P 500 Equal Weight ("RSP as an ETF")
    # OTC ADRs and OTC-quoted foreign issuers: priced on the US quote, no
    # defensible USD benchmark
    _us("BAESY", "OTCM", IND, benchmark=None), _us("SFTBY", "OTCM", COMM, benchmark=None),
    _us("ADYEY", "OTCM", FIN, benchmark=None), _us("BASFY", "OTCM", MAT, benchmark=None),
    _us("CURLF", "OTCM", HC, benchmark=None, name="Curaleaf"),
    _us("GTBIF", "OTCM", HC, benchmark=None, name="Green Thumb"),
]
REGISTRY = {inst.ticker: inst for inst in _CURATED}

# Listing-exchange names the provider catalogue (ticker_resolver) records,
# mapped to MIC + exchange key. Anything else is unresolved.
_CATALOGUE_EXCHANGES = {
    "NYSE": ("XNYS", "us"), "NASDAQ": ("XNAS", "us"), "NYSE ARCA": ("ARCX", "us"), "AMEX": ("XASE", "us"),
    "NYSE AMERICAN": ("XASE", "us"), "OTC": ("OTCM", "us"), "OTCM": ("OTCM", "us"), "CBOE": ("BATS", "us"),
}


def display_label(ticker):
    """
    How a report names a ticker: "SpaceX (SPCX)" when the registry gives the
    instrument a display name (the bare ticker is unclear: T, V, PL, SPCX),
    else the input unchanged — a megacap ("NVDA"), an unknown ticker, None.
    """
    if not isinstance(ticker, str):
        return ticker
    inst = REGISTRY.get(ticker.strip().upper())
    if inst is None or not inst.name:
        return ticker
    return f"{inst.name} ({inst.ticker})"


def _entry_for_ticker(entries, ticker):
    """The learned entry listed under `ticker`, or None. Several spellings may
    share one verified listing ("GOLDMAN SACHS", "GOLDMAN"); entries that
    disagree on the venue answer nothing."""
    if not ticker:
        return None
    matches = [e for e in entries.values()
               if isinstance(e, dict) and (e.get("ticker") or "").strip().upper() == ticker]
    venues = {(e.get("exchange") or "").strip().upper() for e in matches}
    return matches[0] if matches and len(venues) == 1 else None


def _from_catalogue(name, ticker):
    """An Instrument from ticker_resolver's verified map, or None. The entry
    is found by its verified name, else by its ticker: a listing the
    catalogue verified once is the same listing whichever spelling a later
    claim uses ("Rocket Labs", "CoreWeave stock"). The map carries the
    listing exchange (verified against the provider's own catalogue) but no
    sector, so the benchmark is the country index."""
    entries = learned_ticker_entries()
    entry = entries.get(" ".join(name.split()).upper()) if name else None
    if not isinstance(entry, dict) or (ticker and (entry.get("ticker") or "").strip().upper() != ticker):
        entry = _entry_for_ticker(entries, ticker)
    if not isinstance(entry, dict):
        return None
    listed = (entry.get("ticker") or "").strip().upper()
    if not listed or (ticker and listed != ticker):
        return None
    venue = _CATALOGUE_EXCHANGES.get((entry.get("exchange") or "").strip().upper())
    if venue is None:
        return None
    mic, key = venue
    benchmark = None if mic == "OTCM" else "auto"
    return Instrument(f"{mic}:{listed}", listed, f"{listed.lower()}.us", key, mic, "stock", "US", "USD",
                      None, benchmark, source="provider_catalogue")


def resolve_instrument(ticker=None, name=None, asset_type=None):
    """
    The Instrument for a claim's asset, or None when the metadata cannot be
    resolved deterministically. `ticker` is the claim's resolved ticker (the
    curated alias tables are applied again), `name` the canonical entity
    name or spoken subject (for the verified learned map). `asset_type` only
    refuses a curated instrument whose kind contradicts an explicit claim
    type: a coin for a stock / ETF claim or a listing for a crypto claim (an
    "NVDA" the model calls crypto is not scored). Stock and ETF are one kind
    — both exchange listings, and speakers and the model mix them up ("RSP
    as an ETF" typed stock).
    """
    canonical = (canonical_ticker({"ticker": ticker, "name": name}) or "").upper()
    if not canonical or canonical in UNPRICEABLE_TICKERS:
        return None
    inst = REGISTRY.get(canonical) or _from_catalogue(name, canonical)
    if inst is None:
        return None
    kind = {"stock": "listing", "etf": "listing", "crypto": "coin"}
    if asset_type in kind and kind[asset_type] != kind.get(inst.asset_type):
        return None
    return inst


def benchmark_for(instrument, exchange, symbol=None):
    """
    (benchmark symbol, method) for an instrument, from its OWN metadata:
    an explicit per-instrument benchmark, no benchmark at all
    (no_defensible_benchmark), or the asset-type / country / sector table in
    scorecard_pricing. Never from a claim's sector field.
    """
    import scorecard_pricing as sp
    if instrument is None:
        return None, "unresolved_instrument"
    if instrument.benchmark is None:
        return None, "no_defensible_benchmark"
    if instrument.benchmark != "auto":
        if symbol and instrument.benchmark == symbol:
            return None, "self_benchmark"
        return instrument.benchmark, "instrument_override"
    return sp.resolve_benchmark(instrument.asset_type, exchange, instrument.sector, symbol or instrument.symbol)
