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

Anything not in the curated table and not in the verified learned map is
UNRESOLVED: the scorecard excludes the claim rather than guessing an
exchange or a benchmark. Sectors below follow the GICS classification as of
2026-09 for the primary US listing (ADRs/OTC listings are scored on their
US quote; their home-market index is not a defensible benchmark for a USD
quote, so those carry benchmark=None).
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

    def metadata(self):
        return {"instrument_id": self.instrument_id, "instrument_ticker": self.ticker,
                "instrument_symbol": self.symbol, "instrument_mic": self.mic,
                "instrument_asset_type": self.asset_type, "instrument_country": self.country,
                "instrument_currency": self.currency, "instrument_sector": self.sector,
                "instrument_benchmark_rule": self.benchmark, "instrument_source": self.source}


def _us(ticker, mic, sector, benchmark="auto"):
    return Instrument(f"{mic}:{ticker}", ticker, f"{ticker.lower()}.us", "us", mic, "stock", "US", "USD",
                      sector, benchmark)


def _etf(ticker, mic, sector=None):
    return Instrument(f"{mic}:{ticker}", ticker, f"{ticker.lower()}.us", "us", mic, "etf", "US", "USD", sector, "auto")


def _crypto(ticker):
    return Instrument(f"CRYPTO:{ticker}", ticker, f"{ticker.lower()}usd", "crypto", "CRYPTO", "crypto",
                      "GLOBAL", "USD", None, "auto")


# Curated registry: every ticker the alias tables in signals_data can
# produce. OTC ADRs carry benchmark=None (see the module docstring).
_CURATED = [
    # crypto (24/7, UTC-day close)
    _crypto("BTC"), _crypto("ETH"), _crypto("SOL"),
    # megacaps
    _us("NVDA", "XNAS", IT), _us("MSFT", "XNAS", IT), _us("AAPL", "XNAS", IT), _us("AMZN", "XNAS", CD),
    _us("META", "XNAS", COMM), _us("GOOGL", "XNAS", COMM), _us("TSLA", "XNAS", CD), _us("NFLX", "XNAS", COMM),
    # semis / hardware
    _us("MU", "XNAS", IT), _us("INTC", "XNAS", IT), _us("AMD", "XNAS", IT), _us("ASML", "XNAS", IT),
    _us("AVGO", "XNAS", IT), _us("MRVL", "XNAS", IT), _us("TSM", "XNYS", IT), _us("WDC", "XNAS", IT),
    _us("STX", "XNAS", IT), _us("COHR", "XNYS", IT), _us("NBIS", "XNAS", IT),
    # software / fintech / internet
    _us("PLTR", "XNAS", IT), _us("CRM", "XNYS", IT), _us("IBM", "XNYS", IT), _us("ADBE", "XNAS", IT),
    _us("SNOW", "XNYS", IT), _us("NOW", "XNYS", IT), _us("TEAM", "XNAS", IT), _us("ZS", "XNAS", IT),
    _us("TTD", "XNAS", COMM), _us("CRWD", "XNAS", IT), _us("PANW", "XNAS", IT), _us("SOFI", "XNAS", FIN),
    _us("PYPL", "XNAS", FIN), _us("XYZ", "XNYS", FIN), _us("RDDT", "XNYS", COMM), _us("ZETA", "XNYS", None),
    _us("AXON", "XNAS", IND), _us("MELI", "XNAS", CD), _us("UBER", "XNYS", IND), _us("NU", "XNYS", FIN),
    _us("BABA", "XNYS", CD), _us("ORCL", "XNYS", IT), _us("SHW", "XNYS", MAT), _us("RBRK", "XNYS", IT),
    _us("NCNO", "XNAS", IT), _us("MTZ", "XNYS", IND),
    # energy
    _us("OXY", "XNYS", ENE), _us("CVX", "XNYS", ENE), _us("XOM", "XNYS", ENE), _us("TTE", "XNYS", ENE),
    # other observed
    _us("WMT", "XNYS", CS), _us("HD", "XNYS", CD), _us("UAA", "XNYS", CD), _us("URI", "XNYS", IND),
    _us("PAAS", "XNYS", MAT), _us("SPCX", "XNAS", None),
    # broad and sector ETFs (a claim about SPY scores raw: SPY vs SPY is no benchmark)
    _etf("SPY", "ARCX"), _etf("VOO", "ARCX"), _etf("QQQ", "XNAS"), _etf("IWM", "ARCX"), _etf("DIA", "ARCX"),
    _etf("XLK", "ARCX", IT), _etf("XLF", "ARCX", FIN), _etf("XLE", "ARCX", ENE), _etf("XLV", "ARCX", HC),
    _etf("XLY", "ARCX", CD), _etf("XLP", "ARCX", CS), _etf("XLI", "ARCX", IND), _etf("XLU", "ARCX", UTIL),
    _etf("XLB", "ARCX", MAT), _etf("XLRE", "ARCX", RE), _etf("XLC", "ARCX", COMM),
    # OTC ADRs: priced on the US quote, no defensible USD benchmark
    _us("BAESY", "OTCM", IND, benchmark=None), _us("SFTBY", "OTCM", COMM, benchmark=None),
    _us("ADYEY", "OTCM", FIN, benchmark=None), _us("BASFY", "OTCM", MAT, benchmark=None),
]
REGISTRY = {inst.ticker: inst for inst in _CURATED}

# Listing-exchange names the provider catalogue (ticker_resolver) records,
# mapped to MIC + exchange key. Anything else is unresolved.
_CATALOGUE_EXCHANGES = {
    "NYSE": ("XNYS", "us"), "NASDAQ": ("XNAS", "us"), "NYSE ARCA": ("ARCX", "us"), "AMEX": ("XASE", "us"),
    "NYSE AMERICAN": ("XASE", "us"), "OTC": ("OTCM", "us"), "OTCM": ("OTCM", "us"), "CBOE": ("BATS", "us"),
}


def _from_catalogue(name, ticker):
    """An Instrument from ticker_resolver's verified map, or None. The map
    carries the listing exchange (verified against the provider's own
    catalogue) but no sector, so the benchmark is the country index."""
    if not name:
        return None
    entry = learned_ticker_entries().get(" ".join(name.split()).upper())
    if not isinstance(entry, dict):
        return None
    listed = (entry.get("ticker") or "").upper()
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
    refuses a curated instrument whose type contradicts an explicit stock /
    crypto claim (an "NVDA" the model calls crypto is not scored).
    """
    canonical = (canonical_ticker({"ticker": ticker, "name": name}) or "").upper()
    if not canonical or canonical in UNPRICEABLE_TICKERS:
        return None
    inst = REGISTRY.get(canonical) or _from_catalogue(name, canonical)
    if inst is None:
        return None
    if asset_type in ("stock", "etf", "crypto") and inst.asset_type != asset_type \
            and not (asset_type == "etf" and inst.asset_type == "stock"):
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
