"""Ticker tables: the curated aliases, the canonical_ticker rules and the
instrument registry they must agree with (2026-10 audit of the canonical
claims nothing resolved)."""
import re

import pytest

import instruments
import scorecard_pricing as sp
import signals_data
import transcript_normalize as tn
from instruments import REGISTRY, display_label, resolve_instrument
from signals_data import ASSET_ALIASES, TICKER_ALIASES, UNPRICEABLE_TICKERS, canonical_ticker

# Common words, channel quirks and market subjects that must never become
# name aliases (every ASSET_ALIASES key is also matched in free text).
# Couch Investor's NEW/CORE and HKCM's XP are channel-scoped spellings; the
# market subjects are aggregated as subjects, never as one security; NICE and
# GRAB resolve through their registry rows and title tickers instead.
DENY_LIST = {
    "NEW", "NEW HOLDINGS", "IRON", "CORE", "COREE", "CORRE", "XP", "JOO PENG", "TARGET", "CIRCLE",
    "PROGRESSIVE", "EATEN", "IN VIDEO", "HYPE", "NICE", "GRAB", "GM", "BRAND", "COKE", "SONIC", "AX",
    "AVEX", "NAR", "MERCK", "BICO STOCK", "PRICE", "MARKET", "THE MARKET", "STOCKS", "GOLD", "SILVER",
    "OIL", "S&P 500", "NASDAQ", "DAX",
}


def test_every_alias_output_is_registered_or_unpriceable():
    # instruments' docstring promises the registry covers every ticker the
    # alias tables can produce; 34 outputs once had no row and their claims
    # silently fell out of the scorecard as unresolved_instrument.
    outputs = set(ASSET_ALIASES.values()) | set(TICKER_ALIASES.values())
    assert outputs - (set(REGISTRY) | UNPRICEABLE_TICKERS) == set()


def test_every_name_alias_resolves_to_its_own_value():
    # No key is shadowed by suffix stripping, a fold or the ticker rules.
    for key, ticker in ASSET_ALIASES.items():
        assert canonical_ticker({"name": key}) == TICKER_ALIASES.get(ticker, ticker), key


@pytest.mark.parametrize("name, ticker", [
    ("Fortinet", "FTNT"), ("McDonald's", "MCD"), ("Qualcomm", "QCOM"), ("Service Now", "NOW"),
    ("IonQ stock", "IONQ"), ("Marcado Libé", "MELI"), ("Google Alphabet", "GOOGL"), ("Polygon", "POL"),
    ("XRP", "XRP"), ("Space X", "SPCX"), ("SpaceX", "SPCX"), ("Canopy Growth-Aktie", "CGC"),
    ("Waste Management", "WM"), ("Eli Liy", "LLY"), ("The TradeDesk", "TTD"), ("Adlesian", "TEAM"),
    ("Sony Group", "SONY"), ("Barrick Mining Corporation", "B"), ("Fresnillo PLC", "FRESNILLO"),
    # the 2026-10-17 backlog extraction
    ("Duolingo", "DUOL"), ("Zeta Global", "ZETA"), ("Domino's", "DPZ"), ("Datadog", "DDOG"),
    ("Johnson and Johnson", "JNJ"), ("Intuit", "INTU"), ("Lam research", "LRCX"), ("Teladoc Health", "TDOC"),
    ("Duke Energy", "DUK"), ("Core Weave", "CRWV"), ("Robin Hood", "HOOD"), ("On Holding", "ONON"),
    ("Nokia", "NOK"), ("United Health Group", "UNH"), ("Palenteer", "PLTR"), ("Lamentum", "LITE"),
    ("Sandix", "SNDK"), ("Vertive", "VRT"), ("Moody's", "MCO"),
    # claims extracted 2026-10-13..17, and HKCM's spellings of Atlassian
    ("Lowe's", "LOW"), ("Taiwan Semi", "TSM"), ("GE Vernova", "GEV"), ("Workday", "WDAY"),
    ("Applied Digital", "APLD"), ("Comfort System", "FIX"), ("Adlesin", "TEAM"), ("Atläschen", "TEAM"),
])
def test_name_corrections_resolve(name, ticker):
    assert canonical_ticker({"name": name}) == ticker
    assert (resolve_instrument(ticker, name) is None) == (ticker in UNPRICEABLE_TICKERS)


def test_workday_is_a_name_only_in_capitals_or_before_stock():
    assert not tn.alias_mentions("after a long workday I check the charts")
    assert [m.group(1) for m in tn.alias_mentions("does workday stock look cheap")] == ["workday"]
    assert [m.group(1) for m in tn.alias_mentions("WORKDAY beat estimates")] == ["WORKDAY"]


def test_recorded_ticker_contradicting_a_curated_name_is_corrected():
    # Each record names another company: SY is So-Young (~$2.6) against a
    # speaker's $30.40 invalidation for Sony; GOLD is Barrick's pre-2025
    # ticker; CRWE is Crown Equity, a penny stock.
    assert canonical_ticker({"name": "Sony Corporation", "ticker": "SY"}) == "SONY"
    assert canonical_ticker({"name": "Barrick Mining Corporation", "ticker": "GOLD"}) == "B"
    assert canonical_ticker({"name": "CoreWeave", "ticker": "CRWE"}) == "CRWV"
    assert canonical_ticker({"name": "Rivian", "ticker": "RVN"}) == "RIVN"
    # Agreement is judged after folding; agreeing records are unchanged.
    assert canonical_ticker({"name": "Google", "ticker": "GOOG"}) == "GOOGL"
    assert canonical_ticker({"name": "Nvidia", "ticker": "NVDA"}) == "NVDA"


def test_a_recorded_ticker_the_tables_already_decided_is_kept():
    # A registered instrument or an unpriceable decision is not overridden by
    # a name; only a company's own former ticker is (GOLD -> B above).
    assert canonical_ticker({"name": "Nvidia", "ticker": "SPY"}) == "SPY"
    assert canonical_ticker({"name": "Nvidia", "ticker": "GOLD"}) == "GOLD"
    assert resolve_instrument("GOLD", "Nvidia", "commodity") is None


def test_index_names_are_never_tickers():
    for index in ("NASDAQ", "SPX", "DAX", "S&P 500", "NDX", "DXY"):
        assert canonical_ticker({"ticker": index, "name": index}) is None, index
    assert canonical_ticker({"ticker": "SPX", "name": "S&P 500"}) is None
    # The name still decides: the one listed company sharing an index name.
    assert canonical_ticker({"ticker": "DOW", "name": "Dow Jones"}) is None
    assert canonical_ticker({"ticker": "DOW", "name": "Dow Inc."}) == "DOW"
    assert resolve_instrument("DOW") is None and resolve_instrument("DOW", "Dow Inc.").mic == "XNYS"


def test_names_in_the_ticker_field_fold():
    assert canonical_ticker({"ticker": "XPENG"}) == "XPEV"
    assert canonical_ticker({"ticker": "MATIC"}) == "POL"
    assert canonical_ticker({"ticker": "KLA"}) == "KLAC"
    assert canonical_ticker({"ticker": "I-REN"}) == "IREN"
    assert canonical_ticker({"ticker": "SOLANA", "name": "Solana"}) == "SOL"


def test_name_shaped_ticker_aliases_are_names_and_extra_keys_are_ignored():
    assert canonical_ticker({"name": "SpaceX"}) == "SPCX"
    assert canonical_ticker({"name": "MasTec"}) == "MTZ"
    # Ticker-shaped keys are recorded-ticker fixes only: as names they are
    # other things ("Ruby" is not Rubrik).
    assert canonical_ticker({"name": "Ruby"}) is None
    assert canonical_ticker({"name": "Fortinet", "channel": "Parkev Tatevosian, CFA"}) == "FTNT"


def test_banks_resolve_as_subjects_but_are_not_free_text_mentions():
    # In "Goldman expects the stock to reach $200" the bank is the source; a
    # free-text match would hand "the stock" to Goldman.
    for name, ticker in (("Goldman", "GS"), ("Goldman Sachs", "GS"), ("JP Morgan", "JPM"),
                         ("Bank of America", "BAC"), ("Citigroup", "C"), ("Citi", "C")):
        assert canonical_ticker({"name": name}) == ticker, name
    assert not tn._ALIAS_RE.search("Goldman expects the stock to reach $200, says JP Morgan")
    # A ratings agency is quoted the same way.
    assert canonical_ticker({"name": "Moody's"}) == "MCO"
    assert not tn._ALIAS_RE.search("Moody's downgraded the US last week")


def test_curated_name_outranks_the_learned_map():
    signals_data.reset_learned_tickers({"FORTINET": "WRONG"})
    try:
        assert canonical_ticker({"name": "Fortinet"}) == "FTNT"
        assert canonical_ticker({"name": "Fortinet", "ticker": "FTNT"}) == "FTNT"
    finally:
        signals_data.reset_learned_tickers({})


def test_unpriceable_changes():
    for ticker in ("GOLD", "FRESNILLO", "IRNT", "000660.KS", "005930", "2454.TW", "688041.SS", "ETHBTC"):
        assert ticker in UNPRICEABLE_TICKERS, ticker
    # Dead entries: the alias table / learned map fold them to the ADRs first.
    assert "ADYEN" not in UNPRICEABLE_TICKERS and "BASF" not in UNPRICEABLE_TICKERS
    assert canonical_ticker({"ticker": "ADYEN"}) == "ADYEY" and resolve_instrument("ADYEN").mic == "OTCM"
    assert resolve_instrument("GOLD") is None
    assert resolve_instrument(None, "Fresnillo PLC") is None
    assert {"SK HYNIX", "ANTHROPIC", "OPENAI"} <= UNPRICEABLE_TICKERS


def test_common_words_and_market_subjects_are_not_alias_keys():
    name_shaped = {k for k in TICKER_ALIASES if not re.match(r"^[A-Z]{1,5}$", k)}
    assert DENY_LIST & set(ASSET_ALIASES) == set()
    assert DENY_LIST & name_shaped == set()
    for word in ("Target", "Gold", "Oil", "the market", "New", "Core", "Merck", "Coke"):
        assert canonical_ticker({"name": word}) is None, word


def test_registry_rows_and_display_names():
    wmt = resolve_instrument("WMT")
    assert wmt.mic == "XNAS" and wmt.instrument_id == "XNAS:WMT"   # Nasdaq since 2025-12-09
    nke = resolve_instrument("NKE")
    assert (nke.asset_type, nke.symbol, nke.sector) == ("stock", "nke.us", "consumer discretionary")
    xrp = resolve_instrument("XRP")
    assert (xrp.asset_type, xrp.symbol, xrp.exchange_key) == ("crypto", "xrpusd", "crypto")
    crwv = resolve_instrument(None, "CoreWeave stock")
    assert crwv.asset_type == "stock" and crwv.sector is None and crwv.mic == "XNAS"
    assert resolve_instrument(None, "Space X").symbol == "spcx.us"
    assert display_label("SPCX") == "SpaceX (SPCX)"
    assert display_label("PL") == "Planet Labs (PL)"
    assert display_label("NVDA") == "NVDA"
    assert display_label("ZZZZ") == "ZZZZ" and display_label(None) is None


def test_every_registry_row_has_a_calendar_and_short_tickers_have_names():
    for ticker, inst in REGISTRY.items():
        assert sp.exchange_for_instrument(inst) is not None, ticker
        if len(ticker) <= 2:
            assert inst.name, f"{ticker} needs a display name"
    for megacap in ("NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "NFLX"):
        assert REGISTRY[megacap].name is None
    # Positional construction is unchanged by the new trailing field.
    inst = instruments.Instrument("XNAS:ABC", "ABC", "abc.us", "us", "XNAS", "stock", "US", "USD", None)
    assert inst.name is None and inst.source == "curated"


def test_learned_entry_is_found_by_its_ticker(monkeypatch):
    signals_data.reset_learned_tickers({"ZORBLAX INDUSTRIES": "ZRBX"})
    monkeypatch.setattr(signals_data, "_LEARNED_ENTRIES",
                        {"ZORBLAX INDUSTRIES": {"ticker": "ZRBX", "exchange": "NASDAQ"}})
    inst = resolve_instrument("ZRBX", "Zorblax")
    assert inst.source == "provider_catalogue" and inst.instrument_id == "XNAS:ZRBX"
    assert resolve_instrument("ZRBX").instrument_id == "XNAS:ZRBX"
    # Entries that disagree on the venue answer nothing.
    monkeypatch.setattr(signals_data, "_LEARNED_ENTRIES",
                        {"ZORBLAX INDUSTRIES": {"ticker": "ZRBX", "exchange": "NASDAQ"},
                         "ZORBLAX": {"ticker": "ZRBX", "exchange": "NYSE"}})
    assert resolve_instrument("ZRBX", "Zorblax Holdings") is None
