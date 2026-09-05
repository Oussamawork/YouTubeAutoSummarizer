"""Hardening item 3: scorecard metadata comes from the canonical instrument
registry, never from a model-written sector or asset type."""
from datetime import date, timedelta

import instruments
import research_analytics as ra
import scorecard_pricing as sp
import signals_data

from tests.test_scorecard_pricing import JULY, _claim, _fetch


def test_curated_instrument_carries_every_field():
    inst = instruments.resolve_instrument("NVDA")
    assert inst.instrument_id == "XNAS:NVDA" and inst.symbol == "nvda.us" and inst.exchange_key == "us"
    assert (inst.asset_type, inst.country, inst.currency, inst.sector) == ("stock", "US", "USD", "technology")
    assert inst.source == "curated" and inst.benchmark == "auto"
    meta = inst.metadata()
    assert meta["instrument_sector"] == "technology" and meta["instrument_mic"] == "XNAS"
    btc = instruments.resolve_instrument("BTC")
    assert btc.exchange_key == "crypto" and btc.symbol == "btcusd" and sp.exchange_for_instrument(btc) is sp.CRYPTO
    assert instruments.resolve_instrument(None, "Nvidia").ticker == "NVDA"     # via the curated alias table
    assert instruments.resolve_instrument("GOOG").ticker == "GOOGL"            # share classes folded


def test_model_sector_never_selects_the_benchmark():
    # The claim says Tesla is "technology"; the registry says consumer
    # discretionary. The benchmark follows the registry.
    claim = _claim("2026-07-02T14:00:00+00:00", end="2026-07-20", ticker="TSLA", subject_mention="Tesla",
                   sector="technology")
    series = {d: 100.0 + i for i, d in enumerate(sorted(JULY))}
    sc = ra.scorecard([claim], date(2026, 8, 1), _fetch({"tsla.us": series, "xly.us": series, "xlk.us": series}),
                      min_sample=1)
    scored = sc["A"]["scored"][0]
    assert scored["benchmark_symbol"] == "xly.us" and scored["benchmark_method"] == "sector"
    assert scored["instrument_sector"] == "consumer discretionary" and scored["speaker_sector"] == "technology"
    assert scored["instrument_id"] == "XNAS:TSLA" and scored["instrument_currency"] == "USD"
    # A model asset_type that contradicts the registry is not scored either.
    wrong_type = _claim("2026-07-02T14:00:00+00:00", end="2026-07-20", ticker="TSLA", asset_type="crypto")
    assert ra.scorecard([wrong_type], date(2026, 8, 1), _fetch({}), min_sample=1)["_excluded"] == \
        {"unresolved_instrument": 1}


def test_unresolved_instrument_metadata_gives_null_not_a_guess():
    assert instruments.resolve_instrument("ZORB") is None
    assert instruments.resolve_instrument(None, "Zorblax Industries") is None
    assert instruments.resolve_instrument("OPENAI") is None   # unpriceable by the curated table
    assert instruments.benchmark_for(None, sp.EXCHANGES["us"]) == (None, "unresolved_instrument")
    claim = _claim("2026-07-02T14:00:00+00:00", end="2026-07-20", ticker="ZORB", subject_mention="Zorblax",
                   sector="technology")
    sc = ra.scorecard([claim], date(2026, 8, 1), _fetch({"zorb.us": {d: 1.0 for d in JULY}}), min_sample=1)
    assert sc["_excluded"] == {"unresolved_instrument": 1} and "A" not in sc


def test_uncurated_sector_falls_to_the_country_benchmark_and_otc_adrs_get_none():
    zeta = instruments.resolve_instrument("ZETA")
    assert zeta.sector is None
    assert instruments.benchmark_for(zeta, sp.EXCHANGES["us"]) == ("spy.us", "country")
    adr = instruments.resolve_instrument("BAESY")
    assert adr.mic == "OTCM" and adr.benchmark is None
    assert instruments.benchmark_for(adr, sp.EXCHANGES["us"]) == (None, "no_defensible_benchmark")
    spy = instruments.resolve_instrument("SPY")
    assert spy.asset_type == "etf" and instruments.benchmark_for(spy, sp.EXCHANGES["us"], "spy.us") == (None, "self_benchmark")


def test_provider_verified_ticker_map_supplies_exchange_but_no_sector(monkeypatch):
    signals_data.reset_learned_tickers({"AIRBNB": "ABNB"})
    monkeypatch.setattr(signals_data, "_LEARNED_ENTRIES",
                        {"AIRBNB": {"ticker": "ABNB", "exchange": "NASDAQ", "company": "Airbnb, Inc.", "via": "search"},
                         "MYSTERY CO": {"ticker": "MYST", "exchange": "TSX", "company": "Mystery", "via": "search"}})
    inst = instruments.resolve_instrument(None, "Airbnb")
    assert inst.instrument_id == "XNAS:ABNB" and inst.source == "provider_catalogue" and inst.sector is None
    assert instruments.benchmark_for(inst, sp.EXCHANGES["us"]) == ("spy.us", "country")
    # A venue the calendar rules do not cover is unresolved, not guessed as US.
    signals_data.reset_learned_tickers({"MYSTERY CO": "MYST"})
    monkeypatch.setattr(signals_data, "_LEARNED_ENTRIES",
                        {"MYSTERY CO": {"ticker": "MYST", "exchange": "TSX", "company": "Mystery", "via": "search"}})
    assert instruments.resolve_instrument(None, "Mystery Co") is None
