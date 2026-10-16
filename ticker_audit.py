"""
Ticker audit: is every ticker, alias and cached price series the instrument
it claims to be?

A wrong ticker does not fail loudly. It prices another company: btc.us is
the Grayscale Bitcoin Mini Trust (~$30), not bitcoin; sy.us is So-Young
(~$2.6), not Sony; mogu.us is MOGU Inc., not the Moog a TheStreet guest
described. Every report that compares a creator's call with the market then
compares it with the wrong market. This audit finds those mistakes from the
committed data alone — claims, legacy signal rows, the price cache and the
learned ticker map — and applies NOTHING: it never edits a table, the cache
or the map. Its findings are a worklist for whoever curates
signals_data / instruments.

Sections, in report order (severity in brackets):

  alias_registry_gaps  [error]   an alias output with no instruments.REGISTRY
                                 row and no UNPRICEABLE decision: its claims
                                 drop out of the scorecard as unresolved
  known_wrong          [error]   a symbol known to be the wrong instrument
                                 (KNOWN_WRONG_SYMBOLS) that a current claim or
                                 legacy row still produces
                       [info]    one that is only cached: a prune candidate
  orphans              [info]    cached series nothing produces any more
                                 (benchmarks and the tracker's series count
                                 as producers)
  price_level          [warning] a price the speaker stated against the claim
                                 symbol's close: a 10x gap is a wrong ticker,
                                 never a market move
  registry_gaps        [info]    resolved tickers with no instrument: the next
                                 registry rows to curate
  unresolved_backlog   [info]    names with no ticker, most-discussed first,
                                 each with the ticker legacy rows paired with
                                 the same name (a hint, shown, never applied)
  learned_map          [error]   data/ticker_map.json entries whose listing's
                                 company does not match the key, or whose
                                 venue is not one instruments knows
  series_anomalies     [warning] implausible daily moves, flat runs, '.us'
                                 bars on market holidays, partial crypto
                                 candles, stale series a current claim needs
                       [info]    produced symbols with no cached series
  needs_verification   [info]    medium/low-confidence corrections that are
                                 NOT in any table, each with its evidence and
                                 the exact step that would settle it

Deterministic and offline: no network, no LLM. channel_scorecard and
ticker_resolver (which pull in requests) are imported lazily, inside the
checks that need them. Bad data is logged and skipped, never raised.

    python ticker_audit.py [--json PATH|-] [--strict] [--top N]
        [--cache data/prices.json] [--signals data/signals.jsonl]
        [--ticker-map data/ticker_map.json] [--today YYYY-MM-DD]

Exit code 0; with --strict, 1 when the report has errors. `--ticker-map` is
the file the learned-map section audits; ticker resolution itself always
reads signals_data's process-wide map.
"""
import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone

import canonical_claims as cc
import claims as claim_rules
import instruments
import price_cache
import scorecard_pricing as sp
import signals_data as sd
from helpers import write_json_atomic
from log import describe_error, log_error, log_info, log_warn

DEFAULT_TOP = 25            # lines per section in the text report
BACKLOG_TOP = 40            # names kept in the unresolved backlog
PRICE_BAND = (0.67, 1.5)    # stated price / cached close outside this is a warning
PRICE_LAG_DAYS = 5          # the close compared: first on or after publication, within this
MAX_DAILY_MOVE = 0.40       # a larger day-over-day move is implausible for what we track
FLAT_RUN_DAYS = 5           # this many identical closes in a row: a dead or illiquid quote
STALE_SERIES_DAYS = 10      # a series a current claim needs, older than this, needs warming
PARTIAL_RANGE_RATIO = 0.5   # a last bar whose high-low range is under this share of the usual one is partial
SNIPPET_CHARS = 160

SECTIONS = ("alias_registry_gaps", "known_wrong", "orphans", "price_level", "registry_gaps",
            "unresolved_backlog", "learned_map", "series_anomalies", "needs_verification")
REPORT_KEYS = ("today", "counts", "errors", "warnings") + SECTIONS

# Cached or producible symbols verified to be NOT the instrument the speaker
# meant (price level, daily-return correlation and the transcript; the
# 2026-10 price-symbol audit). They must never be priced for a claim.
KNOWN_WRONG_SYMBOLS = {
    "btc.us": "Grayscale Bitcoin Mini Trust, not bitcoin (btcusd)",
    "eth.us": "Grayscale Ethereum Mini Trust, not ether (ethusd)",
    "gold.us": "NYSE 'GOLD' is neither Barrick (NYSE: B since May 2025) nor the metal",
    "sy.us": "So-Young; Sony is SONY",
    "fr.us": "First Industrial Realty; Fresnillo is LSE: FRES",
    "rci.us": "Rogers; the Columbia ETF is RECS (unverified)",
    "mogu.us": "MOGU Inc.; TheStreet's guest meant Moog (MOG.A)",
    "crwe.us": "Crown Equity; CoreWeave is CRWV",
    "new.us": "no listing; Couch Investor's 'New' is Nu (NU)",
}

# The canonical market keys (the shared contract with canonical_claims and
# creator_tracker): subjects that are deliberately not tickers. They are
# compared with the market through a proxy ETF or not priced at all, so an
# unresolved one is by design, not a curation gap.
MARKET_KEYS = frozenset({
    # proxied
    "S&P 500", "US STOCK MARKET", "NASDAQ", "DOW", "RUSSELL 2000", "SMALL CAPS", "MID CAPS", "GOLD",
    "SILVER", "OIL", "GOLD MINERS", "TECHNOLOGY", "SEMICONDUCTORS", "SOFTWARE", "FINANCIALS", "ENERGY",
    "CONSUMER DISCRETIONARY", "CYBERSECURITY",
    # never priced
    "DAX", "NIKKEI 225", "DXY", "BITCOIN DOMINANCE", "USDT DOMINANCE", "COPPER", "FED", "INFLATION",
    "US TREASURIES", "ALTCOINS",
})

# Medium- and low-confidence corrections from the 2026-10 ticker audits.
# None of them is in a table: each needs the step in `verify` first.
# `claims` is the count at the audit (2026-10-16); the report adds the live
# count and current evidence (see _VERIFY_MATCH).
NEEDS_VERIFICATION = [
    {"key": "IRON (FinVid)", "proposed": "IREN, or Iron Mountain (IRM)?", "confidence": "medium", "claims": 4,
     "evidence": "FinVid 2026-09-09..10-13: 'bullish on Iron ... if the world is going to be compute "
                 "constrained through calendar 2028' and 'I've personally been adding to Iron' read as the "
                 "AI-infrastructure stock IREN, but FinVid's 'expected price range of 90 to $120 per share' is "
                 "2-2.6x iren.us (45.37 on 2026-09-09). Iron Mountain also runs data centres (irm.us is not "
                 "cached). The recorded 'IRON' is Disc Medicine, so any alias must be FinVid-scoped.",
     "verify": "python warm_prices.py --find 'IREN,Iron Mountain'; then --probe iren.us,irm.us and compare "
               "each 2026-09-09 close with the 90-120 range"},
    {"key": "XP (Phantom by HKCM)", "proposed": "XPEV, or XRP?", "confidence": "medium", "claims": 2,
     "evidence": "HKCM 2026-10-13 spells it out ('Joo Peng geschrieben. XP ... Elektromobilitätshersteller'): "
                 "XPeng (XPEV), not XP Inc., and HKCM's own 'XP' now resolves through claims.CHANNEL_ALIASES. "
                 "Phantom by HKCM 2026-10-14 ('Langfristig sind hier +1.200 % drin!') "
                 "gives 10.74-11.48 and a 3$ risk level: xpev.us closed 9.47 on 2026-09-29 and xrpusd 1.40 on "
                 "2026-10-11, so the levels fit XPeng, but Phantom has also said 'XP' for XRP (2026-09-20, "
                 "'Allzeithoch ... 3,67').",
     "verify": "Read the 2026-10-14 Phantom transcript around each claim and compare its levels with "
               "python warm_prices.py --probe xpev.us,xrpusd; resolve per claim, never with a global XP alias"},
    {"key": "MOGU", "proposed": "MOG.A (Moog Inc.)", "confidence": "medium", "claims": 1,
     "evidence": "TheStreet 2026-09-09 (Kevin Mahn): 'A company like move, ticker symbol Mogu ... space "
                 "propulsion, space vehicles and the defense side' describes Moog Inc. (NYSE: MOG.A / MOG.B). "
                 "mogu.us is MOGU Inc., a Chinese e-commerce penny stock (1.85 on 2026-09-09). The provider's "
                 "spelling of Moog's class-A share is unverified.",
     "verify": "python warm_prices.py --find 'Moog'; then --probe mog.a.us (twelvedata_symbol sends MOG.A)"},
    {"key": "COLUMBIA RESEARCH ENHANCED CORE ETF (legacy RCI)", "proposed": "RECS", "confidence": "medium",
     "claims": 1,
     "evidence": "A TheStreet 2026-08-04 legacy row records the Columbia Research Enhanced Core ETF as RCI, "
                 "which is Rogers Communications (rci.us 33.85 on 2026-08-04; its daily returns track VZ/T). "
                 "RECS as the fund's ticker comes from memory, not from the data.",
     "verify": "python warm_prices.py --find 'Columbia Research Enhanced Core'"},
    {"key": "SK HYNIX", "proposed": "its 2026 US listing (ticker not stated in the data)", "confidence": "medium",
     "claims": 6,
     "evidence": "FinVid 2026-09-15 lists 'SK Hynix's US listing' among this year's Nasdaq offerings; "
                 "2026-10-13 'American investors didn't know what SKHEX was prior to July'; 2026-10-16 'the one "
                 "that trades in the US, was at $195 10 days ago, now it's at 146 bucks'. It stays in "
                 "UNPRICEABLE_TICKERS until a listing is confirmed.",
     "verify": "python warm_prices.py --find 'SK Hynix'; accept only a US listing whose close is near $195 "
               "around 2026-10-06 or $146 on 2026-10-16 (python warm_prices.py --probe <symbol>)"},
    {"key": "ANTHROPIC", "proposed": "pending IPO (private today)", "confidence": "low", "claims": 10,
     "evidence": "Private today: TheStreet 2026-10-14 'Want Anthropic Exposure Before Its IPO?'; FinVid "
                 "2026-09-10 'probably going to go public ... end October'. Its UNPRICEABLE_TICKERS entry is "
                 "right now and will be wrong once it lists.",
     "verify": "After the listing: python warm_prices.py --find 'Anthropic', then replace the UNPRICEABLE "
               "entry with a registry row"},
    {"key": "CRONOS / CHRONOS / KRONOS", "proposed": "CRO", "confidence": "medium", "claims": 5,
     "evidence": "HKCM 2026-10-12 ('Neue Kaufzonen für diese Top 2 Altcoins!') quotes 58 cents, 80 cents and "
                 "1.10-1.15; Phantom by HKCM 2026-09-17 quotes 'Preisen um die 5 Cent'. The channels' levels "
                 "differ 10-20x, so at most one of them is Cronos (CRO); crousd is not cached.",
     "verify": "python warm_prices.py --probe crousd and compare with both channels' levels before any alias"},
    {"key": "NAR", "proposed": "NEAR", "confidence": "low", "claims": 2,
     "evidence": "More Crypto Online 2026-10-03 ('Bitcoin Above the May High'): 'Look on the near chart ... "
                 "resistance area, $569' and 'the next resistance is $7.77': probably NEAR Protocol with $5.69 "
                 "mis-transcribed; nearusd is not cached.",
     "verify": "python warm_prices.py --probe nearusd and compare its 2026-10-03 close with 5.69 / 7.77"},
    {"key": "TARGET (Parkev Tatevosian)", "proposed": "TGT", "confidence": "medium", "claims": 5,
     "evidence": "All five claims come from 'Target's Management Team Deserves Credit for Accelerating "
                 "Revenue Growth' (2026-09-26); 'its current market price for 160' fits tgt.us (158.43 on "
                 "2026-09-28). TARGET stays ambiguous globally (claims.AMBIGUOUS_MENTIONS).",
     "verify": "Resolve per video from the title (a title-scoped alias), confirmed by this audit's "
               "price-level check against tgt.us"},
    {"key": "BRAND", "proposed": "OIL (Brent)", "confidence": "medium", "claims": 3,
     "evidence": "More Crypto Online 2026-10-08: 'Brand is above $15 this morning. There were more attacks on "
                 "ships around the straight of Hormuse' is Brent crude (probably $115). HKCM 2026-10-14's "
                 "'BRAND' all-time high at 119.50 is less clear.",
     "verify": "Read the HKCM 2026-10-14 transcript; fold BRAND into OIL (canonical_claims.MACRO_ALIASES) "
               "only if both mean Brent"},
    {"key": "XLP (Phantom by HKCM, crypto)", "proposed": "XRP", "confidence": "medium", "claims": 2,
     "evidence": "Phantom by HKCM 2026-10-09 'Wir kaufen aggressiv XRP und Solana!' talks about 'Solana und "
                 "XLP' with a 98-cent support and 5.86-8.88 targets; xrpusd closed 1.40 on 2026-10-09. XLP is "
                 "also the consumer-staples ETF, so only a crypto-typed alias is safe.",
     "verify": "Alias crypto-typed claims only, after checking the 98-cent support with "
               "python warm_prices.py --probe xrpusd"},
    {"key": "AVEX / AX", "proposed": "AVAX", "confidence": "medium", "claims": 2,
     "evidence": "Phantom by HKCM 2026-10-11: 'Avex' with targets 13.32-15.52 reads as Avalanche (AVAX); "
                 "'AX' in the same video, with a 281 USD long-term target, could also be AAVE. Neither "
                 "avaxusd nor aaveusd is cached.",
     "verify": "python warm_prices.py --probe avaxusd,aaveusd and compare with 13.32-15.52 and 281"},
    {"key": "SONIC", "proposed": "S (Sonic, formerly Fantom)", "confidence": "low", "claims": 1,
     "evidence": "Phantom by HKCM 2026-10-11: 'bei 8 Cent in etwa dann Schluss' fits the Sonic token. Its "
                 "ticker S collides with SentinelOne (NYSE: S, a registry row), so it needs its own crypto key.",
     "verify": "python warm_prices.py --find 'Sonic'; then --probe susd before any alias"},
    {"key": "CENTAURI / CENTURY HOLDINGS", "proposed": "CTRI (Centuri Holdings)", "confidence": "medium",
     "claims": 2,
     "evidence": "TheStreet 2026-09-21: 'One stock you'd put fresh money into today ... Centauri Holdings' "
                 "and '... Century Holdings': one interview ingested twice (videos TFZYMNc8N1M and "
                 "NEcbhM7vfW4). Probably Centuri Holdings (NYSE: CTRI); unverified.",
     "verify": "python warm_prices.py --find 'Centuri Holdings'"},
    {"key": "MARATHON OIL", "proposed": "MPC (Marathon Petroleum)", "confidence": "medium", "claims": 1,
     "evidence": "TheStreet 2026-10-09: 'a Phillips 66, Valero or Marathon Oil', named with the refiners PSX "
                 "and VLO. Marathon Oil was acquired by ConocoPhillips in 2024, so the speaker likely meant "
                 "Marathon Petroleum.",
     "verify": "python warm_prices.py --find 'Marathon Petroleum,Marathon Oil'"},
    {"key": "COIN SHARES BITCOIN MINING AND DIGITAL POWER ETF", "proposed": "WGMI", "confidence": "medium",
     "claims": 2,
     "evidence": "HKCM GLOBAL 2026-09-24/29 ('The AI Trade Nobody Is Talking About'). The CoinShares Bitcoin "
                 "Mining ETF (formerly Valkyrie) trades as WGMI; the 'Digital Power' name is unverified.",
     "verify": "python warm_prices.py --find 'CoinShares Bitcoin Mining'"},
    {"key": "CIPHER MINING", "proposed": "CIFR", "confidence": "medium", "claims": 1,
     "evidence": "FinVid 2026-10-14: 'terowolf cipher mining did a dilution today but these are at very "
                 "attractive prices'. Nasdaq: CIFR, unless the company renamed in 2026.",
     "verify": "python warm_prices.py --find 'Cipher Mining'"},
    {"key": "ADIDAS", "proposed": "ADDYY (USD ADR) or unpriceable", "confidence": "medium", "claims": 2,
     "evidence": "HKCM 2026-09-10 trades Adidas on Xetra (ADS, in EUR: 'wir steigen bei Adidas aus, Profit "
                 "3,62 %'). The OTC ADR ADDYY is only a USD proxy for that position.",
     "verify": "Decide the ADR policy; python warm_prices.py --find 'Adidas' for the ADR's venue"},
    {"key": "COKE", "proposed": "KO", "confidence": "medium", "claims": 1,
     "evidence": "TheStreet 2026-10-08: 'Coke is better on company, so if I had to buy one of them, I buy "
                 "Coke' is The Coca-Cola Company (ko.us 87.77 on 2026-10-08). COKE is also Coca-Cola "
                 "Consolidated's ticker, so a global alias is unsafe.",
     "verify": "Resolve the claim from its evidence and title; never a global COKE alias"},
    {"key": "BICO STOCK", "proposed": "FICO", "confidence": "medium", "claims": 1,
     "evidence": "Parkev Tatevosian 2026-10-01 'Massive News for FICO Stock Investors...': 'BICO stock looks "
                 "undervalued with roughly 18% upside' is FICO mis-transcribed (fico.us 661.75 on 2026-10-01).",
     "verify": "A title-scoped resolution ('FICO' in the title); no global BICO alias"},
    {"key": "HYPE", "proposed": "HYPE (Hyperliquid)", "confidence": "low", "claims": 1,
     "evidence": "More Crypto Online 2026-10-06 ('Bitcoin Shows Strength'): spoken HYPE is Hyperliquid, but "
                 "the quoted support 'between $9067 and $93' fits neither coin cleanly (btcusd 85,550 on "
                 "2026-10-06; hypeusd is not cached). 'hype' is also a common word.",
     "verify": "Read the transcript; alias crypto-typed claims only, after python warm_prices.py --probe hypeusd"},
    {"key": "PROGRESSIVE (TheStreet)", "proposed": "PGR", "confidence": "medium", "claims": 1,
     "evidence": "TheStreet 2026-09-15 ('The Biggest Risk Is Not Being Risky Enough'): subject 'Progressive' "
                 "with the model's PGR (Progressive Corp., NYSE: PGR): 'I like that name a lot ... also a growth "
                 "perspective'. The evidence never names the insurer, and 'progressive' is an ordinary word, so "
                 "a global alias is unsafe.",
     "verify": "Read the transcript; then a TheStreet-scoped alias (claims.CHANNEL_ALIASES), never a global one"},
    {"key": "VOYAGER", "proposed": "VOYG (Voyager Technologies)", "confidence": "medium", "claims": 2,
     "evidence": "Couch Investor 2026-09-29 ('Why This AI Loser Is Actually a Massive Buy'): 'could this "
                 "become a 10 billion company over the next couple of years'. Voyager Technologies "
                 "(space/defence, NYSE: VOYG) fits; Voyager Digital (a bankrupt crypto broker) does not.",
     "verify": "Read the transcript; then python warm_prices.py --find 'Voyager Technologies'"},
    {"key": "CIRCLE", "proposed": "CRCL", "confidence": "medium", "claims": 1,
     "evidence": "Phantom by HKCM 2026-10-02 ('Diese Aktien werden MASSIV vom Krypto Hype profitieren!'): "
                 "'Neben Circle, den wir jetzt mit in die Phantom DNA aufgenommen haben'. Circle Internet "
                 "Group listed in June 2025 (NYSE: CRCL) but was typed private_company; 'circle' is a common "
                 "word.",
     "verify": "python warm_prices.py --find 'Circle Internet'; then an exact-subject, stock-typed alias"},
    {"key": "DPZ (venue)", "proposed": "confirm the registry row's MIC (XNAS) for Domino's Pizza",
     "confidence": "medium", "claims": 9,
     "evidence": "Parkev Tatevosian 2026-10-11 'Domino's Analysis: Value Stock or Value Trap? | DPZ Stock "
                 "Analysis': the ticker is right. The registry row (added 2026-10-16) carries XNAS; NYSE and "
                 "Nasdaq share one exchange in scorecard_pricing, so the symbol (dpz.us), calendar and benchmark "
                 "do not depend on it, only the instrument id does.",
     "verify": "python warm_prices.py --find \"Domino's Pizza\" and correct the row's MIC if it reports NYSE"},
    {"key": "klac.us / mrna.us / aph.us / app.us", "proposed": "confirm the corporate actions behind the levels",
     "confidence": "low", "claims": 2,
     "evidence": "klac.us at 171-209 implies a 5-10:1 split since 2025 (KLA traded above $1,000); mrna.us at "
                 "136-204 is about 6x its late-2025 level; aph.us at 77-85 and app.us at 299-339 are "
                 "confirmed by no spoken price.",
     "verify": "python warm_prices.py --probe klac.us,mrna.us,aph.us,app.us and record any confirmed split "
               "beside the registry rows"},
    {"key": "psp.us / cdpi.us", "proposed": "confirm the fund identities", "confidence": "low", "claims": 0,
     "evidence": "psp.us (TheStreet 2026-08-25, 'PSP' ETF) looks like the Invesco Global Listed Private Equity "
                 "ETF (daily returns correlate 0.84 with KKR); cdpi.us ('Columbia High Dividend Premium Income "
                 "ETF', TheStreet 2026-08-04) is a thin series (20.1-20.5, unchanged closes).",
     "verify": "python warm_prices.py --find 'Invesco Global Listed Private Equity,Columbia High Dividend "
               "Premium Income'"},
    {"key": "TTD / NBIS sectors", "proposed": "confirm the GICS sectors (registry: TTD communication services, "
                                              "NBIS technology)", "confidence": "low", "claims": 28,
     "evidence": "The registry's sector picks each claim's benchmark (xlc.us vs xlk.us); neither TTD's nor "
                 "NBIS's GICS sector could be confirmed offline.",
     "verify": "Check each company's GICS classification and update instruments._CURATED"},
    {"key": "THE MARKET / STOCKS (TheStreet, Couch Investor)", "proposed": "US STOCK MARKET (proxy SPY)",
     "confidence": "medium", "claims": 23,
     "evidence": "Bare 'the market', 'markets' and 'stocks' in TheStreet's and Couch Investor's US-equity "
                 "interviews are the overall US market and should be compared with SPY. In More Crypto Online "
                 "'the market' is the crypto market and must stay a placeholder.",
     "verify": "Map per channel through canonical_claims.asset_key (US-equity channels -> US STOCK MARKET; "
               "crypto channels stay placeholders), then check the subjects again here"},
]

# How each needs-verification entry finds its claims in the current data:
# subject names (normalized, suffix-stripped), tickers (recorded or
# resolved), symbols, and an optional channel filter. Legacy rows are
# matched the same way.
_VERIFY_MATCH = {
    "IRON (FinVid)": {"names": {"IRON"}, "channels": {"FinVid"}},
    "XP (Phantom by HKCM)": {"names": {"XP"}, "tickers": {"XP"}, "channels": {"Phantom by HKCM"}},
    "MOGU": {"names": {"MOGU"}, "tickers": {"MOGU"}, "symbols": {"mogu.us"}},
    "COLUMBIA RESEARCH ENHANCED CORE ETF (legacy RCI)": {"names": {"COLUMBIA RESEARCH ENHANCED CORE ETF"},
                                                         "tickers": {"RCI"}, "symbols": {"rci.us"}},
    "SK HYNIX": {"names": {"SK HYNIX", "HINX", "SK HINIX", "SKHEX"}},
    "ANTHROPIC": {"names": {"ANTHROPIC"}},
    "CRONOS / CHRONOS / KRONOS": {"names": {"CRONOS", "CHRONOS", "KRONOS"}, "tickers": {"CRO"}},
    "NAR": {"names": {"NAR"}},
    "TARGET (Parkev Tatevosian)": {"names": {"TARGET"}, "tickers": {"TGT"}},
    "BRAND": {"names": {"BRAND"}},
    "XLP (Phantom by HKCM, crypto)": {"names": {"XLP"}, "channels": {"Phantom by HKCM"}},
    "AVEX / AX": {"names": {"AVEX", "AX"}},
    "SONIC": {"names": {"SONIC"}},
    "CENTAURI / CENTURY HOLDINGS": {"names": {"CENTAURI HOLDINGS", "CENTURY HOLDINGS", "CENTURI HOLDINGS"}},
    "MARATHON OIL": {"names": {"MARATHON OIL"}},
    "COIN SHARES BITCOIN MINING AND DIGITAL POWER ETF": {
        "names": {"COIN SHARES BITCOIN MINING AND DIGITAL POWER ETF", "COINSHARES BITCOIN MINING ETF"}},
    "CIPHER MINING": {"names": {"CIPHER MINING"}, "tickers": {"CIFR"}},
    "ADIDAS": {"names": {"ADIDAS"}},
    "COKE": {"names": {"COKE"}},
    "BICO STOCK": {"names": {"BICO"}},
    "HYPE": {"names": {"HYPE", "HYPERLIQUID"}},
    "PROGRESSIVE (TheStreet)": {"names": {"PROGRESSIVE"}, "channels": {"TheStreet"}},
    "VOYAGER": {"names": {"VOYAGER"}},
    "CIRCLE": {"names": {"CIRCLE"}, "tickers": {"CRCL"}},
    "DPZ (venue)": {"tickers": {"DPZ"}},
    "klac.us / mrna.us / aph.us / app.us": {"symbols": {"klac.us", "mrna.us", "aph.us", "app.us"}},
    "psp.us / cdpi.us": {"symbols": {"psp.us", "cdpi.us"}},
    "TTD / NBIS sectors": {"tickers": {"TTD", "NBIS"}},
    "THE MARKET / STOCKS (TheStreet, Couch Investor)": {
        "names": {"THE MARKET", "MARKET", "MARKETS", "STOCKS", "STOCK MARKET", "THE STOCK MARKET", "U.S. STOCKS",
                  "US STOCKS", "US STOCK MARKET"},
        "channels": {"TheStreet", "Couch Investor"}},
}

_FALLBACK_SUFFIXES = (" STOCK", " STOCKS", " SHARES", " SHARE", " AKTIE", " AKTIEN", "-AKTIE", "-AKTIEN",
                      " INC", " INC.", " CORP", " CORP.", " CORPORATION", " PLC")
_PLURAL_SUFFIXES = (" STOCKS", " AKTIEN", "-AKTIEN")
# Extractor asset types that name one instrument (a ticker can fix the
# claim); sector / macro / bond / commodity / index / other are themes.
_INSTRUMENT_TYPES = {"stock", "etf", "crypto", "private_company"}
_US_TICKER_SHAPE = re.compile(r"^[A-Z]{1,5}(?:\.[A-Z])?$")
_NON_PRICE_UNITS = {"x", "×", "times", "multiple", "percent", "%", "bps", "basis_points", "ratio"}

# Stated-price phrases. The number must follow the phrase directly, an
# approximation word or a "$" aside: 'current (market) price is/of $X',
# 'trading at $X', 'now at $X', 'shares at $X' and close variants.
_NUMBER = r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_STATED_PRICE = re.compile(
    r"\b(?:current(?:ly)?\s+(?:market\s+)?(?:share\s+|stock\s+)?price\s+(?:is|of|for|at)"
    r"|trading\s+at|trades\s+at|currently\s+at|now\s+at|shares\s+at)"
    r"\s+(?:(?:about|around|roughly|approximately|just|only|nearly|near|almost)\s+)?"
    r"(?:(?:below|under|above|over)\s+)?(?:US)?\$?\s?" + _NUMBER + r"(?P<k>\s?[kK]\b)?", re.I)
# A number followed by these is a multiple, a share or a size, not a price.
_NOT_A_PRICE_AFTER = re.compile(
    r"^\s*(?:[x×](?![a-z])|times\b|p/?e\b|%|percent|per\s?cent|billion|bn\b|million|mn\b|trillion|multiple"
    r"|(?:-|to|or)?\s*\d+(?:\.\d+)?\s*(?:[x×](?![a-z])|times\b))", re.I)
# Right after the number: it is a valuation, not a quote ("well below the
# current market price at 175.74 of my fair value").
_VALUATION_AFTER = re.compile(r"^\s*(?:is\s+|as\s+)?(?:of\s+)?(?:my|the|our|his|their|a)\s+"
                              r"(?:fair|intrinsic|target)\b", re.I)
# Earlier in the evidence: a valuation multiple is being discussed ("on a
# forward price to earnings basis, the stock is trading at 16.4").
_MULTIPLE_CONTEXT = re.compile(
    r"price[- ]to[- ](?:earnings|sales|book|free cash)|\bp/?e\b|\bmultiples?\b|times (?:earnings|sales)"
    r"|\bev/?ebitda\b|\bvaluation\b", re.I)
# Just before the phrase: a historical, own-cost or hoped-for entry price,
# not today's ("when it was trading at 175", "I'd buy shares at $80").
_HISTORICAL_CONTEXT = re.compile(
    r"\b(?:was|were|had been|used to|back in|back when|when|bought|sold|added|purchased|paid|buy|buying|would"
    r"|if)\b", re.I)

_warned = set()


def _warn_once(key, message):
    if key not in _warned:
        _warned.add(key)
        log_warn(message)


# --- Small helpers -------------------------------------------------------------------------------


def _norm(text):
    return " ".join(str(text or "").split()).upper()


def _suffixes():
    """signals_data's name suffixes without the plurals: "IonQ stock" is one
    company, "growth stocks" a category (canonical_claims.asset_key groups
    the same way)."""
    found = getattr(sd, "_NAME_SUFFIXES", None) or _FALLBACK_SUFFIXES
    return sorted((x for x in found if x not in _PLURAL_SUFFIXES), key=len, reverse=True)


def _strip_suffix(name):
    """"IONQ STOCK" -> "IONQ", "CANOPY GROWTH-AKTIE" -> "CANOPY GROWTH": one
    trailing word speakers attach to a name."""
    for suffix in _suffixes():
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)].strip(" -")
    return name


def claim_name(claim):
    """The claim's subject (resolved name, else the spoken one), single-spaced
    and upper case."""
    return _norm(claim.get("canonical_entity_name") or claim.get("subject_mention"))


def _date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value or "").strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _snippet(text, limit=SNIPPET_CHARS):
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _is_view(claim):
    return bool(claim_rules.carries_view(claim))


def _is_directional(claim):
    return _is_view(claim) and cc.direction_of(claim) in ("bullish", "bearish")


def _is_crypto_symbol(symbol):
    return "." not in symbol and symbol.endswith("usd")


def _market_keys():
    """MARKET_KEYS plus any key claims.MARKET_SUBJECTS has grown since."""
    return MARKET_KEYS | set((getattr(claim_rules, "MARKET_SUBJECTS", None) or {}).values())


def _rows(rows, what):
    """Iterate `rows`, logging (once) and skipping any that are not dicts."""
    for row in rows or ():
        if isinstance(row, dict):
            yield row
        else:
            _warn_once(f"row:{what}", f"Ticker audit: skipped a malformed {what} row ({type(row).__name__}).")


def _skip(what, error):
    _warn_once(f"skip:{what}", f"Ticker audit: skipped a {what} it could not read: {describe_error(error)}")


def _claim_example(claim):
    return {"kind": "canonical", "claim_id": claim.get("claim_id"), "channel": cc.source_of(claim),
            "date": cc.claim_date(claim), "name": claim.get("canonical_entity_name") or claim.get("subject_mention"),
            "ticker": claim.get("ticker"), "asset_type": claim.get("asset_type"),
            "evidence": _snippet(claim.get("evidence_text"))}


def _legacy_assets(records):
    """(record, asset) for every valid legacy asset, as the legacy jobs see
    them (signals_data._iter_assets); a row that cannot be read is skipped."""
    out = []
    for record in _rows(records, "legacy signal"):
        try:
            out.extend(sd._iter_assets([record]))
        except Exception as e:  # bad data must never stop the audit
            _skip("legacy signal row", e)
    return out


def _legacy_example(record, asset):
    return {"kind": "legacy", "video_id": record.get("video_id"), "channel": record.get("channel_name"),
            "date": _date(record.get("date")), "name": asset.get("name"), "ticker": asset.get("ticker"),
            "asset_type": asset.get("type")}


# --- Symbols -------------------------------------------------------------------------------------


def claim_symbol(claim):
    """
    The internal price symbol a canonical claim produces: its instrument's
    symbol (instruments.resolve_instrument over the resolved ticker), else
    canonical_claims.priceable_symbol — the claim-typed fallback that the
    warm job and the pulse use, and that produced mogu.us.
    """
    name = claim.get("canonical_entity_name") or claim.get("subject_mention")
    inst = instruments.resolve_instrument(cc.resolved_ticker(claim), name, claim.get("asset_type"))
    if inst is not None:
        return inst.symbol
    return cc.priceable_symbol(claim)


def legacy_symbol(asset, channel=None):
    """The symbol a legacy signals.jsonl asset produces
    (channel_scorecard.symbol_for, with the row's channel: Couch Investor's
    "New" is nu.us, never new.us)."""
    try:
        import channel_scorecard  # lazily: it imports requests and the Telegram sender
    except Exception as e:  # pragma: no cover - a broken environment, not bad data
        _warn_once("import:channel_scorecard",
                   f"Ticker audit: legacy symbols unavailable (channel_scorecard: {describe_error(e)}).")
        return None
    return channel_scorecard.symbol_for(asset, channel)


def alias_registry_gaps():
    """
    Alias outputs (ASSET_ALIASES and TICKER_ALIASES values) with neither an
    instruments.REGISTRY row nor an UNPRICEABLE_TICKERS entry. Each one is an
    error: instruments promises a row for every alias output, and a claim on
    such a ticker drops out of the scorecard as unresolved_instrument.
    """
    produced = defaultdict(list)
    for table_name in ("ASSET_ALIASES", "TICKER_ALIASES"):
        for key, value in (getattr(sd, table_name, None) or {}).items():
            if isinstance(value, str) and value.strip():
                produced[value.strip().upper()].append(f"{table_name}[{key!r}]")
    registry, unpriceable = instruments.REGISTRY, sd.UNPRICEABLE_TICKERS
    return [{"ticker": ticker, "aliases": sorted(keys), "severity": "error"}
            for ticker, keys in sorted(produced.items()) if ticker not in registry and ticker not in unpriceable]


def _reference_symbols(claims, today):
    """
    {symbol: {label: n}} for series something other than a claim's own
    instrument needs: each claim instrument's benchmark ("benchmark"), every
    benchmark-table symbol ("benchmark table", 0), and the creator tracker's
    planned requests ("creator tracker"). They are producers too, so a
    benchmark-only series is never called an orphan.
    """
    ref = defaultdict(dict)
    for symbol in set(sp.BENCHMARKS.values()):
        ref[symbol]["benchmark table"] = 0
    for c in _rows(claims, "claim"):
        try:
            name = c.get("canonical_entity_name") or c.get("subject_mention")
            inst = instruments.resolve_instrument(cc.resolved_ticker(c), name, c.get("asset_type"))
            if inst is None:
                continue
            bench, _ = instruments.benchmark_for(inst, sp.exchange_for_instrument(inst), inst.symbol)
        except Exception as e:  # bad data must never stop the audit
            _skip("claim", e)
            continue
        if bench:
            ref[bench]["benchmark"] = ref[bench].get("benchmark", 0) + 1
    try:
        import creator_tracker  # optional: the tracker's series are producers when it is installed
        for symbol, _start, _end in creator_tracker.needed_price_ranges(list(_rows(claims, "claim")), today):
            ref[symbol]["creator tracker"] = 1
    except Exception as e:
        _warn_once("tracker", f"Ticker audit: creator tracker needs not counted ({describe_error(e)}).")
    return dict(ref)


def symbol_trace(claims, records, cache, reference=None):
    """
    {symbol: {producers: {normalized name: count}, canonical: n, legacy: n,
    directional: n, reference: {label: n}, cached: bool, from, to,
    last_date, last_close, examples: [...]}} for every symbol a canonical
    claim (claim_symbol) or a legacy asset (legacy_symbol) produces, every
    reference series and every cached symbol. `directional` counts the
    canonical claims that are bullish or bearish views: the calls a report
    marks to market. A cached symbol with no producer and no reference is
    an orphan: nothing will ever read it.
    """
    trace = {}

    def slot(symbol):
        return trace.setdefault(symbol, {"producers": Counter(), "canonical": 0, "legacy": 0, "directional": 0,
                                         "reference": {}, "examples": []})

    for c in _rows(claims, "claim"):
        try:
            symbol = claim_symbol(c)
            directional = _is_directional(c) if symbol else False
        except Exception as e:  # bad data must never stop the audit
            _skip("claim", e)
            continue
        if not symbol:
            continue
        entry = slot(symbol)
        entry["producers"][claim_name(c) or "?"] += 1
        entry["canonical"] += 1
        entry["directional"] += directional
        if len(entry["examples"]) < 3:
            entry["examples"].append(_claim_example(c))
    for record, asset in _legacy_assets(records):
        try:
            symbol = legacy_symbol(asset, record.get("channel_name"))
        except Exception as e:  # bad data must never stop the audit
            _skip("legacy asset", e)
            continue
        if not symbol:
            continue
        entry = slot(symbol)
        entry["producers"][_norm(asset.get("name")) or "?"] += 1
        entry["legacy"] += 1
        if len(entry["examples"]) < 3:
            entry["examples"].append(_legacy_example(record, asset))
    for symbol, labels in (reference or {}).items():
        slot(symbol)["reference"].update(labels or {})
    for symbol in cache or {}:
        slot(symbol)
    for symbol, entry in trace.items():
        cached = (cache or {}).get(symbol)
        closes = (cached or {}).get("closes") or {}
        last = max(closes) if closes else None
        entry.update({
            "producers": dict(entry["producers"].most_common()), "cached": bool(cached),
            "from": (cached or {}).get("from"), "to": (cached or {}).get("to"),
            "last_date": last, "last_close": closes.get(last) if last else None,
        })
    return dict(sorted(trace.items()))


def known_wrong_findings(trace):
    """
    One row per KNOWN_WRONG_SYMBOLS entry: an error when a current claim or a
    legacy row still produces the symbol (it will be priced again), info
    when it is only cached (prune it), ok ("clear") when it is gone.
    """
    out = []
    for symbol, why in sorted(KNOWN_WRONG_SYMBOLS.items()):
        entry = (trace or {}).get(symbol) or {}
        canonical, legacy = entry.get("canonical", 0), entry.get("legacy", 0)
        if canonical or legacy:
            severity, status = "error", "produced"
        elif entry.get("cached"):
            severity, status = "info", "cached_only"
        else:
            severity, status = "ok", "clear"
        out.append({"symbol": symbol, "why": why, "status": status, "severity": severity,
                    "canonical": canonical, "legacy": legacy, "producers": entry.get("producers") or {},
                    "examples": entry.get("examples") or [], "cached": bool(entry.get("cached")),
                    "from": entry.get("from"), "to": entry.get("to"), "last_close": entry.get("last_close")})
    return out


def orphan_symbols(trace):
    """Cached symbols nothing produces or references: prune candidates."""
    return [{"symbol": symbol, "from": e.get("from"), "to": e.get("to"), "last_date": e.get("last_date"),
             "last_close": e.get("last_close"), "known_wrong": KNOWN_WRONG_SYMBOLS.get(symbol), "severity": "info"}
            for symbol, e in sorted((trace or {}).items())
            if e.get("cached") and not e.get("canonical") and not e.get("legacy") and not e.get("reference")]


# --- Stated prices -------------------------------------------------------------------------------


def _stated_values(claim):
    """[(source, value)] for the current prices a claim states: a price
    claim's baseline_value, and evidence phrases like 'trading at $X'."""
    unit = str(claim.get("target_unit") or "").strip().lower()
    if unit in _NON_PRICE_UNITS or claim.get("currency") not in (None, "USD"):
        return []   # a multiple, or a price in another currency than the symbol's
    found = []
    base = claim.get("baseline_value")
    if (isinstance(base, (int, float)) and not isinstance(base, bool) and base > 0
            and claim.get("forecast_metric") in (None, "price")):
        found.append(("baseline", float(base)))
    text = str(claim.get("evidence_text") or "")
    for m in _STATED_PRICE.finditer(text):
        after = text[m.end("num"):m.end("num") + 32]
        if _NOT_A_PRICE_AFTER.match(after) or _VALUATION_AFTER.match(after) or re.match(r",\d", after):
            continue  # a multiple, a size, a valuation, or a mangled decimal comma ("$1,43")
        if _MULTIPLE_CONTEXT.search(text[:m.start()]):
            continue
        if _HISTORICAL_CONTEXT.search(text[max(0, m.start() - 40):m.start()]):
            continue
        try:
            value = float(m.group("num").replace(",", ""))
        except ValueError:
            continue
        if m.group("k"):
            value *= 1000
        if value > 0:
            found.append(("evidence", value))
    seen, out = set(), []
    for source, value in found:
        if value not in seen:
            seen.add(value)
            out.append((source, value))
    return out


def _close_on_or_after(entry, day, lag=PRICE_LAG_DAYS):
    closes = (entry or {}).get("closes") or {}
    for offset in range(lag + 1):
        d = day + timedelta(days=offset)
        if d in closes:
            return d, closes[d]
    return None, None


def price_level_checks(claims, cache, band=PRICE_BAND):
    """
    Prices the speakers stated ("the current market price is $262",
    "trading at $42.64", a price claim's baseline_value) against the claim
    symbol's first cached close on or after publication (within
    PRICE_LAG_DAYS). A ratio outside `band` is a warning: a wrong ticker
    shows up as a 10x gap (So-Young at $2.6 for a Sony at $24), a market
    move never does. Multiples ("trading at 14 times earnings"), shares,
    sizes and historical prices are skipped.

    {"band", "checked", "in_band", "symbols", "corroborated",
     "contradicted", "no_close", "outliers": [...]} — corroborated symbols
    had only in-band statements, contradicted ones only outliers.
    """
    lo, hi = band
    out = {"band": [lo, hi], "checked": 0, "in_band": 0, "symbols": 0, "corroborated": [], "contradicted": [],
           "no_close": 0, "outliers": []}
    good, bad = Counter(), Counter()
    for c in _rows(claims, "claim"):
        try:
            values = _stated_values(c)
            if not values:
                continue
            symbol, published = claim_symbol(c), cc.claim_date(c)
        except Exception as e:  # bad data must never stop the audit
            _skip("claim", e)
            continue
        if not symbol or published is None:
            out["no_close"] += len(values)
            continue
        day, close = _close_on_or_after((cache or {}).get(symbol), published)
        if not close:
            out["no_close"] += len(values)
            continue
        for source, value in values:
            ratio = value / close
            out["checked"] += 1
            if lo <= ratio <= hi:
                out["in_band"] += 1
                good[symbol] += 1
                continue
            bad[symbol] += 1
            out["outliers"].append({
                "claim_id": c.get("claim_id"), "symbol": symbol, "ticker": cc.resolved_ticker(c),
                "name": c.get("canonical_entity_name") or c.get("subject_mention"), "source": source,
                "stated": value, "close": round(close, 4), "close_date": day, "ratio": round(ratio, 3),
                "channel": cc.source_of(c), "published": published,
                "evidence": _snippet(c.get("evidence_text")), "severity": "warning"})
    out["symbols"] = len(set(good) | set(bad))
    out["corroborated"] = sorted(s for s in good if not bad[s])
    out["contradicted"] = sorted(s for s in bad if not good[s])
    out["outliers"].sort(key=lambda r: (-abs(math.log(r["ratio"])) if r["ratio"] > 0 else 0, r["claim_id"] or ""))
    return out


# --- Tickers without an instrument, names without a ticker ------------------------------------------


def registry_gaps(claims, cache=None):
    """
    Resolved tickers with no instrument (not in the registry, not in the
    verified learned map) and not unpriceable: the next registry rows to
    curate, most view claims first. A gap whose symbol is a known-wrong one
    (NEW, MOGU) is a wrong ticker to correct, not a row to add.
    """
    gaps = {}
    for c in _rows(claims, "claim"):
        try:
            ticker = cc.resolved_ticker(c)
            if not ticker or ticker in sd.UNPRICEABLE_TICKERS:
                continue
            name = c.get("canonical_entity_name") or c.get("subject_mention")
            if instruments.resolve_instrument(ticker, name) is not None:
                continue
            symbol, view, directional = claim_symbol(c), _is_view(c), _is_directional(c)
        except Exception as e:  # bad data must never stop the audit
            _skip("claim", e)
            continue
        gap = gaps.setdefault(ticker, {"ticker": ticker, "claims": 0, "views": 0, "directional": 0,
                                       "channels": Counter(), "names": Counter(), "symbols": Counter()})
        gap["claims"] += 1
        gap["views"] += view
        gap["directional"] += directional
        gap["channels"][cc.source_of(c)] += 1
        gap["names"][claim_name(c)] += 1
        if symbol:
            gap["symbols"][symbol] += 1
    out = []
    for gap in gaps.values():
        symbol = gap["symbols"].most_common(1)[0][0] if gap["symbols"] else None
        out.append({"ticker": gap["ticker"], "claims": gap["claims"], "views": gap["views"],
                    "directional": gap["directional"], "channels": dict(gap["channels"].most_common()),
                    "names": dict(gap["names"].most_common(3)), "symbol": symbol,
                    "cached": (symbol in cache) if (symbol and cache is not None) else None,
                    "us_ticker_shape": bool(_US_TICKER_SHAPE.match(gap["ticker"])),
                    "known_wrong": KNOWN_WRONG_SYMBOLS.get(symbol), "severity": "info"})
    return sorted(out, key=lambda g: (-g["views"], -g["claims"], g["ticker"]))


def _legacy_pairs(records):
    """{stripped name: {recorded ticker: [count, first asset]}} from legacy rows."""
    pairs = defaultdict(dict)
    for _record, asset in _legacy_assets(records):
        name = _strip_suffix(_norm(asset.get("name")))
        ticker = _norm(asset.get("ticker"))
        if not name or not ticker:
            continue
        slot = pairs[name].setdefault(ticker, [0, asset])
        slot[0] += 1
    return pairs


def _legacy_hint(pairs, name, cache):
    """The ticker legacy rows paired with `name`: shown, never applied."""
    options = pairs.get(name)
    if not options:
        return None
    ranked = sorted(options.items(), key=lambda kv: (-kv[1][0], kv[0]))
    ticker, (rows, asset) = ranked[0]
    try:
        symbol = legacy_symbol(asset)
    except Exception as e:  # bad data must never stop the audit
        _skip("legacy asset", e)
        symbol = None
    return {"ticker": ticker, "rows": rows, "symbol": symbol,
            "cached": (symbol in cache) if (symbol and cache is not None) else None,
            "others": {t: n for t, (n, _a) in ranked[1:]}}


def _matches(spec, names, tickers, symbol, channel):
    if spec.get("channels") and channel not in spec["channels"]:
        return False
    return bool((names & spec.get("names", set())) or (tickers & spec.get("tickers", set()))
                or (symbol and symbol in spec.get("symbols", set())))


def _verification_key(name):
    """The needs-verification entry a backlog name belongs to, if any."""
    for key, spec in _VERIFY_MATCH.items():
        if name in spec.get("names", ()):
            return key
    return None


def unresolved_backlog(claims, records=None, cache=None, top=BACKLOG_TOP):
    """
    The curation worklist: canonical subjects with no ticker, keyed by the
    suffix-stripped name and ranked by directional view claims, then view
    claims, then claims. Each entry carries its channels, two evidence
    snippets, the ticker the speakers or the model said (`spoken_hint`) and
    `legacy_hint` — the ticker legacy signal rows paired with the same name,
    with whether its symbol is cached. Hints are shown, never applied.

    `entries` holds the names the extractor typed as an instrument (stock,
    ETF, crypto, private company): the ones a ticker can fix. Subjects that
    are not tickers are counted apart: themes typed sector / macro / bond /
    commodity / index / other ("TEN YEAR YIELD", "MEMORY"), canonical market
    keys (MARKET_KEYS: S&P 500, GOLD, OIL, ...), placeholders, baskets and
    metric subjects (canonical_claims.asset_key is None: "the company",
    "price", "Google and Meta") and names already decided unpriceable
    (ANTHROPIC, OPENAI).
    """
    groups, market, decided = {}, Counter(), Counter()
    placeholders = Counter()
    market_keys = _market_keys()
    for c in _rows(claims, "claim"):
        try:
            if cc.resolved_ticker(c):
                continue
            key = cc.asset_key(c)
            view, directional = _is_view(c), _is_directional(c)
        except Exception as e:  # bad data must never stop the audit
            _skip("claim", e)
            continue
        if key is None:
            placeholders[claim_name(c) or "?"] += 1
            continue
        name = _strip_suffix(_norm(key))
        if key in market_keys or name in market_keys:
            market[name if name in market_keys else key] += 1
            continue
        if name in sd.UNPRICEABLE_TICKERS:
            decided[name] += 1
            continue
        g = groups.setdefault(name, {"name": name, "claims": 0, "views": 0, "directional": 0,
                                     "channels": Counter(), "asset_types": Counter(), "spoken": Counter(),
                                     "rows": []})
        g["claims"] += 1
        g["views"] += view
        g["directional"] += directional
        g["channels"][cc.source_of(c)] += 1
        g["asset_types"][str(c.get("asset_type") or "unknown")] += 1
        spoken = _norm(c.get("ticker_spoken"))
        if spoken:
            g["spoken"][spoken] += 1
        g["rows"].append(c)
    pairs = _legacy_pairs(records) if records is not None else {}
    names, themes = [], []
    for g in groups.values():
        dominant = g["asset_types"].most_common(1)[0][0]
        (names if dominant in _INSTRUMENT_TYPES else themes).append(g)

    def rank(g):
        return -g["directional"], -g["views"], -g["claims"], g["name"]

    entries = []
    for g in sorted(names, key=rank)[:max(0, top)]:
        rows = sorted(g["rows"], key=lambda c: (not _safe_bool(_is_directional, c), not _safe_bool(_is_view, c),
                                                str(c.get("published_at") or "")))
        examples, seen = [], set()
        for c in rows:
            text = _snippet(c.get("evidence_text"))
            if text and text not in seen:
                seen.add(text)
                examples.append({"claim_id": c.get("claim_id"), "channel": cc.source_of(c),
                                 "date": cc.claim_date(c), "text": text})
            if len(examples) == 2:
                break
        spoken = g["spoken"].most_common(1)
        entries.append({
            "name": g["name"], "claims": g["claims"], "views": g["views"], "directional": g["directional"],
            "channels": dict(g["channels"].most_common()), "asset_types": dict(g["asset_types"].most_common()),
            "examples": examples,
            "spoken_hint": {"ticker": spoken[0][0], "claims": spoken[0][1]} if spoken else None,
            "legacy_hint": _legacy_hint(pairs, g["name"], cache) if records is not None else None,
            "needs_verification": _verification_key(g["name"]), "severity": "info"})
    return {"entries": entries, "names": len(names), "claims": sum(g["claims"] for g in names),
            "views": sum(g["views"] for g in names), "directional": sum(g["directional"] for g in names),
            "themes": {"names": len(themes), "claims": sum(g["claims"] for g in themes),
                       "directional": sum(g["directional"] for g in themes),
                       "top": {g["name"]: g["claims"] for g in sorted(themes, key=rank)[:8]}},
            "market_subjects": dict(market.most_common()), "decided_unpriceable": dict(decided.most_common()),
            "placeholders": {"claims": sum(placeholders.values()), "top": dict(placeholders.most_common(8))}}


# --- The learned map ----------------------------------------------------------------------------------


def load_learned_entries(path=None):
    """The learned ticker map's raw entries ({KEY: {ticker, exchange,
    company, via, checked}}), read as plain JSON; {} when absent or
    unreadable. Never raises."""
    path = path or sd.LEARNED_TICKERS_FILE
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError) as e:
        log_warn(f"Ticker audit: could not read the learned ticker map at {path}: {e}")
        return {}
    if not isinstance(payload, dict) or payload.get("version") != 1 or not isinstance(payload.get("tickers"), dict):
        log_warn(f"Ticker audit: ignoring the learned ticker map at {path}: unrecognised format.")
        return {}
    return payload["tickers"]


def learned_map_issues(entries):
    """
    Learned-map entries that would put a wrong listing into the data: the
    listing's company fails ticker_resolver.names_match against the key (the
    resolver's own identity rule), or its exchange is not one
    instruments._CATALOGUE_EXCHANGES maps to a venue. Each is an error.
    Entries with no ticker are "no US listing" verdicts and are skipped.
    """
    try:
        import ticker_resolver  # lazily: it imports the price provider and the LLM client
        names_match = ticker_resolver.names_match
    except Exception as e:  # pragma: no cover - a broken environment, not bad data
        _warn_once("import:ticker_resolver",
                   f"Ticker audit: learned-map name check unavailable (ticker_resolver: {describe_error(e)}).")
        names_match = None
    venues = getattr(instruments, "_CATALOGUE_EXCHANGES", {})
    issues = []
    for key, entry in sorted((entries or {}).items(), key=lambda kv: str(kv[0])):
        if not isinstance(entry, dict):
            issues.append({"key": key, "ticker": None, "company": None, "exchange": None, "via": None,
                           "problems": ["malformed_entry"], "severity": "error"})
            continue
        if not entry.get("ticker"):
            continue
        problems = []
        try:
            if names_match is not None and not names_match(str(key), entry.get("company") or ""):
                problems.append("company_mismatch")
        except Exception as e:  # bad data must never stop the audit
            _skip("learned-map entry", e)
            problems.append("company_unreadable")
        if str(entry.get("exchange") or "").strip().upper() not in venues:
            problems.append("unknown_exchange")
        if problems:
            issues.append({"key": key, "ticker": entry.get("ticker"), "company": entry.get("company"),
                           "exchange": entry.get("exchange"), "via": entry.get("via"),
                           "checked": entry.get("checked"), "problems": problems, "severity": "error"})
    return issues


# --- Price series -------------------------------------------------------------------------------------


def _range_ratio(entry, day, days, window=30):
    """The bar's high-low range relative to the median range of the bars
    before it (up to `window`), or None without enough daily bars."""
    highs, lows, closes = entry.get("highs") or {}, entry.get("lows") or {}, entry["closes"]

    def spread(d):
        if d in highs and d in lows and closes.get(d):
            return (highs[d] - lows[d]) / closes[d]
        return None

    usual = [x for x in (spread(d) for d in days if d < day) if x is not None][-window:]
    last = spread(day)
    if last is None or len(usual) < 5:
        return None
    median = sorted(usual)[len(usual) // 2]
    return round(last / median, 3) if median > 0 else None


def series_anomalies(cache, today, trace=None):
    """
    Price-series problems that make a mark-to-market wrong (warnings unless
    noted):

      big_move            a day-over-day move above MAX_DAILY_MOVE
      flat_run            FLAT_RUN_DAYS or more identical closes in a row
      non_trading_day     '.us' bars on a weekend or an NYSE holiday
      partial_candle      a crypto bar dated on the entry's requested 'to'
                          day: the warm may have run before that UTC day
                          closed. A warning when the bar is dated on or after
                          the audit day, when there are no daily bars to
                          judge by, or when its high-low range is under
                          PARTIAL_RANGE_RATIO of the series' usual range
                          (the 2026-10-11 bars: 0.5% against 2-4%); info
                          when the range looks like a full day's, as it does
                          once the warm stops at the last completed day
      stale               the last close is more than STALE_SERIES_DAYS
                          old while the series is still needed: the creator
                          tracker plans it, or a current directional view
                          claim (a call a report marks to market) produces
                          it (needs `trace`)

    With `trace`, symbols a claim, a legacy row, a benchmark or the tracker
    needs but the cache has never held are listed as `no_series` (info): a
    junk symbol (a foreign code with ".us" appended) is re-requested by
    every warm run, a real one is simply not warmed yet.
    """
    out = []
    us = sp.EXCHANGES["us"]
    for symbol, entry in sorted((cache or {}).items()):
        try:
            closes = (entry or {}).get("closes") or {}
            days = sorted(d for d in closes if isinstance(d, date))
            for prev, cur in zip(days, days[1:]):
                a, b = closes[prev], closes[cur]
                if a and a > 0 and abs(b / a - 1) > MAX_DAILY_MOVE:
                    out.append({"kind": "big_move", "symbol": symbol, "date": cur, "from_date": prev,
                                "from_close": a, "close": b, "change": round(b / a - 1, 4), "severity": "warning"})
            run_start, run = 0, 1
            for i in range(1, len(days) + 1):
                if i < len(days) and closes[days[i]] == closes[days[i - 1]]:
                    run += 1
                    continue
                if run >= FLAT_RUN_DAYS:
                    out.append({"kind": "flat_run", "symbol": symbol, "start": days[run_start],
                                "end": days[i - 1], "days": run, "close": closes[days[i - 1]],
                                "severity": "warning"})
                run_start, run = i, 1
            if symbol.endswith(".us"):
                off = [d for d in days if not sp.is_trading_day(us, d)]
                if off:
                    out.append({"kind": "non_trading_day", "symbol": symbol, "dates": off, "severity": "warning"})
            if _is_crypto_symbol(symbol) and entry.get("to") in closes:
                day = entry["to"]
                ratio = _range_ratio(entry, day, days)
                likely = ratio is None or ratio < PARTIAL_RANGE_RATIO or (today is not None and day >= today)
                out.append({"kind": "partial_candle", "symbol": symbol, "date": day, "close": closes[day],
                            "range_ratio": ratio, "severity": "warning" if likely else "info"})
        except Exception as e:  # bad data must never stop the audit
            _skip("price series", e)
            continue
    for symbol, t in sorted((trace or {}).items()):
        tracker = bool((t.get("reference") or {}).get("creator tracker"))
        if t.get("cached") and (t.get("directional") or tracker) and t.get("last_date") and today:
            age = (today - t["last_date"]).days
            if age > STALE_SERIES_DAYS:
                out.append({"kind": "stale", "symbol": symbol, "last_date": t["last_date"], "days": age,
                            "canonical": t.get("canonical", 0), "directional": t.get("directional", 0),
                            "tracker": tracker, "severity": "warning"})
        reference = t.get("reference") or {}
        wanted = (t.get("canonical") or t.get("legacy") or reference.get("benchmark")
                  or reference.get("creator tracker"))
        if not t.get("cached") and wanted:
            out.append({"kind": "no_series", "symbol": symbol, "canonical": t.get("canonical", 0),
                        "legacy": t.get("legacy", 0), "reference": sorted(k for k, v in reference.items() if v),
                        "producers": dict(list((t.get("producers") or {}).items())[:3]),
                        "known_wrong": KNOWN_WRONG_SYMBOLS.get(symbol), "severity": "info"})
    order = {"warning": 0, "info": 1}
    kinds = ("big_move", "partial_candle", "non_trading_day", "flat_run", "stale", "no_series")
    return sorted(out, key=lambda a: (order.get(a["severity"], 2), kinds.index(a["kind"]),
                                      -a.get("directional", 0), -(a.get("canonical", 0) + a.get("legacy", 0)),
                                      a["symbol"]))


# --- Needs verification ---------------------------------------------------------------------------------


def needs_verification_status(claims, records=None, cache=None):
    """
    NEEDS_VERIFICATION with the current state of each entry: `current` =
    {claims, legacy, channels, resolved, unresolved, examples} — how many
    canonical claims and legacy rows it still covers, what the matched
    claims resolve to today (a ticker or a market key) and two pieces of
    today's evidence. An entry whose claims all resolve, or are gone, has
    been settled (or re-extracted) since.
    """
    market_keys = _market_keys()
    profiles = []
    for c in _rows(claims, "claim"):
        try:
            names = {claim_name(c), _strip_suffix(claim_name(c)), _norm(c.get("subject_mention")),
                     _strip_suffix(_norm(c.get("subject_mention")))} - {""}
            resolved = cc.resolved_ticker(c)
            tickers = {_norm(c.get("ticker")), _norm(resolved)} - {""}
            if not resolved:
                key = cc.asset_key(c)
                resolved = key if key in market_keys else None
            profiles.append((c, names, tickers, claim_symbol(c), cc.source_of(c), resolved))
        except Exception as e:  # bad data must never stop the audit
            _skip("claim", e)
    legacy = []
    for record, asset in _legacy_assets(records) if records is not None else []:
        try:
            name = _norm(asset.get("name"))
            legacy.append((record, asset, {name, _strip_suffix(name)} - {""},
                           {_norm(asset.get("ticker")), _norm(sd.canonical_ticker(asset))} - {""},
                           legacy_symbol(asset, record.get("channel_name")), record.get("channel_name")))
        except Exception as e:  # bad data must never stop the audit
            _skip("legacy asset", e)
    out = []
    for item in NEEDS_VERIFICATION:
        spec = _VERIFY_MATCH.get(item["key"], {})
        matched = [(c, resolved) for c, names, tickers, symbol, channel, resolved in profiles
                   if _matches(spec, names, tickers, symbol, channel)]
        hits = [c for c, _resolved in matched]
        resolved = Counter(r for _c, r in matched if r)
        old = [(r, a) for r, a, names, tickers, symbol, channel in legacy
               if _matches(spec, names, tickers, symbol, channel)]
        channels = Counter(cc.source_of(c) for c in hits) + Counter(r.get("channel_name") or "?" for r, _a in old)
        hits.sort(key=lambda c: (not _safe_bool(_is_directional, c), str(c.get("published_at") or "")))
        examples = [{"claim_id": c.get("claim_id"), "channel": cc.source_of(c), "date": cc.claim_date(c),
                     "text": _snippet(c.get("evidence_text"))} for c in hits[:2]]
        row = dict(item)
        row["current"] = {"claims": len(hits), "legacy": len(old), "channels": dict(channels.most_common()),
                          "resolved": dict(resolved.most_common()), "unresolved": len(hits) - sum(resolved.values()),
                          "examples": examples}
        row["severity"] = "info"
        out.append(row)
    return out


# --- The report -------------------------------------------------------------------------------------------


def run_audit(claims, records, cache, learned_entries, today, top=BACKLOG_TOP, reference=None):
    """
    The full audit over the data passed in — the only other thing read is
    the learned ticker map that ticker resolution itself consults — as a
    dict with exactly REPORT_KEYS. `errors` and `warnings` are counts; a
    section that fails is logged, left empty and counted as an error.
    `reference` ({symbol: {label: n}}) overrides the benchmark and tracker
    producers (_reference_symbols), for tests.
    """
    claims = list(_rows(claims, "claim"))
    records = list(_rows(records, "legacy signal")) if records is not None else []
    cache = _clean_cache(cache)
    failed = []

    def section(name, compute, empty):
        try:
            return compute()
        except Exception as e:  # one broken check must not cost the others
            log_error(f"Ticker audit: the {name} check failed: {describe_error(e)}", exc_info=True)
            failed.append(name)
            return empty

    if reference is None:
        reference = section("reference", lambda: _reference_symbols(claims, today), {})
    trace = section("symbol_trace", lambda: symbol_trace(claims, records, cache, reference), {})
    report = {"today": today, "counts": {}, "errors": 0, "warnings": 0}
    report["alias_registry_gaps"] = section("alias_registry_gaps", alias_registry_gaps, [])
    report["known_wrong"] = section("known_wrong", lambda: known_wrong_findings(trace), [])
    report["orphans"] = section("orphans", lambda: orphan_symbols(trace), [])
    report["price_level"] = section("price_level", lambda: price_level_checks(claims, cache),
                                    {"band": list(PRICE_BAND), "checked": 0, "in_band": 0, "symbols": 0,
                                     "corroborated": [], "contradicted": [], "no_close": 0, "outliers": []})
    report["registry_gaps"] = section("registry_gaps", lambda: registry_gaps(claims, cache), [])
    report["unresolved_backlog"] = section(
        "unresolved_backlog", lambda: unresolved_backlog(claims, records, cache, top=top),
        {"entries": [], "names": 0, "claims": 0, "views": 0, "directional": 0,
         "themes": {"names": 0, "claims": 0, "directional": 0, "top": {}}, "market_subjects": {},
         "decided_unpriceable": {}, "placeholders": {"claims": 0, "top": {}}})
    report["learned_map"] = section("learned_map", lambda: learned_map_issues(learned_entries), [])
    report["series_anomalies"] = section("series_anomalies", lambda: series_anomalies(cache, today, trace), [])
    report["needs_verification"] = section("needs_verification",
                                           lambda: needs_verification_status(claims, records, cache), [])

    known = report["known_wrong"]
    anomalies = report["series_anomalies"]
    price = report["price_level"]
    backlog = report["unresolved_backlog"]
    produced = [s for s, t in trace.items() if t.get("canonical") or t.get("legacy")]
    report["counts"] = {
        "claims": len(claims),
        "view_claims": sum(1 for c in claims if _safe_bool(_is_view, c)),
        "claims_without_ticker": sum(1 for c in claims if not _safe_value(cc.resolved_ticker, c)),
        "legacy_records": len(records),
        "legacy_assets": len(_legacy_assets(records)),
        "cached_symbols": len(cache),
        "produced_symbols": len(produced),
        "produced_uncached": sum(1 for s in produced if not trace[s].get("cached")),
        "learned_map_entries": len(learned_entries or {}),
        "alias_registry_gaps": len(report["alias_registry_gaps"]),
        "known_wrong_produced": sum(1 for k in known if k["severity"] == "error"),
        "known_wrong_cached_only": sum(1 for k in known if k["severity"] == "info"),
        "orphans": len(report["orphans"]),
        "price_checks": price["checked"],
        "price_checks_in_band": price["in_band"],
        "price_level_outliers": len(price["outliers"]),
        "price_symbols_corroborated": len(price["corroborated"]),
        "registry_gaps": len(report["registry_gaps"]),
        "registry_gap_claims": sum(g["claims"] for g in report["registry_gaps"]),
        "unresolved_names": backlog["names"],
        "unresolved_name_claims": backlog["claims"],
        "theme_claims": backlog["themes"]["claims"],
        "market_subject_claims": sum(backlog["market_subjects"].values()),
        "unpriceable_name_claims": sum(backlog["decided_unpriceable"].values()),
        "placeholder_claims": backlog["placeholders"]["claims"],
        "learned_map_issues": len(report["learned_map"]),
        "series_warnings": sum(1 for a in anomalies if a["severity"] == "warning"),
        "series_without_cache": sum(1 for a in anomalies if a["kind"] == "no_series"),
        "needs_verification": len(report["needs_verification"]),
        "failed_sections": failed,
    }
    report["errors"] = (len(report["alias_registry_gaps"]) + report["counts"]["known_wrong_produced"]
                        + len(report["learned_map"]) + len(failed))
    report["warnings"] = len(price["outliers"]) + report["counts"]["series_warnings"]
    return {key: report[key] for key in REPORT_KEYS}


def _clean_cache(cache):
    """The price cache without entries no check can read (price_cache.load
    never returns one; a hand-built cache might)."""
    clean = {}
    for symbol, entry in (cache if isinstance(cache, dict) else {}).items():
        if isinstance(symbol, str) and isinstance(entry, dict) and isinstance(entry.get("closes") or {}, dict):
            clean[symbol] = entry
        else:
            _warn_once("cache", f"Ticker audit: skipped a malformed price-cache entry ({symbol!r}).")
    return clean


def _safe_bool(fn, row):
    try:
        return bool(fn(row))
    except Exception:  # bad data must never stop the audit
        return False


def _safe_value(fn, row):
    try:
        return fn(row)
    except Exception:  # bad data must never stop the audit
        return None


# --- Text ---------------------------------------------------------------------------------------------------


def _fmt_date(value):
    return value.isoformat() if isinstance(value, (date, datetime)) else (str(value) if value else "?")


def _fmt_num(value):
    if value is None:
        return "?"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return f"{value:.4g}"


def _counts(mapping, limit=3):
    items = list((mapping or {}).items())
    text = ", ".join(f"{k} {v}" for k, v in items[:limit])
    return text + (f", +{len(items) - limit} more" if len(items) > limit else "")


def _capped(lines, rows, top, render):
    for row in rows[:top]:
        lines.extend(render(row))
    if len(rows) > top:
        lines.append(f"  …and {len(rows) - top} more (see --json).")


def _example_text(example):
    if example.get("kind") == "legacy":
        return (f"legacy {example.get('channel') or '?'} {_fmt_date(example.get('date'))}: "
                f"{example.get('name')!s} as {example.get('ticker')!s} typed {example.get('asset_type')!s}")
    return (f"{example.get('channel') or '?'} {_fmt_date(example.get('date'))}: {example.get('name')!s} "
            f"as {example.get('ticker')!s} typed {example.get('asset_type')!s} ({example.get('claim_id')})")


def format_text(report, top=DEFAULT_TOP):
    """The plain-text report: a header, then every section in SECTIONS
    order, each with its count; at most `top` rows per section."""
    counts = report.get("counts") or {}
    lines = [
        f"Ticker audit — {_fmt_date(report.get('today'))} (offline and read-only: nothing below is applied)",
        f"{counts.get('claims', 0)} canonical claims ({counts.get('claims_without_ticker', 0)} without a ticker) · "
        f"{counts.get('legacy_records', 0)} legacy signal rows ({counts.get('legacy_assets', 0)} assets) · "
        f"{counts.get('cached_symbols', 0)} cached series · {counts.get('learned_map_entries', 0)} learned tickers",
        f"Result: {report.get('errors', 0)} errors · {report.get('warnings', 0)} warnings"
        + (f" · failed checks: {', '.join(counts['failed_sections'])}" if counts.get("failed_sections") else ""),
    ]

    def header(title, count, note):
        lines.extend(["", f"{title} ({count}) — {note}"])

    gaps = report.get("alias_registry_gaps") or []
    header("ALIAS OUTPUTS WITHOUT AN INSTRUMENT", len(gaps), "error: add a registry row or an UNPRICEABLE entry")
    if not gaps:
        lines.append("• none: every alias output has a registry row or an unpriceable decision")
    _capped(lines, gaps, top, lambda g: [f"• {g['ticker']} ← {', '.join(g['aliases'][:4])}"
                                         + (f" (+{len(g['aliases']) - 4})" if len(g['aliases']) > 4 else "")])

    known = report.get("known_wrong") or []
    findings = sorted((k for k in known if k["severity"] != "ok"),
                      key=lambda k: (k["severity"] != "error", k["symbol"]))
    produced = sum(1 for k in known if k["severity"] == "error")
    header("KNOWN-WRONG SYMBOLS", len(findings),
           f"{produced} still produced (error), {len(findings) - produced} cached only (info: prune)")

    def known_line(k):
        where = (f"cached {_fmt_date(k['from'])}…{_fmt_date(k['to'])}, last {_fmt_num(k['last_close'])}"
                 if k["cached"] else "not cached")
        if k["severity"] == "error":
            by = [f"{k['canonical']} canonical claim(s)"] if k["canonical"] else []
            by += [f"{k['legacy']} legacy row(s)"] if k["legacy"] else []
            out = [f"• {k['symbol']} — {k['why']} — STILL PRODUCED by {' and '.join(by)} "
                   f"({_counts(k['producers'])}) — {where}"]
            out.extend(f"    {_example_text(e)}" for e in k["examples"][:2])
            return out
        return [f"• {k['symbol']} — {k['why']} — cached only, nothing produces it ({where})"]

    _capped(lines, findings, top, known_line)
    clear = [k["symbol"] for k in known if k["severity"] == "ok"]
    if clear:
        lines.append(f"  clear (neither produced nor cached): {', '.join(clear)}")

    orphans = report.get("orphans") or []
    header("ORPHANED CACHE SERIES", len(orphans), "info: cached, nothing produces or needs them (prune candidates)")
    _capped(lines, orphans, top, lambda o: [
        f"• {o['symbol']} {_fmt_date(o['from'])}…{_fmt_date(o['to'])}, last {_fmt_num(o['last_close'])}"
        + (f" — known wrong: {o['known_wrong']}" if o.get("known_wrong") else "")])

    price = report.get("price_level") or {}
    outliers = price.get("outliers") or []
    band = price.get("band") or list(PRICE_BAND)
    header("STATED PRICES VS THE CACHED CLOSE", len(outliers),
           f"warning: outside {band[0]}–{band[1]}x of the first close after publication")
    lines.append(f"  {price.get('checked', 0)} stated prices checked on {price.get('symbols', 0)} symbols: "
                 f"{price.get('in_band', 0)} within the band; {len(price.get('corroborated') or [])} symbols "
                 f"corroborated, {len(price.get('contradicted') or [])} contradicted"
                 + (f" ({', '.join(price['contradicted'][:8])})" if price.get("contradicted") else "")
                 + f"; {price.get('no_close', 0)} with no cached close.")
    _capped(lines, outliers, top, lambda o: [
        f"• {o['symbol']} ({o['ticker'] or o['name']}): stated {_fmt_num(o['stated'])} ({o['source']}) vs close "
        f"{_fmt_num(o['close'])} on {_fmt_date(o['close_date'])} = {o['ratio']}x — {o['channel']} "
        f"{_fmt_date(o['published'])} ({o['claim_id']})", f"    \"{o['evidence']}\""])

    rgaps = report.get("registry_gaps") or []
    header("RESOLVED TICKERS WITHOUT AN INSTRUMENT", len(rgaps),
           f"info: the next registry rows to curate ({sum(g['claims'] for g in rgaps)} claims)")

    def gap_line(g):
        symbol = g["symbol"] or "no symbol"
        cached = "" if g["symbol"] is None or g["cached"] is None else (" (cached)" if g["cached"] else " (not cached)")
        notes = []
        if g.get("known_wrong"):
            notes.append(f"KNOWN WRONG — {g['known_wrong']}: correct the ticker, do not add a row")
        if not g.get("us_ticker_shape"):
            notes.append("not a US ticker shape")
        return [f"• {g['ticker']} — {g['claims']} claims ({g['views']} views, {g['directional']} directional) · "
                f"{_counts(g['channels'], 2)} · {symbol}{cached} · as {_counts(g['names'], 2)}"
                + (f" — {'; '.join(notes)}" if notes else "")]

    _capped(lines, rgaps, top, gap_line)

    backlog = report.get("unresolved_backlog") or {}
    entries = backlog.get("entries") or []
    header("UNRESOLVED NAMES", backlog.get("names", 0),
           f"info: instruments with no ticker, most directional views first ({backlog.get('claims', 0)} claims, "
           f"{backlog.get('directional', 0)} directional views); hints are never applied")

    def backlog_line(e):
        hints = []
        if e.get("legacy_hint"):
            h = e["legacy_hint"]
            state = "" if h["symbol"] is None else f", {h['symbol']} {'cached' if h['cached'] else 'not cached'}"
            hints.append(f"legacy rows say {h['ticker']} ×{h['rows']}{state}"
                         + (f" (also {_counts(h['others'], 2)})" if h.get("others") else ""))
        if e.get("spoken_hint"):
            hints.append(f"spoken {e['spoken_hint']['ticker']} ×{e['spoken_hint']['claims']}")
        if e.get("needs_verification"):
            hints.append(f"see needs-verification '{e['needs_verification']}'")
        out = [f"• {e['name']} — {e['claims']} claims, {e['views']} views ({e['directional']} directional) · "
               f"{_counts(e['channels'], 2)}" + (f" · {'; '.join(hints)}" if hints else "")]
        out.extend(f"    {x['channel']} {_fmt_date(x['date'])}: \"{x['text']}\"" for x in e.get("examples") or [])
        return out

    for entry in entries[:top]:
        lines.extend(backlog_line(entry))
    hidden = backlog.get("names", 0) - min(top, len(entries))
    if hidden > 0:
        lines.append(f"  …and {hidden} more names with fewer claims"
                     + (f" ({len(entries) - top} of them in --json)." if len(entries) > top else "."))
    themes = backlog.get("themes") or {}
    market = backlog.get("market_subjects") or {}
    decided = backlog.get("decided_unpriceable") or {}
    placeholders = backlog.get("placeholders") or {}
    lines.append(f"  Not single instruments: themes {themes.get('names', 0)} names, {themes.get('claims', 0)} "
                 f"claims ({_counts(themes.get('top'), 5)}); market subjects {sum(market.values())} claims "
                 f"({_counts(market, 6)}); unpriceable names {sum(decided.values())} claims ({_counts(decided, 4)}); "
                 f"placeholders, baskets and metrics {placeholders.get('claims', 0)} claims "
                 f"({_counts(placeholders.get('top'), 4)}).")

    learned = report.get("learned_map") or []
    header("LEARNED TICKER MAP", len(learned), f"error: of {counts.get('learned_map_entries', 0)} entries")
    if not learned:
        lines.append("• none: every learned listing matches its name and a known venue")
    _capped(lines, learned, top, lambda i: [
        f"• {i['key']} → {i['ticker']} ({i['company']}, {i['exchange']}, via {i['via']}): "
        f"{', '.join(i['problems'])}"])

    anomalies = report.get("series_anomalies") or []
    warnings = [a for a in anomalies if a["severity"] == "warning"]
    missing = [a for a in anomalies if a["kind"] == "no_series"]
    complete = [a for a in anomalies if a["kind"] == "partial_candle" and a["severity"] != "warning"]
    header("PRICE-SERIES ANOMALIES", len(warnings),
           f"warning; plus {len(missing)} needed symbols with no cached series (info)"
           + (f" and {len(complete)} crypto bars on the requested day that look complete" if complete else ""))

    def anomaly_line(a):
        kind = a["kind"]
        if kind == "big_move":
            return [f"• big move: {a['symbol']} {_fmt_date(a['from_date'])} {_fmt_num(a['from_close'])} → "
                    f"{_fmt_date(a['date'])} {_fmt_num(a['close'])} ({a['change']:+.0%})"]
        if kind == "flat_run":
            return [f"• flat run: {a['symbol']} {a['days']} identical closes {_fmt_num(a['close'])} "
                    f"{_fmt_date(a['start'])}…{_fmt_date(a['end'])}"]
        if kind == "non_trading_day":
            return [f"• non-trading-day bars: {a['symbol']} {', '.join(_fmt_date(d) for d in a['dates'][:5])}"
                    + (f" (+{len(a['dates']) - 5})" if len(a["dates"]) > 5 else "")]
        if kind == "partial_candle":
            spread = (f"; its high-low range is {a['range_ratio']:.0%} of the usual one"
                      if a.get("range_ratio") is not None else "; no daily bars to judge it by")
            return [f"• partial candle: {a['symbol']} bar dated {_fmt_date(a['date'])}, the day the warm "
                    f"requested up to ({_fmt_num(a['close'])}){spread}"]
        if kind == "stale":
            who = [f"{a['directional']} directional view(s)"] if a.get("directional") else []
            who += ["the creator tracker"] if a.get("tracker") else []
            return [f"• stale: {a['symbol']} last close {_fmt_date(a['last_date'])} ({a['days']} days old), "
                    f"needed by {' and '.join(who)}"]
        return [f"• {kind}: {a['symbol']}"]

    def needed_by(a):
        parts = [f"{a['canonical']}c"] if a.get("canonical") else []
        parts += [f"{a['legacy']}l"] if a.get("legacy") else []
        parts += ["tracker" if r == "creator tracker" else r for r in a.get("reference") or []]
        return f"{a['symbol']}({'+'.join(parts)}{', KNOWN WRONG' if a.get('known_wrong') else ''})"

    _capped(lines, warnings, top, anomaly_line)
    if missing:
        shown = ", ".join(needed_by(a) for a in missing[:top * 2])
        more = f", +{len(missing) - top * 2} more" if len(missing) > top * 2 else ""
        lines.append(f"  No cached series yet (c = canonical claims, l = legacy rows): {shown}{more}. "
                     f"Real symbols are warmed by the next run; a junk one is re-requested by every run.")

    verify = report.get("needs_verification") or []
    header("NEEDS VERIFICATION", len(verify), "info: medium/low-confidence corrections, applied nowhere")

    def verify_head(v):
        cur = v.get("current") or {}
        now = (f"now {cur.get('claims', 0)} claim(s)"
               + (f" + {cur['legacy']} legacy row(s)" if cur.get("legacy") else "")
               + (f" ({_counts(cur.get('channels'), 2)})" if cur.get("channels") else ""))
        if cur.get("claims"):
            state = [f"resolved {_counts(cur['resolved'], 3)}"] if cur.get("resolved") else []
            state += [f"unresolved {cur['unresolved']}"] if cur.get("unresolved") else []
            now += f" — {'; '.join(state)}"
        return f"• {v['key']} → {v['proposed']} [{v['confidence']}] — {now}"

    for v in verify[:top]:
        lines.extend([verify_head(v), f"    evidence: {v['evidence']}", f"    verify: {v['verify']}"])
        lines.extend(f"    today: {x['channel']} {_fmt_date(x['date'])}: \"{x['text']}\""
                     for x in (v.get("current") or {}).get("examples") or [])
    # The list is curated and short: the rest still get one line each.
    lines.extend(f"{verify_head(v)} (evidence and verify step in --json)" for v in verify[top:])
    return "\n".join(lines)


# --- JSON and the CLI ---------------------------------------------------------------------------------------


def _jsonable(value):
    """The report with dates as ISO strings, sets as sorted lists."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _load_claims():
    try:
        return cc.load_canonical_claims()
    except Exception as e:  # the audit must still report on the cache and the tables
        log_error(f"Ticker audit: canonical claims unavailable: {describe_error(e)}")
        return []


def _parse_day(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a YYYY-MM-DD date: {value!r}") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit tickers, aliases and cached price series (read-only).")
    parser.add_argument("--json", metavar="PATH", help="also write the report as JSON to PATH ('-' = stdout only)")
    parser.add_argument("--strict", action="store_true", help="exit 1 when the report has errors")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP, help="rows per section in the text report")
    parser.add_argument("--cache", default=price_cache.CACHE_FILE, help="the daily-close cache to audit")
    parser.add_argument("--signals", default=sd.SIGNALS_FILE, help="the legacy signal rows (producers too)")
    parser.add_argument("--ticker-map", dest="ticker_map", default=sd.LEARNED_TICKERS_FILE,
                        help="the learned ticker map the learned-map section audits")
    parser.add_argument("--today", type=_parse_day, help="the audit date (default: today, UTC)")
    args = parser.parse_args(argv)
    top = max(1, args.top)
    today = args.today or datetime.now(timezone.utc).date()
    try:
        report = run_audit(_load_claims(), sd.load_signals(args.signals), price_cache.load(args.cache),
                           load_learned_entries(args.ticker_map), today, top=max(BACKLOG_TOP, top))
    except Exception as e:  # pragma: no cover - run_audit guards every check; this is the last line
        log_error(f"Ticker audit failed: {describe_error(e)}", exc_info=True)
        return 1 if args.strict else 0
    if args.json == "-":
        sys.stdout.write(json.dumps(_jsonable(report), indent=1, ensure_ascii=False) + "\n")
    else:
        print(format_text(report, top=top))
        if args.json and not write_json_atomic(args.json, _jsonable(report), indent=1, sort_keys=False):
            report["errors"] += 1
    log_info(f"Ticker audit: {report['errors']} errors, {report['warnings']} warnings.")
    return 1 if args.strict and report["errors"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
