"""
The one loader every production analytics job reads canonical claims through.

`data/research/claims.jsonl` is append-only and carries every run ever made;
what analytics may count is narrower, and every consumer used to have to
know the rules. They live here instead:

  load_canonical_claims  active run per video only (superseded runs out),
                         no legacy-imported rows (schema_version="legacy"),
                         no cross-chunk repeats (repeat_of_claim_id), with
                         the latest observed condition outcome overlaid
  headline_claims        + claims.is_headline_claim (reviewed-clean, fully
                         covered, evidence-located, the source's own view)
  view_claims            + claims.carries_view (a view-bearing claim type
                         with a bullish / bearish / neutral / mixed stance)

`data/signals.jsonl` is a backward-compatible VIEW derived from these claims
(one legacy stance per asset, chosen by claims.claims_to_legacy_signals). It
is read by nothing in this module and by no headline analytics: the weekly
pulse, consensus, flips, source comparisons, scorecards and charts all start
from `view_claims`, so a reduction the old schema forces can never shape a
new result. Consumers still on the legacy shape are listed in
docs/tdd-full-transcript-claims.md § 11.

The aggregation below groups views per (asset, horizon bucket): a short-term
bearish claim and a long-term bullish claim from the same source are two
rows, never one averaged stance.

Entity resolution is brought up to date at load time: `reresolve` re-runs
claims.resolve_entity on every stored row resolved under an older
claims.RESOLUTION_VERSION, so an alias or rule added after extraction corrects
stored claims (ticker, subject kind, review reasons) without re-extracting.
"""
import re
from collections import Counter, defaultdict
from datetime import datetime

import claims as _claims
from claims import carries_view, is_headline_claim
import instruments
import research_state
from signals_data import CONVICTION_WEIGHTS, UNPRICEABLE_TICKERS, canonical_ticker

BUCKETS = ("short", "medium", "long", "unspecified")
# How a bucket reads in the report. The unspecified bucket carries no tag: it
# is most claims (a speaker rarely names a horizon), and "[unspecified]" on
# every row buried the asset name and was clipped off the charts.
HORIZON_LABELS = {"short": "short-term", "medium": "medium-term", "long": "long-term"}

# Unresolved subjects that are references, not assets: the extractor keeps the
# spoken phrase when coreference fails ("this business", "the company's",
# German "die Aktie"), and a portfolio or cash position is not an instrument.
# They carry no ticker and would otherwise count as distinct "assets" in
# consensus and new-on-radar.
_PLACEHOLDER_FIRST_WORDS = {"THIS", "THAT", "THE", "THESE", "THOSE", "IT", "ITS", "MY", "OUR", "THEIR", "HIS", "HER",
                            "THEY", "THEM", "DIE", "DER", "DAS", "DEN", "DEM", "ES", "EACH"}
# Leading articles after which a name the tables know is still that name
# ("THE S&P 500", "THE TRADEDESK", German "DER BITCOIN", "DER DAX").
_ARTICLES = {"THE", "DER", "DIE", "DAS", "DEN", "DEM"}
_NON_ASSETS = {"CASH", "PORTFOLIO", "MY PORTFOLIO", "WHITE COUNT", "PRICE", "RECESSION",
               "COMPUTE", "CLARITY ACT", "MARKET", "THE MARKET", "MARKETS", "THE MARKETS", "STOCKS", "EVERYTHING",
               "THEY", "THEM", "STOCK", "COMPANY", "BUSINESS", "MANAGEMENT TEAM", "REVENUE", "DEMAND", "AI",
               "THE WORLD", "KURS", "AKTIE", "TITEL", "ETF", "ETFS", "OTHERS", "SHORT TERM", "LONG TERM",
               "INTERNATIONAL"}
# Chart-reading vocabulary the extractor sometimes keeps as the subject
# ("wave two or B wave", "the SOL/BTC chart", "the support zone", "the 50-week
# SMA", "Ethereum against Bitcoin", "the monthly candle").
_NON_ASSET_PATTERN = re.compile(
    r"\b(WAVES?|CHART|COUNT|SUPPORTS?|RESISTANCE|SCENARIO|UPTREND|DOWNTREND|RETRACEMENT|FIBONACCI|SMA|EMA|RSI|"
    r"RALLY|RATIO|AGAINST|CANDLES?)\b")

# Spellings of one un-tickered market folded to one key, so oil talk is not
# split across OIL, OIL PRICES and WTI (each a single creator on its own). The
# table is claims.MARKET_SUBJECTS, whose keys are the contract the creator
# tracker's market proxies price against.
MACRO_ALIASES = dict(_claims.MARKET_SUBJECTS)

# Price targets: only a price level the speaker expects the asset to reach.
# A recommendation's number is an entry level ("Nvidia at 200 or under,
# easy"), and a forecast of revenue, market cap, margin or a multiple is not
# a share price (a $50bn revenue figure once read as a +22,495,163,440% target).
_TARGET_CLAIM_TYPES = {"price_target", "forecast"}
_NON_PRICE_UNITS = {"percent", "%", "x", "multiple", "bps", "basis_points"}


def display_label(asset, bucket):
    tag = HORIZON_LABELS.get(bucket)
    return f"{asset} ({tag})" if tag else asset


def is_placeholder_asset(key, ticker=None):
    """True for an unresolved reference phrase or a non-instrument: a
    placeholder word, chart vocabulary, or a pair such as "SOL/BTC". A name
    after an article that the market or alias tables know ("THE S&P 500",
    "THE TRADEDESK", "THE FED", German "DER BITCOIN") is a name, not a
    reference."""
    if ticker or not key:
        return not key
    if "/" in key:
        return True
    if key.split()[0] in _ARTICLES and (_claims.market_key(key) or _claims.name_ticker(key)):
        return False
    if key in _NON_ASSETS or key.endswith(" PORTFOLIO") or _NON_ASSET_PATTERN.search(key):
        return True
    return key.split()[0] in _PLACEHOLDER_FIRST_WORDS


def price_target_of(claim):
    """The USD price level a claim targets, or None when it isn't one."""
    if claim.get("claim_type") not in _TARGET_CLAIM_TYPES:
        return None
    if claim.get("currency") not in (None, "USD"):
        return None
    if claim.get("forecast_metric") not in (None, "price"):
        return None
    unit = (claim.get("target_unit") or "").strip().lower()
    if unit and unit not in ("usd", "$", "dollar", "dollars"):
        return None
    if claim.get("target_kind") == "absolute_value" and isinstance(claim.get("target_value"), (int, float)):
        value = float(claim["target_value"])
    elif claim.get("target_kind") == "range" and all(
            isinstance(claim.get(k), (int, float)) for k in ("target_low", "target_high")):
        value = (claim["target_low"] + claim["target_high"]) / 2.0
    else:
        return None
    return value if value > 0 else None
LEGACY_TYPE = {"stock": "stock", "crypto": "crypto", "etf": "etf", "index": "index",
               "commodity": "commodity", "macro": "macro"}


# Review reasons that only ever described the model's ticker field. Under the
# current rules an unverifiable model ticker is dropped with a non-blocking
# resolution note instead, and only a ticker that CONTRADICTS the subject's
# curated one is reviewed (ticker_subject_mismatch).
_STALE_RESOLUTION_REASONS = frozenset({"ticker_not_in_evidence"})


def _resolution_method(row, ent):
    """(entity_resolution_method, confidence) after re-resolution, following
    validate_claims' labels as far as the stored fields allow (the segment
    text is not stored, so an in-segment mention cannot be told apart)."""
    method = row.get("entity_resolution_method")
    confidence = row.get("entity_resolution_confidence")
    status = ent["entity_resolution_status"]
    if method == "local_coreference":
        return method, confidence   # the transcript resolved it; the stored name is its result
    if ent["resolution_method_hint"] == "title":
        return "title", 0.6 if status == "confirmed" else 0.5
    if status in ("unresolved", "ambiguous"):
        return "unresolved", 0.0
    if method in (None, "unresolved") or (method == "explicit_mention" and _claims._is_generic(
            row.get("canonical_entity_name") or row.get("subject_mention"))):
        subject = _claims._fold(row.get("canonical_entity_name") or row.get("subject_mention") or "")[0]
        if subject and subject in _claims._fold(row.get("evidence_text") or "")[0]:
            return "explicit_mention", 1.0 if status == "confirmed" else 0.8
        if subject and subject in _claims._fold(row.get("video_title") or "")[0]:
            return "title", 0.6
        return "curated_mapping", 0.5
    return method, confidence


def reresolve(row):
    """
    A stored claim with its entity resolution brought up to the current rules
    (claims.RESOLUTION_VERSION), without re-extraction: claims.resolve_entity
    re-runs on the claim's own fields (resolved name or spoken subject, the
    model's ticker, evidence, title, asset type, channel) and its verdict
    replaces ticker, ticker_source, entity_resolution_status, subject_kind and
    resolution_notes (and canonical_entity_name for a market subject). The
    retired review reason ticker_not_in_evidence is dropped, a new one is
    added, and review_required is recomputed from what is left the way
    validate_claims computes it. A stored verdict that is ambiguous, or one
    that would turn ambiguous (AMBIGUOUS_MENTIONS grew), is kept until a
    re-extraction. testable / testability_issues are not recomputed.

    Pure: returns a new dict, or `row` itself when it was resolved under the
    current version.
    """
    if row.get("resolution_version") == _claims.RESOLUTION_VERSION:
        return row
    subject = row.get("canonical_entity_name") or row.get("subject_mention")
    ent = _claims.resolve_entity(subject, row.get("ticker_spoken"), row.get("evidence_text"), row.get("video_title"),
                                 row.get("asset_type"), channel=row.get("channel_name"))
    if row.get("entity_resolution_status") == "ambiguous" or ent["entity_resolution_status"] == "ambiguous":
        # Coreference ambiguity needs the transcript, and a newly ambiguous
        # name is the table's change of mind: the stored verdict stands.
        ent = {"canonical_entity_name": row.get("canonical_entity_name"), "ticker": row.get("ticker"),
               "ticker_source": row.get("ticker_source"),
               "entity_resolution_status": row.get("entity_resolution_status"), "review_reason": None,
               "subject_kind": row.get("subject_kind") or "instrument",
               "resolution_notes": list(row.get("resolution_notes") or []), "resolution_method_hint": None}
    out = dict(row)
    out.update(ticker=ent["ticker"], ticker_source=ent["ticker_source"],
               entity_resolution_status=ent["entity_resolution_status"], subject_kind=ent["subject_kind"],
               resolution_notes=list(ent["resolution_notes"]), resolution_version=_claims.RESOLUTION_VERSION)
    if ent["subject_kind"] == "market":
        out["canonical_entity_name"] = ent["canonical_entity_name"]
    out["entity_resolution_method"], out["entity_resolution_confidence"] = _resolution_method(row, ent)
    reasons = [r for r in (row.get("review_reasons") or []) if r not in _STALE_RESOLUTION_REASONS]
    if ent["review_reason"] and ent["review_reason"] not in reasons:
        reasons.append(ent["review_reason"])
    out["review_reasons"] = reasons
    out["review_required"] = bool(reasons) or row.get("evidence_start_character") is None \
        or out["entity_resolution_status"] == "ambiguous" or row.get("extraction_confidence") == "low"
    return out


def load_canonical_claims(state=None, path=None, conditions=None):
    """Active, non-legacy, non-repeated claims with entity resolution brought
    up to the current rules (reresolve) and condition outcomes applied."""
    state = state or research_state.load_state()
    rows = research_state.load_active_claims(state, path=path, include_legacy=False)
    evaluations = conditions if conditions is not None else research_state.load_condition_evaluations()
    out = []
    for row in rows:
        if row.get("repeat_of_claim_id"):
            continue
        row = reresolve(row)
        ev = evaluations.get(row.get("claim_id"))
        if ev:
            row = dict(row)
            for k in ("condition_status", "condition_evaluation_date", "condition_evidence"):
                row[k] = ev.get(k)
            if ev.get("condition_data_source"):
                row["condition_data_source"] = ev["condition_data_source"]
        out.append(row)
    return out


def headline_claims(claims):
    return [c for c in claims if is_headline_claim(c)]


def view_claims(claims):
    return [c for c in claims if carries_view(c)]


def claim_date(claim):
    for key in ("published_at", "extracted_at"):
        try:
            return datetime.fromisoformat(str(claim.get(key)).replace("Z", "+00:00")).date()
        except (TypeError, ValueError):
            continue
    return None


def in_window(claim, start, end):
    """(start, end] like the legacy dataset's window convention."""
    d = claim_date(claim)
    return d is not None and start < d <= end


def source_of(claim):
    return claim.get("channel_name") or claim.get("channel_id") or "unknown"


def asset_key(claim):
    """The asset identity analytics group on: the resolved ticker, folded
    through the curated alias tables, else the market key ("the S&P 500" ->
    "S&P 500", "Silber" -> "SILVER"), else the spoken name without trailing
    words such as " stock". None for a placeholder phrase, and for a basket
    or a metric subject: "Google and Meta" is not one asset and a company's
    gross margin is not its share price, so neither may vote as one in view
    analytics, the pulse or the tracker."""
    if claim.get("subject_kind") in ("basket", "metric"):
        return None
    ticker = resolved_ticker(claim)
    if ticker:
        return ticker
    name = claim.get("canonical_entity_name") or claim.get("subject_mention") or ""
    key = _claims.market_key(name) or _claims.strip_name(name)
    key = MACRO_ALIASES.get(key, key)
    return None if is_placeholder_asset(key) else key


def resolved_ticker(claim):
    """The claim's ticker after the curated tables, or None: the claim's
    channel's own spelling first (claims.CHANNEL_ALIASES: Couch Investor's
    "New" is Nu Holdings), then signals_data.canonical_ticker."""
    name = claim.get("canonical_entity_name") or claim.get("subject_mention")
    scoped = _claims.channel_ticker(name, claim.get("channel_name"))
    if scoped:
        return scoped
    return canonical_ticker({"ticker": claim.get("ticker"), "name": name, "channel": claim.get("channel_name")})


UP = {"increase", "recover", "outperform"}
DOWN = {"decrease", "decline", "underperform"}


def direction_of(claim):
    """bullish / bearish / neutral / None from the stance, then the forecast
    direction. Only view-bearing claims should be asked."""
    stance = claim.get("stance")
    if stance in ("bullish", "bearish"):
        return stance
    d = claim.get("forecast_direction")
    if d in UP:
        return "bullish"
    if d in DOWN:
        return "bearish"
    if stance in ("neutral", "mixed"):
        return "neutral"
    return None


def latest_views(claims):
    """{(asset, bucket): {source: latest view claim}} over view claims."""
    views = defaultdict(dict)
    ordered = sorted(view_claims(claims), key=lambda c: (c.get("published_at") or "", c.get("extracted_at") or ""))
    for c in ordered:
        key = asset_key(c)
        if not key or direction_of(c) is None:
            continue
        bucket = c.get("horizon_bucket") if c.get("horizon_bucket") in BUCKETS else "unspecified"
        views[(key, bucket)][source_of(c)] = c
    return views


def aggregate_views(claims, channel_weights=None):
    """
    Per (asset, horizon bucket) entries in the shape the pulse and its
    charts consume: label, ticker, type, mentions (view claims), channels,
    bull / bear / neutral SOURCE counts (one current view per source),
    conviction-weighted sums, actions, USD price targets, plus the asset
    and bucket the entry is for and the ids of the claims behind it.
    """
    channel_weights = channel_weights or {}
    entries = {}
    mentions = Counter()
    all_actions = defaultdict(Counter)
    all_targets = defaultdict(list)
    claim_ids = defaultdict(list)
    for c in view_claims(claims):
        key = asset_key(c)
        if not key:
            continue
        bucket = c.get("horizon_bucket") if c.get("horizon_bucket") in BUCKETS else "unspecified"
        mentions[(key, bucket)] += 1
        claim_ids[(key, bucket)].append(c.get("claim_id"))
        if c.get("recommendation_action") not in (None, "none", "unclear"):
            all_actions[(key, bucket)][c["recommendation_action"]] += 1
        target = price_target_of(c)
        if target is not None:
            all_targets[(key, bucket)].append(target)
    for (key, bucket), per_source in latest_views(claims).items():
        sample = next(iter(per_source.values()))
        ticker = resolved_ticker(sample)
        entry = {
            "asset": key, "horizon_bucket": bucket,
            "label": display_label(ticker or key, bucket),
            "ticker": ticker,
            "type": _entry_type(ticker, [sample]),
            "mentions": mentions[(key, bucket)], "channels": set(per_source),
            "bull": 0, "bear": 0, "neutral": 0,
            "bull_w": 0.0, "bear_w": 0.0, "neutral_w": 0.0,
            "actions": all_actions[(key, bucket)], "targets": all_targets[(key, bucket)],
            "claim_ids": claim_ids[(key, bucket)],
        }
        for source, c in per_source.items():
            weight = channel_weights.get(source, 1.0) * CONVICTION_WEIGHTS.get(
                {"high": "high", "medium": "medium", "low": "low"}.get(c.get("certainty_level"), "unspecified"), 0.75)
            d = direction_of(c)
            if d == "bullish":
                entry["bull"] += 1
                entry["bull_w"] += weight
            elif d == "bearish":
                entry["bear"] += 1
                entry["bear_w"] += weight
            else:
                entry["neutral"] += 1
                entry["neutral_w"] += weight
        entries[(key, bucket)] = entry
    return entries


def lean(bull, bear):
    """
    The one rule for turning directional counts into a lean, shared by a
    video's tone and a creator's vote on an asset: bullish when at least two
    thirds of the directional claims are bullish (bull >= 2 x bear), bearish
    when at least two thirds are bearish, mixed in between, neutral with no
    directional claim at all. Before it, a single dissenting remark made an
    8-to-3 bullish video "mixed".
    """
    total = bull + bear
    if not total:
        return "neutral"
    if 3 * bull >= 2 * total:
        return "bullish"
    if 3 * bear >= 2 * total:
        return "bearish"
    return "mixed"


def video_tone(claims):
    """
    {video_id: bullish|bearish|neutral|mixed} from each video's view claims
    under `lean`: the speaker's balance of directional views, never a model's
    "overall tone" field.
    """
    per_video = defaultdict(Counter)
    for c in view_claims(claims):
        per_video[c.get("video_id")][direction_of(c) or "neutral"] += 1
    return {vid: lean(counts.get("bullish", 0), counts.get("bearish", 0))
            for vid, counts in per_video.items()}


def _entry_type(ticker, rows):
    """An aggregate entry's asset type: the instrument's when the ticker
    resolves to one (a SpaceX claim the model typed private_company is the
    SPCX stock; a Bitcoin typed stock is the coin), else the latest claim's
    model-written type."""
    inst = instruments.resolve_instrument(ticker, None, None) if ticker else None
    if inst is not None:
        return inst.asset_type
    return LEGACY_TYPE.get(next((c.get("asset_type") for c in reversed(rows) if c.get("asset_type")), None), "other")


# Contractions and possessives stay lower case after an apostrophe
# ("McDonald's", "Dick's"); a name part does not ("O'Reilly").
_APOSTROPHE_TAILS = {"", "S", "T", "D", "M", "LL", "RE", "VE"}


def _capitalize_part(part):
    titled = part.title()
    if len(part) > 3 and titled.startswith("Mc"):
        titled = "Mc" + titled[2:].capitalize()      # McDonald, McKesson
    return titled


def _display_word(word):
    if len(word) <= 2:
        return word
    if "'" in word:
        head, _, tail = word.partition("'")
        return _capitalize_part(head) + "'" + (tail.lower() if tail in _APOSTROPHE_TAILS else _capitalize_part(tail))
    return _capitalize_part(word)


# Market keys whose readable form title case gets wrong ("Dxy", "NIKKEI 225").
_MARKET_LABELS = {
    "DXY": "DXY", "DAX": "DAX", "NIKKEI 225": "Nikkei 225", "RUSSELL 2000": "Russell 2000",
    "USDT DOMINANCE": "USDT dominance", "BITCOIN DOMINANCE": "Bitcoin dominance", "US STOCK MARKET": "US stocks",
    "US TREASURIES": "US Treasuries", "GOLD MINERS": "Gold miners", "SMALL CAPS": "Small caps",
    "MID CAPS": "Mid caps", "CONSUMER DISCRETIONARY": "Consumer discretionary",
    "CONSUMER STAPLES": "Consumer staples", "HEALTHCARE": "Health care",
}


def _display_name(key):
    """Readable label for an un-tickered asset key: "OIL" -> "Oil",
    "MCDONALD'S" -> "McDonald's", "DXY" -> "DXY", keeping short all-caps
    index names ("S&P 500") as they are."""
    if key in _MARKET_LABELS:
        return _MARKET_LABELS[key]
    if any(ch.isdigit() or ch == "&" for ch in key):
        return key
    return " ".join(_display_word(w) for w in key.split())


def aggregate_views_by_asset(claims, channel_weights=None):
    """
    The reader-facing aggregation the weekly pulse uses: one entry per ASSET
    (not per horizon bucket), each creator casting one vote.

    A creator's vote is the `lean` of the directional view claims in their
    most recent video on the asset within `claims` (callers pass one window):
    several claims in one video no longer resolve to whichever sorted last,
    and a creator bearish short-term but bullish long-term votes "mixed" —
    reported as split, never averaged into one direction. The horizons they
    named are kept per entry (`horizons`) so the report can say "mostly
    long-term". 86% of view claims name no horizon, which is why a per-bucket
    row split (aggregate_views, kept for research analytics) fragmented the
    clearest consensus of the week into weaker rows.

    Entries carry the aggregate_views shape the charts consume (label,
    ticker, type, mentions, channels, bull / bear / neutral, *_w, actions,
    targets) plus `votes` {creator: lean}, `videos`, `horizons` and
    `target_by` {creator: target}. Price targets are one per creator (their
    latest), so one creator repeating a number does not become a consensus
    target; target_by says whose each one is, so a report can keep a
    dissenter's level off the majority's row.
    """
    channel_weights = channel_weights or {}
    per_asset = defaultdict(list)
    for c in view_claims(claims):
        if c.get("review_required"):
            continue
        key = asset_key(c)
        if key:
            per_asset[key].append(c)
    entries = {}
    for key, rows in per_asset.items():
        rows.sort(key=lambda c: (c.get("published_at") or "", c.get("extracted_at") or ""))
        by_source = defaultdict(list)
        for c in rows:
            by_source[source_of(c)].append(c)
        ticker = next((t for t in (resolved_ticker(c) for c in reversed(rows)) if t), None)
        entry = {
            "asset": key, "label": ticker or _display_name(key), "ticker": ticker,
            "type": _entry_type(ticker, rows),
            "mentions": len(rows), "videos": len({c.get("video_id") for c in rows}),
            "channels": set(by_source), "votes": {},
            "bull": 0, "bear": 0, "neutral": 0, "bull_w": 0.0, "bear_w": 0.0, "neutral_w": 0.0,
            "actions": Counter(), "targets": [], "target_by": {}, "horizons": Counter(),
            "claim_ids": [c.get("claim_id") for c in rows],
        }
        for source, own in by_source.items():
            latest_video = own[-1].get("video_id")
            latest = [c for c in own if c.get("video_id") == latest_video]
            directions = Counter(direction_of(c) for c in latest)
            vote = lean(directions.get("bullish", 0), directions.get("bearish", 0))
            entry["votes"][source] = vote
            weight = channel_weights.get(source, 1.0)
            slot = {"bullish": "bull", "bearish": "bear"}.get(vote, "neutral")
            entry[slot] += 1
            entry[slot + "_w"] += weight
            for c in latest:
                if c.get("horizon_bucket") in HORIZON_LABELS and direction_of(c) in ("bullish", "bearish"):
                    entry["horizons"][c["horizon_bucket"]] += 1
            action = next((c["recommendation_action"] for c in reversed(own)
                           if c.get("recommendation_action") not in (None, "none", "unclear")), None)
            if action:
                entry["actions"][action] += 1
            target = next((t for t in (price_target_of(c) for c in reversed(own)) if t is not None), None)
            if target is not None:
                entry["targets"].append(target)
                entry["target_by"][source] = target
        entries[key] = entry
    return entries


def video_dates(claims):
    dates = {}
    for c in claims:
        d = claim_date(c)
        if d and (c.get("video_id") not in dates or d < dates[c.get("video_id")]):
            dates[c.get("video_id")] = d
    return dates


_NOT_A_POSITION = {"no_position", "not_stated", "unclear"}
# A disclosure must say the speaker holds or traded something; the extractor
# once read "I'm not invested in SpaceX", "a company I used to own" and "we
# know Macy's" as ownership.
_OWNERSHIP_CUE = re.compile(
    r"\b(own|owned|holding|holdings|position|bought|buy|buying|invest\w*|shares|stake|portfolio|added|"
    r"sold|trimmed|long|investiert|gekauft|halten|halte|tranche|eingestiegen|depot|positionen)\b", re.I)
# Contractions ("I don't already own", "it isn't in my portfolio", "I don't
# have it") are read clause by clause by claims.portfolio_position: a "n't"
# here never matched (no word boundary inside "don't"), and a 30-character
# window would reach across "but" into the next clause.
_NEGATED_OWNERSHIP = re.compile(
    r"\b(not|never|no longer|used to)\b[^.]{0,30}\b(own|owned|invest\w*|hold\w*|position)\b|"
    r"\bnicht\b[^.]{0,30}\b(investiert|drin|gekauft)\b|\bkeine?n?\s+position", re.I)


def _states_a_position(evidence, subject=None):
    """The evidence says the speaker holds (or held) the asset. Stored claims
    keep the position their extraction gave them, so this re-reads the
    evidence: a disclosure whose evidence says the speaker holds none
    ("this is not in my portfolio") is never reported as a holding."""
    if not evidence:
        return True  # nothing to check against; hand-built rows in tests
    return (bool(_OWNERSHIP_CUE.search(evidence)) and not _NEGATED_OWNERSHIP.search(evidence)
            and _claims.portfolio_position(evidence, subject) != "no_position")


# What the creator says they DID, read from the evidence, in priority order:
# "I sold half" is a trim, not an exit. A stored position says what they hold
# ("owns"); an action outranks it, so "I sold my PayPal stock earlier this
# year" reads "sold PYPL", never "owns PYPL".
_DISCLOSED_ACTIONS = (
    ("trimmed", re.compile(
        r"\b(sold (half|a (third|quarter|portion|part|bit|little)|some|part|\d+ ?/ ?\d+|\d+ ?(%|percent))|"
        r"trimm(ed|ing)|cut\b[^.]{0,30}\bposition|reduced (my|our) (position|stake)|teilweise verkauft)", re.I)),
    ("sold", re.compile(
        r"\b(i sold|we sold|i'?ve sold|sold (my|our|all|out|everything)|no longer (own|hold)|exited|verkauft)\b",
        re.I)),
    ("added", re.compile(
        r"\b(bought|added|add(ing)? (more|to (my|our))|buying more|purchased|doubled (my|our) (position|stake)|"
        r"nachgekauft|gekauft|aufgestockt)\b", re.I)),
)
_ACTION_PRIORITY = {action: i for i, (action, _) in enumerate(_DISCLOSED_ACTIONS)}
_TRADE_ACTIONS = frozenset(_ACTION_PRIORITY)
# A negated action ("I haven't sold") or one describing a holding ("every
# share that I've bought", "the stock I bought") rather than reporting a
# trade ("I announced that I added to my position" is still a trade).
_ACTION_NEGATION = re.compile(
    r"(\b(not|never|nicht|nie|kein\w*)\b|n't\b)[^.!?,;]{0,20}$"
    r"|\b(shares?|stocks?|positions?|everything|which)\s+(that\s+)?(i|we)\s*('ve|'d|\s+have|\s+had)?\s*$", re.I)


def disclosed_action(evidence, subject=None):
    """
    "sold", "trimmed" or "added" when the evidence says the creator traded
    the asset, else None. A negated action ("I haven't sold") does not
    count. When the evidence states two different actions ("I sold Tesla and
    bought Nvidia"), the one nearest the subject wins.
    """
    ev = evidence if isinstance(evidence, str) else ""
    found = {}
    for action, pattern in _DISCLOSED_ACTIONS:
        for m in pattern.finditer(ev):
            if not _ACTION_NEGATION.search(ev[max(0, m.start() - 30):m.start()]):
                found[action] = m.start()
                break
    if "trimmed" in found and "sold" in found and abs(found["trimmed"] - found["sold"]) <= 12:
        del found["sold"]   # "I sold half": one trim, read twice
    if not found:
        return None
    position = None
    if len(found) > 1 and isinstance(subject, str) and subject.strip():
        position = ev.lower().find(" ".join(subject.split()).lower())
    if position is not None and position >= 0:
        return min(found, key=lambda a: (abs(found[a] - position), _ACTION_PRIORITY[a]))
    return min(found, key=_ACTION_PRIORITY.get)


def portfolio_disclosures(claims, start=None, end=None):
    """
    The separate portfolio-disclosure report: who disclosed what position in
    which asset. Read from claim_type=portfolio_disclosure records only —
    these never enter stance analytics, and a view never enters this report.
    `position` is the action the evidence states (disclosed_action: sold /
    trimmed / added) when it states one, else the stored position; a stated
    exit ("I no longer own it") is reported as sold, not dropped as a
    negated holding.
    """
    rows = []
    for c in claims:
        if c.get("claim_type") != "portfolio_disclosure" or c.get("schema_version") == "legacy":
            continue
        if start is not None and not in_window(c, start, end):
            continue
        action = disclosed_action(c.get("evidence_text"), c.get("subject_mention"))
        if action is None and (c.get("portfolio_disclosure") in _NOT_A_POSITION
                               or not _states_a_position(c.get("evidence_text"), c.get("subject_mention"))):
            continue
        rows.append({
            "source": source_of(c), "asset": asset_key(c), "ticker": resolved_ticker(c),
            "position": action or c.get("portfolio_disclosure"),
            "date": (claim_date(c).isoformat() if claim_date(c) else None),
            "evidence": c.get("evidence_text"), "claim_id": c.get("claim_id"),
            "review_required": bool(c.get("review_required")),
        })
    return sorted(rows, key=lambda r: (r["date"] or "", r["source"], r["asset"] or ""))


POSITION_LABELS = {
    "owns_unspecified": "owns", "long": "long", "short": "short", "bought": "bought",
    "sold": "sold", "added": "added to", "trimmed": "trimmed", "exited": "exited",
}
MAX_DISCLOSURE_SOURCES = 10


def format_portfolio_disclosures(rows):
    """One line per source, grouped by what they said they did: "• Source —
    owns ADBE, MSFT, TSLA*". Unresolved and non-instrument subjects are left
    out, a repeated (source, asset) disclosure is listed once, and * marks a
    disclosure still awaiting review."""
    grouped = disclosure_groups(rows)
    if not grouped:
        return ""
    lines = ["💼 Portfolio disclosures (what creators say they hold; not counted in consensus):"]
    for source, parts in grouped[:MAX_DISCLOSURE_SOURCES]:
        lines.append(f"• {source} — {'; '.join(f'{p} {names}' for p, names in parts)}")
    if len(grouped) > MAX_DISCLOSURE_SOURCES:
        lines.append(f"…and {len(grouped) - MAX_DISCLOSURE_SOURCES} more creators.")
    return "\n".join(lines)


def disclosure_groups(rows):
    """[(source, [(position, "A, B")])], most assets first. Only reviewed-clean
    rows with a resolved ticker reach a reader: an unverified or unresolved
    disclosure ("UNITED HEALTH AKTIE*") is maintainer material, not news.
    One line per (source, ticker): the latest disclosure wins (by date; on
    one date a trade outranks a holding statement, then row order), so a
    creator who bought and later sold reads "sold"."""
    def rank(r):
        return r.get("date") or "", r.get("position") in _TRADE_ACTIONS

    latest = {}
    for r in rows:
        if r.get("review_required") or not r.get("ticker") or is_placeholder_asset(r["asset"]):
            continue
        key = (r["source"], r["ticker"])
        if key not in latest or rank(r) >= rank(latest[key]):
            latest[key] = r
    grouped = defaultdict(lambda: defaultdict(set))
    for (source, ticker), r in latest.items():
        position = POSITION_LABELS.get(r["position"], str(r["position"] or "mentions").replace("_", " "))
        grouped[source][position].add(ticker)
    ordered = sorted(grouped, key=lambda s: (-sum(len(a) for a in grouped[s].values()), s))
    return [(s, [(p, ", ".join(sorted(a))) for p, a in sorted(grouped[s].items())]) for s in ordered]


# Tickers that are both a coin and a US listing (Stacks / Seagate, Chainlink /
# Interlink, ...): only for these does a claim's explicit type choose the
# namespace over the instrument registry.
COLLIDING_TICKERS = frozenset({"STX", "LINK", "SOL", "COMP", "UNI", "DOT", "NEAR", "ONE",
                               # coins whose ticker is also a NYSE listing: Sun Communities (SUI),
                               # Vermilion Energy (VET), AXIS Capital (AXS), Forum Energy (FET),
                               # TRX Gold (TRX)
                               "SUI", "VET", "AXS", "FET", "TRX"})


def priceable_symbol(entry_or_claim):
    """
    Internal price symbol for a view entry / claim, or None. The resolved
    ticker decides, registry first (instruments.resolve_instrument): a
    SpaceX claim the model typed private_company prices as spcx.us, and a
    Bitcoin typed "stock" as btcusd, never the btc.us Grayscale trust. The
    model's asset_type only matters where the registry cannot decide:
      - a COLLIDING_TICKERS ticker whose explicit stock / etf / crypto type
        contradicts the registry is priced in the type's namespace (a listing
        as "<t>.us", a coin only when it is a known coin), never guessed;
      - a known coin (claims.crypto_tickers) or a crypto-typed ticker prices
        as "<t>usd";
      - any other unregistered stock / etf / private_company / other /
        untyped ticker as "<t>.us".
    UNPRICEABLE_TICKERS and market types (index, commodity, ...) give None.
    """
    ticker = (resolved_ticker(entry_or_claim) or "").upper()
    if not ticker or ticker in UNPRICEABLE_TICKERS:
        return None
    asset_type = entry_or_claim.get("type") or entry_or_claim.get("asset_type")
    explicit = {"stock": "listing", "etf": "listing", "crypto": "coin"}.get(asset_type)
    inst = instruments.resolve_instrument(ticker, None, None)
    if inst is not None:
        kind = "coin" if inst.asset_type == "crypto" else "listing"
        if not (ticker in COLLIDING_TICKERS and explicit and explicit != kind):
            return inst.symbol
    coins = _claims.crypto_tickers()
    if ticker in COLLIDING_TICKERS:
        if explicit == "listing":
            return f"{ticker.lower()}.us"
        if explicit == "coin" and ticker in coins:
            return f"{ticker.lower()}usd"
        return None
    if ticker in coins or asset_type == "crypto":
        return f"{ticker.lower()}usd"
    if asset_type in ("stock", "etf", "private_company", "other", None):
        return f"{ticker.lower()}.us"
    return None
