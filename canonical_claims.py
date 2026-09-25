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
"""
import re
from collections import Counter, defaultdict
from datetime import datetime

from claims import carries_view, is_headline_claim
import research_state
from signals_data import CONVICTION_WEIGHTS, UNPRICEABLE_TICKERS, canonical_ticker

BUCKETS = ("short", "medium", "long", "unspecified")
# How a bucket reads in the report. The unspecified bucket carries no tag: it
# is most claims (a speaker rarely names a horizon), and "[unspecified]" on
# every row buried the asset name and was clipped off the charts.
HORIZON_LABELS = {"short": "short-term", "medium": "medium-term", "long": "long-term"}

# Unresolved subjects that are references, not assets: the extractor keeps the
# spoken phrase when coreference fails ("this business", "the company's"), and
# a portfolio or cash position is not an instrument. They carry no ticker and
# would otherwise count as distinct "assets" in consensus and new-on-radar.
_PLACEHOLDER_FIRST_WORDS = {"THIS", "THAT", "THE", "THESE", "THOSE", "IT", "ITS", "MY", "OUR", "THEIR", "HIS", "HER"}
_NON_ASSETS = {"CASH", "PORTFOLIO", "MY PORTFOLIO", "WHITE COUNT", "PRICE", "RECESSION",
               "COMPUTE", "CLARITY ACT", "MARKET", "THE MARKET", "STOCKS", "EVERYTHING"}
# Chart-reading vocabulary the extractor sometimes keeps as the subject
# ("wave two or B wave", "the SOL/BTC chart").
_NON_ASSET_PATTERN = re.compile(r"\bWAVES?\b|\bCHART\b|\bCOUNT\b")

# Spellings of one un-tickered theme folded to one key, so oil talk is not
# split across OIL, OIL PRICES and WTI (each a single creator on its own).
MACRO_ALIASES = {
    "OIL PRICES": "OIL", "OIL PRICE": "OIL", "CRUDE OIL": "OIL", "CRUDE": "OIL", "WTI": "OIL",
    "BRENT": "OIL", "ÖL": "OIL", "ÖLPREIS": "OIL",
    "MIDCAPS": "MID CAPS", "MID-CAPS": "MID CAPS",
    "BITCOIN DOMINANZ": "BITCOIN DOMINANCE", "BTC DOMINANCE": "BITCOIN DOMINANCE",
    "NASDAC": "NASDAQ", "NASDAQ 100": "NASDAQ", "NASDAQ-100": "NASDAQ",
    "S&P": "S&P 500", "SP500": "S&P 500", "S&P500": "S&P 500",
    "GOLD PRICE": "GOLD", "GOLDPREIS": "GOLD",
}

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
    """True for an unresolved reference phrase or a non-instrument."""
    if ticker or not key:
        return not key
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


def load_canonical_claims(state=None, path=None, conditions=None):
    """Active, non-legacy, non-repeated claims with condition outcomes applied."""
    state = state or research_state.load_state()
    rows = research_state.load_active_claims(state, path=path, include_legacy=False)
    evaluations = conditions if conditions is not None else research_state.load_condition_evaluations()
    out = []
    for row in rows:
        if row.get("repeat_of_claim_id"):
            continue
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
    through the curated alias tables, else the spoken name."""
    ticker = canonical_ticker({"ticker": claim.get("ticker"), "name": claim.get("canonical_entity_name")
                               or claim.get("subject_mention")})
    if ticker:
        return ticker
    key = " ".join((claim.get("canonical_entity_name") or claim.get("subject_mention") or "").split()).upper() or None
    key = MACRO_ALIASES.get(key, key)
    return None if is_placeholder_asset(key) else key


def resolved_ticker(claim):
    """The claim's ticker after the curated alias tables, or None."""
    return canonical_ticker({"ticker": claim.get("ticker"), "name": claim.get("canonical_entity_name")
                             or claim.get("subject_mention")})


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
        ticker = canonical_ticker({"ticker": sample.get("ticker"), "name": sample.get("canonical_entity_name")
                                   or sample.get("subject_mention")})
        entry = {
            "asset": key, "horizon_bucket": bucket,
            "label": display_label(ticker or key, bucket),
            "ticker": ticker,
            "type": LEGACY_TYPE.get(sample.get("asset_type"), "other"),
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


def _display_name(key):
    """Readable label for an un-tickered asset key: "OIL" -> "Oil", keeping
    short all-caps index names ("S&P 500") as they are."""
    if any(ch.isdigit() or ch == "&" for ch in key):
        return key
    return " ".join(w if len(w) <= 2 else w.title() for w in key.split())


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
    targets) plus `votes` {creator: lean}, `videos`, and `horizons`.
    Price targets are one per creator (their latest), so one creator
    repeating a number does not become a consensus target.
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
            "type": LEGACY_TYPE.get(next((c.get("asset_type") for c in reversed(rows) if c.get("asset_type")), None), "other"),
            "mentions": len(rows), "videos": len({c.get("video_id") for c in rows}),
            "channels": set(by_source), "votes": {},
            "bull": 0, "bear": 0, "neutral": 0, "bull_w": 0.0, "bear_w": 0.0, "neutral_w": 0.0,
            "actions": Counter(), "targets": [], "horizons": Counter(),
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
_NEGATED_OWNERSHIP = re.compile(
    r"\b(not|n't|never|no longer|used to)\b[^.]{0,30}\b(own|owned|invest\w*|hold\w*|position)\b|"
    r"\bnicht\b[^.]{0,30}\b(investiert|drin|gekauft)\b|\bkeine?n?\s+position", re.I)


def _states_a_position(evidence):
    if not evidence:
        return True  # nothing to check against; hand-built rows in tests
    return bool(_OWNERSHIP_CUE.search(evidence)) and not _NEGATED_OWNERSHIP.search(evidence)


def portfolio_disclosures(claims, start=None, end=None):
    """
    The separate portfolio-disclosure report: who disclosed what position in
    which asset. Read from claim_type=portfolio_disclosure records only —
    these never enter stance analytics, and a view never enters this report.
    """
    rows = []
    for c in claims:
        if c.get("claim_type") != "portfolio_disclosure" or c.get("schema_version") == "legacy":
            continue
        if start is not None and not in_window(c, start, end):
            continue
        if c.get("portfolio_disclosure") in _NOT_A_POSITION or not _states_a_position(c.get("evidence_text")):
            continue
        rows.append({
            "source": source_of(c), "asset": asset_key(c), "ticker": resolved_ticker(c),
            "position": c.get("portfolio_disclosure"), "date": (claim_date(c).isoformat() if claim_date(c) else None),
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
    disclosure ("UNITED HEALTH AKTIE*") is maintainer material, not news."""
    grouped = defaultdict(lambda: defaultdict(set))
    for r in rows:
        if r.get("review_required") or not r.get("ticker") or is_placeholder_asset(r["asset"]):
            continue
        position = POSITION_LABELS.get(r["position"], str(r["position"] or "mentions").replace("_", " "))
        grouped[r["source"]][position].add(r["ticker"])
    ordered = sorted(grouped, key=lambda s: (-sum(len(a) for a in grouped[s].values()), s))
    return [(s, [(p, ", ".join(sorted(a))) for p, a in sorted(grouped[s].items())]) for s in ordered]


def priceable_symbol(entry_or_claim):
    """Internal price symbol for a view entry / claim, or None."""
    ticker = (entry_or_claim.get("ticker") or "").upper()
    if not ticker or ticker in UNPRICEABLE_TICKERS:
        return None
    asset_type = entry_or_claim.get("type") or entry_or_claim.get("asset_type")
    if asset_type in ("stock", "etf"):
        return f"{ticker.lower()}.us"
    if asset_type == "crypto":
        return f"{ticker.lower()}usd"
    return None
