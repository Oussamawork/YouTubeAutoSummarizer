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
from collections import Counter, defaultdict
from datetime import datetime

from claims import carries_view, is_headline_claim
import research_state
from signals_data import CONVICTION_WEIGHTS, UNPRICEABLE_TICKERS, canonical_ticker

BUCKETS = ("short", "medium", "long", "unspecified")
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
    return " ".join((claim.get("canonical_entity_name") or claim.get("subject_mention") or "").split()).upper() or None


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
        if c.get("currency") in (None, "USD"):
            if c.get("target_kind") == "absolute_value" and isinstance(c.get("target_value"), (int, float)):
                all_targets[(key, bucket)].append(float(c["target_value"]))
            elif c.get("target_kind") == "range" and all(isinstance(c.get(k), (int, float)) for k in ("target_low", "target_high")):
                all_targets[(key, bucket)].append((c["target_low"] + c["target_high"]) / 2.0)
    for (key, bucket), per_source in latest_views(claims).items():
        sample = next(iter(per_source.values()))
        ticker = canonical_ticker({"ticker": sample.get("ticker"), "name": sample.get("canonical_entity_name")
                                   or sample.get("subject_mention")})
        entry = {
            "asset": key, "horizon_bucket": bucket,
            "label": f"{ticker or key} [{bucket}]",
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


def video_tone(claims):
    """
    {video_id: bullish|bearish|neutral|mixed} from each video's view claims:
    the speaker's balance of directional views, never a model's "overall
    tone" field. Videos with view claims of neither direction are neutral.
    """
    per_video = defaultdict(Counter)
    for c in view_claims(claims):
        per_video[c.get("video_id")][direction_of(c) or "neutral"] += 1
    tone = {}
    for vid, counts in per_video.items():
        bull, bear = counts.get("bullish", 0), counts.get("bearish", 0)
        tone[vid] = "mixed" if bull and bear else "bullish" if bull else "bearish" if bear else "neutral"
    return tone


def video_dates(claims):
    dates = {}
    for c in claims:
        d = claim_date(c)
        if d and (c.get("video_id") not in dates or d < dates[c.get("video_id")]):
            dates[c.get("video_id")] = d
    return dates


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
        rows.append({
            "source": source_of(c), "asset": asset_key(c), "ticker": c.get("ticker"),
            "position": c.get("portfolio_disclosure"), "date": (claim_date(c).isoformat() if claim_date(c) else None),
            "evidence": c.get("evidence_text"), "claim_id": c.get("claim_id"),
            "review_required": bool(c.get("review_required")),
        })
    return sorted(rows, key=lambda r: (r["date"] or "", r["source"], r["asset"] or ""))


def format_portfolio_disclosures(rows):
    if not rows:
        return ""
    lines = ["💼 Portfolio disclosures (ownership statements; not views, not counted in consensus):"]
    for r in rows[:20]:
        flag = " (unverified)" if r["review_required"] else ""
        lines.append(f"• {r['source']} — {r['asset'] or 'unresolved'}: {r['position']}{flag}")
    if len(rows) > 20:
        lines.append(f"…and {len(rows) - 20} more.")
    return "\n".join(lines)


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
