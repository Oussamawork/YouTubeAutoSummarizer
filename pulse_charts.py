"""Weekly pulse charts: renders the signals aggregation as PNG images for the
Telegram photo album that accompanies the text pulse (see market_pulse.py),
and the two creators-vs-market charts drawn from a creator_tracker record
(build_tracker_chart_data / render_tracker_charts): each creator's calls
against the market over the same days, and how the calls creators agree on
have moved since.

Written for a non-technical reader: every chart carries a plain-English
headline, a one-line "How to read" hint on the image itself, word-based axis
labels ("all bearish / split / all bullish" rather than -1/0/+1), and direct
value labels instead of a legend wherever possible. The tracker charts never
rank creators: rows follow the number of calls, small samples are faded, and
every rate keeps its denominator.

The data-prep functions are pure (no matplotlib import) so they stay testable
in environments without the plotting stack; rendering imports matplotlib
lazily and every chart is best-effort — a failed chart is logged and skipped,
never raised, so the text pulse always goes out.
"""
import os
import textwrap
from datetime import date, datetime, timedelta

import creator_tracker
from log import log_warn

from signals_data import (
    CONVICTION_WEIGHTS,  # noqa: F401  (documented dependency of net_stance)
    NET_THRESHOLD,
    net_stance,
    _direction,
    _directional_mentions,
    _overall_tone,
    _parse_date,
    _in_window,
)

# Light-surface palette shared with the artifact mockups; PNGs render one
# fixed theme (Telegram photos have no dark mode).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BULL = "#2a78d6"
BEAR = "#e34948"
NEUTRAL = "#e6e5e0"
MIXED = "#c9c8c1"

DISCLAIMER = "Creator opinions, not investment advice."
# The tracker charts carry the tracker's own label too: a call marked to
# market is not a verdict on a forecast whose horizon has not ended.
TRACKER_DISCLAIMER = (f"{creator_tracker.REPORT_LABEL[:1].upper()}{creator_tracker.REPORT_LABEL[1:]}. "
                      f"{DISCLAIMER}")

# Telegram shows an album photo about 380px wide before it is tapped. The
# first charts were 10in wide, so 10pt text arrived ~5px tall; at 6.4in the
# same point sizes land ~1.6x larger. Header and footer are sized in inches.
FIG_WIDTH = 6.4
DPI = 200
HEADER_IN = 1.35   # title + subtitle + two-line how-to
FOOTER_IN = 0.35

MAX_CONSENSUS_ROWS = 10
MAX_FLIP_ROWS = 8
MAX_UPSIDE_ROWS = 8
MAX_MAP_POINTS = 15
# A scatter point needs this many directional calls to be worth plotting;
# below that the position is mostly noise.
MIN_MAP_DIRECTIONAL = 3
TONE_WEEKS = 4
# The bull-bear spread runs over a rolling quarter — it is a trend line, and a
# trend needs room. Below MATURE_SPREAD_WEEKS the chart says so on its face
# rather than inviting the reader to over-read a three-point line.
SPREAD_WEEKS = 12
MATURE_SPREAD_WEEKS = 9  # ~2 months of weekly points

# Creators vs the market (the tracker charts, from a creator_tracker record).
MAX_TRACKER_CREATOR_ROWS = 10
MAX_OUTCOME_ROWS = 8
# Each tracker chart needs enough rows to compare; fewer is a sentence, and
# the text report already carries it.
MIN_TRACKER_CREATORS = 3
MIN_OUTCOME_ROWS = 3
# A creator whose median call is within this many percentage points of the
# market is level with it, neither ahead nor behind: +0.03 points is noise,
# and the title and the bar colour say so.
LEVEL_BAND_POINTS = 0.5

# The 6.4in figure fits ~42 bold characters at 14.5pt whatever the glyphs.
TITLE_MAX_CHARS = 42
# The upside title is mostly lowercase ("Best-supported target: ..."):
# measured, 46 characters of it render at most ~5.85in in DejaVu Sans Bold
# 14.5pt, inside the ~6.0in the header has.
UPSIDE_TITLE_MAX_CHARS = 46


# ---------------------------------------------------------------------------
# Pure data prep (no matplotlib)

def window_label(start, end):
    """Human date range for a (start, end] window: 'Aug 10 - 16, 2026'.
    `start` is exclusive (the dedup-window convention), so the first shown
    day is start+1."""
    first = start + timedelta(days=1)
    if first.year != end.year:
        return f"{first.strftime('%b %-d, %Y')} - {end.strftime('%b %-d, %Y')}"
    if first.month != end.month:
        return f"{first.strftime('%b %-d')} - {end.strftime('%b %-d')}, {end.year}"
    return f"{first.strftime('%b %-d')} - {end.day}, {end.year}"


def consensus_rows(current, limit=MAX_CONSENSUS_ROWS, min_creators=1, houses=None):
    """Top assets for the consensus board, ranked like the text pulse (most
    directional calls first). Assets nobody took a side on are skipped — an
    empty bar says nothing — and so are assets with fewer than
    `min_creators` creators taking a side; given `houses`
    (creator_tracker.creator_houses), two channels of one house count once
    toward that minimum, as in the text pulse. For per-creator entries
    (canonical_claims.aggregate_views_by_asset) `neutral` counts the creators
    who were split."""
    ranked = sorted(
        current.items(),
        key=lambda kv: (-_directional_mentions(kv[1]), -abs(net_stance(kv[1])),
                        -kv[1]["mentions"], kv[1]["label"]),
    )
    rows = []
    for _, entry in ranked:
        if not _directional_mentions(entry):
            continue
        split = (sum(1 for v in entry["votes"].values() if v == "mixed")
                 if "votes" in entry else entry["neutral"])
        if houses is not None and "votes" in entry:
            sided = len({houses.get(c, c) for c, v in entry["votes"].items() if v != "neutral"})
        else:
            sided = _directional_mentions(entry) + ("votes" in entry) * split
        if sided < min_creators:
            continue
        rows.append({
            "label": entry["label"],
            "bull": entry["bull"],
            "bear": entry["bear"],
            "neutral": split,
            "channels": len(entry["channels"]),
        })
        if len(rows) >= limit:
            break
    return rows


def flip_rows(current, previous, limit=MAX_FLIP_ROWS, min_votes=1):
    """Consensus flips with the evidence behind them: weighted net stance in
    both windows plus the directional-call counts, so the chart can draw a
    one-vote flip thinner than a well-attended reversal."""
    rows = []
    for key, entry in current.items():
        prev = previous.get(key)
        if not prev:
            continue
        cur_score, prev_score = net_stance(entry), net_stance(prev)
        if {_direction(cur_score), _direction(prev_score)} != {"bullish", "bearish"}:
            continue
        if min(_directional_mentions(prev), _directional_mentions(entry)) < min_votes:
            continue
        rows.append({
            "label": entry["label"],
            "from_score": prev_score,
            "to_score": cur_score,
            "from_votes": _directional_mentions(prev),
            "to_votes": _directional_mentions(entry),
        })
    rows.sort(key=lambda r: (-min(r["from_votes"], r["to_votes"]), r["label"]))
    return rows[:limit]


def tone_weeks(records, today, num_weeks=TONE_WEEKS):
    """Weekly market-sentiment buckets for the trailing `num_weeks` 7-day
    windows, oldest first. Windows with no records are dropped; a window the
    dataset only partially covers is flagged so the chart can say so. Each
    week carries `weeks_ago` (0 = the current window) so a chart plotting
    weeks on a time axis can tell a skipped week from a consecutive one."""
    dates = sorted(d for d in (_parse_date(r.get("date")) for r in records) if d)
    if not dates:
        return []
    earliest = dates[0]
    weeks = []
    end = today
    for weeks_ago in range(num_weeks):
        start = end - timedelta(days=7)
        bucket = [r for r in records if _in_window(r, start, end)]
        if bucket:
            tone = _overall_tone(bucket)
            # No year in the row label: it's a rolling window, and the
            # shorter text keeps the row labels inside the figure.
            first = start + timedelta(days=1)
            weeks.append({
                "label": f"{first.strftime('%b %-d')} - {end.strftime('%b %-d')}",
                "short_label": end.strftime("%b %-d"),
                "weeks_ago": weeks_ago,
                "n": len(bucket),
                "bullish": tone.get("bullish", 0),
                "bearish": tone.get("bearish", 0),
                "neutral": tone.get("neutral", 0),
                "mixed": tone.get("mixed", 0),
                "partial": earliest > start + timedelta(days=1),
            })
        end = start
    weeks.reverse()
    return weeks


def spread_from_weeks(weeks):
    """The bull-bear spread rows from tone-week rows (see spread_weeks)."""
    return [
        {
            "label": week["label"],
            "short_label": week["short_label"],
            "weeks_ago": week["weeks_ago"],
            "n": week["n"],
            "spread": (week["bullish"] - week["bearish"]) / week["n"] * 100.0,
            "partial": week["partial"],
        }
        for week in weeks
    ]


def spread_weeks(records, today, num_weeks=SPREAD_WEEKS):
    """
    The bull-bear spread per week: share of videos expecting a rise minus the
    share expecting a fall, in percentage points. This is the headline number
    sentiment surveys (AAII's weekly investor survey being the reference) lead
    with, because one signed line answers "is optimism building or fading?"
    without the reader decoding a stack of shares.

    Runs over a rolling quarter rather than the tone chart's four weeks: the
    spread is a trend instrument and only says something once there is a trend
    to see. With a handful of weeks it is honest but thin — expect it to earn
    its place around the two-to-three-month mark, when seasonal noise starts
    averaging out and a turn in the line is distinguishable from one loud week.
    Until then the chart labels itself as early days (see MATURE_SPREAD_WEEKS).
    """
    return spread_from_weeks(tone_weeks(records, today, num_weeks=num_weeks))


def conviction_points(current, limit=MAX_MAP_POINTS, min_directional=MIN_MAP_DIRECTIONAL):
    """Assets for the agreement-vs-attention scatter: net stance (x) and
    directional-call count (y), for assets with enough calls to place. The cap
    keeps a crowded week readable, but never at the cost of the story: assets
    that are NOT part of the bullish pile (bearish or contested) are re-added
    past the cap — they are rare, and they are what the chart exists to show."""
    points = [
        {"label": e["label"], "net": net_stance(e), "directional": _directional_mentions(e)}
        for e in current.values()
        if _directional_mentions(e) >= min_directional
    ]
    points.sort(key=lambda p: (-p["directional"], p["label"]))
    kept = points[:limit]
    extra = [p for p in points[limit:] if p["net"] < NET_THRESHOLD][:3]
    return kept + extra


# A stated target this far from today's price is almost never a share-price
# target: it is a revenue or market-cap figure, a percentage, or a price in
# another unit that slipped past extraction. One such number once put a
# +22,495,163,440% "implied move" in the pulse and flattened the whole chart.
PLAUSIBLE_TARGET_RATIO = (0.2, 5.0)


def plausible_targets(targets, price):
    """The targets within PLAUSIBLE_TARGET_RATIO of the latest price; every
    target when no price is known (nothing to check against)."""
    if not price:
        return list(targets)
    lo, hi = PLAUSIBLE_TARGET_RATIO
    return [t for t in targets if lo * price <= t <= hi * price]


def upside_rows(current, latest_prices, limit=MAX_UPSIDE_ROWS):
    """Implied move from the latest close to the creator price targets, for
    assets that have both. Alongside the average, the low and high targets are
    kept — the industry convention (low/average/high range vs current price)
    and the honest display when one moonshot target would dominate a bare
    average. Percentages compare cleanly across price scales where raw dollar
    targets don't.

    Shared targets lead: rows are ordered by how many targets back them, then
    by implied move, then by label. Ordered by move alone, a single creator's
    multi-year target (+323%) used to top the chart and its title."""
    rows = []
    for key, entry in current.items():
        price = latest_prices.get(key)
        targets = plausible_targets(entry["targets"], price)
        if not price or not targets:
            continue
        target = sum(targets) / len(targets)
        t_lo, t_hi = min(targets), max(targets)
        pct = lambda t: (t - price) / price * 100.0
        rows.append({
            "label": entry["label"],
            "price": price,
            "target": target,
            "t_lo": t_lo,
            "t_hi": t_hi,
            "pct": pct(target),
            "lo": pct(t_lo),
            "hi": pct(t_hi),
            "n_targets": len(targets),
        })
    rows.sort(key=_upside_order)
    return rows[:limit]


def _upside_order(row):
    return -row["n_targets"], -row["pct"], row["label"]


def build_chart_data(records, current, previous, window_start, today, tone=None, videos=None,
                     channels=None, min_creators=1, houses=None):
    """Everything the renderer needs, as plain dicts/lists. `current` and
    `previous` are per-asset entries (canonical_claims.aggregate_views for
    the production pulse, aggregate_assets for the legacy view) for the two
    windows. `tone` (week rows) and the video/channel counts are supplied by
    the canonical pulse, whose tone comes from each video's own view claims;
    without them they are computed from legacy signal records."""
    if tone is None:
        current_records = [r for r in records if _in_window(r, window_start, today)]
        channel_names = {r.get("channel_name") for r in current_records if r.get("channel_name")}
        tone = tone_weeks(records, today)
        videos = len(current_records) if videos is None else videos
        channels = len(channel_names) if channels is None else channels
    return {
        "window": window_label(window_start, today),
        "videos": videos or 0,
        "channels": channels or 0,
        "consensus": consensus_rows(current, min_creators=min_creators, houses=houses),
        "flips": flip_rows(current, previous, min_votes=min_creators),
        "tone": tone[-TONE_WEEKS:] if tone else [],
        "spread": spread_from_weeks(tone),
        "map": conviction_points(current),
        "upside": [],  # filled by the caller once latest prices are known
    }


# ---------------------------------------------------------------------------
# Creators vs the market: pure data prep over a creator_tracker record
# (dict shapes in creator_tracker.track_record; returns there are fractions,
# here they are percentages).

def _as_date(value):
    """A date from a date, datetime or ISO string (a record read back from
    creator_tracker's --json dump); None otherwise."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


def _day(value, weekday=False):
    """'Oct 9' (or 'Fri Oct 9'); None for a missing date."""
    day = _as_date(value)
    if day is None:
        return None
    return f"{day:%a %b} {day.day}" if weekday else f"{day:%b} {day.day}"


def _count(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def creators_vs_market_rows(record, limit=MAX_TRACKER_CREATOR_ROWS, min_calls=None):
    """
    One row per creator for the creators-vs-market chart: label (the
    creator), calls (scored calls), bench_n (those compared with a
    benchmark), beat, median_dir_excess (the creator's median call: its move
    in the called direction minus the market's over the same days, in
    percentage points), small_sample and mostly_guests (a guest-panel
    channel, tagged like the text report does).

    Only creators with a market comparison (bench_n >= 1) and at least
    `min_calls` scored calls (default creator_tracker.MIN_CREATOR_CALLS_SHOWN:
    the text report puts fewer on its 'too few calls' line, and a one-call
    bar would set the chart's scale). Ordered by number of calls, then name,
    never by result.
    """
    min_calls = creator_tracker.MIN_CREATOR_CALLS_SHOWN if min_calls is None else min_calls
    rows = []
    for creator, s in ((record or {}).get("creators") or {}).items():
        calls, median = s.get("calls") or 0, s.get("median_dir_excess")
        if not s.get("bench_n") or median is None or calls < min_calls:
            continue
        rows.append({
            "label": creator,
            "calls": calls,
            "bench_n": s["bench_n"],
            "beat": s.get("beat") or 0,
            "median_dir_excess": 100.0 * median,
            "small_sample": bool(s.get("small_sample")),
            "mostly_guests": bool(s.get("mostly_guests")),
        })
    rows.sort(key=lambda r: (-r["calls"], r["label"]))
    return rows[:max(0, limit)]


def _not_charted(record, rows, min_calls):
    """The creators with scored calls but no bar, and why: too_few calls,
    no_benchmark (only calls on the S&P 500 or Bitcoin themselves) or
    beyond the row limit. Ordered like the rows."""
    shown = {r["label"] for r in rows}
    out = []
    for creator, s in ((record or {}).get("creators") or {}).items():
        if creator in shown:
            continue
        calls = s.get("calls") or 0
        if calls < min_calls:
            reason = "too_few"
        elif not s.get("bench_n") or s.get("median_dir_excess") is None:
            reason = "no_benchmark"
        else:
            reason = "limit"
        out.append({"label": creator, "calls": calls, "reason": reason})
    out.sort(key=lambda r: (-r["calls"], r["label"]))
    return out


def consensus_outcome_rows(record, limit=MAX_OUTCOME_ROWS):
    """
    One row per asset that 2+ creators agree on (record['assets'] with a
    bullish or bearish consensus and creators from at least two houses on
    that side):
    label, direction, n_creators, move (the asset's move since the first of
    their calls, in %), bench_move (the market over the same days, in %, or
    None for the market itself or a missing close), benchmark_label, since
    (that call's entry date) and right (the move went the called way).
    Split assets and single-creator calls are left out. Ordered by number of
    creators, then label.
    """
    rows = []
    for asset, s in ((record or {}).get("assets") or {}).items():
        direction = s.get("consensus")
        if direction not in ("bullish", "bearish"):
            continue
        side = s.get("bulls") if direction == "bullish" else s.get("bears")
        # Two channels of one house (creator_tracker.creator_houses) are one
        # opinion: the row needs creators from two houses.
        if s.get("houses", len(side or ())) < 2 or s.get("move") is None:
            continue
        bench = s.get("bench_move")
        rows.append({
            "label": s.get("label") or asset,
            "direction": direction,
            "n_creators": len(side),
            "move": 100.0 * s["move"],
            "bench_move": None if bench is None else 100.0 * bench,
            "benchmark_label": s.get("benchmark_label"),
            "since": s.get("since"),
            "right": bool(s.get("right")),
        })
    rows.sort(key=lambda r: (-r["n_creators"], r["label"]))
    return rows[:max(0, limit)]


def _fit_title(candidates, limit, fallback):
    """The first candidate within `limit` characters, else the fallback."""
    return next((title for title in candidates if len(title) <= limit), fallback)


def creators_vs_market_title(rows):
    """Takeaway headline for the creators-vs-market chart: how many of the
    creators with a full sample (creator_tracker.MIN_CREATOR_SAMPLE calls or
    more; the faded rows, and the title says so) have a median call more than
    LEVEL_BAND_POINTS ahead of the market so far. A count, never a ranking."""
    fallback = "Creators' calls vs the market so far"
    solid = [r for r in rows if not r.get("small_sample")]
    if not solid:
        return fallback
    ahead = sum(r["median_dir_excess"] > LEVEL_BAND_POINTS for r in solid)
    who = f"{ahead} of {len(solid)} with {creator_tracker.MIN_CREATOR_SAMPLE}+ calls"
    return _fit_title((f"{who} ahead of the market", f"{who} ahead"), TITLE_MAX_CHARS, fallback)


def consensus_outcomes_title(rows):
    """Takeaway headline for the consensus-outcomes chart: how many shared
    calls have moved the way the creators called."""
    fallback = "Shared calls vs the market so far"
    if not rows:
        return fallback
    right = sum(bool(r.get("right")) for r in rows)
    return _fit_title((f"{right} of {_count(len(rows), 'shared call')} moving their way",), TITLE_MAX_CHARS,
                      fallback)


def _prices_stale(record):
    """The tracker report's stale-price rule: no S&P 500 close near the
    requested as-of, or the newest one more than a week before it."""
    if not record or not record.get("calls"):
        return False
    if not record.get("market_data"):
        return True
    requested, as_of = _as_date(record.get("requested_as_of")), _as_date(record.get("as_of"))
    return bool(requested and as_of and as_of < requested - timedelta(days=7))


def build_tracker_chart_data(record):
    """
    Everything the tracker renderer needs from a creator_tracker record, as
    plain dicts/lists: as_of (the effective as-of every call is marked to),
    since (the first call), stale (the record's prices are over a week old),
    creators (creators_vs_market_rows), not_charted (creators with scored
    calls but no bar, and why) and outcomes (consensus_outcome_rows).
    """
    record = record if isinstance(record, dict) else {}
    min_calls = creator_tracker.MIN_CREATOR_CALLS_SHOWN
    creators = creators_vs_market_rows(record, min_calls=min_calls)
    return {
        "as_of": _as_date(record.get("as_of")),
        "since": _as_date(record.get("since")),
        "stale": _prices_stale(record),
        "creators": creators,
        "not_charted": _not_charted(record, creators, min_calls),
        "outcomes": consensus_outcome_rows(record),
    }


# ---------------------------------------------------------------------------
# Rendering (matplotlib, lazy import, best-effort per chart)

def _new_figure(plt, height):
    fig = plt.figure(figsize=(FIG_WIDTH, height), facecolor=SURFACE)
    return fig


def _plot_box(height, bottom_in, left, right=0.95, extra_top_in=0.0):
    """Axes rect [left, bottom, width, height] below the fixed header."""
    top = 1 - (HEADER_IN + extra_top_in) / height
    bottom = bottom_in / height
    return [left, bottom, right - left, top - bottom]


def _chrome(ax):
    """Recessive axes: no box, hairline grid only where a chart asks for it."""
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)


# Wrap widths in characters at each font size. matplotlib's own `wrap=True`
# only breaks at the figure edge, which clips the last word; wrapping here
# keeps every header inside the margin.
META_WRAP = 72
HOWTO_WRAP = 66
NOTE_WRAP = 88   # 8.5pt footnotes
# Line heights of the wrapped context (9.5pt) and how-to (10pt) lines. The
# fixed HEADER_IN fits a one-line context and a two-line how-to; a chart
# with more asks _extra_header_in for the difference.
META_LINE_IN = 0.16
HOWTO_LINE_IN = 0.17


def _label_margin(labels, minimum, fontsize=11.5):
    """Left margin (figure fraction) wide enough for the bold row labels: a
    fixed margin clipped long names off the left edge, leaving only their
    tail visible. ~0.62em per bold character."""
    widest = max((len(str(l)) for l in labels), default=0)
    char_in = fontsize * 0.62 / 72
    return min(0.45, max(minimum, (0.12 + widest * char_in) / FIG_WIDTH))


def _extra_header_in(meta, howto):
    """Inches a header needs beyond HEADER_IN (which fits a one-line context
    and a two-line how-to); pass it as the plot box's extra_top_in and add
    it to the figure height."""
    meta_lines = len(textwrap.wrap(meta, META_WRAP)) or 1
    howto_lines = len(textwrap.wrap(howto, HOWTO_WRAP)) or 1
    return (meta_lines - 1) * META_LINE_IN + max(0, howto_lines - 2) * HOWTO_LINE_IN


def _finish(fig, path, title, meta, howto, disclaimer=DISCLAIMER):
    """Shared header (takeaway title + context + how-to-read) and footer,
    placed in inches from the top so the spacing is the same on every chart
    height. The header lives on the figure, not the axes. A context line
    that wraps pushes the how-to down by its extra lines."""
    h = fig.get_figheight()
    meta = textwrap.fill(meta, META_WRAP)
    howto_top = 0.74 + meta.count("\n") * META_LINE_IN
    fig.text(0.05, 1 - 0.14 / h, title, fontsize=14.5, fontweight="bold", color=INK, va="top")
    fig.text(0.05, 1 - 0.47 / h, meta, fontsize=9.5, color=MUTED, va="top")
    fig.text(0.05, 1 - howto_top / h, textwrap.fill(howto, HOWTO_WRAP), fontsize=10, color=INK2, va="top")
    fig.text(0.05, 0.1 / h, disclaimer, fontsize=8.5, color=MUTED)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)


def _zero_line(ax, n_rows, label):
    """The x = 0 reference line of a row chart (ylim -0.6 .. n_rows - 0.1),
    named just above the top row; the line stops under its name rather than
    running through it."""
    ax.plot([0, 0], [-0.6, n_rows - 0.32], color=MUTED, lw=1, zorder=1)
    ax.text(0, n_rows - 0.25, label, ha="center", va="bottom", fontsize=8.5, color=MUTED)


def _join(labels):
    labels = list(labels)
    return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]


def consensus_title(rows):
    """Takeaway headline for the consensus board from its own rows."""
    bulls = [r["label"] for r in rows if r["bull"] >= 2 * max(r["bear"], 1) and r["bull"] >= 2][:2]
    bears = [r["label"] for r in rows if r["bear"] >= 2 * max(r["bull"], 1) and r["bear"] >= 2][:1]
    if bulls and bears:
        title = f"Bullish on {_join(bulls)}; bearish on {bears[0]}"
        return title if len(title) <= TITLE_MAX_CHARS else f"Bullish on {bulls[0]}; bearish on {bears[0]}"
    if bulls:
        return f"Creators lean bullish on {_join(bulls)}"
    if bears:
        return f"Creators lean bearish on {bears[0]}"
    return "Where creators stand this week"


def _render_consensus(plt, data, path):
    rows = data["consensus"]
    height = HEADER_IN + 0.36 * len(rows) + 0.95
    fig = _new_figure(plt, height)
    left = _label_margin([r["label"] for r in rows], 0.16)
    ax = fig.add_axes(_plot_box(height, 0.72, left, right=0.74))
    _chrome(ax)

    ys = range(len(rows) - 1, -1, -1)
    max_bull = max(r["bull"] for r in rows)
    max_bear = max(r["bear"] for r in rows)
    for y, row in zip(ys, rows):
        if row["bear"]:
            ax.barh(y, -row["bear"], height=0.62, color=BEAR)
            ax.text(-row["bear"] - 0.15, y, str(row["bear"]), ha="right", va="center",
                    fontsize=10.5, fontweight="bold", color=INK)
        if row["bull"]:
            ax.barh(y, row["bull"], height=0.62, color=BULL)
            ax.text(row["bull"] + 0.15, y, str(row["bull"]), ha="left", va="center",
                    fontsize=10.5, fontweight="bold", color=INK)
        note = f"{row['channels']} creator{'s' if row['channels'] != 1 else ''}"
        if row["neutral"]:
            note += f" · {row['neutral']} split"
        ax.text(1.03, y, note, transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=9, color=MUTED)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(list(ys), [r["label"] for r in rows], fontsize=11.5, color=INK)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.set_xlim(-max(max_bear, 1) - 0.9, max(max_bull, 1) + 0.9)
    ax.set_xticks([])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.text(-0.25, -0.03, "← going down", transform=ax.get_xaxis_transform(),
            ha="right", va="top", fontsize=9.5, color=BEAR)
    ax.text(0.25, -0.03, "going up →", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=9.5, color=BULL)

    _finish(
        fig, path,
        consensus_title(rows),
        f"{data['window']} · assets with 2+ creators taking a side",
        "Each number is one creator's view: blue = going up, red = going down.",
    )
    plt.close(fig)


def _render_flips(plt, data, path):
    rows = data["flips"]
    fig = _new_figure(plt, 5.2)
    ax = fig.add_axes([0.25, 0.13, 0.3, 0.6])
    _chrome(ax)

    for score in (-1, 0, 1):
        ax.axhline(score, color=GRID if score else MUTED, lw=1, zorder=0)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xlim(-0.02, 1.02)
    ax.set_yticks([-1, 0, 1], ["all say down", "split", "all say up"],
                  fontsize=10.5, color=INK2)
    ax.set_xticks([0, 1], ["last week", "this week"], fontsize=11, color=INK2)

    # Deterministic label placement: a top-down sweep keeps a minimum gap
    # between labels, then the whole stack shifts up if it ran past the floor.
    def place(labels):
        out, prev = [], None
        for anchor, text, strong, color in sorted(labels, key=lambda l: -l[0]):
            y = anchor if prev is None else min(anchor, prev - 0.16)
            out.append([y, text, strong, color])
            prev = y
        if out and out[-1][0] < -1.2:
            shift = -1.2 - out[-1][0]
            for item in out:
                item[0] += shift
        return out

    # Labels live on the right side only (name + evidence); the left edge
    # keeps just the word scale, so the two never collide.
    right = []
    for row in rows:
        strong = min(row["from_votes"], row["to_votes"]) >= 2
        color = BULL if row["to_score"] > 0 else BEAR
        alpha, lw = (1.0, 3.0) if strong else (0.45, 1.4)
        ax.plot([0, 1], [row["from_score"], row["to_score"]], color=color,
                lw=lw, alpha=alpha, solid_capstyle="round", zorder=2)
        ax.plot(0, row["from_score"], "o", ms=8 if strong else 5,
                color=BULL if row["from_score"] > 0 else BEAR,
                alpha=alpha, mec=SURFACE, mew=1.5, zorder=3)
        ax.plot(1, row["to_score"], "o", ms=8 if strong else 5, color=color,
                alpha=alpha, mec=SURFACE, mew=1.5, zorder=3)
        votes = f"{row['from_votes']}→{row['to_votes']} calls"
        right.append((row["to_score"], f"{row['label']} ({votes})", strong, color))
    for y, text, strong, _ in place(right):
        ax.text(1.06, y, text, ha="left", va="center", fontsize=10.5,
                color=INK if strong else MUTED,
                fontweight="bold" if strong else "normal")

    _finish(
        fig, path,
        f"Consensus flipped on {len(rows)} asset{'s' if len(rows) != 1 else ''}",
        f"{data['window']} vs the week before",
        "Each line is one asset. Thin, faint lines rest on a single creator.",
    )
    plt.close(fig)


def tone_title(weeks):
    """Takeaway headline for the mood chart: this week's bullish share and
    its move against last week."""
    now = weeks[-1]
    share = now["bullish"] / now["n"] if now["n"] else 0
    title = f"{share:.0%} of this week's videos expect a rise"
    if len(weeks) > 1 and weeks[-2]["n"] and weeks[-2]["weeks_ago"] == now["weeks_ago"] + 1:
        before = weeks[-2]["bullish"] / weeks[-2]["n"]
        delta = round((share - before) * 100)
        if abs(delta) >= 3:
            title += f" ({'up' if delta > 0 else 'down'} from {before:.0%})"
    return title


def _render_tone(plt, data, path):
    weeks = data["tone"]
    height = HEADER_IN + 0.62 * len(weeks) + 1.0
    fig = _new_figure(plt, height)
    ax = fig.add_axes(_plot_box(height, 0.78, 0.30, right=0.86))
    _chrome(ax)

    # Edge-anchored 100% bars (the survey-share convention): bearish is
    # anchored to the left edge and bullish to the right, so both headline
    # shares line up across weeks. Mixed and no-lean videos share one gray:
    # two near-identical grays read as one band anyway. Newest week on top.
    any_partial = False
    for i, week in enumerate(weeks):
        n = week["n"]
        bear, bull = week["bearish"] / n, week["bullish"] / n
        middle = (week["mixed"] + week["neutral"]) / n
        left = 0.0
        for share, color in ((bear, BEAR), (middle, MIXED), (bull, BULL)):
            if share > 0:
                ax.barh(i, share, left=left, height=0.6, color=color,
                        edgecolor=SURFACE, linewidth=1.5)
                if color == MIXED and share >= 0.12:
                    ax.text(left + share / 2, i, f"{share:.0%}", ha="center", va="center",
                            fontsize=9, color=INK2)
                left += share
        ax.text(-0.02, i, f"{bear:.0%}", ha="right", va="center",
                fontsize=10.5, fontweight="bold", color=BEAR)
        ax.text(1.02, i, f"{bull:.0%}", ha="left", va="center",
                fontsize=10.5, fontweight="bold", color=BULL)
        label = ("This week" if week["weeks_ago"] == 0 else week["label"]) + ("*" if week["partial"] else "")
        any_partial = any_partial or week["partial"]
        ax.text(-0.14, i + 0.1, label, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=10, fontweight="bold", color=INK)
        ax.text(-0.14, i - 0.24, f"{n} videos", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8.5, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(weeks) - 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BEAR, MIXED, BULL)]
    ax.legend(handles, ["Expect a fall", "Mixed / no lean", "Expect a rise"],
              loc="upper center", bbox_to_anchor=(0.45, -0.03), ncol=3, frameon=False,
              fontsize=9, labelcolor=INK2, handlelength=1.1, handleheight=1.1)
    if any_partial:
        fig.text(0.05, 0.3 / height, "* data collection started mid-week", fontsize=8.5, color=MUTED)

    _finish(
        fig, path,
        tone_title(weeks),
        "Share of analyzed videos expecting a rise (blue) or a fall (red)",
        "Each bar is one week of videos, oldest at the bottom.",
    )
    plt.close(fig)


def _render_spread(plt, data, path):
    """The bull-bear spread as one signed line over a rolling quarter, filled
    to the zero baseline (blue above = net optimism, red below = net
    pessimism). Gaps in the dataset break the line rather than being drawn
    through, so a quiet week never reads as a smooth trend."""
    rows = data["spread"]
    fig = _new_figure(plt, 5.4)
    ax = fig.add_axes([0.14, 0.17, 0.8, 0.5])
    _chrome(ax)

    # x is "weeks ago" negated, so the newest week sits at the right edge and
    # a skipped week leaves a real gap on the axis.
    xs = [-r["weeks_ago"] for r in rows]
    ys = [r["spread"] for r in rows]
    limit = max(60.0, max(abs(v) for v in ys) * 1.25)
    ax.set_ylim(-limit, limit)
    ax.set_xlim(min(xs) - 0.35, max(xs) + 0.35)

    for level in (-50, -25, 25, 50):
        if abs(level) < limit:
            ax.axhline(level, color=GRID, lw=1, zorder=0)
    ax.axhline(0, color=MUTED, lw=1.2, zorder=1)

    # Split the series into runs of consecutive weeks; each run draws as its
    # own filled line so missing weeks stay visible as breaks.
    runs, run = [], [0]
    for i in range(1, len(rows)):
        if rows[i]["weeks_ago"] == rows[i - 1]["weeks_ago"] - 1:
            run.append(i)
        else:
            runs.append(run)
            run = [i]
    runs.append(run)
    for run in runs:
        rx = [xs[i] for i in run]
        ry = [ys[i] for i in run]
        if len(run) > 1:
            ax.fill_between(rx, ry, 0, where=[v >= 0 for v in ry],
                            color=BULL, alpha=0.16, interpolate=True, zorder=1)
            ax.fill_between(rx, ry, 0, where=[v <= 0 for v in ry],
                            color=BEAR, alpha=0.16, interpolate=True, zorder=1)
            ax.plot(rx, ry, color=BULL, lw=2.5, solid_capstyle="round", zorder=2)
        for x, y in zip(rx, ry):
            ax.plot(x, y, "o", ms=8, color=BULL if y >= 0 else BEAR,
                    mec=SURFACE, mew=1.5, zorder=3)

    # Only the latest point carries a number — the line carries the shape.
    last_x, last_y = xs[-1], ys[-1]
    ax.annotate(f"{last_y:+.0f}", (last_x, last_y),
                textcoords="offset points", xytext=(0, 14 if last_y >= 0 else -22),
                ha="center", fontsize=13, fontweight="bold",
                color=BULL if last_y >= 0 else BEAR)

    ax.set_xticks(xs, [r["short_label"] + ("*" if r["partial"] else "") for r in rows],
                  fontsize=9.5, color=MUTED)
    ax.set_yticks([-50, 0, 50], ["−50", "even", "+50"], fontsize=9.5, color=INK2)
    ax.tick_params(colors=MUTED, labelsize=9.5, length=0)

    notes = []
    if any(r["partial"] for r in rows):
        notes.append("* data collection started mid-week")
    if len(rows) < MATURE_SPREAD_WEEKS:
        notes.append(
            f"Only {len(rows)} week{'s' if len(rows) != 1 else ''} of history so far — "
            "this line starts telling a real story after about 2–3 months."
        )
    for i, note in enumerate(notes):
        fig.text(0.05, 0.075 - i * 0.032, note, fontsize=9, color=MUTED)

    _finish(
        fig, path,
        f"Optimism gap: {rows[-1]['spread']:+.0f} points",
        "% of videos expecting a rise minus % expecting a fall",
        "Above the middle line: more optimists than pessimists.",
    )
    plt.close(fig)


def _render_map(plt, data, path):
    points = data["map"]
    fig = _new_figure(plt, 6.0)
    ax = fig.add_axes([0.12, 0.13, 0.82, 0.6])
    _chrome(ax)

    max_dir = max(p["directional"] for p in points)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(0, max_dir * 1.15)
    for x in (-1, 0, 1):
        ax.axvline(x, color=GRID if x else MUTED, lw=1, zorder=0)
    ax.set_xticks([-1, 0, 1], ["all say down", "split", "all say up"],
                  fontsize=10.5, color=INK2)
    ax.set_yticks([])
    ax.set_ylabel("how much creators talked about it →", fontsize=10, color=MUTED)
    ax.text(0, max_dir * 1.08, "much discussed, no agreement", ha="center",
            fontsize=9.5, color=MUTED, style="italic")

    for p in points:
        color = (BULL if p["net"] > NET_THRESHOLD
                 else BEAR if p["net"] < -NET_THRESHOLD else MUTED)
        ax.plot(p["net"], p["directional"], "o", ms=10, color=color,
                mec=SURFACE, mew=1.5, zorder=3)

    # Labels stack per half-unit column of the x axis: a top-down sweep keeps
    # a minimum gap inside each column, so the crowded "everyone agrees"
    # edge becomes a tidy list beside its dots instead of a pile-up. A stack
    # that would run below the axis shifts up as a block, and any label that
    # drifted from its dot gets a hairline leader so ownership stays clear.
    step = max_dir * 0.09
    columns = {}
    for p in points:
        columns.setdefault(round(p["net"] * 2), []).append(p)
    top_needed = max_dir * 1.15
    for column in columns.values():
        column.sort(key=lambda p: -p["directional"])
        ys, prev = [], None
        for p in column:
            y = p["directional"] if prev is None else min(p["directional"], prev - step)
            ys.append(y)
            prev = y
        floor = step * 0.7
        if ys and ys[-1] < floor:
            shift = floor - ys[-1]
            ys = [y + shift for y in ys]
        top_needed = max(top_needed, ys[0] + step * 0.6 if ys else 0)
        for p, y in zip(column, ys):
            left = p["net"] > 0.75
            lx = p["net"] - 0.045 if left else p["net"] + 0.045
            if abs(y - p["directional"]) > step * 0.5:
                ax.plot([p["net"], lx + (0.01 if left else -0.01)],
                        [p["directional"], y], color=GRID, lw=1, zorder=1)
            ax.text(lx, y, p["label"], ha="right" if left else "left", va="center",
                    fontsize=10, fontweight="bold", color=INK)
    ax.set_ylim(0, top_needed)

    _finish(
        fig, path,
        "Agreement vs. attention",
        f"{data['window']} · assets with at least {MIN_MAP_DIRECTIONAL} up-or-down calls",
        "How to read: the higher a dot, the more calls an asset drew; the further "
        "right, the more creators agree it's going up.",
    )
    plt.close(fig)


RANGE_BLUE = "#9ec5f4"  # light step of the bull blue, for the low-high span


def _money(value):
    return f"\\${value:,.2f}" if value < 20 else f"\\${value:,.0f}"


def upside_title(rows):
    """Takeaway headline for the target chart: the target the most creators
    back (ties: the larger implied move, then the label). A target only one
    creator named never headlines the chart: the biggest number on it is
    usually one creator's multi-year call."""
    fallback = "Creator price targets vs the last close"
    shared = [r for r in rows if r["n_targets"] >= 2]
    if not shared:
        return fallback
    top = min(shared, key=_upside_order)
    move = f"{top['label']} {top['pct']:+.0f}%"
    return _fit_title((f"Best-supported target: {move} ({top['n_targets']} creators)",
                       f"Best-supported target: {move}"), UPSIDE_TITLE_MAX_CHARS, fallback)


def _render_upside(plt, data, path):
    """Low / average / high target range vs the last close — the convention
    analyst-forecast pages use. The range strip keeps a lone moonshot target
    visible as disagreement instead of letting it silently inflate a bare
    average. A target one creator named is drawn lighter than a shared one,
    and shared targets are listed first (see upside_rows)."""
    rows = data["upside"]
    height = HEADER_IN + 0.5 * len(rows) + 0.85
    fig = _new_figure(plt, height)
    left = _label_margin([r["label"] for r in rows], 0.14)
    ax = fig.add_axes(_plot_box(height, 0.72, left, right=0.66))
    _chrome(ax)

    ys = range(len(rows) - 1, -1, -1)
    lo_min = min(0, min(r["lo"] for r in rows))
    hi_max = max(0, max(r["hi"] for r in rows))
    span = (hi_max - lo_min) or 1
    for y, row in zip(ys, rows):
        shared = row["n_targets"] > 1
        if shared and round(row["t_lo"]) != round(row["t_hi"]):
            ax.plot([row["lo"], row["hi"]], [y, y], color=RANGE_BLUE, lw=6,
                    solid_capstyle="round", zorder=2)
        base = BULL if row["pct"] >= 0 else BEAR
        ax.plot(row["pct"], y, "o", ms=10, color=base, alpha=1.0 if shared else 0.55,
                mec=SURFACE, mew=1.5, zorder=3)
        ax.text(row["pct"], y + 0.3, f"{row['pct']:+.0f}%", ha="center",
                va="bottom", fontsize=10, fontweight="bold", color=INK)
        who = f"{row['n_targets']} creator{'s' if shared else ''}"
        if shared and round(row["t_lo"]) != round(row["t_hi"]):
            note = f"{_money(row['price'])} close\n{_money(row['t_lo'])}–{_money(row['t_hi'])} · {who}"
        else:
            note = f"{_money(row['price'])} close\ntarget {_money(row['target'])} · {who}"
        ax.text(1.04, y, note, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=8.5, color=MUTED, linespacing=1.3)
    _zero_line(ax, len(rows), "last close")
    ax.set_yticks(list(ys), [r["label"] for r in rows], fontsize=11.5, color=INK)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.set_xlim(lo_min - span * 0.12, hi_max + span * 0.12)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xticks([])

    _finish(
        fig, path,
        upside_title(rows),
        f"{data['window']} · price targets vs the last close",
        "Dot = average target as a move from the last close. Faded dots rest on one creator.",
    )
    plt.close(fig)


# --- Creators vs the market (tracker charts) -------------------------------

TRACKER_ROW_IN = 0.42   # inches per creator / asset row
FADED = 0.45            # alpha of a small-sample creator's bar
# One character of a bold 10pt value label ("+17.1%": digits, sign, percent)
# is ~0.09in in DejaVu Sans Bold, the widest of the fonts the charts meet.
VALUE_CHAR_IN = 0.09
ARROWS = {"bullish": "▲", "bearish": "▼"}
NBSP = " "   # textwrap never breaks at it; it renders as a space


def _pct_label(value):
    """'+1.2%'; two decimals under 0.1, so a near-zero bar's colour is explained."""
    return f"{value:+.2f}%" if abs(value) < 0.1 else f"{value:+.1f}%"


def _labels_in(texts):
    """Inches the widest bold 10pt value label needs past its bar, gap included."""
    return 0.1 + VALUE_CHAR_IN * max((len(t) for t in texts), default=0)


def _half_span(extent, axis_in, label_in):
    """Half-width of a zero-centred axis whose longest bar (`extent`) still
    leaves `label_in` inches for its value label on either side."""
    share = min(0.6, label_in / max(axis_in / 2, 0.1))
    return max(extent, 0.5) / (1 - share)


def _padded_span(lo, hi, axis_in, label_in):
    """x limits showing [lo, hi] (lo <= 0 <= hi) with `label_in` inches for
    a value label past each side that holds bars, and a slim margin on a
    side that holds none."""
    span = (hi - lo) or 1.0
    share = min(0.3, label_in / max(axis_in, 0.1))
    left, right = (share if lo < 0 else 0.04), (share if hi > 0 else 0.04)
    width = span / (1 - left - right)
    return lo - left * width, hi + right * width


def _axis_hint(ax, x, text, ha, color):
    """A word label just under the axes, a fixed few points below them."""
    ax.annotate(text, xy=(x, 0), xycoords="axes fraction", xytext=(0, -5), textcoords="offset points",
                ha=ha, va="top", fontsize=9.5, color=color)


def _creators_meta(data):
    parts = []
    since, as_of = _day(data.get("since")), _day(data.get("as_of"), weekday=True)
    if since:
        parts.append(f"Calls since {since}")
    if as_of:
        parts.append(f"marked to {as_of} close" + (" (stale prices)" if data.get("stale") else ""))
    parts.append("median of each call's move minus the market's over the same days, sign-flipped for bearish "
                 "calls (S&P 500; crypto: Bitcoin)")
    return " · ".join(parts)


def _not_charted_note(entries):
    """One footnote line naming the creators the chart leaves out, and why."""
    groups = {"too_few": [], "no_benchmark": [], "limit": []}
    for entry in entries or ():
        groups.setdefault(entry["reason"], []).append(entry)
    parts = []
    if groups["too_few"]:
        parts.append("Too few calls to chart yet: " + ", ".join(f"{e['label']} {e['calls']}" for e in groups["too_few"]))
    if groups["no_benchmark"]:
        parts.append("Only calls on the S&P 500 or Bitcoin themselves: "
                     + ", ".join(e["label"] for e in groups["no_benchmark"]))
    if groups["limit"]:
        parts.append(f"+{_count(len(groups['limit']), 'more creator')} with fewer calls")
    return textwrap.fill(" · ".join(parts), NOTE_WRAP)


def _render_creators_vs_market(plt, data, path):
    """Each creator's median call against the market over the same days, as
    one bar per creator around a 'market' line: blue when their median call
    beat the market in the called direction by more than LEVEL_BAND_POINTS,
    red when it trailed by more, grey when level; faded under
    creator_tracker.MIN_CREATOR_SAMPLE calls. Rows follow the number of
    calls; the right-hand note keeps every count's denominator."""
    rows = data["creators"]
    meta = _creators_meta(data)
    # A no-break space keeps each "=" with the word before it when wrapped.
    howto = (f"Right of the line{NBSP}= their calls beat the market in the direction they called. "
             f"Faded{NBSP}= fewer than {creator_tracker.MIN_CREATOR_SAMPLE} calls. Mark-to-market, not a verdict.")
    note = _not_charted_note(data.get("not_charted"))
    foot = 0.2 + 0.14 * note.count("\n") if note else 0.0
    extra = _extra_header_in(meta, howto)
    height = HEADER_IN + extra + TRACKER_ROW_IN * len(rows) + 0.9 + foot
    fig = _new_figure(plt, height)
    labels = [r["label"] + (" (mostly guests)" if r.get("mostly_guests") else "") for r in rows]
    left, right = _label_margin(labels, 0.2, fontsize=10.5), 0.8
    ax = fig.add_axes(_plot_box(height, 0.7 + foot, left, right=right, extra_top_in=extra))
    _chrome(ax)

    values = [r["median_dir_excess"] for r in rows]
    half = _half_span(max(abs(v) for v in values), (right - left) * FIG_WIDTH,
                      _labels_in([_pct_label(v) for v in values]))
    gap = half * 0.03
    ys = range(len(rows) - 1, -1, -1)
    for y, row in zip(ys, rows):
        value, faded = row["median_dir_excess"], row["small_sample"]
        ahead = value > 0
        color = BULL if value > LEVEL_BAND_POINTS else BEAR if value < -LEVEL_BAND_POINTS else MIXED
        ax.barh(y, value, height=0.6, color=color, alpha=FADED if faded else 1.0)
        ax.text(value + gap if ahead else value - gap, y, _pct_label(value), ha="left" if ahead else "right",
                va="center", fontsize=10, fontweight="bold", color=MUTED if faded else INK)
        ax.text(1.04, y, f"{row['beat']}/{row['bench_n']} beat\n{_count(row['calls'], 'call')}",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8.5, color=MUTED,
                linespacing=1.3)
    _zero_line(ax, len(rows), "market")
    ax.set_yticks(list(ys), labels, fontsize=10.5, color=INK)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.set_xlim(-half, half)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xticks([])
    _axis_hint(ax, 0.48, "← behind the market", "right", BEAR)
    _axis_hint(ax, 0.52, "ahead of the market →", "left", BULL)
    if note:
        fig.text(0.05, 0.3 / height, note, fontsize=8.5, color=MUTED, va="bottom", linespacing=1.3)

    _finish(fig, path, creators_vs_market_title(rows), meta, howto, disclaimer=TRACKER_DISCLAIMER)
    plt.close(fig)


def _outcomes_meta(data):
    as_of = _day(data.get("as_of"), weekday=True)
    market = "vs the S&P 500 (crypto: Bitcoin)"
    if as_of:
        stale = " (stale prices)" if data.get("stale") else ""
        return (f"Assets 2+ creators agree on · move since the first of their calls, to the {as_of} "
                f"close{stale}, {market}")
    return f"Assets 2+ creators agree on · move since the first of their calls {market}"


def _outcome_note(row):
    bench, label = row.get("bench_move"), row.get("benchmark_label")
    if bench is not None:
        first = f"{label or 'market'} {bench:+.1f}%"
    elif label:
        first = f"{label}: no close"          # the benchmark's close is missing
    else:
        first = "the market itself"           # the S&P 500 or Bitcoin: no benchmark
    since = _day(row.get("since"))
    return f"{first}\nsince {since}" if since else first


def _render_consensus_outcomes(plt, data, path):
    """Per asset 2+ creators agree on, the move since the first of their
    calls (bar, coloured by the call: blue bullish, red bearish) against the
    market over the same days (grey tick)."""
    rows = data["outcomes"]
    meta = _outcomes_meta(data)
    howto = ("Blue bars should end right of their tick for a bullish call to be ahead of the market; "
             "red bars left.")
    extra = _extra_header_in(meta, howto)
    height = HEADER_IN + extra + TRACKER_ROW_IN * len(rows) + 1.0
    fig = _new_figure(plt, height)
    labels = [f"{r['label']} {ARROWS[r['direction']]} {r['n_creators']}" for r in rows]
    left, right = _label_margin(labels, 0.16), 0.76
    ax = fig.add_axes(_plot_box(height, 0.8, left, right=right, extra_top_in=extra))
    _chrome(ax)

    points = [0.0] + [r["move"] for r in rows] + [r["bench_move"] for r in rows if r["bench_move"] is not None]
    xmin, xmax = _padded_span(min(points), max(points), (right - left) * FIG_WIDTH,
                              _labels_in([f"{r['move']:+.1f}%" for r in rows]))
    gap = (xmax - xmin) * 0.015
    ys = range(len(rows) - 1, -1, -1)
    for y, row in zip(ys, rows):
        move, bench = row["move"], row["bench_move"]
        ax.barh(y, move, height=0.6, color=BULL if row["direction"] == "bullish" else BEAR, zorder=2)
        if bench is not None:
            ax.plot([bench, bench], [y - 0.4, y + 0.4], color=INK2, lw=2.4, solid_capstyle="butt", zorder=3)
        # The value sits past whichever ends further out on its side, bar or
        # tick, so the two never overlap.
        ends = [move] + ([bench] if bench is not None else [])
        if move >= 0:
            x, ha = max(ends) + gap, "left"
        else:
            x, ha = min(ends) - gap, "right"
        ax.text(x, y, f"{move:+.1f}%", ha=ha, va="center", fontsize=10, fontweight="bold", color=INK)
        ax.text(1.04, y, _outcome_note(row), transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=8.5, color=MUTED, linespacing=1.3)
    _zero_line(ax, len(rows), "no change")
    ax.set_yticks(list(ys), labels, fontsize=11.5, color=INK)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=BULL), plt.Rectangle((0, 0), 1, 1, color=BEAR),
               plt.Line2D([], [], color=INK2, marker="|", markersize=11, markeredgewidth=2.4, linestyle="None")]
    ax.legend(handles, ["bullish call", "bearish call", "market, same days"], loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=9, labelcolor=INK2,
              handlelength=1.1, handleheight=1.1)

    _finish(fig, path, consensus_outcomes_title(rows), meta, howto, disclaimer=TRACKER_DISCLAIMER)
    plt.close(fig)


# Album order: the three charts that carry the week first; the rest appear
# only when they have something to say (see _chart_ready).
CHARTS = [
    ("consensus", "1-consensus.png", _render_consensus),
    ("tone", "2-tone.png", _render_tone),
    ("upside", "3-upside.png", _render_upside),
    ("flips", "4-flips.png", _render_flips),
    ("map", "5-map.png", _render_map),
    ("spread", "6-spread.png", _render_spread),
]

# The tone chart needs history to compare; a single week says nothing.
MIN_TONE_WEEKS = 2
# The spread line is the tone chart's numbers drawn as a trend; until it has
# a real trend to show (~2 months) it only repeats the tone chart.
MIN_SPREAD_WEEKS = MATURE_SPREAD_WEEKS
MIN_CONSENSUS_ROWS = 3
# The flip chart is drawn only when at least one reversal rests on two or
# more calls on both sides; one creator against one creator is noise and
# the text already lists real consensus changes.
MIN_FLIP_VOTES = 2
# The attention map needs enough points to show a spread, and a bearish or
# contested one — a pile of bullish dots only repeats the consensus board.
MIN_MAP_POINTS = 6


def _chart_ready(key, section):
    if not section:
        return False
    if key == "tone":
        return len(section) >= MIN_TONE_WEEKS
    if key == "spread":
        return len(section) >= MIN_SPREAD_WEEKS
    if key == "consensus":
        return len(section) >= MIN_CONSENSUS_ROWS
    if key == "flips":
        # One reversal is a sentence, and the text carries it; the slope
        # chart earns its place with two or more well-attended reversals.
        return sum(min(r["from_votes"], r["to_votes"]) >= MIN_FLIP_VOTES for r in section) >= 2
    if key == "map":
        return len(section) >= MIN_MAP_POINTS and any(p["net"] < NET_THRESHOLD for p in section)
    return True


def _pyplot(what):
    """matplotlib.pyplot on the Agg backend, or None (logged) when the
    plotting stack cannot load."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log_warn(f"matplotlib unavailable; skipping {what}: {e}")
        return None
    return plt


def render_charts(data, out_dir):
    """Render every chart whose data section is non-empty. Returns the list of
    written paths, in album order; a chart that fails is logged and skipped so
    one bad render never blocks the rest."""
    plt = _pyplot("pulse charts")
    if plt is None:
        return []
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for key, filename, renderer in CHARTS:
        if not _chart_ready(key, data.get(key)):
            continue
        path = os.path.join(out_dir, filename)
        try:
            renderer(plt, data, path)
            paths.append(path)
        except Exception as e:
            log_warn(f"Pulse chart '{key}' failed to render: {e}")
    return paths


# The creators-vs-market album, for the Friday report: separate from the
# pulse's CHARTS, whose album and order are unchanged.
TRACKER_CHARTS = [
    ("creators", "tracker-1-creators.png", _render_creators_vs_market),
    ("outcomes", "tracker-2-consensus.png", _render_consensus_outcomes),
]


def _tracker_ready(key, section):
    if not section:
        return False
    if key == "creators":
        return len(section) >= MIN_TRACKER_CREATORS
    if key == "outcomes":
        return len(section) >= MIN_OUTCOME_ROWS
    return True


def render_tracker_charts(data, out_dir):
    """
    Render the creators-vs-market charts (TRACKER_CHARTS) from
    build_tracker_chart_data output. Returns the written paths in album
    order; each chart is drawn only when its own section is ready
    (MIN_TRACKER_CREATORS creators, MIN_OUTCOME_ROWS shared calls). Entirely
    best-effort: matplotlib missing, an unwritable directory or a failed
    render is logged and that chart skipped, never raised.
    """
    if not isinstance(data, dict):
        return []
    ready = [(key, filename, renderer) for key, filename, renderer in TRACKER_CHARTS
             if _tracker_ready(key, data.get(key))]
    if not ready:
        return []
    plt = _pyplot("tracker charts")
    if plt is None:
        return []
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        log_warn(f"Tracker charts: cannot write to {out_dir}: {e}")
        return []
    paths = []
    for key, filename, renderer in ready:
        path = os.path.join(out_dir, filename)
        try:
            renderer(plt, data, path)
            paths.append(path)
        except Exception as e:
            log_warn(f"Tracker chart '{key}' failed to render: {e}")
    return paths
