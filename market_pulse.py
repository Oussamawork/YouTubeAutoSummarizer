import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from log import log_info, log_warn, log_error
from sendToTelegram import send_telegram_text

# Weekly market pulse: aggregates the per-video signals accumulated in
# data/signals.jsonl (see signals.py) into one Telegram report — top mentioned
# assets with net stance, consensus flips vs the prior window, and assets newly
# on the radar. Runs from its own weekly workflow; read-only over the dataset.
# The output is research over creator opinions, not investment advice.

load_dotenv('.env')

SIGNALS_FILE = "data/signals.jsonl"
STANCE_SCORE = {"bullish": 1, "bearish": -1, "neutral": 0}
# An asset is "new on the radar" when it appears in the current window but in
# none of the records this many days before the window started.
NEW_ASSET_LOOKBACK_DAYS = 30
MAX_ASSETS_IN_REPORT = 8
# Net-stance thresholds: mean score above/below this counts as a directional
# consensus; in between reads as mixed.
NET_THRESHOLD = 0.15

DISCLAIMER = "⚠️ Aggregated creator opinions — research input, not investment advice."


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


def _asset_key(asset):
    ticker = asset.get("ticker")
    if ticker:
        return ticker.upper()
    return (asset.get("name") or "").strip().upper()


def _iter_assets(records):
    """Yield (record, asset) for every valid asset entry in the records."""
    for record in records:
        signals = record.get("signals")
        if not isinstance(signals, dict):
            continue
        for asset in signals.get("assets", []):
            if isinstance(asset, dict) and _asset_key(asset):
                yield record, asset


def aggregate_assets(records):
    """
    Fold records into per-asset stats:
    {key: {label, mentions, channels, bull, bear, neutral, actions, targets}}
    """
    stats = {}
    for record, asset in _iter_assets(records):
        key = _asset_key(asset)
        entry = stats.setdefault(key, {
            "label": asset.get("ticker") or asset.get("name") or key,
            "mentions": 0, "channels": set(),
            "bull": 0, "bear": 0, "neutral": 0,
            "actions": Counter(), "targets": [],
        })
        entry["mentions"] += 1
        if record.get("channel_name"):
            entry["channels"].add(record["channel_name"])
        stance = asset.get("stance")
        if stance == "bullish":
            entry["bull"] += 1
        elif stance == "bearish":
            entry["bear"] += 1
        else:
            entry["neutral"] += 1
        action = asset.get("action")
        if action and action != "none":
            entry["actions"][action] += 1
        target = asset.get("price_target")
        if isinstance(target, (int, float)) and not isinstance(target, bool):
            entry["targets"].append(target)
    return stats


def net_stance(entry):
    """Mean stance score in [-1, 1] for one aggregated asset entry."""
    total = entry["bull"] + entry["bear"] + entry["neutral"]
    if not total:
        return 0.0
    return (entry["bull"] - entry["bear"]) / total


def _direction(score):
    if score > NET_THRESHOLD:
        return "bullish"
    if score < -NET_THRESHOLD:
        return "bearish"
    return "mixed"


def find_flips(current, previous):
    """
    Assets whose directional consensus flipped between the previous and current
    windows (bullish -> bearish or vice versa). A move to/from "mixed" is not a
    flip — only a real sign change is worth an alert.
    """
    flips = []
    for key, entry in current.items():
        prev = previous.get(key)
        if not prev:
            continue
        cur_dir = _direction(net_stance(entry))
        prev_dir = _direction(net_stance(prev))
        if {cur_dir, prev_dir} == {"bullish", "bearish"}:
            flips.append({
                "label": entry["label"],
                "from": prev_dir, "to": cur_dir,
                "mentions": entry["mentions"],
            })
    return sorted(flips, key=lambda f: -f["mentions"])


def find_new_assets(current, older_records):
    """Asset keys present in the current window but absent from the lookback
    records that preceded it."""
    seen_before = {_asset_key(a) for _, a in _iter_assets(older_records)}
    return {key: entry for key, entry in current.items() if key not in seen_before}


def _overall_tone(records):
    tone = Counter()
    for record in records:
        signals = record.get("signals")
        if isinstance(signals, dict) and signals.get("market_sentiment"):
            tone[signals["market_sentiment"]] += 1
    return tone


def _format_asset_line(entry):
    score = net_stance(entry)
    parts = [
        f"• {entry['label']} — {entry['mentions']} mention{'s' if entry['mentions'] != 1 else ''}",
        f"{len(entry['channels'])} channel{'s' if len(entry['channels']) != 1 else ''}",
        f"net {_direction(score)} ({entry['bull']}↑/{entry['bear']}↓)",
    ]
    if entry["actions"]:
        actions = " ".join(f"{a}×{n}" for a, n in entry["actions"].most_common())
        parts.append(actions)
    if entry["targets"]:
        avg = sum(entry["targets"]) / len(entry["targets"])
        parts.append(f"avg target {avg:,.0f}")
    return " · ".join(parts)


def build_pulse(current_records, previous_records, older_records, start, end):
    """
    Render the plain-text weekly pulse. Returns "" when the current window has
    no analyzable records (caller skips the send).
    """
    if not current_records:
        return ""

    current = aggregate_assets(current_records)
    previous = aggregate_assets(previous_records)

    channels = {r.get("channel_name") for r in current_records if r.get("channel_name")}
    lines = [
        f"📈 Weekly Market Pulse — {start.isoformat()} → {end.isoformat()}",
        f"Videos analyzed: {len(current_records)} · Channels: {len(channels)}",
    ]

    tone = _overall_tone(current_records)
    if tone:
        tone_line = " · ".join(f"{k} {v}" for k, v in tone.most_common())
        lines.append(f"Overall tone: {tone_line}")

    if current:
        lines.append("")
        lines.append("Top assets:")
        ranked = sorted(
            current.values(),
            key=lambda e: (-e["mentions"], -abs(net_stance(e)), e["label"]),
        )
        for entry in ranked[:MAX_ASSETS_IN_REPORT]:
            lines.append(_format_asset_line(entry))
        if len(ranked) > MAX_ASSETS_IN_REPORT:
            lines.append(f"…and {len(ranked) - MAX_ASSETS_IN_REPORT} more.")

    flips = find_flips(current, previous)
    if flips:
        lines.append("")
        lines.append("🔄 Consensus flips vs prior week:")
        for flip in flips:
            lines.append(f"• {flip['label']}: {flip['from']} → {flip['to']}")

    new_assets = find_new_assets(current, older_records)
    if new_assets:
        lines.append("")
        lines.append(f"🆕 New on the radar (past {NEW_ASSET_LOOKBACK_DAYS} days):")
        for entry in sorted(new_assets.values(), key=lambda e: -e["mentions"])[:5]:
            lines.append(f"• {entry['label']} — {_direction(net_stance(entry))}")

    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def generate_pulse(days=7, today=None, path=SIGNALS_FILE):
    """Load the dataset and build the pulse text for the trailing `days` window."""
    today = today or datetime.now(timezone.utc).date()
    records = load_signals(path)
    window_start = today - timedelta(days=days)
    prev_start = window_start - timedelta(days=days)
    lookback_start = window_start - timedelta(days=NEW_ASSET_LOOKBACK_DAYS)

    current = [r for r in records if _in_window(r, window_start, today)]
    previous = [r for r in records if _in_window(r, prev_start, window_start)]
    older = [r for r in records if _in_window(r, lookback_start, window_start)]
    return build_pulse(current, previous, older, window_start, today)


def main():
    parser = argparse.ArgumentParser(description="Weekly market pulse from data/signals.jsonl")
    parser.add_argument("--days", type=int, default=7, help="Window size in days (default 7)")
    parser.add_argument("--dry-run", action="store_true", help="Print the pulse instead of sending it")
    args = parser.parse_args()

    pulse = generate_pulse(days=args.days)
    if not pulse:
        log_info("No signal records in the window; skipping the pulse this week.")
        return 0

    if args.dry_run:
        print(pulse)
        return 0

    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHANNEL_ID")
    if not token or not chat_id:
        log_error("TELEGRAM_TOKEN and TELEGRAM_CHANNEL_ID must be set to send the pulse.")
        return 1

    if send_telegram_text(token, chat_id, pulse):
        log_info("Weekly market pulse sent.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
