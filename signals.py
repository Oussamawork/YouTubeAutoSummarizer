import json
import re

from log import log_info, log_warn
from summarizer import complete, QUOTA_EXHAUSTED_SENTINEL

# Structured market-signal extraction from finance-video summaries.
#
# One extra (cheap) LLM call per summarized video turns the free-text summary
# into machine-readable signals — which assets the speaker discussed and what
# stance they took — appended to data/signals.jsonl for later aggregation
# (sentiment time series, consensus flips, per-channel scorecards). Extraction
# is strictly best-effort: any failure returns None and must never affect
# summary delivery or dedup state. The output is research data, not advice.

ASSET_TYPES = {"stock", "crypto", "etf", "index", "commodity", "macro"}
STANCES = {"bullish", "bearish", "neutral"}
CONVICTIONS = {"low", "medium", "high"}
ACTIONS = {"buy", "sell", "hold", "watch", "none"}
HORIZONS = {"short", "medium", "long", "unspecified"}
MARKET_SENTIMENTS = {"bullish", "bearish", "neutral", "mixed"}

SIGNALS_SYSTEM_PROMPT = (
    "You extract structured market signals from the summary of a finance-related "
    "YouTube video. Output ONLY a JSON object — no markdown fences, no prose — "
    "with exactly this shape:\n"
    "{\n"
    '  "assets": [\n'
    "    {\n"
    '      "name": "<company/asset name as stated>",\n'
    '      "ticker": "<ticker symbol if stated or unambiguous, else null>",\n'
    '      "type": "stock" | "crypto" | "etf" | "index" | "commodity" | "macro",\n'
    '      "stance": "bullish" | "bearish" | "neutral",\n'
    '      "conviction": "low" | "medium" | "high",\n'
    '      "action": "buy" | "sell" | "hold" | "watch" | "none",\n'
    '      "catalysts": ["<short phrase per reason/catalyst the speaker gives>"],\n'
    '      "price_target": <number or null>,\n'
    '      "horizon": "short" | "medium" | "long" | "unspecified"\n'
    "    }\n"
    "  ],\n"
    '  "market_sentiment": "bullish" | "bearish" | "neutral" | "mixed",\n'
    '  "topics": ["<2-5 short topic tags>"]\n'
    "}\n"
    "\n"
    "Rules:\n"
    "- Report ONLY what the speaker actually says in the summary. Never infer, "
    "extrapolate, or invent stances, tickers, price targets, or reasons.\n"
    '- "action" is the speaker\'s own stated action or recommendation; use '
    '"none" when they state no action.\n'
    '- "market_sentiment" is the speaker\'s overall tone about markets in this '
    "video, not your own view.\n"
    "- If the summary contains no market-relevant content, output exactly "
    '{"assets": [], "market_sentiment": "neutral", "topics": []}.'
)

SIGNALS_USER_TEMPLATE = (
    "Channel: {channel_name}\n"
    "Video title: {video_title}\n"
    "\n"
    "Video summary:\n"
    "{summary}"
)

# Matches an optional ```json ... ``` fence around the model's output.
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_code_fences(text):
    match = _FENCE_RE.match(text.strip())
    return match.group(1) if match else text


def _normalize_asset(asset):
    """Validate/normalize one asset entry; None drops entries missing the core
    fields (a signal without a named asset and a stance is unusable)."""
    if not isinstance(asset, dict):
        return None
    name = asset.get("name")
    stance = asset.get("stance")
    if not isinstance(name, str) or not name.strip():
        return None
    if stance not in STANCES:
        return None

    ticker = asset.get("ticker")
    if not isinstance(ticker, str) or not ticker.strip():
        ticker = None
    price_target = asset.get("price_target")
    if not isinstance(price_target, (int, float)) or isinstance(price_target, bool):
        price_target = None
    catalysts = asset.get("catalysts")
    if not isinstance(catalysts, list):
        catalysts = []
    catalysts = [c.strip() for c in catalysts if isinstance(c, str) and c.strip()]

    return {
        "name": name.strip(),
        "ticker": ticker.strip().upper() if ticker else None,
        "type": asset.get("type") if asset.get("type") in ASSET_TYPES else "other",
        "stance": stance,
        "conviction": asset.get("conviction") if asset.get("conviction") in CONVICTIONS else "unspecified",
        "action": asset.get("action") if asset.get("action") in ACTIONS else "none",
        "catalysts": catalysts,
        "price_target": price_target,
        "horizon": asset.get("horizon") if asset.get("horizon") in HORIZONS else "unspecified",
    }


def _parse_signals(text):
    """
    Parse the model's output into a validated signals dict, or None when it
    isn't usable JSON of the expected shape. Never raises.
    """
    if not text:
        return None
    try:
        data = json.loads(_strip_code_fences(text))
    except (ValueError, TypeError) as e:
        log_warn(f"Signal extraction returned unparseable JSON: {e}")
        return None
    if not isinstance(data, dict):
        log_warn("Signal extraction returned JSON that is not an object; discarding.")
        return None

    raw_assets = data.get("assets")
    assets = []
    if isinstance(raw_assets, list):
        for asset in raw_assets:
            normalized = _normalize_asset(asset)
            if normalized is not None:
                assets.append(normalized)

    sentiment = data.get("market_sentiment")
    if sentiment not in MARKET_SENTIMENTS:
        sentiment = "neutral"

    topics = data.get("topics")
    if not isinstance(topics, list):
        topics = []
    topics = [t.strip() for t in topics if isinstance(t, str) and t.strip()]

    return {"assets": assets, "market_sentiment": sentiment, "topics": topics}


def extract_signals(summary, video_title=None, channel_name=None):
    """
    Extract structured market signals from a video summary via the shared LLM
    provider chain. Returns the signals dict, or None when the summary is empty,
    providers are unavailable/quota-limited, or the output can't be parsed.
    Best-effort by contract: never raises.
    """
    summary = (summary or "").strip()
    if not summary:
        return None

    user_message = SIGNALS_USER_TEMPLATE.format(
        channel_name=(channel_name or "unknown").strip() or "unknown",
        video_title=(video_title or "unknown").strip() or "unknown",
        summary=summary,
    )
    text = complete(SIGNALS_SYSTEM_PROMPT, user_message)
    if text == QUOTA_EXHAUSTED_SENTINEL:
        log_warn("Signal extraction skipped: LLM quota exhausted.")
        return None
    if not text:
        log_warn("Signal extraction produced no output.")
        return None

    parsed = _parse_signals(text)
    if parsed is not None:
        log_info(
            f"Extracted signals for {len(parsed['assets'])} asset(s), "
            f"market sentiment: {parsed['market_sentiment']}."
        )
    return parsed
