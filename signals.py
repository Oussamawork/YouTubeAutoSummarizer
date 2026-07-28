import json
import re

from helpers import env_int
from log import log_info, log_warn
from summarizer import (
    complete,
    QUOTA_EXHAUSTED_SENTINEL,
    INSUFFICIENT_TRANSCRIPT_SENTINEL,
    SUMMARY_SYSTEM_PROMPT,
    COMPACT_SUMMARY_SYSTEM_PROMPT,
    _build_user_message,
    _truncate_transcript,
)

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

# The signals object's shape and rules, shared by the standalone extraction
# prompt and the combined summarize+extract prompt so they can never drift.
SIGNALS_SCHEMA = (
    "{\n"
    '  "assets": [\n'
    "    {\n"
    '      "name": "<company/asset name as stated>",\n'
    '      "ticker": "<the ticker ONLY if the speaker says it aloud or it appears in the video title; otherwise null. Never supply one from your own knowledge of the company>",\n'
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
    "- If there is no market-relevant content, use "
    '{"assets": [], "market_sentiment": "neutral", "topics": []}.'
)

SIGNALS_SYSTEM_PROMPT = (
    "You extract structured market signals from the summary of a finance-related "
    "YouTube video. Output ONLY a JSON object — no markdown fences, no prose — "
    "with exactly this shape:\n" + SIGNALS_SCHEMA
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
    isn't usable JSON of the expected shape. Tolerates code fences and prose
    around the JSON object (models add preambles despite instructions — seen
    in production with gemini-2.5-flash). Never raises.
    """
    if not text:
        return None
    stripped = _strip_code_fences(text)
    try:
        data = json.loads(stripped)
    except (ValueError, TypeError):
        # Fall back to the outermost {...} span — handles "Here is the JSON:"
        # preambles, trailing commentary, and fences the regex didn't match.
        start, end = stripped.find("{"), stripped.rfind("}")
        if start == -1 or end <= start:
            log_warn("Signal extraction returned no JSON object; discarding.")
            return None
        try:
            data = json.loads(stripped[start:end + 1])
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


# Combined summarize+extract: one call returns both the Telegram summary and
# the structured signals, halving per-video LLM requests (which is what the
# per-minute rate limit actually counts). It carries the summary AND the signals
# object AND the JSON escaping of both, so it needs more room than a plain
# summary call — but not much more: observed summaries run well under 2.5k chars
# and the signals object adds ~1k tokens. The truncations this budget was once
# blamed for came from thinking tokens (see LLM_REASONING_EFFORT), not from the
# summary competing with the signals for space.
COMBINED_MAX_TOKENS = env_int("LLM_COMBINED_MAX_TOKENS", 3000)

COMBINED_SUFFIX = (
    "\n\n"
    "=== OUTPUT ENVELOPE ===\n"
    "Everything above describes the TEXT that belongs in the \"summary\" field: "
    "still plain text with \"• \" bullets and the asset roster, no markdown, no "
    "preamble. The RESPONSE as a whole is ONE JSON object and nothing else (no "
    "fences, no prose outside it):\n"
    '{"summary": "<the summary described above, as a single JSON string using '
    '\\n for line breaks>", "signals": <the object described below>}\n'
    "\n"
    "If the transcript is unsummarizable, the single-token rule above applies "
    "to the \"summary\" FIELD, not to the response. Return exactly: "
    '{"summary": "INSUFFICIENT_TRANSCRIPT", "signals": {"assets": [], '
    '"market_sentiment": "neutral", "topics": []}}\n'
    "\n"
    "The \"signals\" value captures the market content of the TRANSCRIPT — "
    "including assets the summary text had no room to spell out in prose. It "
    "is the complete record; the summary is the readable one. Exactly this "
    "shape:\n" + SIGNALS_SCHEMA
)


# The base prompts close by saying "output only the summary itself". Appending a
# JSON envelope after that leaves two contradictory answers to "what is the
# response?", so the trailing rule is removed rather than argued with.
TRAILING_OUTPUT_RULE = (
    "\n\nOutput only the summary itself — no preamble, no sign-off, and no "
    "phrases like \"Here is the summary\"."
)


def _build_combined_prompt(compact=False):
    """Summary rules + the JSON envelope that also carries the signals."""
    base = COMPACT_SUMMARY_SYSTEM_PROMPT if compact else SUMMARY_SYSTEM_PROMPT
    return base.replace(TRAILING_OUTPUT_RULE, "") + COMBINED_SUFFIX


def summarize_with_signals(transcript, title=None, compact=False, channel_name=None):
    """
    One LLM call producing both the summary and its market signals.

    Returns (summary, signals) on success — where `summary` may be the
    INSUFFICIENT_TRANSCRIPT sentinel and `signals` may be None — or None when
    the combined path didn't work, telling the caller to fall back to the
    separate summarize/extract calls. Never raises.
    """
    transcript = (transcript or "").strip()
    if not transcript:
        return None

    text = complete(
        _build_combined_prompt(compact),
        _build_user_message(_truncate_transcript(transcript), title),
        json_mode=True,
        max_tokens=COMBINED_MAX_TOKENS,
    )
    if not text or text == QUOTA_EXHAUSTED_SENTINEL:
        # Quota/failure is the provider chain's verdict, not a parsing problem:
        # falling back would just burn another request against the same wall.
        return (text, None) if text == QUOTA_EXHAUSTED_SENTINEL else None

    stripped = _strip_code_fences(text)
    try:
        data = json.loads(stripped)
    except (ValueError, TypeError):
        start, end = stripped.find("{"), stripped.rfind("}")
        if start == -1 or end <= start:
            log_warn("Combined summary+signals call returned no JSON; using the separate calls.")
            return None
        try:
            data = json.loads(stripped[start:end + 1])
        except (ValueError, TypeError) as e:
            log_warn(f"Combined summary+signals JSON unparseable ({e}); using the separate calls.")
            return None

    if not isinstance(data, dict):
        return None
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        log_warn("Combined call produced no summary; using the separate calls.")
        return None

    summary = summary.strip()
    if summary.upper().startswith(INSUFFICIENT_TRANSCRIPT_SENTINEL):
        return INSUFFICIENT_TRANSCRIPT_SENTINEL, None

    raw_signals = data.get("signals")
    signals = _parse_signals(json.dumps(raw_signals)) if isinstance(raw_signals, dict) else None
    log_info(
        "Combined summary+signals call succeeded"
        + (f" ({len(signals['assets'])} asset(s))." if signals else " (no signals).")
    )
    return summary, signals


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
    text = complete(SIGNALS_SYSTEM_PROMPT, user_message, json_mode=True)
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
