# YouTube Latest Video Fetcher and Summarizer

⚠️ This project is tested and compatible with Python 3.11.

This Python script fetches the latest video from a YouTube channel using the YouTube Data API v3, extracts the transcript, summarizes the content using advanced natural language processing (NLP) techniques, and optionally sends the summarized content to a Telegram channel.

## Features
- Fetches each channel's recent videos from its **public RSS feed** (no API quota; catches channels that upload more than once a day), with the YouTube Data API as fallback.
- **Deduplicates** across runs: only videos newer than the per-channel watermark are processed, capped per run so a backlog can't flood Telegram.
- **Retries videos whose captions aren't up yet** — auto-generated captions often appear hours after upload, so "no transcript" videos are silently retried for a few runs before a notice is sent.
- Displays the **channel name**, **video title**, **video URL**, and **publication date**.
- Extracts the **transcript** of the video (via Supadata or `youtube-transcript-api`).
- Summarizes the transcript with an **LLM via the OpenAI-compatible Chat Completions API** — provider-agnostic (Gemini, Groq, OpenRouter, OpenAI, local servers, …), configured by environment variables. The prompt grounds the model on the video title, translates to English when needed, and guards against over-long transcripts.
- Generates a one-line **TL;DR plus bullet-point key takeaways**, in English regardless of the source language.
- Supports sending the summarized content to a specified **Telegram channel** — one message per video, or one combined **daily digest** (`DAILY_DIGEST=true`).
- **On-demand mode**: `python scraper.py --video-url <url>` (or the workflow's `video_url` dispatch input) summarizes any single video immediately, bypassing the channel scan and dedup.
- Uses environment variables to securely store API keys and tokens.

## Requirements
Before running the script, ensure you have the required Python dependencies:

- `requests` for HTTP calls to the YouTube API, the LLM provider, and Telegram.
- `youtube-transcript-api` for fetching video transcripts.
- `python-dotenv` for loading environment variables from the `.env` file.
- `colorama` for colored logging.

### LLM provider configuration

The summarizer tries providers in this order, skipping any whose key is unset:

1. **Generic** (any OpenAI-compatible endpoint): `LLM_API_KEY` + `LLM_BASE_URL` (+ optional `LLM_MODEL`, `LLM_NAME`)
2. **Gemini** (free tier): `GEMINI_API_KEY` (+ optional `GEMINI_MODEL`, default `gemini-2.5-flash`)
3. **Groq** (free tier): `GROQ_API_KEY` (+ optional `GROQ_MODEL`, default `llama-3.3-70b-versatile`)

If no provider is configured, transcripts are fetched but no summary is produced (a notice is sent to Telegram).

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Configuration

### 1. Obtain YouTube API Key:
   - Visit the [Google Developers Console](https://console.developers.google.com/).
   - Create a project and enable the **YouTube Data API v3**.
   - Obtain your **API key** from the **Credentials** section of the Google Console.

### 2. Obtain Telegram Bot Token:
   - Create a new bot using [BotFather](https://t.me/botfather) on Telegram.
   - Obtain the bot token after setting up your bot.

### 3. Create `.env` File:
   In the root of your project directory, create a file named `.env` and add the following variables:

   ```env
   YOUTUBE_API_KEY=YOUR_YOUTUBE_API_KEY
   TELEGRAM_TOKEN=YOUR_TELEGRAM_BOT_TOKEN
   TELEGRAM_CHANNEL_ID=YOUR_TELEGRAM_CHANNEL_ID
   ```

### 4. Format of `channel_ids.txt`:
   One YouTube channel ID **or `@handle`** per line, optionally followed by per-channel options. Blank lines and `#` comments are ignored. For example:

   ```txt
   UC_x5XG1OV2P6uZZ5FSM9Ttw
   UCBR8-60-B28hp2BmDPdntcQ digest
   @somechannel digest max=5
   ```

   Handles are resolved to channel IDs at run time (1 YouTube API quota unit each, resolved once per run). Dedup state is always keyed by the resolved ID, so switching a line between a handle and its ID never re-sends old videos. A handle that can't be resolved (typo, renamed channel) is logged and skipped for that run without affecting the other channels.

   | Option | Effect |
   | --- | --- |
   | `digest` | For prolific channels: instead of one full-summary message per video, bundle the channel's new videos into **one compact TL;DR digest message per run** (short TL;DR + up to 3 bullets each), so the chat isn't flooded. |
   | `max=N` | Per-run video cap for this channel (overrides `MAX_VIDEOS_PER_RUN`); `0` means no cap. |
   | `only=a,b,c` | Process a video only when its **title** mentions one of these keywords. Matching is case-insensitive and **whole-word**, so `eth` does not match "wh*eth*er" and `sol` does not match "*sol*ve" — which also means `sol` does not match "solana", so list each alias you want (e.g. `only=sol,solana`). Filtering happens before the transcript fetch, so skipped videos cost no transcript credits and no LLM calls. |

## Usage

### Fetching and Summarizing Videos:
1. Add YouTube channel IDs to `channel_ids.txt` (one per line).
2. Run the script to fetch each channel's new videos, extract the transcripts, summarize them, and send the summaries to Telegram:

   ```bash
   python scraper.py
   ```

### Summarizing a single video on demand:

```bash
python scraper.py --video-url "https://www.youtube.com/watch?v=example123"
```

This skips the channel scan and dedup state entirely — useful for any video, subscribed channel or not. On GitHub, run the *Daily YouTube Summary* workflow manually and fill in the `video_url` input.

### Behavior tuning (all optional):

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_MAX_TOKENS` | `2000` | Response budget per summary call. A response cut off at the cap is retried with double the budget and deferred (never delivered half-written) if it still truncates. |
| `LLM_MAX_TOKENS_CEILING` / `LLM_MAX_ESCALATIONS` | `8000` / `2` | Escalation bound. The effective ceiling is at least `start budget x 2^escalations`, so a call configured above `8000` can still escalate rather than silently losing the feature. |
| `LLM_REASONING_EFFORT` | `low` | Thinking budget for reasoning models. Gemini 3 counts thinking tokens against `max_tokens`, so an uncapped budget spends the response allowance on thinking and returns a few hundred characters cut mid-sentence. Set empty to omit the parameter. |
| `SUMMARY_MAX_OUTPUT_TOKENS` | `4000` | Output allowance for a plain summary call (alias of `LLM_MAX_TOKENS`). Caps what the model **writes**, never what it reads: the complete transcript is always sent. |
| `COMBINED_MAX_OUTPUT_TOKENS` | `8000` | Output allowance for the combined summary+claims call (alias `LLM_COMBINED_MAX_TOKENS`); `CLAIMS_MAX_OUTPUT_TOKENS` (`6000`) for a claims-only call. |
| `CONTEXT_SAFETY_MARGIN_TOKENS` | `2048` | Headroom kept under a model's computed input capacity. Whether a request fits is decided per model, in tokens, over the complete request (`token_budget.py`); a request that does not fit is chunked with full coverage, never trimmed. |
| `LLM_LIVE_CAPABILITIES` / `LLM_LIVE_COUNT_TOKENS` | `true` / `true` | Use Gemini's `models.get` (input/output limits, cached a week in `data/model_capabilities.json`) and `countTokens` (exact request size). Off falls back to the dated registry in `model_capabilities.py` and a conservative 3-chars-per-token estimate. `MODEL_CAPABILITIES_JSON` declares limits for a proxy or unknown model. |
| `GEMINI_FREE_TIER_TPM` | `250000` | Free-tier tokens-per-minute cap, applied as a per-request input bound for Gemini models (a single request over it is rejected outright). `0` disables. |
| `MAX_SUMMARY_CHUNKS` / `CHUNK_OVERLAP_TOKENS` | `12` / `300` | Bounds for complete-coverage chunked summarization of a transcript that does not fit one request (each chunk is one metered request; a quota pre-check defers the video when the remaining requests cannot finish it). |
| `PERSIST_TRANSCRIPTS` | `true` | Store raw transcripts gzip'd under `data/transcripts/` before any cleaning, so research can be reprocessed without another transcript credit. |
| `EXHAUSTIVE_RESEARCH_MODE` / `RESEARCH_CHUNK_TOKENS` | `false` / `12000` | High-recall research: extract claims chunk by chunk even when the transcript fits. Multiplies requests; affects only the research branch. |
| `RESEARCH_BACKFILL_MAX_VIDEOS` | `5` | Videos per run for `research_backfill.py --retry` (research retries from stored transcripts). |
| `MAX_VIDEOS_PER_RUN` | `0` (no cap) | Max videos processed per channel per run; older ones go first, the rest wait for the next run. `0` processes everything the channel has due. |
| `NO_TRANSCRIPT_MAX_ATTEMPTS` | `8` | Runs to retry a video whose captions aren't up yet. Giving up needs **this and** `NO_TRANSCRIPT_MIN_HOURS` to be satisfied. |
| `NO_TRANSCRIPT_MIN_HOURS` | `36` | Never write a video off before it has been chased this long, whatever the polling rate. An attempt count alone is the wrong unit: at one run every two hours, three attempts is six hours, and auto-captions routinely take longer to appear. |
| `PENDING_RETRY_MIN_HOURS` | `3` | Minimum gap between retries of one deferred video. A "no captions" answer costs a transcript credit, so without this a frequent schedule spends one per run on every pending video. `0` disables it. |
| `RUN_DEADLINE_MINUTES` | `35` | Wall-clock budget for one run. Videos not started by then defer to the next run, so a slow run flushes its digests and saves state instead of being killed by the workflow's 45-minute timeout mid-video. `0` disables it. |
| `DAILY_DIGEST` | off | `true` bundles all of a run's summaries into one combined Telegram message. |
| `MIN_VIDEO_SECONDS` | `90` | Skip videos shorter than this before fetching a transcript — Shorts and clips rarely carry usable captions and aren't worth a transcript credit. Checked with one `videos.list` call per 50 videos (1 quota unit of 10,000/day). Videos whose metadata can't be read are kept. `0` disables the check. |
| `SKIP_UNCAPTIONED` | off | Also skip videos the API reports as having no captions. **Off by default**: the API's `caption` flag tracks *uploaded* captions and is commonly `false` for videos that only have auto-generated ones, which Supadata can still fetch. Enable only if a run's failure breakdown shows it is safe. |

### Free/premium channel split (optional):

Set `TELEGRAM_FREE_CHANNEL_ID` (secret) to a second — typically public — Telegram
channel and every successfully summarized video also produces a short teaser
there: the summary's TL;DR first line plus the video link. Full summaries keep
going to `TELEGRAM_CHANNEL_ID`, which becomes your premium channel. Optionally
set `PREMIUM_INVITE_URL` (repo variable) to append a "🔓 Full summary: <link>"
call-to-action to each teaser, pointing readers at the premium channel's invite
or subscription link. Warning/deferral notices are never teased, and a failed
teaser send never affects dedup state. Leave `TELEGRAM_FREE_CHANNEL_ID` unset to
keep the original single-channel behavior.

### Transcript budget (Supadata free tier):

Supadata's free tier gives a fixed number of transcript fetches per key per
month (100 by default), and a spent pool means no transcripts at all — the
`youtube-transcript-api` fallback is blocked from CI IPs. So usage is metered:
each configured key adds `SUPADATA_CREDITS_PER_KEY` to a monthly budget, and
keys are used in order, rotating to the next one whenever a key fails to
deliver. Counters live in `data/supadata_usage.json`, committed back by the
daily workflow, and every run logs `Transcript budget: used/allowed today, N
left this month`.

**By default a run spends whatever the cycle has left**, so every video
published today is summarized today. The trade-off is that a busy stretch can
exhaust the pool before the reset date; when it does, videos defer via the
budget flag (no "manual review" message, no retry attempt consumed) and are
picked up once credits return, so nothing is written off — there is just a
quiet gap. Set `SUPADATA_DAILY_PACING=true` to ration instead: each day may
then use `remaining ÷ days left in the cycle`, capped at `budget ÷ 28`, so the
credits last to the reset date.

**Channel order is priority order**: channels are processed top to bottom in
`channel_ids.txt` and credits are spent in that order, so list the channels you
least want to miss first — the ones at the bottom absorb whatever is left.

| Variable | Default | Purpose |
| --- | --- | --- |
| `SUPADATA_API_KEY`, `SUPADATA_API_KEY_2`, `SUPADATA_API_KEY_3` (secrets) | — | Free-tier keys; used in order, rotating when one reports no credits. `SUPADATA_API_KEYS` also accepts a comma-separated list. |
| `SUPADATA_CREDITS_PER_KEY` | `100` | Monthly credits each key contributes to the budget. |
| `SUPADATA_MONTHLY_BUDGET` | keys × credits | Explicit override for the whole cycle's budget. |
| `SUPADATA_DAILY_PACING` | `false` | Ration the cycle's credits across its remaining days instead of spending what's needed each run. Off means a day's videos are all processed that day. |
| `TWELVEDATA_API` (secret) | — | Price data for implied upside, the accuracy scorecard and the price-target chart ([twelvedata.com](https://twelvedata.com), free tier: 800 requests/day, 8/min). Without it prices are unavailable and the jobs say so once per run (the keyless Stooq source it replaced sat behind a browser check and returned nothing to a server, verified 2026-08-17, and has been removed). |
| `TWELVEDATA_MAX_REQUESTS` | `120` | Ceiling on price requests per run, so a paced run can't outlast its workflow timeout. The Sunday cache warmer raises it to 600. |
| `SUPADATA_RESET_DAY` | `1` | Day of the month the plan's credits reset. Supadata resets on the plan's anniversary, not the 1st — the dashboard shows it ("Credits reset on 08/17" → set `17`). Only consulted when pacing is enabled: it makes the pacing think the cycle ends sooner than it does; a per-day ceiling of budget ÷ 28 limits the damage, but set it correctly. |

### Research dataset (claims) and market signals (on by default):

Every delivered summary also produces **atomic, evidence-backed claims** — one record per asset × metric × direction × target × horizon × condition, each with a verbatim excerpt located in the normalized transcript, the speaker attribution (own view / guest / quoted analyst / question / retrospective), the ticker only when spoken or curated, and a deterministic testability verdict. They live in `data/research/claims.jsonl`; the research ledger `data/research/research_state.json` tracks extraction independently of Telegram delivery (a failed extraction is `failed_retryable` and retried by `research_backfill.py --retry` from the stored transcript, never recorded as "no claims"). `python research_analytics.py --dry-run` prints the data-quality header, consensus (one current view per source per asset per horizon bucket), stance changes and the scorecard. `data/signals.jsonl` continues to be written, now **derived** from the validated claims (`claims.claims_to_legacy_signals` documents the reduction).

### Market signals (on by default):

Every delivered summary is also
analyzed by one extra LLM call that extracts structured market signals — the
assets discussed, the speaker's stance (bullish/bearish/neutral), conviction,
stated action, catalysts, price targets, and overall market sentiment. Each
video appends one JSON line to `data/signals.jsonl` (date, video metadata,
summary, signals), which the daily workflow commits back so the dataset grows
run over run. Disable it by setting the repo variable `MARKET_SIGNALS=false`.
Extraction is best-effort: failures are logged and never affect
Telegram delivery or dedup state. The data is research input for later
aggregation (sentiment trends, consensus flips, per-channel track records) —
it is not investment advice.

Once records accumulate, a **weekly market pulse** is sent every Monday by
`.github/workflows/weekly-pulse.yml` (also runnable on demand, or locally with
`python market_pulse.py --dry-run`): the trailing week's top mentioned assets
with net stance and stated actions, consensus flips versus the prior week, and
assets newly on the radar. Weeks with no data are skipped silently. Once the
Friday scorecard has enough history (5+ evaluated calls for a channel), the
pulse's consensus becomes **accuracy-weighted** — each channel's stance counts
at 0.5 + its 7-day hit rate, shown in a report footer — and price targets are
annotated with the **implied move** versus the latest close (best-effort).
The text pulse is followed by a **photo album of six charts** built for
non-technical reading — each carries a plain-English headline and a "how to
read" line on the image itself. Charts are best-effort: any that fails, or has
too little history to mean anything, is skipped and never blocks the text.

Prices are not fetched by those jobs on the hot path. The provider's free tier
allows 8 requests/minute and the dataset spans hundreds of symbols, so
`.github/workflows/warm-prices.yml` runs every **Sunday** and tops up a
committed cache of daily closes (`data/prices.json`, `python warm_prices.py
--dry-run` to see what it would fetch). Closes are immutable history, so a
symbol whose calls have all elapsed is fetched once and never again — after the
first warm, a normal week only pays for new tickers and still-open horizons.
The Monday pulse and Friday scorecard read the cache and only reach the network
for gaps.

Every Friday, `.github/workflows/weekly-scorecard.yml` sends a **per-channel
accuracy scorecard** (`python channel_scorecard.py --dry-run` locally): each
channel's directional calls are checked against free daily prices — was the
price higher after a bullish call, lower after a bearish one — at 7- and
30-day horizons, with hit rates and average move in the called direction.
It waits automatically until the dataset spans at least a week, and always
shows sample sizes (small samples are noise, not skill).

### Dedup state (`seen_videos.json`):

The daily workflow commits this file back to the repo after each run. It stores a per-channel watermark (`last_video_id` + `last_published`) plus a `pending` map of videos deferred for retry (captions not up yet, LLM quota exhausted). A video counts as done only once Telegram has accepted its message: a finished summary that could not be delivered is kept in its pending record and re-sent on the next run at no transcript or LLM cost. Legacy flat `{channel_id: video_id}` files are migrated automatically.

### Example Output:
The script prints a per-run log like the following to the console:

```
[INFO] Processing channel ID: UC123456789
[INFO] Found video: "Example Video Title"
[INFO] Fetching transcript...
[INFO] Cleaning transcript...
[INFO] Summarizing transcript...
[INFO] Sending summary to Telegram...
[INFO] Process completed successfully.
```

The generated JSON file will have the following structure:

```json
[
  {
    "channel_name": "Example Channel",
    "video_title": "Example Video Title",
    "video_url": "https://www.youtube.com/watch?v=example123",
    "published_at": "2024-12-25T17:36:31Z",
    "transcript": "Full transcript text here...",
    "summary": "Overview line.\n\n• Key takeaway 1.\n• Key takeaway 2."
  }
]
```

## Implementation Details

### Summarization:
The transcript is summarized by a large language model through the OpenAI-compatible Chat Completions API, so the project is provider-agnostic (Gemini, Groq, OpenRouter, OpenAI, local servers, …) — see [LLM provider configuration](#llm-provider-configuration). The prompt instructs the model to produce a reader-friendly, plain-text summary:

- A single-sentence **TL;DR / overview** on the first line.
- A short list of **key takeaways** as `• ` bullets, scaling with the content (typically 3-7).
- Always in **English**, regardless of the transcript's source language.
- **Faithful to the transcript** — the model is told not to invent facts, names, or numbers, and to say so briefly if the transcript is too garbled to summarize.

The video **title** is passed alongside the transcript to ground the model on the topic. Output is kept plain-text (no markdown) because the Telegram sender HTML-escapes the summary, so `**bold**`/`#` markers would not render.

The **complete transcript** is sent every time. Before a request leaves the process it is measured in tokens — system prompt, title, transcript and the JSON envelope together — against the selected model's own input limit (live metadata when available, a dated registry otherwise). A request that does not fit is never trimmed: the transcript is summarized in ordered, overlapping chunks that cover every character, the chunk notes are merged by one final call, and partial coverage is never delivered. The old 120,000-character cap that cut the middle out of long videos is gone; see `docs/tdd-full-transcript-claims.md`.

Tunable summarization environment variables (all optional, with sensible defaults):

| Variable | Default | Purpose |
| --- | --- | --- |
| `SUMMARY_MAX_OUTPUT_TOKENS` | `4000` | Output allowance for the generated summary (input capacity is the model's own limit). |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature (lower = more faithful). |
| `LLM_TIMEOUT` | `120` | Per-request timeout in seconds. |

### Telegram Integration:
The script uses the Telegram Bot API to send summarized content directly to a Telegram channel.

## Contribution
Feel free to fork the repository and submit pull requests for enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
