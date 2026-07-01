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
   The `channel_ids.txt` file should contain one YouTube channel ID per line. For example:

   ```txt
   UC_x5XG1OV2P6uZZ5FSM9Ttw
   UCBR8-60-B28hp2BmDPdntcQ
   UC123456789
   ```

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
| `MAX_VIDEOS_PER_RUN` | `3` | Max videos processed per channel per run; older ones go first, the rest wait for the next run. |
| `NO_TRANSCRIPT_MAX_ATTEMPTS` | `3` | Runs to retry a video whose captions aren't up yet before notifying and giving up. |
| `DAILY_DIGEST` | off | `true` bundles all of a run's summaries into one combined Telegram message. |

### Dedup state (`seen_videos.json`):

The daily workflow commits this file back to the repo after each run. It stores a per-channel watermark (`last_video_id` + `last_published`) plus a `pending` map of videos deferred for retry (captions not up yet, LLM quota exhausted). Legacy flat `{channel_id: video_id}` files are migrated automatically.

### Example Output:
The script outputs a JSON file containing details of the latest videos and their summaries. Additionally, it prints the following to the console:

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

To avoid blowing past model context windows and to keep token cost predictable, very long transcripts are **truncated** to a configurable character cap before being sent, with a `...[transcript truncated]` marker appended and a warning logged.

Tunable summarization environment variables (all optional, with sensible defaults):

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_MAX_TOKENS` | `1500` | Max tokens for the generated summary. |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature (lower = more faithful). |
| `LLM_TIMEOUT` | `60` | Per-request timeout in seconds. |
| `LLM_MAX_TRANSCRIPT_CHARS` | `48000` | Character cap on transcript text sent to the model. |

### Telegram Integration:
The script uses the Telegram Bot API to send summarized content directly to a Telegram channel.

## Contribution
Feel free to fork the repository and submit pull requests for enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
