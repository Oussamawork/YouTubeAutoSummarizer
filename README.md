# YouTube Latest Video Fetcher and Summarizer

⚠️ This project is tested and compatible with Python 3.11.

This Python script fetches the latest video from a YouTube channel using the YouTube Data API v3, extracts the transcript, summarizes the content using advanced natural language processing (NLP) techniques, and optionally sends the summarized content to a Telegram channel.

## Features
- Fetches the latest video from any YouTube channel.
- Displays the **channel name**, **video title**, **video URL**, and **publication date**.
- Extracts the **transcript** of the video (via Supadata or `youtube-transcript-api`).
- Summarizes the transcript with an **LLM via the OpenAI-compatible Chat Completions API** — provider-agnostic (Gemini, Groq, OpenRouter, OpenAI, local servers, …), configured by environment variables. The prompt grounds the model on the video title, translates to English when needed, and guards against over-long transcripts.
- Generates a one-line **TL;DR plus bullet-point key takeaways**, in English regardless of the source language.
- Supports sending the summarized content to a specified **Telegram channel**.
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
   The `channel_ids.txt` file lists one channel per line. Each line may be a
   **channel ID** (`UC...`), a **channel URL** (`https://youtube.com/@handle` or
   `/channel/UC...`), or a bare **@handle** — handles and URLs are resolved to
   channel IDs automatically via the YouTube Data API. Blank lines and lines
   starting with `#` are ignored. For example:

   ```txt
   # Channel ID, URL, or @handle — one per line
   UC_x5XG1OV2P6uZZ5FSM9Ttw
   https://www.youtube.com/@GoogleDevelopers
   @veritasium
   ```

## Usage

### Fetching and Summarizing Videos:
1. Add YouTube channel IDs to `channel_ids.txt` (one per line).
2. Run the script to fetch the latest video, extract the transcript, clean it, summarize it, and optionally send the summary to Telegram:

   ```bash
   python scraper.py
   ```

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
