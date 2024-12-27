# YouTube Latest Video Fetcher and Summarizer

This Python script fetches the latest video from a YouTube channel using the YouTube Data API v3, extracts the transcript, summarizes the content using advanced natural language processing (NLP) techniques, and optionally sends the summarized content to a Telegram channel.

## Features
- Fetches the latest video from any YouTube channel.
- Displays the **channel name**, **video title**, **video URL**, and **publication date**.
- Extracts the **transcript** of the video and translates it into English if needed.
- Cleans the transcript by removing timestamps, repeated lines, and unnecessary whitespace to ensure a high-quality summary.
- Summarizes the transcript using `facebook/bart-large-cnn` with a **sliding window approach** to preserve context.
- Generates **bullet-point summaries** of the transcript with actionable insights.
- Supports sending the summarized content to a specified **Telegram channel**.
- Uses environment variables to securely store API keys and tokens.

## Requirements
Before running the script, ensure you have the required Python dependencies:

- `transformers` for NLP and summarization.
- `langdetect` for language detection.
- `requests` for making HTTP requests to the YouTube API.
- `python-dotenv` for loading environment variables from the `.env` file.
- `loguru` for structured logging.

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
    "summary": "Summarized text here...",
    "bullet_points": [
      "Key idea 1 from the summary.",
      "Key idea 2 from the summary.",
      "Key idea 3 from the summary."
    ]
  }
]
```

## Implementation Details

### Sliding Window Chunking:
The script uses a sliding window mechanism to handle long transcripts that exceed the token limit of the summarization model (typically 1024 tokens for `facebook/bart-large-cnn`). The mechanism works as follows:

1. **Window Size**: Each chunk is limited to 1024 characters (or tokens) to ensure compatibility with the model.
2. **Overlap Size**: A 256-character overlap is introduced between consecutive chunks to retain context from the previous chunk.
3. **Context Retention**: The overlap ensures that key ideas that span across chunk boundaries are not lost, preserving the coherence of the summary.

This approach balances the model's input limitations with the need for comprehensive summaries of long texts.

### NLP Models:
- **Summarization**: `facebook/bart-large-cnn`
- **Translation**: `Helsinki-NLP/opus-mt-[detected_language]-en`

### Transcript Cleaning:
Before summarization, the script removes timestamps, redundant lines, and excessive whitespace to improve the quality of the input data for the summarization model.

### Telegram Integration:
The script uses the Telegram Bot API to send summarized content directly to a Telegram channel.

## Contribution
Feel free to fork the repository and submit pull requests for enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
