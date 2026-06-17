import requests
import time
from dotenv import load_dotenv
from transcript import get_transcript_from_video
from helpers import read_channel_ids, save_to_json, clean_summary, load_seen_videos, save_seen_videos
from summarizer import summarize_transcript
from log import log_info, log_error, log_warn, log_debug
from sendToTelegram import send_telegram_message
import os
from datetime import datetime

# Load environment variables from .env file
load_dotenv('.env')

# YouTube API request tuning
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/search"
REQUEST_TIMEOUT = 15          # seconds before a hung request is abandoned
MAX_RETRIES = 3               # attempts for transient failures
RETRY_BACKOFF = 2            # base seconds, multiplied by the attempt number
TRANSIENT_STATUS = {429, 500, 502, 503, 504}

# File that remembers the last video summarized per channel (dedup state)
SEEN_VIDEOS_FILE = "seen_videos.json"


def _parse_latest_video(data):
    """Turn a YouTube search response into our video dict, or None if empty."""
    items = data.get("items") if isinstance(data, dict) else None
    if not items:
        log_warn("No videos found for this channel.")
        return None

    video = items[0]
    video_id = video["id"]["videoId"]
    snippet = video["snippet"]
    log_info(f"Found video: {snippet['title']} | Channel: {snippet['channelTitle']}")
    return {
        "video_id": video_id,
        "channel_name": snippet["channelTitle"],
        "video_title": snippet["title"],
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "published_at": snippet["publishedAt"],
    }


def get_latest_video(YOUTUBE_api_key, channel_id):
    log_info(f"get_latest_video called for channel_id={channel_id}")

    params = {
        "part": "snippet",
        "channelId": channel_id,
        "order": "date",
        "type": "video",
        "maxResults": 1,
        "key": YOUTUBE_api_key,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(YOUTUBE_API_URL, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            log_warn(f"YouTube API request error (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
                continue
            log_error("YouTube API unreachable after retries.")
            return None

        log_debug(f"GET request to: {response.url}")

        if response.status_code == 200:
            log_debug("Received 200 OK from YouTube API")
            try:
                return _parse_latest_video(response.json())
            except (ValueError, KeyError) as e:
                log_error(f"Unexpected YouTube API response shape: {e}")
                return None

        if response.status_code in TRANSIENT_STATUS and attempt < MAX_RETRIES:
            log_warn(
                f"Transient YouTube API status {response.status_code} "
                f"(attempt {attempt}/{MAX_RETRIES}); retrying."
            )
            time.sleep(RETRY_BACKOFF * attempt)
            continue

        log_error(f"YouTube API returned status code {response.status_code} | {response.text}")
        return None

if __name__ == "__main__":
    log_info("Starting main script.")
    
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")

    # Fail fast if any required credential is missing, rather than discovering it
    # mid-run when every YouTube lookup or Telegram send fails.
    required = {
        "YOUTUBE_API_KEY": YOUTUBE_API_KEY,
        "TELEGRAM_TOKEN": TELEGRAM_TOKEN,
        "TELEGRAM_CHANNEL_ID": TELEGRAM_CHANNEL_ID,
    }
    missing = [name for name, value in required.items() if not value]

    if missing:
        log_error(
            "Missing required environment variables: "
            f"{', '.join(missing)}. Set them as repository secrets (CI) or in .env (local)."
        )
    else:
        log_info("All required environment variables loaded.")
        if not os.getenv("SUPADATA_API_KEY"):
            log_warn(
                "SUPADATA_API_KEY not set; transcript fetching may fail on CI "
                "where YouTube blocks the runner IP."
            )
        
        # Read channel IDs from the file
        channel_ids = read_channel_ids("channel_ids.txt")
        if not channel_ids:
            log_warn("No channel IDs found. Check your channel_ids.txt file.")
        else:
            log_info(f"Beginning process to fetch video details for each channel.")
            results = []
            seen_videos = load_seen_videos(SEEN_VIDEOS_FILE)

            for channel_id in channel_ids:
                transcript = ''
                log_info(f"Processing channel ID: {channel_id}")
                video_details = get_latest_video(YOUTUBE_API_KEY, channel_id)
                if video_details:
                    # Skip channels whose latest video was already processed, so
                    # the same summary isn't re-sent every day.
                    if seen_videos.get(channel_id) == video_details['video_id']:
                        log_info(
                            f"No new video for channel {channel_id} "
                            f"(latest already processed: {video_details['video_id']}). Skipping."
                        )
                        continue

                    log_info(f"Video details retrieved: {video_details['video_title']} (published: {video_details['published_at']})")

                    log_info(f"Fetching transcript for {video_details['video_url']} ...")
                    transcript = get_transcript_from_video(video_details['video_url'])

                    # Check the actual transcript TEXT, not the dict (a dict is always truthy).
                    transcript_text = transcript.get('transcript', '') if isinstance(transcript, dict) else ''
                    transcript_text = transcript_text.strip() if transcript_text else ''

                    if transcript_text:
                        log_info("Transcript fetched successfully.")
                        log_info("Summarizing transcript...")
                        summary = clean_summary(summarize_transcript(transcript_text))
                        video_details['transcript'] = transcript

                        if summary:
                            video_details['summary_facebook_bart'] = summary
                            telegram_body = summary
                            log_info("Summary generated and sent to Telegram.")
                        else:
                            # Transcript existed but the summarizer produced nothing.
                            video_details['summary_facebook_bart'] = "Summary not available."
                            telegram_body = "⚠️ A transcript was found, but summarization produced no output. Manual review needed."
                            log_warn("Empty summary despite a transcript. Notifying Telegram.")
                    else:
                        # No transcript at all (kome.ai failed or the video has no captions).
                        video_details['transcript'] = "Transcript not found."
                        video_details['summary_facebook_bart'] = "Summary not available."
                        telegram_body = "⚠️ No transcript available for this video, so no summary could be generated. Manual review needed."
                        log_warn("Transcript empty or unavailable. Notifying Telegram.")

                    # Always notify Telegram so empty summaries are never silent.
                    send_telegram_message(TELEGRAM_TOKEN, TELEGRAM_CHANNEL_ID, video_details['channel_name'], video_details['video_title'], video_details['video_url'], video_details['published_at'], telegram_body)

                    # Mark this video as processed and persist immediately, so a
                    # later crash doesn't cause already-sent videos to be re-sent.
                    seen_videos[channel_id] = video_details['video_id']
                    save_seen_videos(SEEN_VIDEOS_FILE, seen_videos)

                    results.append(video_details)
                else:
                    log_warn(f"No video details returned for channel ID: {channel_id}.")

            # Save the results to a JSON file
            if results:
                filename = 'video_details_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.json'
                log_info(f"Saving results to {filename} ...")
                save_to_json(results, filename)
                log_info(f"Process completed successfully.")
            else:
                log_warn("No results to save.")

    log_info(f"Main script finished.")
