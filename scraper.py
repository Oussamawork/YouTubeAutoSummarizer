import requests
from dotenv import load_dotenv
from transcript import get_transcript_from_video
from helpers import read_channel_ids, save_to_json, clean_summary
from summarizer import summarize_transcript, second_pass_summarize
from log import log_info, log_error, log_warn, log_debug
from sendToTelegram import send_telegram_message
import os
from datetime import datetime

# Load environment variables from .env file
load_dotenv('.env')

def get_latest_video(YOUTUBE_api_key, channel_id):
    log_info(f"get_latest_video called for channel_id={channel_id}")

    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "channelId": channel_id,
        "order": "date",
        "type": "video",
        "maxResults": 1,
        "key": YOUTUBE_api_key,
    }
    response = requests.get(url, params=params)
    log_debug(f"GET request to: {response.url}")

    if response.status_code == 200:
        data = response.json()
        log_debug("Received 200 OK from YouTube API")
        if "items" in data and len(data["items"]) > 0:
            video = data["items"][0]
            video_id = video["id"]["videoId"]
            video_title = video["snippet"]["title"]
            channel_name = video["snippet"]["channelTitle"]
            published_at = video["snippet"]["publishedAt"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"

            log_info(f"Found video: {video_title} | Channel: {channel_name}")

            return {
                'channel_name': channel_name,
                'video_title': video_title,
                'video_url': video_url,
                'published_at': published_at
            }
        else:
            log_warn("No videos found for this channel.")
            return None
    else:
        log_error(f"YouTube API returned status code {response.status_code} | {response.text}")
        return None

if __name__ == "__main__":
    log_info("Starting main script.")
    
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")


    if not YOUTUBE_API_KEY:
        log_error("API Key not found. Make sure it's set in the .env file.")
    else:
        log_info("Successfully loaded API key from environment.")
        
        # Read channel IDs from the file
        channel_ids = read_channel_ids("channel_ids.txt")
        if not channel_ids:
            log_warn("No channel IDs found. Check your channel_ids.txt file.")
        else:
            log_info(f"Beginning process to fetch video details for each channel.")
            results = []

            for channel_id in channel_ids:
                transcript = ''
                log_info(f"Processing channel ID: {channel_id}")
                video_details = get_latest_video(YOUTUBE_API_KEY, channel_id)
                if video_details:
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

                    results.append(video_details)
                else:
                    log_warn("No video details returned for channel ID: {channel_id}.")

            # Save the results to a JSON file
            if results:
                filename = 'video_details_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.json'
                log_info(f"Saving results to {filename} ...")
                save_to_json(results, filename)
                log_info(f"Process completed successfully.")
            else:
                log_warn("No results to save.")

    log_info(f"Main script finished.")
