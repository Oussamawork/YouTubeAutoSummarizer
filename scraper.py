import requests
from dotenv import load_dotenv
from transcript import get_transcript_from_video
from helpers import read_channel_ids, save_to_json
from summarizer import summarize_transcript
from log import log_info, log_error, log_warn, log_debug
import os
from datetime import datetime

# Load environment variables from .env file
load_dotenv('.env')

def get_latest_video(api_key, channel_id):
    log_info("get_latest_video called for channel_id={channel_id}")

    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "channelId": channel_id,
        "order": "date",
        "type": "video",
        "maxResults": 1,
        "key": api_key,
    }
    response = requests.get(url, params=params)
    log_debug("GET request to: {response.url}")

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

            log_info("Found video: {video_title} | Channel: {channel_name}")

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
        log_error("YouTube API returned status code {response.status_code} | {response.text}")
        return None

if __name__ == "__main__":
    log_info("Starting main script.")
    
    API_KEY = os.getenv("YOUTUBE_API_KEY")
    if not API_KEY:
        log_error("API Key not found. Make sure it's set in the .env file.")
    else:
        log_info("Successfully loaded API key from environment.")
        
        # Read channel IDs from the file
        channel_ids = read_channel_ids("channel_ids.txt")
        if not channel_ids:
            log_warn("No channel IDs found. Check your channel_ids.txt file.")
        else:
            log_info("Beginning process to fetch video details for each channel.")
            results = []

            for channel_id in channel_ids:
                transcript = ''
                log_info("Processing channel ID: {channel_id}")
                video_details = get_latest_video(API_KEY, channel_id)
                if video_details:
                    log_info("Video details retrieved: {video_details['video_title']} (published: {video_details['published_at']})")
                    
                    log_info("Fetching transcript for {video_details['video_url']} ...")
                    transcript = get_transcript_from_video(video_details['video_url'])

                    if transcript:
                        log_info("Transcript fetched successfully.")
                        log_info("Summarizing transcript...")
                        summary = summarize_transcript(transcript['transcript'])
                        video_details['transcript'] = transcript
                        video_details['summary'] = summary.replace("\u00a0", " ")
                        log_info("Summary generated.")
                    else:
                        log_warn("Transcript not found.")
                        video_details['transcript'] = "Transcript not found."
                        video_details['summary'] = "Summary not available."

                    results.append(video_details)
                else:
                    log_warn("No video details returned for channel ID: {channel_id}.")

            # Save the results to a JSON file
            if results:
                filename = 'video_details_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.json'
                log_info("Saving results to {filename} ...")
                save_to_json(results, filename)
                log_info("Process completed successfully.")
            else:
                log_warn("No results to save.")

    log_info("Main script finished.")
