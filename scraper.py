import requests
from dotenv import load_dotenv
from transcript import get_transcript_from_video
from helpers import read_channel_ids, save_to_json
from summarizer import summarize_transcript
import os
from datetime import datetime

# Load environment variables from .env file
load_dotenv('.env')

def get_latest_video(api_key, channel_id):
    print(f"[INFO] get_latest_video called for channel_id={channel_id}")
    
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
    print(f"[DEBUG] GET request to: {response.url}")
    
    if response.status_code == 200:
        data = response.json()
        print("[DEBUG] Received 200 OK from YouTube API")
        if "items" in data and len(data["items"]) > 0:
            video = data["items"][0]
            video_id = video["id"]["videoId"]
            video_title = video["snippet"]["title"]
            channel_name = video["snippet"]["channelTitle"]
            published_at = video["snippet"]["publishedAt"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"

            print(f"[INFO] Found video: {video_title} | Channel: {channel_name}")

            return {
                'channel_name': channel_name,
                'video_title': video_title,
                'video_url': video_url,
                'published_at': published_at
            }
        else:
            print("[WARN] No videos found for this channel.")
            return None
    else:
        print(f"[ERROR] YouTube API returned status code {response.status_code} | {response.text}")
        return None

if __name__ == "__main__":
    print("[INFO] Starting main script.")
    
    API_KEY = os.getenv("YOUTUBE_API_KEY")
    if not API_KEY:
        print("[ERROR] API Key not found. Make sure it's set in the .env file.")
    else:
        print("[INFO] Successfully loaded API key from environment.")
        
        # Read channel IDs from the file
        channel_ids = read_channel_ids("channel_ids.txt")
        if not channel_ids:
            print("[WARN] No channel IDs found. Check your channel_ids.txt file.")
        else:
            print("[INFO] Beginning process to fetch video details for each channel.")
            results = []

            for channel_id in channel_ids:
                print(f"\n[INFO] Processing channel ID: {channel_id}")
                video_details = get_latest_video(API_KEY, channel_id)

                if video_details:
                    print(f"[INFO] Video details retrieved: {video_details['video_title']} (published: {video_details['published_at']})")
                    
                    print(f"[INFO] Fetching transcript for {video_details['video_url']} ...")
                    transcript = get_transcript_from_video(video_details['video_url'])
                    
                    if transcript:
                        print("[INFO] Transcript fetched successfully.")
                        print("[INFO] Summarizing transcript...")
                        summary = summarize_transcript(transcript)
                        video_details['transcript'] = transcript
                        video_details['summary'] = summary
                        print("[INFO] Summary generated.")
                    else:
                        print("[WARN] Transcript not found.")
                        video_details['transcript'] = "Transcript not found."
                        video_details['summary'] = "Summary not available."

                    results.append(video_details)
                else:
                    print(f"[WARN] No video details returned for channel ID: {channel_id}.")

            # Save the results to a JSON file
            if results:
                filename = 'video_details_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.json'
                print(f"[INFO] Saving results to {filename} ...")
                save_to_json(results, filename)
                print("[INFO] Process completed successfully.")
            else:
                print("[WARN] No results to save.")

    print("[INFO] Main script finished.")
