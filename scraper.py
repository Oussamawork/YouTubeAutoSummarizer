import requests
from dotenv import load_dotenv
from transcript import get_transcript_from_video
from helpers import read_channel_ids
from helpers import save_to_json 
import os
import json
from datetime import datetime


# Load environment variables from .env file
load_dotenv('.env')

# Function to get the latest video from a channel
def get_latest_video(api_key, channel_id):
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
    if response.status_code == 200:
        data = response.json()
        if "items" in data and len(data["items"]) > 0:
            video = data["items"][0]
            video_id = video["id"]["videoId"]
            video_title = video["snippet"]["title"]
            channel_name = video["snippet"]["channelTitle"]
            published_at = video["snippet"]["publishedAt"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            return {
                    'channel_name' : channel_name, 
                    'video_title' : video_title,
                    'video_url' : video_url,
                    'published_at' : published_at
                }
        else:
            print("No videos found for this channel.")
            return None
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None
    

# Main script
if __name__ == "__main__":
    # Replace with your actual API key
    API_KEY = os.getenv("YOUTUBE_API_KEY")
    
    if not API_KEY:
        print("API Key not found. Make sure it's set in the .env file.")
    else:
        # Read channel IDs from the file
        channel_ids = read_channel_ids("channel_ids.txt")
        if not channel_ids:
            print("No channel IDs found. Please check your channel_ids.txt file.")
        else:
            results = []
            # Loop through the list of channel IDs and get the latest video
            for channel_id in channel_ids:
                video_details = get_latest_video(API_KEY, channel_id)
                if video_details:
                    # Fetch the transcript for the video
                    transcript = get_transcript_from_video(video_details['video_url'])
                    if transcript:
                        video_details['transcript'] = transcript
                    else:
                        video_details['transcript'] = "Transcript not found."
                    
                    results.append(video_details)
                
            # Save the results to a JSON file
            if results:
                filename = 'video_details_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.json'
                save_to_json(results, filename)
            
            
