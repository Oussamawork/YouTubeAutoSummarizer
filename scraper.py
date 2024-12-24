import requests
from dotenv import load_dotenv
from transcript import get_transcript_from_video
import os


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
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            return {
                    'channel_name' : channel_name, 
                    'video_title' : video_title,
                    'video_url' : video_url
                }
        else:
            print("No videos found for this channel.")
            return None
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# Function to read channel IDs from a file
def read_channel_ids(file_path):
    with open(file_path, "r") as file:
        return [line.strip() for line in file.readlines()]
    
# Main script
if __name__ == "__main__":
    # Replace with your actual API key
    API_KEY = os.getenv("YOUTUBE_API_KEY")
    
    if not API_KEY:
        print("API Key not found. Make sure it's set in the .env file.")
    else:
        # Read channel IDs from the file
        channel_ids = read_channel_ids("channel_ids.txt")
        
        # Loop through the list of channel IDs and get the latest video
        for channel_id in channel_ids:
            video_details = get_latest_video(API_KEY, channel_id)
            transcript = get_transcript_from_video(video_details['video_url'])
            print(transcript)
            
            
