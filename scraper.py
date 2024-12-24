import requests

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
            print(f"Channel Name: {channel_name}\nLatest Video: {video_title}\nURL: {video_url}")
            return video_url
        else:
            print("No videos found for this channel.")
            return None
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# Main script
if __name__ == "__main__":
    # Replace with your actual API key
    API_KEY = "AIzaSyDvu5ArsXiu1pvYDQUUUNzw1JPDMHH8sVI"
    
    # List of channel IDs
    channel_ids = [
        "UCrGLm-Drgv0vbbemwwHeXJw",  
    ]
    
    # Loop through the list of channel IDs and get the latest video
    for channel_id in channel_ids:
        get_latest_video(API_KEY, channel_id)