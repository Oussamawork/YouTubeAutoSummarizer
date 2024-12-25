import requests

def get_transcript_from_video(video_id):
    url = 'https://api.kome.ai/api/tools/youtube-transcripts'
    params = {
        "format": True,
        "video_id": video_id
    }
    response = requests.post(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None