import requests
from log import log_info, log_warn, log_error  # assuming you're using structured logging

def get_transcript_from_video(video_id):
    url = 'https://api.kome.ai/api/tools/youtube-transcripts'
    params = {
        "format": True,
        "video_id": video_id
    }

    try:
        response = requests.post(url, params=params)
        log_info(f"Transcript request status: {response.status_code}")

        if response.status_code != 200:
            log_warn(f"Non-200 status code: {response.status_code}")
            log_warn(f"Response body: {response.text[:200]}")  # Short preview
            return {"transcript": ""}

        # Attempt to parse JSON only if response has proper content-type
        if 'application/json' in response.headers.get('Content-Type', ''):
            return response.json()
        else:
            log_error("Response content is not JSON.")
            log_warn(f"Response content: {response.text[:200]}")
            return {"transcript": ""}

    except ValueError as json_err:
        log_error(f"JSON decoding error: {json_err}")
        return {"transcript": ""}
    except Exception as e:
        log_error(f"Exception during transcript fetch: {str(e)}")
        return {"transcript": ""}
