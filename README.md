# YouTube Latest Video Fetcher

This Python script fetches the latest video from a YouTube channel using the YouTube Data API v3. The script uses an API key to authenticate requests and fetch the latest video from a list of specified channel IDs.

## Features
- Fetches the latest video from any YouTube channel.
- Displays the **channel name**, **video title**, and **video URL**.
- Uses environment variables to securely store the API key.
  
## Requirements
Before running the script, ensure you have the required Python dependencies:

- `requests` for making HTTP requests to the YouTube API.
- `python-dotenv` for loading the API key from the `.env` file.

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Configuration

### 1. Obtain YouTube API Key:
   - Visit the [Google Developers Console](https://console.developers.google.com/).
   - Create a project and enable the **YouTube Data API v3**.
   - Obtain your **API key** from the **Credentials** section of the Google Console.

### 2. Create `.env` File:
   In the root of your project directory, create a file named `.env` and add your YouTube API key like this:

   ```env
   YOUTUBE_API_KEY=YOUR_API_KEY
