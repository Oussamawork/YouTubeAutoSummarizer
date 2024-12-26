import requests

def send_telegram_message(bot_token, chat_id, channel_name, video_title, video_url, published_at, summary):

     # Format the message
    message = (
        f"🎥 *Channel*: {channel_name}\n"
        f"📅 *Published At*: {published_at}\n"
        f"📌 *Video Title*: {video_title}\n"
        f"🔗 [Watch on YouTube]({video_url})\n\n"
        f"📜 *Summary*:\n{summary}"
    )

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",  # Supports bold, italics, and links
    }
    response = requests.post(url, data=data)
    if response.status_code != 200:
        print(f"[ERROR] Could not send message. Response: {response.text}")
    else:
        print("[INFO] Message sent successfully to Telegram channel.")
