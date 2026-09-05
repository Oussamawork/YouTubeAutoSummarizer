import log


def test_redact_hides_bot_token_and_api_key():
    url = "https://api.telegram.org/bot123456:ABC-def_ghi/sendMessage"
    assert "123456" not in log.redact(url)
    assert "/bot<redacted>/sendMessage" in log.redact(url)
    yt = "https://www.googleapis.com/youtube/v3/search?part=snippet&key=AIzaSecret&maxResults=1"
    assert "AIzaSecret" not in log.redact(yt)
    assert "maxResults=1" in log.redact(yt)


def test_describe_error_names_the_type_and_redacts():
    err = ValueError("boom at https://api.telegram.org/bot1:tok/sendMessage")
    text = log.describe_error(err)
    assert text.startswith("ValueError: ")
    assert "1:tok" not in text
