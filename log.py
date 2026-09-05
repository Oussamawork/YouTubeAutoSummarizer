import logging
import os
import re

import colorama
from colorama import Fore, Style

# Initialize colorama (especially important on Windows)
colorama.init()

# Short, colored level labels matching the project's original style.
_LEVEL_LABELS = {
    logging.DEBUG: f"{Fore.BLUE}[DEBUG]{Style.RESET_ALL}",
    logging.INFO: f"{Fore.GREEN}[INFO]{Style.RESET_ALL}",
    logging.WARNING: f"{Fore.YELLOW}[WARN]{Style.RESET_ALL}",
    logging.ERROR: f"{Fore.RED}[ERROR]{Style.RESET_ALL}",
    logging.CRITICAL: f"{Fore.RED}[CRITICAL]{Style.RESET_ALL}",
}


class _ColorFormatter(logging.Formatter):
    def format(self, record):
        label = _LEVEL_LABELS.get(record.levelno, f"[{record.levelname}]")
        return f"{label} {record.getMessage()}"


_logger = logging.getLogger("youtube_auto_summarizer")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(_ColorFormatter())
    _logger.addHandler(_handler)
    _logger.propagate = False

# Verbosity is configurable via LOG_LEVEL (DEBUG/INFO/WARNING/ERROR); defaults to INFO.
_logger.setLevel((os.getenv("LOG_LEVEL") or "INFO").upper())


def log_info(message):
    _logger.info(message)


def log_debug(message):
    _logger.debug(message)


def log_warn(message):
    _logger.warning(message)


def log_error(message, exc_info=False):
    """Log an error; `exc_info=True` appends the current traceback, for the
    catch-all handlers whose one-line message otherwise hides the cause."""
    _logger.error(message, exc_info=exc_info)


# Credentials travel in URLs: the Telegram bot token is a path segment and the
# YouTube API key a query parameter. `requests` embeds the request URL in its
# exception text, so a naive f"{e}" prints them. GitHub Actions masks exact
# secret values in its logs, but a local run or a copied log line does not.
_SECRET_PATTERNS = (
    re.compile(r"/bot[0-9]+:[A-Za-z0-9_-]+"),
    re.compile(r"([?&](?:key|apikey|api_key)=)[^&\s'\"]+", re.IGNORECASE),
)


def redact(text):
    """`text` with bot tokens and API keys replaced by a placeholder."""
    text = str(text)
    text = _SECRET_PATTERNS[0].sub("/bot<redacted>", text)
    return _SECRET_PATTERNS[1].sub(r"\1<redacted>", text)


def describe_error(exc):
    """A redacted one-line description of an exception, for log messages."""
    return f"{type(exc).__name__}: {redact(exc)}"
