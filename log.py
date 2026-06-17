import logging
import os

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


def log_error(message):
    _logger.error(message)
