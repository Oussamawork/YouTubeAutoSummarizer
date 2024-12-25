import colorama
from colorama import Fore, Style

# Initialize colorama (especially important on Windows)
colorama.init()

def log_info(message):
    print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} {message}")

def log_debug(message):
    print(f"{Fore.BLUE}[DEBUG]{Style.RESET_ALL} {message}")

def log_warn(message):
    print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} {message}")

def log_error(message):
    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {message}")
