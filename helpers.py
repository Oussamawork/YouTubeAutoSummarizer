import json
import re

from log import log_error

# Function to read channel IDs from a file
def read_channel_ids(file_path):
    """
    Read channel IDs, one per line. Blank lines and lines starting with '#'
    (comments) are ignored. Returns [] if the file is missing.
    """
    try:
        with open(file_path, "r") as file:
            ids = []
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                ids.append(line)
            return ids
    except FileNotFoundError:
        log_error(f"Channel IDs file not found: {file_path}")
        return []
    
# Function to save results to a JSON file
def save_to_json(data, filename):
    try:
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data to JSON file: {e}")

def clean_summary(summary: str) -> str:
    # 1. Replace non-breaking spaces with a regular space
    summary_clean = summary.replace("\u00a0", " ")
    
    # 2. Condense any sequence of whitespace (spaces, tabs, newlines, etc.) into a single space
    summary_clean = re.sub(r"\s+", " ", summary_clean)
    
    # 3. Strip leading and trailing spaces
    summary_clean = summary_clean.strip()
    
    return summary_clean