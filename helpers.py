import json
import re

# Function to read channel IDs from a file
def read_channel_ids(file_path):
    with open(file_path, "r") as file:
        return [line.strip() for line in file.readlines()]
    
# Function to save results to a JSON file
def save_to_json(data, filename):
    try:
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data to JSON file: {e}")

def clean_summary(summary: str) -> str:
    """
    Replaces non-breaking spaces (\u00a0) with regular spaces and
    condenses all runs of whitespace into a single space.

    Args:
        summary (str): The raw summary text to clean.

    Returns:
        str: The cleaned summary.
    """
    # 1. Replace non-breaking spaces with a regular space
    summary_clean = summary.replace("\u00a0", " ")
    
    # 2. Condense any sequence of whitespace (spaces, tabs, newlines, etc.) into a single space
    summary_clean = re.sub(r"\s+", " ", summary_clean)
    
    # 3. Strip leading and trailing spaces
    summary_clean = summary_clean.strip()
    
    return summary_clean