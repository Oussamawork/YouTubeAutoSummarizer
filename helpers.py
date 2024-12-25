import json

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
