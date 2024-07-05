import json

def find_invalid_json_lines(file_path):
    with open(file_path, 'r') as file:
        print(f"Total len = {len(file.readlines())}")
        for line_number, line in enumerate(file, start=1):
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON at line {line_number}: {e.msg}")
                # Optionally, print the problematic line to help with debugging
                print(f"Problematic line content: {line}")

# Replace 'your_file.json' with the path to your JSON file
file_path = '/network/eldar/datasets/shubhra_deduplicated/train.jsonl'
find_invalid_json_lines(file_path)

