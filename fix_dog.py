import json

# Load the JSON file
with open('future.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Replace NaN values with "Nan"
def replace_nan(obj):
    if isinstance(obj, dict):
        return {k: replace_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan(elem) for elem in obj]
    elif obj != obj:  # Check for NaN
        return "Nan"
    return obj

data = replace_nan(data)

# Save the updated JSON file
with open('future.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
