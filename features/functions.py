import json

def load_lottie_file(path: str):
    """Utility to read lottie JSON files safely."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)