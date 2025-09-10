"""
Convert Kokoro-ONNX voices bin file to JSON format that matches the library's expectations.

The binary file contains embeddings for each voice as numpy arrays.
Kokoro-ONNX expects these to be preserved in the JSON as numeric arrays.

Run this script with:
uv run python scripts/convert_voices_to_json_fixed.py
"""

import json
import numpy as np
from pathlib import Path


def convert_bin_to_json(bin_path, json_path):
    """Convert binary voices file to JSON format required by Kokoro-ONNX."""
    try:
        # Load the binary voices file which is a .npz archive
        print(f"Loading binary voices file from {bin_path}")
        voices_data = np.load(bin_path, allow_pickle=True)

        # Extract voice names and their corresponding embeddings
        voice_names = list(voices_data.item().keys())
        print(f"Found {len(voice_names)} voices")

        # Create a dictionary mapping voice names to their embeddings
        voices_dict = {}
        for voice_name in voice_names:
            # Convert the numpy array to a regular Python list for JSON serialization
            voice_array = voices_data.item()[voice_name].tolist()
            voices_dict[voice_name] = voice_array

        # Write the JSON file
        with open(json_path, "w") as f:
            json.dump(voices_dict, f, indent=2)

        print(f"Successfully created {json_path} with {len(voices_dict)} voices")
        # Debug: Show a sample of the data
        sample_voice = list(voices_dict.keys())[0]
        print(
            f"Sample voice '{sample_voice}' has embedding of length {len(voices_dict[sample_voice])}"
        )
        return True
    except Exception as e:
        print(f"Failed to convert voices file: {str(e)}")
        return False


def main():
    # Define paths
    tts_dir = Path("tts")
    bin_path = tts_dir / "voices.bin"
    json_path = tts_dir / "voices.json"

    # Ensure the TTS directory exists
    tts_dir.mkdir(exist_ok=True)

    # Convert the file
    if not bin_path.exists():
        print(f"Error: Binary voices file not found at {bin_path}")
        print("Run fetch_voices.py first to download the voices")
        return

    success = convert_bin_to_json(bin_path, json_path)

    if success:
        print("✅ Voice file successfully converted to JSON format")
        print(f"   - Source: {bin_path}")
        print(f"   - Target: {json_path}")
        print(
            "\nNow update your TTS_VOICE_FILE environment variable to point to this JSON file."
        )
    else:
        print("❌ Failed to convert voices file")


if __name__ == "__main__":
    main()
