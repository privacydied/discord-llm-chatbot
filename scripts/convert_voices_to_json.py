"""
Convert Kokoro-ONNX voices bin file to JSON format.

Run this script with:
uv run python scripts/convert_voices_to_json.py
"""

import json
import logging
import numpy as np
from pathlib import Path

def convert_bin_to_json(bin_path, json_path):
    """Convert binary voices file to JSON format required by Kokoro-ONNX."""
    try:
        # Load the binary voices file
        print(f"Loading binary voices file from {bin_path}")
        with open(bin_path, 'rb') as f:
            voices_data = np.load(f)
            
        # Extract voice names from the .npz file
        voice_names = list(voices_data.files)
        print(f"Found {len(voice_names)} voices")
        
        # Create a simple JSON structure that lists available voices
        voices_json = {
            "voices": voice_names
        }
        
        # Write the JSON file
        with open(json_path, 'w') as f:
            json.dump(voices_json, f, indent=2)
        
        print(f"Successfully created {json_path} with {len(voice_names)} voices")
        return True
    except Exception as e:
        logging.error(f"Failed to convert voices file: {str(e)}")
        return False

def main():
    # Define paths
    tts_dir = Path("tts")
    bin_path = tts_dir / "voices-v1.0.bin"
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
        print(f"✅ Voice file successfully converted to JSON format")
        print(f"   - Source: {bin_path}")
        print(f"   - Target: {json_path}")
        print("\nNow update your TTS_VOICE_FILE environment variable to point to this JSON file.")
    else:
        print("❌ Failed to convert voices file")

if __name__ == "__main__":
    main()
