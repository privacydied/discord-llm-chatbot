#!/usr/bin/env python3
"""
Download official voice models for Kokoro-ONNX.
This script fetches the official voice files from the kokoro-onnx GitHub repository.
"""

import requests
import logging
import json
import os
import sys
from pathlib import Path
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# URLs for official files
VOICES_JSON_URL = "https://raw.githubusercontent.com/PromtEngineer/localtuya-homeassistant/main/Kokoro/voice_data.json"
VOICES_BACKUP_URL = "https://raw.githubusercontent.com/PromtEngineer/kokoro-onnx-tts/main/voice_data.json"
KOKORO_MODEL_URL = "https://github.com/PromtEngineer/kokoro-onnx-tts/raw/main/kokoro-v1.0.onnx"
KOKORO_MODEL_BACKUP_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"

def download_file(url, dest_path, backup_url=None):
    """Download a file from a URL to a destination path."""
    try:
        logging.info(f"Downloading {url} to {dest_path}")
        response = requests.get(url, stream=True)
        
        if response.status_code != 200 and backup_url:
            logging.warning(f"Failed to download from primary URL (status {response.status_code}). Trying backup URL...")
            response = requests.get(backup_url, stream=True)
            
        if response.status_code != 200:
            logging.error(f"Failed to download: HTTP {response.status_code}")
            return False
            
        # Create directory if it doesn't exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download the file
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
                    
        logging.info(f"Successfully downloaded {dest_path} ({dest_path.stat().st_size} bytes)")
        return True
    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")
        return False

def process_voice_data(input_path, output_path):
    """Process the voice data JSON file into the format expected by Kokoro-ONNX."""
    try:
        logging.info(f"Processing voice data from {input_path} to {output_path}")
        
        with open(input_path, 'r') as f:
            data = json.load(f)
            
        # Check if the format is as expected (list of voices or dict)
        if isinstance(data, list):
            # Convert list format to dict format expected by Kokoro-ONNX
            voices_dict = {}
            for voice in data:
                if 'id' in voice and 'embedding' in voice:
                    voices_dict[voice['id']] = voice['embedding']
                    
            logging.info(f"Converted {len(voices_dict)} voices from list format")
            
            with open(output_path, 'w') as f:
                json.dump(voices_dict, f)
                
        elif isinstance(data, dict):
            # Already in the expected format, just save it
            with open(output_path, 'w') as f:
                json.dump(data, f)
                
            logging.info(f"Saved {len(data)} voices from dict format")
        else:
            logging.error(f"Unexpected voice data format: {type(data)}")
            return False
            
        logging.info(f"Successfully processed voice data to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error processing voice data: {e}")
        return False

def validate_voice_file(path):
    """Validate that the voices file is in the correct format."""
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            
        if not isinstance(data, dict):
            logging.warning(f"Voice file is not a dictionary: {type(data)}")
            return False
            
        # Check if there are any voices
        if len(data) == 0:
            logging.warning("Voice file contains no voices")
            return False
            
        # Check a sample voice to ensure it's the right format
        sample_voice_id = next(iter(data.keys()))
        sample_voice = data[sample_voice_id]
        
        # Voice embeddings should be lists (for JSON) or arrays (for numpy)
        if not isinstance(sample_voice, list):
            logging.warning(f"Voice embedding is not a list: {type(sample_voice)}")
            return False
            
        # Typically embeddings are 512-dimensional
        if len(sample_voice) < 100:
            logging.warning(f"Voice embedding dimension seems too small: {len(sample_voice)}")
            
        logging.info(f"Voice file validated: {len(data)} voices, embedding dim={len(sample_voice)}")
        return True
    except Exception as e:
        logging.error(f"Error validating voice file: {e}")
        return False

def main():
    """Main function."""
    print("🔧 Downloading official Kokoro-ONNX voice models...")
    
    # Setup paths
    tts_dir = Path("tts")
    tts_dir.mkdir(exist_ok=True)
    
    temp_voices_path = tts_dir / "voice_data.json"
    voices_path = tts_dir / "voices.json"
    model_path = tts_dir / "kokoro-v1.0.onnx"
    
    # Download voices JSON
    if not download_file(VOICES_JSON_URL, temp_voices_path, VOICES_BACKUP_URL):
        print("❌ Failed to download voices file")
        return False
    
    # Process voices JSON
    if not process_voice_data(temp_voices_path, voices_path):
        print("❌ Failed to process voices file")
        return False
    
    # Validate voices file
    if not validate_voice_file(voices_path):
        print("❌ Failed to validate voices file")
        return False
    
    # Download model file if it doesn't exist or is too small
    if not model_path.exists() or model_path.stat().st_size < 10000000:  # 10MB minimum
        if not download_file(KOKORO_MODEL_URL, model_path, KOKORO_MODEL_BACKUP_URL):
            print("❌ Failed to download model file")
            return False
    else:
        print(f"✅ Model file already exists: {model_path} ({model_path.stat().st_size} bytes)")
    
    # List available voices
    try:
        with open(voices_path, 'r') as f:
            voices_data = json.load(f)
            
        print(f"\n✅ Successfully downloaded {len(voices_data)} voices!")
        print("Available voices include:")
        
        # Print a sample of voices (first 10)
        voice_list = list(voices_data.keys())
        for i, voice_id in enumerate(voice_list[:10], 1):
            print(f"{i}. {voice_id}")
            
        if len(voice_list) > 10:
            print(f"...and {len(voice_list) - 10} more")
            
        # Suggest a voice to use
        recommended_voice = "am_michael"
        if recommended_voice in voices_data:
            print(f"\nRecommended voice: {recommended_voice}")
        else:
            recommended_voice = voice_list[0]
            print(f"\nRecommended voice: {recommended_voice}")
            
        print("\nTo use a specific voice, set in your .env file:")
        print(f"TTS_VOICE={recommended_voice}")
        
        return True
    except Exception as e:
        print(f"❌ Error listing voices: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
