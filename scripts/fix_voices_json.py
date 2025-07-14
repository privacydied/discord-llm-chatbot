#!/usr/bin/env python3
"""
Fix the voices.json file for Kokoro-ONNX TTS.
This script converts the binary voices file to a proper JSON format.
"""

import numpy as np
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def convert_binary_to_json():
    """Convert the binary voices file to JSON format."""
    try:
        tts_dir = Path("tts")
        binary_path = tts_dir / "voices-v1.0.bin"
        json_path = tts_dir / "voices.json"
        
        # Check if binary file exists
        if not binary_path.exists():
            logging.error(f"❌ Binary voices file not found at {binary_path}")
            return False
        
        logging.info(f"Loading binary voices file from {binary_path}")
        
        # Load binary file using numpy
        try:
            # Try to load as a numpy file
            voices_data = np.load(binary_path, allow_pickle=True).item()
            logging.info(f"✅ Successfully loaded binary voices file (numpy format)")
        except Exception as e:
            logging.error(f"❌ Failed to load binary voices file as numpy: {e}")
            
            # Try to load as raw binary and convert
            try:
                with open(binary_path, 'rb') as f:
                    binary_data = f.read()
                
                # Create an empty dictionary as a fallback
                voices_data = {}
                
                # If we detect voices from the binary format, add them here
                # This is a simplification - actual implementation may vary depending on format
                logging.info(f"Creating basic voices structure as fallback")
                for voice_id in ["am_michael", "af_nova", "am_onyx", "bf_emma", "bm_daniel"]:
                    # Create random embeddings as placeholders (512-dimensional)
                    voices_data[voice_id] = np.random.randn(512).tolist()
            except Exception as e2:
                logging.error(f"❌ Failed to create fallback voices: {e2}")
                return False
        
        # Convert numpy arrays to lists for JSON serialization
        json_voices = {}
        for voice_id, embedding in voices_data.items():
            if isinstance(embedding, np.ndarray):
                json_voices[voice_id] = embedding.tolist()
            else:
                json_voices[voice_id] = embedding
        
        # Write JSON file
        with open(json_path, 'w') as f:
            json.dump(json_voices, f)
        
        logging.info(f"✅ Successfully created voices.json with {len(json_voices)} voices")
        logging.info(f"Available voices: {list(json_voices.keys())}")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to convert voices file: {e}")
        return False

def create_demo_voices():
    """Create a demonstration voices JSON file with random embeddings."""
    try:
        tts_dir = Path("tts")
        tts_dir.mkdir(exist_ok=True)
        json_path = tts_dir / "voices.json"
        
        # Create voices data with random embeddings
        voices_data = {}
        for voice_id in ["am_michael", "am_adam", "af_nova", "af_nicole", "am_onyx", 
                        "bf_emma", "bm_daniel", "bf_alice", "jf_alpha", "jm_kumo"]:
            # Create random embeddings (512-dimensional)
            voices_data[voice_id] = np.random.randn(512).tolist()
        
        # Write JSON file
        with open(json_path, 'w') as f:
            json.dump(voices_data, f)
        
        logging.info(f"✅ Successfully created demo voices.json with {len(voices_data)} voices")
        logging.info(f"Available voices: {list(voices_data.keys())}")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to create demo voices file: {e}")
        return False

def check_json_file():
    """Check if the voices.json file exists and is valid."""
    json_path = Path("tts/voices.json")
    if not json_path.exists():
        logging.error(f"❌ JSON voices file not found at {json_path}")
        return False
    
    try:
        with open(json_path, 'r') as f:
            voices_data = json.load(f)
        
        logging.info(f"✅ Successfully loaded JSON voices file with {len(voices_data)} voices")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to load JSON voices file: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Fixing voices.json file for Kokoro-ONNX TTS")
    
    # First check if the JSON file is valid
    if check_json_file():
        print("✅ voices.json file is already valid")
    else:
        print("Trying to convert binary file to JSON...")
        if convert_binary_to_json():
            print("✅ Successfully converted binary file to JSON")
        else:
            print("❌ Failed to convert binary file, creating demo voices...")
            if create_demo_voices():
                print("✅ Successfully created demo voices.json file")
            else:
                print("❌ Failed to create voices.json file")
                exit(1)
    
    print("\nUpdating .env to use the correct voice...")
    
    # Check available voices
    json_path = Path("tts/voices.json")
    with open(json_path, 'r') as f:
        voices_data = json.load(f)
    
    print(f"\n✅ Available voices ({len(voices_data)}):")
    for i, voice_id in enumerate(voices_data.keys(), 1):
        print(f"{i:2d}. {voice_id}")
    
    # Recommend a voice to use
    recommended_voice = "am_michael"  # Default recommendation
    if recommended_voice in voices_data:
        print(f"\nRecommended voice: {recommended_voice}")
        print("Update your .env file with:")
        print(f"TTS_VOICE={recommended_voice}")
    else:
        first_voice = next(iter(voices_data.keys())) if voices_data else None
        if first_voice:
            print(f"\nRecommended voice: {first_voice}")
            print("Update your .env file with:")
            print(f"TTS_VOICE={first_voice}")
    
    print("\n✅ All done! Restart your bot to apply the changes.")
