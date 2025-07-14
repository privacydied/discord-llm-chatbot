#!/usr/bin/env python3
"""
Diagnostic script for troubleshooting Kokoro-ONNX TTS issues.
This script attempts to:
1. Import kokoro_onnx
2. Initialize Kokoro with model files
3. Generate a sample audio file
4. Report any issues encountered
"""

import sys
import json
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_imports():
    """Check if required modules can be imported."""
    try:
        import kokoro_onnx
        logging.info(f"✅ kokoro_onnx imported successfully. Version: {kokoro_onnx.__version__ if hasattr(kokoro_onnx, '__version__') else 'unknown'}")
        
        try:
            from misaki import en, espeak
            logging.info("✅ misaki imported successfully")
        except ImportError as e:
            logging.error(f"❌ Failed to import misaki: {e}")
            return False
            
        try:
            import soundfile as sf
            logging.info("✅ soundfile imported successfully")
        except ImportError as e:
            logging.error(f"❌ Failed to import soundfile: {e}")
            return False
            
        return True
    except ImportError as e:
        logging.error(f"❌ Failed to import kokoro_onnx: {e}")
        logging.info("Checking if package is installed...")
        
        # Check if package is installed
        import subprocess
        result = subprocess.run(["pip", "list"], capture_output=True, text=True)
        if "kokoro-onnx" in result.stdout:
            logging.info("📦 kokoro-onnx is installed, but module import failed")
            logging.info(f"Python path: {sys.path}")
        else:
            logging.error("❌ kokoro-onnx package is not installed")
            
        return False

def check_model_files():
    """Check if required model files exist."""
    model_dir = Path("tts")
    model_file = model_dir / "kokoro-v1.0.onnx"
    voices_file = model_dir / "voices.json"
    
    if not model_dir.exists():
        logging.error(f"❌ Model directory {model_dir} does not exist")
        return False
    
    # Check model file
    if not model_file.exists():
        logging.error(f"❌ Model file {model_file} does not exist")
        return False
    logging.info(f"✅ Found model file: {model_file} ({model_file.stat().st_size} bytes)")
    
    # Check voices file
    if not voices_file.exists():
        logging.error(f"❌ Voices file {voices_file} does not exist")
        return False
    logging.info(f"✅ Found voices file: {voices_file} ({voices_file.stat().st_size} bytes)")
    
    # Validate voices file format
    try:
        with open(voices_file, "r") as f:
            voices_data = json.load(f)
        
        # Check format
        if not isinstance(voices_data, dict):
            logging.warning(f"⚠️ voices.json is not a dictionary. Type: {type(voices_data)}")
            return False
        
        logging.info(f"✅ voices.json contains {len(voices_data)} voices")
        first_voice = list(voices_data.keys())[0] if voices_data else None
        if first_voice:
            logging.info(f"First voice: {first_voice}")
            logging.info(f"First voice embedding type: {type(voices_data[first_voice])}")
        
        return True
    except json.JSONDecodeError as e:
        logging.error(f"❌ voices.json is not valid JSON: {e}")
        return False
    except Exception as e:
        logging.error(f"❌ Error reading voices.json: {e}")
        return False

def initialize_tts():
    """Attempt to initialize Kokoro-ONNX TTS."""
    try:
        from kokoro_onnx import Kokoro
        from misaki import en, espeak
        
        model_path = Path("tts/kokoro-v1.0.onnx")
        voices_path = Path("tts/voices.json")
        
        # Initialize G2P
        fallback = espeak.EspeakFallback(british=False)
        g2p = en.G2P(trf=False, british=False, fallback=fallback)
        logging.info("✅ Misaki G2P initialized")
        
        # Initialize Kokoro
        logging.info(f"Initializing Kokoro with model={model_path}, voices={voices_path}")
        engine = Kokoro(str(model_path), str(voices_path))
        logging.info("✅ Kokoro engine initialized")
        
        # Get available voices
        if hasattr(engine, 'voices'):
            available_voices = list(engine.voices)
            logging.info(f"Available voices: {available_voices}")
            if available_voices:
                # Use the first voice for testing
                test_voice = available_voices[0]
                logging.info(f"Testing with voice: {test_voice}")
                
                # Phonemize text
                text = "This is a test of the Kokoro-ONNX TTS system."
                phonemes, _ = g2p(text)
                logging.info("✅ Text phonemized")
                
                # Generate audio
                samples, sample_rate = engine.create(text, test_voice, phonemes=phonemes)
                logging.info(f"✅ Audio generated: {len(samples)} samples at {sample_rate}Hz")
                
                # Save to file for testing
                import soundfile as sf
                output_path = Path("tts_test_output.wav")
                sf.write(output_path, samples, sample_rate)
                logging.info(f"✅ Audio saved to {output_path}")
                
                return True
            else:
                logging.error("❌ No voices available in engine")
                return False
        else:
            logging.error("❌ Engine has no voices attribute")
            return False
    except Exception as e:
        logging.error(f"❌ Failed to initialize Kokoro-ONNX: {e}", exc_info=True)
        return False

def check_config():
    """Check the .env configuration."""
    env_path = Path(".env")
    if not env_path.exists():
        logging.error("❌ .env file not found")
        return False
    
    try:
        with open(env_path, "r") as f:
            env_content = f.read()
        
        # Check TTS settings
        if "TTS_BACKEND=" not in env_content:
            logging.warning("⚠️ TTS_BACKEND not found in .env")
        else:
            for line in env_content.splitlines():
                if line.startswith("TTS_BACKEND="):
                    backend = line.split("=", 1)[1].strip()
                    logging.info(f"TTS_BACKEND: {backend}")
                    if backend != "kokoro-onnx":
                        logging.warning(f"⚠️ TTS_BACKEND is not set to kokoro-onnx: {backend}")
                
                if line.startswith("TTS_VOICE="):
                    voice = line.split("=", 1)[1].strip()
                    logging.info(f"TTS_VOICE: {voice}")
                    
                if line.startswith("TTS_VOICE_FILE="):
                    voice_file = line.split("=", 1)[1].strip()
                    logging.info(f"TTS_VOICE_FILE: {voice_file}")
                    
                if line.startswith("TTS_MODEL_FILE="):
                    model_file = line.split("=", 1)[1].strip()
                    logging.info(f"TTS_MODEL_FILE: {model_file}")
        
        return True
    except Exception as e:
        logging.error(f"❌ Error reading .env: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Diagnosing Kokoro-ONNX TTS...")
    
    # Check environment
    print("\n--- Environment ---")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    
    # Check configuration
    print("\n--- Configuration ---")
    check_config()
    
    # Check imports
    print("\n--- Import Check ---")
    imports_ok = check_imports()
    if not imports_ok:
        print("\n❌ Import check failed. Please install required packages:")
        print("uv pip install -U kokoro-onnx soundfile \"misaki[en]\"")
        sys.exit(1)
    
    # Check model files
    print("\n--- Model Files Check ---")
    files_ok = check_model_files()
    if not files_ok:
        print("\n❌ Model files check failed.")
        sys.exit(1)
    
    # Initialize TTS
    print("\n--- TTS Initialization ---")
    tts_ok = initialize_tts()
    if not tts_ok:
        print("\n❌ TTS initialization failed.")
        sys.exit(1)
    
    print("\n✅ All checks passed! TTS should be working properly.")
    sys.exit(0)
