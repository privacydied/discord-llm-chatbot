#!/usr/bin/env python
"""
Script to inspect the ONNX model inputs for kokoro-onnx 0.4.9
"""

import os
import sys
import onnxruntime as ort
from pathlib import Path

def main():
    # Get model path from environment or use default
    model_path = os.environ.get('TTS_MODEL_FILE', 'tts/onnx/kokoro-v1.0.onnx')
    print(f"Using model path: {model_path}")
    
    # Check if model file exists
    if not Path(model_path).exists():
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    
    try:
        # Load the ONNX model
        providers = ort.get_available_providers()
        print(f"Available ONNX providers: {providers}")
        
        sess = ort.InferenceSession(model_path, providers=providers)
        
        # Print model inputs
        print("\nONNX model inputs:")
        for input in sess.get_inputs():
            print(f"- Name: {input.name}, Shape: {input.shape}, Type: {input.type}")
        
        # Print model outputs
        print("\nONNX model outputs:")
        for output in sess.get_outputs():
            print(f"- Name: {output.name}, Shape: {output.shape}, Type: {output.type}")
            
    except Exception as e:
        print(f"Error inspecting model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
