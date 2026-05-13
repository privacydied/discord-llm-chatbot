#!/usr/bin/env python3
"""Measure actual runtime memory by loading the bot heavy components."""
import os
os.environ.setdefault("KOKORO_SKIP_TOKENIZER_PROBE", "1")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import gc
import psutil

def mem(label):
    gc.collect()
    proc = psutil.Process(os.getpid())
    rss = proc.memory_info().rss / 1024 / 1024
    print(f"  {label:40s}: RSS = {rss:.1f} MB")

mem("baseline (Python + stdlib)")

# Load onnxruntime and create Kokoro session
import onnxruntime as ort
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
sess_opts.enable_cpu_mem_arena = False
sess_opts.intra_op_num_threads = 1
sess_opts.inter_op_num_threads = 1
providers = [("CPUExecutionProvider", {})]

print()
print("Creating Kokoro ONNX session (arena=False)...")
model_path = "/volume1/py/discord-llm-chatbot/tts/kokoro-v1.0.onnx"
if os.path.exists(model_path):
    sess = ort.InferenceSession(model_path, sess_opts, providers=providers)
    gc.collect()
    mem("After Kokoro ONNX session (arena=False)")
else:
    print(f"  Model not found: {model_path}")

# Load sentence-transformers
print()
print("Loading sentence-transformers model...")
from sentence_transformers import SentenceTransformer
st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
gc.collect()
mem("After sentence-transformers loaded")

# Load faster-whisper
print()
print("Loading whisper model...")
from faster_whisper import WhisperModel
whisper = WhisperModel("base", device="cpu", compute_type="int8", download_root="/volume1/py/discord-llm-chatbot/stt/cache")
gc.collect()
mem("After whisper model loaded")

mem("FINAL TOTAL")
