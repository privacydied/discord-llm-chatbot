#!/usr/bin/env python3
"""Measure memory per module import."""
import gc
import tracemalloc

modules = [
    'discord',
    'numpy',
    'onnxruntime',
    'sentence_transformers',
    'faster_whisper',
    'torch',
]

for name in modules:
    gc.collect()
    tracemalloc.stop()
    tracemalloc.start()
    try:
        __import__(name)
    except Exception as e:
        print(f"  {name}: IMPORT ERROR {e}")
        continue
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  {name}: {current/1024/1024:.1f} MB traced, {peak/1024/1024:.1f} MB peak")
