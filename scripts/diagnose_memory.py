#!/usr/bin/env python3
"""Diagnostic script to find memory leaks in the bot."""
import gc
import sys
import os
import tracemalloc

os.chdir('/volume1/py/discord-llm-chatbot')
tracemalloc.start()

# Load all modules and check memory
import importlib
modules_before = set(sys.modules.keys())

# Import the heavy modules individually and check memory delta
heavy_modules = [
    ('bot.config', 'Config'),
    ('bot.memory.profiles', 'Memory Profiles'),
    ('bot.context', 'Context Store'),
    ('bot.memory.context_manager', 'ContextManager'),
    ('bot.memory.enhanced_context_manager', 'EnhancedContextManager'),
    ('bot.rag.chroma_backend', 'ChromaDB Backend'),
    ('bot.rag.embedding_interface', 'Embedding Interface'),
    ('bot.rag.lazy_load_methods', 'Lazy Load Methods'),
    ('bot.concurrency_manager', 'Concurrency Manager'),
    ('bot.request_coalescing', 'Request Coalescing'),
    ('bot.single_flight_cache', 'Single Flight Cache'),
    ('bot.tts.kokoro_direct', 'TTS Kokoro'),
    ('bot.tts.manager', 'TTS Manager'),
    ('bot.stt', 'STT'),
    ('bot.hear', 'Hear/STT Pipeline'),
    ('bot.see', 'See/Vision'),
    ('bot.router', 'Router'),
    ('bot.memory.service', 'Memory Service'),
    ('bot.memory.curator', 'Memory Curator'),
    ('bot.memory.persistent_store', 'Persistent Store'),
    ('bot.memory.semantic_store', 'Semantic Store'),
    ('bot.core.session_cache', 'Session Cache'),
    ('bot.core.template_cache', 'Template Cache'),
]

print("=" * 70)
print("MEMORY DIAGNOSTIC")
print("=" * 70)

# Track memory per import
for module_path, name in heavy_modules:
    gc.collect()
    before = tracemalloc.get_traced_memory()
    try:
        mod = importlib.import_module(module_path)
        after = tracemalloc.get_traced_memory()
        delta = (after[0] - before[0]) / 1024
        total = after[0] / 1024 / 1024
        print(f"  {name:30s}: +{delta:8.0f} KB  (total: {total:.1f} MB)")
    except Exception as e:
        print(f"  {name:30s}: FAILED - {e}")

gc.collect()
current, peak = tracemalloc.get_traced_memory()
print(f"\nTotal traced: {current / 1024 / 1024:.1f} MB")
print(f"Peak traced:  {peak / 1024 / 1024:.1f} MB")

# Check for largest objects
print("\n--- Top 20 largest allocations ---")
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:20]:
    print(f"  {stat}")

# Check the size of key module-global dicts
print("\n--- Global dict sizes ---")

def dict_memory_size(d, name):
    import sys
    try:
        total = sys.getsizeof(d)
        for k, v in getattr(d, 'items', lambda: [])():
            total += sys.getsizeof(k) + sys.getsizeof(v)
        print(f"  {name:40s}: {total / 1024:.1f} KB ({len(d)} entries)")
    except:
        print(f"  {name:40s}: (error calculating)")

# Check context stores
try:
    from bot import context
    dict_memory_size(context.conversation_store, "context.conversation_store")
    dict_memory_size(context.last_message_time, "context.last_message_time")
except Exception as e:
    print(f"  context module: {e}")

try:
    from bot.memory import profiles
    dict_memory_size(profiles.user_cache, "profiles.user_cache")
    dict_memory_size(profiles.server_cache, "profiles.server_cache")
    dict_memory_size(profiles.user_profiles_last_saved, "profiles.user_profiles_last_saved")
    dict_memory_size(profiles.server_profiles_last_saved, "profiles.server_profiles_last_saved")
except Exception as e:
    print(f"  profiles module: {e}")

try:
    pass
except:
    pass

# Check file sizes for JSON stores
import os
files_to_check = [
    'runtime/context.json',
    'runtime/enhanced_context.json',
    'user_profiles/',
    'server_profiles/',
]
print("\n--- Storage file sizes ---")
for f in files_to_check:
    full_path = os.path.join('/volume1/py/discord-llm-chatbot', f)
    try:
        if os.path.isfile(full_path):
            size = os.path.getsize(full_path)
            print(f"  {f:40s}: {size / 1024:.1f} KB")
        elif os.path.isdir(full_path):
            total = sum(os.path.getsize(os.path.join(full_path, fn)) for fn in os.listdir(full_path))
            count = len(os.listdir(full_path))
            print(f"  {f:40s}: {total / 1024:.1f} KB ({count} files)")
    except Exception as e:
        print(f"  {f:40s}: {e}")

tracemalloc.stop()
print("\nDiagnostic complete.")
