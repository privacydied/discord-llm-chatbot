import sys
import os
import importlib

def run_diagnostics():
    print("--- [DIAGNOSTIC START] ---", flush=True)

    try:
        print(f"[INFO] Python Executable: {sys.executable}", flush=True)
        print(f"[INFO] Python Version: {sys.version}", flush=True)
        print(f"[INFO] Working Directory: {os.getcwd()}", flush=True)
        print(f"[INFO] Sys Path: {sys.path}", flush=True)

        standard_libs = [
            'os', 'sys', 'argparse', 'asyncio', 'hashlib', 'typing', 'pathlib'
        ]

        for lib in standard_libs:
            print(f"[TEST] Importing standard library: {lib}...", flush=True)
            try:
                importlib.import_module(lib)
                print(f"[SUCCESS] Imported {lib}", flush=True)
            except Exception as e:
                print(f"[FATAL] CRITICAL FAILURE importing '{lib}': {e}", flush=True)
                sys.exit(1)

        print("[TEST] All standard libraries imported successfully.", flush=True)

        print("[TEST] Checking for conflicting packages...", flush=True)
        try:
            import pkg_resources
            installed_packages = {pkg.key for pkg in pkg_resources.working_set}
            if 'onnxruntime-gpu' in installed_packages:
                print("[FATAL] Conflicting package 'onnxruntime-gpu' is still installed.", flush=True)
            else:
                print("[SUCCESS] 'onnxruntime-gpu' not found.", flush=True)
        except ImportError:
            print("[WARN] `pkg_resources` not found, cannot check for conflicting packages.", flush=True)
        except Exception as e:
            print(f"[ERROR] Could not check for conflicting packages: {e}", flush=True)

    except Exception as e:
        print(f"[FATAL] An unexpected error occurred during diagnostics: {e}", flush=True)
    finally:
        print("--- [DIAGNOSTIC END] ---", flush=True)

if __name__ == "__main__":
    run_diagnostics()
