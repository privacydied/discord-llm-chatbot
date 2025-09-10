#!/usr/bin/env python3
"""
Kill any hanging RAG rebuild processes.

Usage: python kill_hanging_processes.py
"""

import os
import signal
import subprocess


def kill_hanging_processes():
    """Kill any hanging Python processes related to RAG rebuild."""
    print("🔍 Looking for hanging processes...")

    try:
        # Find processes containing rebuild_rag_collection
        result = subprocess.run(
            ["pgrep", "-f", "rebuild_rag_collection"], capture_output=True, text=True
        )

        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            print(f"Found {len(pids)} hanging processes:")

            for pid in pids:
                try:
                    pid = int(pid.strip())
                    print(f"  Killing PID {pid}...")
                    os.kill(pid, signal.SIGTERM)
                    print(f"  ✅ Killed PID {pid}")
                except (ValueError, ProcessLookupError) as e:
                    print(f"  ⚠️ Could not kill PID {pid}: {e}")
        else:
            print("✅ No hanging processes found")

    except FileNotFoundError:
        print("⚠️ pgrep not available, trying alternative method...")

        # Alternative: use ps and grep
        try:
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True)

            lines = result.stdout.split("\n")
            hanging_pids = []

            for line in lines:
                if "rebuild_rag_collection" in line and "python" in line:
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pid = int(parts[1])
                            hanging_pids.append(pid)
                        except ValueError:
                            continue

            if hanging_pids:
                print(f"Found {len(hanging_pids)} hanging processes:")
                for pid in hanging_pids:
                    try:
                        print(f"  Killing PID {pid}...")
                        os.kill(pid, signal.SIGTERM)
                        print(f"  ✅ Killed PID {pid}")
                    except ProcessLookupError:
                        print(f"  ⚠️ PID {pid} already terminated")
            else:
                print("✅ No hanging processes found")

        except Exception as e:
            print(f"❌ Error finding processes: {e}")


if __name__ == "__main__":
    kill_hanging_processes()
    print("\n🔄 You can now safely run the rebuild script again")
