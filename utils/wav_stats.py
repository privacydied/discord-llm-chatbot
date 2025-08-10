#!/usr/bin/env python3
from __future__ import annotations
import sys
import wave
import struct
import math
from pathlib import Path


def wav_stats(path: Path) -> None:
    with wave.open(str(path), 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        duration = n_frames / float(framerate) if framerate else 0.0
        frames = wf.readframes(n_frames)

    if sampwidth != 2:
        print(f"Unsupported sample width: {sampwidth} bytes")
        return

    # Interpret as little-endian signed 16-bit
    count = len(frames) // 2
    samples = struct.unpack('<' + 'h' * count, frames)

    # If stereo, downmix to mono for stats
    if n_channels > 1:
        mono = []
        for i in range(0, len(samples), n_channels):
            chunk = samples[i:i+n_channels]
            mono.append(int(sum(chunk) / len(chunk)))
        samples = mono

    # Peak and RMS
    peak = max(abs(s) for s in samples) if samples else 0
    if samples:
        rms = math.sqrt(sum((s*s) for s in samples) / len(samples))
    else:
        rms = 0.0

    # Normalize to 0..1 for readability
    peak_norm = peak / 32767.0 if 32767 else 0.0
    rms_norm = rms / 32767.0 if 32767 else 0.0

    print(f"channels={n_channels} sr={framerate}Hz width={sampwidth*8}bit duration={duration:.3f}s frames={n_frames}")
    print(f"peak={peak} ({peak_norm:.3f}) rms={rms:.1f} ({rms_norm:.3f})")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: wav_stats.py <wav_path>")
        sys.exit(2)
    wav_path = Path(sys.argv[1])
    wav_stats(wav_path)
