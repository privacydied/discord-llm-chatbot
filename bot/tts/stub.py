import wave
import math
import struct


def generate_stub_wav(path: str, duration=0.25, freq=440):
    rate = 16000
    frames = int(rate * duration)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        for i in range(frames):
            val = int(32767 * 0.2 * math.sin(2 * math.pi * freq * i / rate))
            w.writeframes(struct.pack("<h", val))
