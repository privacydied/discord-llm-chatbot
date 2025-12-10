from __future__ import annotations

import base64
import contextlib
import subprocess
import tempfile
import wave
from pathlib import Path


def _decode_ogg_to_pcm(ogg_path: Path) -> tuple[bytes, int, int, int]:
    """Decode OGG/Opus to raw PCM using ffmpeg.
    
    Returns (raw_pcm_bytes, n_channels, sample_width, n_frames).
    [REH]
    """
    # Decode to 16-bit signed LE mono WAV in memory
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(ogg_path),
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", "48000",
        "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:200]}")
        raw = result.stdout
        # 16-bit mono at 48kHz
        n_frames = len(raw) // 2
        return raw, 1, 2, n_frames
    except Exception:
        raise


def compute_waveform_b64(audio_path: str | Path, bins: int = 256) -> str:
    """
    Compute a compact waveform byte array (max 256 bytes) and return base64-encoded string.

    Implementation details [PA][IV]:
    - Reads PCM from a WAV or OGG file (mono or stereo). If stereo, downmix by averaging channels.
    - For OGG/Opus, decodes via ffmpeg first.
    - Normalizes samples to [0, 255] representing amplitude; packs into bytes.
    - Returns base64-encoded bytes as required by Discord voice message "waveform" field.
    """
    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    # Detect OGG/Opus by extension or magic bytes [REH]
    is_ogg = p.suffix.lower() in (".ogg", ".opus")
    if not is_ogg:
        try:
            with open(p, "rb") as f:
                magic = f.read(4)
                is_ogg = magic == b"OggS"
        except Exception:
            pass

    if is_ogg:
        # Decode OGG to PCM via ffmpeg
        try:
            raw, n_channels, sample_width, n_frames = _decode_ogg_to_pcm(p)
        except Exception:
            # Fallback to flat waveform if decoding fails
            return base64.b64encode(bytes([128] * min(bins, 256))).decode("ascii")
    else:
        # WAV path
        with contextlib.closing(wave.open(str(p), "rb")) as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            wf.getframerate()

            if sample_width not in (1, 2, 3, 4):
                # Unsupported width; bail to flat waveform [REH]
                return base64.b64encode(bytes([0] * min(bins, 256))).decode("ascii")

            # Read raw frames
            raw = wf.readframes(n_frames)

    # Convert raw PCM to list of ints per sample (mono)
    # We avoid numpy to reduce deps.
    def _to_samples_le(buf: bytes, width: int) -> list[int]:
        out: list[int] = []
        if width == 1:
            # unsigned 8-bit
            out = [b - 128 for b in buf]
        else:
            # signed little-endian
            step = width
            for i in range(0, len(buf), step):
                chunk = buf[i : i + step]
                if len(chunk) < step:
                    break
                # Pad to 4 bytes for int.from_bytes sign handling
                pad = chunk + (b"\x00" * (4 - step))
                val = int.from_bytes(pad, byteorder="little", signed=True)
                # Shift for 24-bit stored in 32-bit val
                if step == 3 and val & 0x800000:
                    val -= 1 << 24
                out.append(val)
        return out

    samples = _to_samples_le(raw, sample_width)

    # Downmix stereo to mono if needed
    if n_channels == 2:
        mono: list[int] = []
        for i in range(0, len(samples), 2):
            try:
                mono.append((samples[i] + samples[i + 1]) // 2)
            except IndexError:
                break
        samples = mono

    if not samples:
        return base64.b64encode(bytes([0] * min(bins, 256))).decode("ascii")

    # Normalize to [-1.0, 1.0]
    max_abs = max(1, max(abs(s) for s in samples))
    norm = [s / max_abs for s in samples]

    # Bin to fixed length
    bins = max(1, min(bins, 256))
    stride = max(1, len(norm) // bins)
    binned: list[int] = []
    for i in range(0, len(norm), stride):
        if len(binned) >= bins:
            break
        window = norm[i : i + stride]
        # take mean of absolute values as amplitude
        amp = sum(abs(x) for x in window) / max(1, len(window))
        # map to 0..255
        binned.append(int(max(0, min(255, round(amp * 255)))))

    # pad if short
    if len(binned) < bins:
        binned.extend([0] * (bins - len(binned)))

    return base64.b64encode(bytes(binned[:bins])).decode("ascii")
