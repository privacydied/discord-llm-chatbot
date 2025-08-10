import logging
import inspect
import io
import os
import wave
from typing import Any, Optional, Tuple
from kokoro_onnx import Kokoro
from .base import BaseEngine
from bot.tts.errors import TTSError

logger = logging.getLogger(__name__)

class KokoroONNXEngine(BaseEngine):
    def __init__(self, model_path: str, voices_path: str, tokenizer: str = "espeak", voice: Optional[str] = None):
        self.model_path = model_path
        self.voices_path = voices_path
        self.tokenizer = tokenizer
        self.voice = voice or os.getenv("TTS_VOICE", "af_heart")
        self.engine = None
        # Misaki G2P
        self._g2p_initialized = False
        self._g2p = None
        
    def load(self):
        try:
            self.engine = Kokoro(
                model_path=self.model_path,
                voices_path=self.voices_path
            )
            logger.info("KokoroEngine loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load KokoroEngine: {e}", exc_info=True)
            raise TTSError(f"Failed to load KokoroEngine: {e}") from e
            
    async def synthesize(self, text: str) -> bytes:
        if not self.engine:
            self.load()

        try:
            # Prefer Misaki G2P -> engine.create(is_phonemes=True)
            if not self._g2p_initialized:
                try:
                    from misaki import en as misaki_en  # type: ignore
                    try:
                        from misaki import espeak as misaki_espeak  # type: ignore
                        fallback = misaki_espeak.EspeakFallback(british=False)
                        logger.info("Misaki espeak fallback available")
                    except Exception:
                        fallback = None
                        logger.info("Misaki espeak fallback not available; proceeding without it")
                    self._g2p = misaki_en.G2P(trf=False, british=False, fallback=fallback)
                except Exception:
                    self._g2p = None
                    logger.warning("Misaki G2P unavailable; will use Kokoro text methods", exc_info=True)
                finally:
                    self._g2p_initialized = True

            # If Misaki is available, try phoneme path first
            if self._g2p is not None:
                try:
                    phon = self._g2p(text)
                    # Misaki may return (phonemes, meta) or just phonemes
                    if isinstance(phon, tuple) and len(phon) >= 1:
                        phonemes = phon[0]
                    else:
                        phonemes = phon
                    create = getattr(self.engine, "create", None)
                    if callable(create):
                        logger.info("Kokoro engine using 'create' with phonemes via Misaki")
                        result = create(phonemes, self.voice, is_phonemes=True)
                        if inspect.isawaitable(result):
                            result = await result
                        wav_bytes = self._normalize_audio_to_wav_bytes(result)
                        if wav_bytes is not None:
                            return wav_bytes
                except Exception:
                    # Fall through to generic probing
                    logger.warning("Phoneme path failed; falling back to generic probing", exc_info=True)

            # Probe common method names across versions
            candidates = (
                "generate_audio",
                "synthesize",
                "tts",
                "generate",
                "infer",
                "speak",
                "_create_audio",
                "create_audio",
            )

            for name in candidates:
                meth = getattr(self.engine, name, None)
                if not callable(meth):
                    continue
                try:
                    call_result = meth(text)
                    if inspect.isawaitable(call_result):
                        logger.info(f"Kokoro engine using async method '{name}'")
                        result = await call_result
                    else:
                        logger.info(f"Kokoro engine using sync method '{name}'")
                        result = call_result

                    # Normalize output to WAV bytes
                    wav_bytes = self._normalize_audio_to_wav_bytes(result)
                    if wav_bytes is not None:
                        return wav_bytes
                except TypeError:
                    # Signature mismatch, try next
                    continue

            # Log available callable attributes to aid debugging
            try:
                callables = [a for a in dir(self.engine) if callable(getattr(self.engine, a, None)) and not a.startswith('__')]
                logger.debug(f"Kokoro engine callable methods: {callables}")
            except Exception:
                pass
            raise TTSError("No compatible synthesis method found on Kokoro engine")

        except Exception as e:
            logger.error(f"Kokoro synthesis failed: {e}", exc_info=True)
            raise TTSError(f"Synthesis failed: {e}") from e

    def _normalize_audio_to_wav_bytes(self, data: Any) -> bytes | None:
        """Attempt to convert various audio outputs to WAV bytes.
        Supports bytes, bytearray, numpy arrays, and (audio, sr) or (sr, audio) tuples.
        Returns None if the format is unrecognized.
        """
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)

        # Lazy import numpy to avoid hard dependency if not present
        try:
            import numpy as np  # type: ignore
        except Exception:
            np = None  # type: ignore

        sr = getattr(self.engine, "sample_rate", None)
        audio = None

        # Tuple formats: (audio, sr) or (sr, audio)
        if isinstance(data, tuple) and len(data) == 2:
            a, b = data
            # Heuristic: sr is int-like, audio is array-like
            if isinstance(a, (int, float)):
                sr = int(a)
                audio = b
            elif isinstance(b, (int, float)):
                sr = int(b)
                audio = a
            else:
                audio = a

        # Numpy array directly
        if audio is None:
            audio = data

        if np is not None and isinstance(audio, np.ndarray):  # type: ignore
            # Ensure mono int16 PCM
            arr = audio
            if arr.dtype.kind == 'f':
                # Float in [-1,1] -> int16
                arr = (arr.clip(-1.0, 1.0) * 32767.0).astype(np.int16)  # type: ignore
            elif arr.dtype != np.int16:  # type: ignore
                arr = arr.astype(np.int16)  # type: ignore

            if arr.ndim > 1:
                arr = arr[:, 0]

            sr = int(sr) if sr else 24000
            with io.BytesIO() as buf:
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(arr.tobytes())
                return buf.getvalue()

        return None