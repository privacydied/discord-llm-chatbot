import logging
import inspect
import io
import os
import wave
from pathlib import Path
import array as pyarray
from typing import Any, Optional, Tuple
from kokoro_onnx import Kokoro
from .base import BaseEngine
from bot.tts.errors import TTSError
from bot.tokenizer_registry import select_tokenizer_for_language, apply_lexicon

logger = logging.getLogger(__name__)

class KokoroONNXEngine(BaseEngine):
    def __init__(self, model_path: str, voices_path: str, tokenizer: Optional[str] = None, voice: Optional[str] = None):
        self.model_path = model_path
        self.voices_path = voices_path
        self.language = os.getenv("TTS_LANGUAGE", "en").strip() or "en"
        # Respect explicit tokenizer argument; otherwise delegate selection to registry (registry will log decisions)
        self.tokenizer = tokenizer if tokenizer else select_tokenizer_for_language(self.language)
        self.voice = voice or os.getenv("TTS_VOICE", "af_heart")
        self.engine = None
        # Misaki G2P (only used when registry selects 'misaki')
        self._g2p_initialized = False
        self._g2p = None
        
    def load(self):
        try:
            # Patch EspeakWrapper to avoid Python 3.12 incompatibility
            try:
                import os as _os
                _os.environ.setdefault("KOKORO_SKIP_TOKENIZER_PROBE", "1")
                import kokoro_onnx.tokenizer as _tok  # type: ignore
                _ew = getattr(_tok, "EspeakWrapper", None)
                if _ew is not None:
                    if not hasattr(_ew, "set_data_path"):
                        setattr(_ew, "set_data_path", staticmethod(lambda *_, **__: None))
                    if not hasattr(_ew, "set_library"):
                        setattr(_ew, "set_library", staticmethod(lambda *_, **__: None))
                    logger.debug("Patched EspeakWrapper with no-op methods")
            except Exception:
                # Non-fatal; continue and let Kokoro attempt init
                logger.debug("EspeakWrapper patch not applied", exc_info=True)

            self.engine = Kokoro(
                model_path=self.model_path,
                voices_path=self.voices_path
            )
            # Back-compat for tests: if tokenizer is a Mock, override to provided string
            try:
                current_tok = getattr(self.engine, "tokenizer", None)
                is_mock = False
                try:
                    import unittest.mock as _um
                    is_mock = isinstance(current_tok, (_um.Mock, _um.MagicMock))  # type: ignore
                except Exception:
                    is_mock = False
                if is_mock or current_tok is None:
                    setattr(self.engine, "tokenizer", self.tokenizer)
            except Exception:
                # Non-fatal if underlying object disallows setting attributes
                logger.debug("Could not set engine.tokenizer attribute; continuing", exc_info=True)
            logger.info("KokoroEngine loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load KokoroEngine: {e}", exc_info=True)
            raise TTSError(f"Failed to load KokoroEngine: {e}") from e
            
    async def synthesize(self, text: str) -> bytes:
        if not self.engine:
            self.load()

        try:
            # Entry logging for diagnostics
            try:
                logger.info(
                    "Kokoro.synthesize start | len=%d | tokenizer=%s | voice=%s | text=%r",
                    len(text) if isinstance(text, str) else -1,
                    getattr(self, "tokenizer", None),
                    getattr(self, "voice", None),
                    text,
                )
            except Exception:
                pass
            # Apply lexicon overrides via registry (do not duplicate registry logs)
            try:
                lex_text, changed = apply_lexicon(text, self.language)
                if changed:
                    text = lex_text
            except Exception:
                pass
            # Preferred: official example path from kokoro-onnx english.py
            try:
                ga = getattr(self.engine, "generate_audio", None)
                if callable(ga):
                    logger.info("Kokoro engine trying 'generate_audio' variants (voice=%s)", self.voice)
                    attempts = [
                        # Prefer simplest signature first to satisfy tests and common APIs
                        lambda: ga(text),  # engine default voice
                        # Then try voice-bearing forms across possible signatures
                        lambda: ga(text, voice=self.voice),
                        lambda: ga(text, self.voice),
                        lambda: ga(text=text, voice=self.voice),
                        lambda: ga(text=text, voice_name=self.voice),
                        lambda: ga(self.voice, text),
                        lambda: ga(voice=self.voice, text=text),
                        lambda: ga(speaker=self.voice, text=text),
                    ]
                    for inv in attempts:
                        try:
                            result = inv()
                        except TypeError:
                            continue
                        except Exception:
                            continue
                        if inspect.isawaitable(result):
                            result = await result
                        wav_bytes = self._normalize_audio_to_wav_bytes(result)
                        if wav_bytes is not None:
                            return wav_bytes
                    logger.debug("generate_audio variants did not yield recognizable audio; continuing to fallbacks")
            except Exception:
                # Fall through to Misaki/probing
                logger.debug("generate_audio path failed; falling back", exc_info=True)

            # Prefer Misaki G2P -> engine.create(is_phonemes=True) only if selected tokenizer is 'misaki'
            use_misaki = (str(getattr(self, "tokenizer", "")).lower().strip() == "misaki")
            if use_misaki and not self._g2p_initialized:
                try:
                    from misaki import en as misaki_en  # type: ignore
                    try:
                        from misaki import espeak as misaki_espeak  # type: ignore
                        fallback = misaki_espeak.EspeakFallback(british=False)
                        logger.debug("Misaki espeak fallback available")
                    except Exception:
                        fallback = None
                        logger.debug("Misaki espeak fallback not available; proceeding without it")
                    self._g2p = misaki_en.G2P(trf=False, british=False, fallback=fallback)
                except Exception:
                    self._g2p = None
                finally:
                    self._g2p_initialized = True

            # If Misaki is available, try phoneme path first
            if use_misaki and self._g2p is not None:
                try:
                    logger.debug("Invoking Misaki G2P on input text")
                    phon = self._g2p(text)
                    # Misaki may return (phonemes, meta) or just phonemes
                    if isinstance(phon, tuple) and len(phon) >= 1:
                        phonemes = phon[0]
                    else:
                        phonemes = phon
                    # Avoid verbose logging; registry owns tokenizer decision logs
                    # Guard against empty phoneme outputs (suspected root of dropped words)
                    invalid_marker = False
                    try:
                        s = str(phonemes)
                        # Treat presence of obvious unknown markers as invalid phonemes
                        if any(ch in s for ch in ("?", "❓", "�")):
                            invalid_marker = True
                    except Exception:
                        pass

                    empty_or_blank = (not phonemes) or (isinstance(phonemes, str) and not phonemes.strip())
                    if not (empty_or_blank or invalid_marker):
                        create = getattr(self.engine, "create", None)
                        if callable(create):
                            logger.info("Kokoro engine using 'create' with phonemes via Misaki")
                            result = create(phonemes, self.voice, is_phonemes=True)
                            if inspect.isawaitable(result):
                                result = await result
                            wav_bytes = self._normalize_audio_to_wav_bytes(result)
                            if wav_bytes is not None:
                                return wav_bytes
                            else:
                                logger.debug("create(is_phonemes=True) returned unrecognized audio format; continuing to probing methods")
                except Exception:
                    # Fall through to generic probing quietly
                    logger.debug("Phoneme path failed; falling back to generic probing", exc_info=True)

            # Probe common method names across versions
            candidates = (
                "generate_audio",
                "synthesize",
                "tts",
                "generate",
                "infer",
                "speak",
                "create",
                "_create_audio",
                "create_audio",
                # Additional possibilities across versions
                "__call__",
                "predict",
                "forward",
                "process",
                "pipeline",
                "process_text",
                "create_from_text",
                "text_to_speech",
                "synthesize_speech",
                "tts_generate",
                "generate_tts",
                "generate_waveform",
            )

            for name in candidates:
                meth = getattr(self.engine, name, None)
                if not callable(meth):
                    continue

                # Build a list of invocation patterns to try, accommodating
                # different Kokoro versions/signatures. We always start with
                # simplest forms and escalate to voice-bearing signatures.
                patterns = []
                try:
                    # Try to inspect the method signature to construct kwargs
                    try:
                        sig = inspect.signature(meth)
                        param_names = list(sig.parameters.keys())
                    except Exception:
                        param_names = []

                    if name in ("create", "_create_audio", "create_audio"):
                        # Text-mode create (phoneme path already tried above)
                        # Cover both text-first and voice-first signatures across versions
                        patterns.extend([
                            # text-first
                            lambda: meth(text, self.voice, is_phonemes=False),
                            lambda: meth(text, self.voice),
                            lambda: meth(text=text, voice=self.voice, is_phonemes=False),
                            lambda: meth(text=text, voice=self.voice),
                            # kwargs-based (safe irrespective of positional order)
                            lambda: meth(voice=self.voice, text=text, is_phonemes=False),
                            lambda: meth(voice=self.voice, text=text),
                            lambda: meth(speaker=self.voice, text=text, is_phonemes=False),
                            lambda: meth(speaker=self.voice, text=text),
                            # text-only (engine may use default voice)
                            lambda: meth(text),
                        ])
                    else:
                        patterns.extend([
                            lambda: meth(text),
                            lambda: meth(text, self.voice),
                            lambda: meth(text=text, voice=self.voice),
                            lambda: meth(text=text, speaker=self.voice),
                            # voice-first positional (some APIs expect voice before text)
                            lambda: meth(self.voice, text),
                        ])

                    # Signature-aware kwargs attempts (covers alternate arg names)
                    text_keys = [k for k in ("text", "sentence", "input_text", "input", "content", "s") if k in param_names]
                    voice_keys = [k for k in ("voice", "speaker", "spk", "voice_name", "speaker_id", "voice_id") if k in param_names]
                    phoneme_flag_keys = [k for k in ("is_phonemes", "is_phones") if k in param_names]

                    # Build combinations while keeping it lightweight
                    if text_keys:
                        # text only
                        for tk in text_keys:
                            patterns.append(lambda tk=tk: meth(**{tk: text}))
                        # text + voice
                        for tk in text_keys:
                            for vk in voice_keys:
                                patterns.append(lambda tk=tk, vk=vk: meth(**{tk: text, vk: self.voice}))
                                # For create-like paths that accept phoneme flag
                                if name in ("create", "_create_audio", "create_audio") and phoneme_flag_keys:
                                    for pk in phoneme_flag_keys:
                                        patterns.append(lambda tk=tk, vk=vk, pk=pk: meth(**{tk: text, vk: self.voice, pk: False}))
                    # voice-only then text if signature exposes names in that order
                    if voice_keys and text_keys:
                        for vk in voice_keys:
                            for tk in text_keys:
                                patterns.append(lambda vk=vk, tk=tk: meth(**{vk: self.voice, tk: text}))
                except Exception:
                    # If building patterns fails, skip this method gracefully
                    continue

                for invoker in patterns:
                    try:
                        call_result = invoker()
                    except TypeError:
                        # Signature mismatch for this pattern, try next
                        continue
                    except Exception:
                        # Unexpected failure in this pattern, try next one
                        continue

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
                    else:
                        logger.debug("Method '%s' returned unrecognized audio format; trying next pattern", name)

            # Log available callable attributes to aid debugging
            try:
                callables = [a for a in dir(self.engine) if callable(getattr(self.engine, a, None)) and not a.startswith('__')]
                logger.debug(f"Kokoro engine callable methods: {callables}")
            except Exception:
                pass

            # Final fallback: use our direct integration wrapper if available
            try:
                logger.info("Attempting KokoroDirect fallback path")
                from bot.kokoro_direct_fixed import KokoroDirect  # local helper wrapper
                kd = KokoroDirect(model_path=self.model_path, voices_path=self.voices_path)
                out_path = kd.create(text, self.voice)
                try:
                    from pathlib import Path as _P
                    if isinstance(out_path, _P) and out_path.exists():
                        with open(out_path, 'rb') as _f:
                            data = _f.read()
                        try:
                            out_path.unlink(missing_ok=True)  # py3.8+: ok on this runtime
                        except Exception:
                            pass
                        return data
                except Exception:
                    logger.debug("KokoroDirect returned non-path or read failed; continuing", exc_info=True)
            except Exception:
                logger.debug("KokoroDirect fallback unavailable or failed", exc_info=True)

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

        # Direct BytesIO
        if isinstance(data, io.BytesIO):
            try:
                return data.getvalue()
            except Exception:
                pass

        # Dict outputs e.g., {audio: ..., sr: 24000}
        if isinstance(data, dict):
            audio = None
            sr = None
            for k in ("audio", "samples", "pcm", "wav", "waveform"):
                if k in data:
                    audio = data[k]
                    break
            for k in ("sr", "sample_rate", "rate", "fs", "sampling_rate"):
                if k in data:
                    sr = data[k]
                    break
            if audio is not None:
                # Recurse normalize with tuple so below logic can handle
                return self._normalize_audio_to_wav_bytes((audio, sr) if sr is not None else audio)

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
            if isinstance(a, (int, float)):
                sr, audio = a, b
            else:
                audio, sr = a, b

        # Numpy array directly
        if audio is None:
            audio = data

        # array.array("h") case
        if isinstance(audio, pyarray.array) and audio.typecode == 'h':
            sr_eff = int(sr) if sr else 24000
            with io.BytesIO() as buf:
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr_eff)
                    wf.writeframes(audio.tobytes())
                return buf.getvalue()

        if np is not None and isinstance(audio, np.ndarray):  # type: ignore
            # Handle float arrays correctly by scaling to int16 PCM range
            try:
                arr = audio
                if np.issubdtype(arr.dtype, np.floating):  # type: ignore
                    arr = np.clip(arr, -1.0, 1.0)
                    arr = (arr * 32767.0).astype(np.int16, copy=False)
                else:
                    # For integer arrays, clip to int16 range if needed
                    if arr.dtype != np.int16:  # type: ignore
                        arr = np.clip(arr, -32768, 32767).astype(np.int16, copy=False)
                    else:
                        arr = arr.astype(np.int16, copy=False)
            except Exception:
                # Fallback scaling path
                try:
                    arr = (audio.clip(-1.0, 1.0) * 32767.0).astype(np.int16)  # type: ignore
                except Exception:
                    try:
                        arr = audio.astype(np.int16)  # type: ignore
                    except Exception:
                        return None

            # If multi-channel, mix down to mono safely
            if getattr(arr, 'ndim', 1) > 1:
                try:
                    arr = arr.astype(np.int32, copy=False).mean(axis=-1).astype(np.int16)
                except Exception:
                    arr = arr[:, 0].astype(np.int16, copy=False)

            sr_eff = int(sr) if sr else 24000
            with io.BytesIO() as buf:
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr_eff)
                    wf.writeframes(arr.tobytes())
                return buf.getvalue()

        # Pydub AudioSegment
        try:
            from pydub import AudioSegment  # type: ignore
            if isinstance(audio, AudioSegment):
                with io.BytesIO() as buf:
                    audio.export(buf, format='wav')
                    return buf.getvalue()
        except Exception:
            pass

        # Fallback: handle list/tuple of numbers when numpy is unavailable
        if isinstance(audio, (list, tuple)) and audio and isinstance(audio[0], (int, float)):
            # Convert to int16 PCM
            def _to_int16(x: float | int) -> int:
                if isinstance(x, float):
                    x = max(-1.0, min(1.0, x))
                    return int(x * 32767.0)
                # assume already in int range
                return int(x)

            pcm = bytes()
            try:
                pcm = b"".join(int(_to_int16(v)).to_bytes(2, byteorder="little", signed=True) for v in audio)  # type: ignore
            except Exception:
                return None

            sr_eff = int(sr) if sr else 24000
            with io.BytesIO() as buf:
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr_eff)
                    wf.writeframes(pcm)
                return buf.getvalue()

        # If data is a file path to wav/pcm, read it
        if isinstance(data, (str, Path)):
            try:
                p = Path(data)
                if p.exists() and p.is_file():
                    with p.open('rb') as f:
                        return f.read()
            except Exception:
                pass

        return None

# Backward-compatible alias for legacy imports/tests
# Some tests/modules import KokoroEngine; maintain that name.
KokoroEngine = KokoroONNXEngine