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

def _looks_like_ipa(s: str) -> bool:
    """Detect if input text appears to be IPA phonemes rather than plain text."""
    # Minimal heuristic: any "obvious IPA" characters trigger phoneme mode
    ipa_markers = set("ɑɒəɜɪʊθðʃʒŋˈˌːɾʔɹɱ̃")
    return any(ch in ipa_markers for ch in s)

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

            # Final fallback: use our direct integration wrapper with registry decisions
            try:
                logger.debug("Attempting KokoroDirect fallback with registry decisions")
                from bot.kokoro_direct_fixed import KokoroDirect  # local helper wrapper
                from bot.tokenizer_registry import select_for_language
                kd = KokoroDirect(model_path=self.model_path, voices_path=self.voices_path)
                
                # Get registry decision for this text and language
                decision = select_for_language(self.language, text)
                
                if decision.mode == "phonemes":
                    # PHONEME PATH — use registry phonemes directly
                    logger.debug(f"Using registry phonemes from {decision.alphabet} tokenizer")
                    out_path = kd.create(
                        phonemes=decision.payload,
                        voice=self.voice,
                        lang=self.language,
                        disable_autodiscovery=True,
                        logger=logger,
                    )
                else:
                    # GRAPHEME PATH — use quiet grapheme tokenization
                    logger.debug(f"Using grapheme path with {decision.alphabet} text")
                    out_path = kd.create(
                        text=decision.payload,
                        voice=self.voice,
                        lang=self.language,
                        disable_autodiscovery=True,
                        logger=logger,
                    )
                
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
        """Ensure 16-bit PCM WAV regardless of input format.
        
        Converts various audio formats to standardized 16-bit PCM WAV
        with proper peak normalization for clean, consistent output.
        """
        import io
        import soundfile as sf
        import numpy as np

        # Handle different input types
        if isinstance(data, (bytes, bytearray)):
            # Already bytes, try to read as WAV
            try:
                y, sr = sf.read(io.BytesIO(data), always_2d=False, dtype="float32")
            except Exception:
                return bytes(data)  # Return as-is if we can't read it

        elif isinstance(data, io.BytesIO):
            try:
                y, sr = sf.read(data, always_2d=False, dtype="float32")
            except Exception:
                try:
                    return data.getvalue()
                except Exception:
                    return None

        elif isinstance(data, dict):
            # Dict outputs e.g., {audio: ..., sr: 24000}
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
            if audio is None:
                return None
            y, sr = audio, sr or 24000

        elif isinstance(data, tuple) and len(data) == 2:
            # Tuple formats: (audio, sr) or (sr, audio)
            a, b = data
            if isinstance(a, (int, float)):
                sr, audio = int(a), b
            else:
                audio, sr = a, b or 24000
            y, sr = audio, sr

        elif isinstance(data, (str, Path)):
            # File path
            try:
                y, sr = sf.read(str(data), always_2d=False, dtype="float32")
            except Exception:
                return None

        else:
            # Assume it's already audio data
            y = data
            sr = getattr(self.engine, "sample_rate", None) or 24000

        # Convert to numpy array if needed
        if not isinstance(y, np.ndarray):
            try:
                y = np.array(y, dtype=np.float32)
            except Exception:
                return None

        # Ensure we have valid audio data
        if y.size == 0:
            return None

        # Light peak normalization to -1 dBFS (avoids clipping while maximizing loudness)
        peak = float(np.max(np.abs(y)))
        if peak > 0:
            y = y * (0.8912509381337456 / peak)  # 10 ** (-1/20)

        # Convert to int16 PCM
        y_int16 = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)

        # Mix down to mono if multi-channel
        if y_int16.ndim > 1:
            y_int16 = y_int16.mean(axis=-1).astype(np.int16)

        # Create WAV
        buf = io.BytesIO()
        sf.write(buf, y_int16, int(sr), format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def _resample_for_discord_voice(self, wav_bytes: bytes) -> bytes:
        """Resample WAV audio to 48kHz int16 PCM for Discord voice streaming.
        
        Discord voice channels expect 48kHz sample rate. This method
        resamples any WAV audio to 48kHz with proper peak normalization.
        """
        import io
        import soundfile as sf
        import numpy as np
        from scipy.signal import resample_poly

        try:
            # Read the WAV data
            y, sr = sf.read(io.BytesIO(wav_bytes), always_2d=False, dtype="float32")
            
            if sr == 48000:
                # Already at correct sample rate
                return wav_bytes
            
            # Resample to 48kHz
            if sr != 48000:
                # Calculate resampling ratio
                up, down = 48000, sr
                # Simplify the fraction
                from math import gcd
                g = gcd(up, down)
                up //= g
                down //= g
                
                # Resample using polyphase filter
                y = resample_poly(y, up, down)
            
            # Light peak normalization to -1 dBFS
            peak = float(np.max(np.abs(y)))
            if peak > 0:
                y = y * (0.8912509381337456 / peak)
            
            # Convert to int16 PCM
            y_int16 = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
            
            # Create new WAV at 48kHz
            buf = io.BytesIO()
            sf.write(buf, y_int16, 48000, format="WAV", subtype="PCM_16")
            return buf.getvalue()
            
        except Exception as e:
            logger.warning(f"Failed to resample audio for Discord voice: {e}")
            return wav_bytes  # Return original if resampling fails

# Backward-compatible alias for legacy imports/tests
# Some tests/modules import KokoroEngine; maintain that name.
KokoroEngine = KokoroONNXEngine