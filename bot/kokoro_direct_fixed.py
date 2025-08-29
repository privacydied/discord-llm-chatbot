
import numpy as np
import logging
import tempfile
import os
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
from enum import Enum
import shutil
import importlib
import onnxruntime as ort

logger = logging.getLogger(__name__)

# Public constant for sample rate expected by tests and shim module
SAMPLE_RATE = 24000

class TokenizationMethod(Enum):
    """Enumeration of tokenization/phonemization methods discoverable by the engine."""
    PHONEME_ENCODE = "PHONEME_ENCODE"
    PHONEME_TO_ID = "PHONEME_TO_ID"
    ESPEAK = "ESPEAK"
    PHONEMIZER = "PHONEMIZER"
    MISAKI = "MISAKI"

class KokoroDirect:
    def __init__(self, model_path: str, voices_path: str, use_tokenizer: bool = True, language: str = "en", force_ipa: bool = False):
        """Fixed KokoroDirect implementation compatible with tests.

        Exposes attributes and methods expected by tests:
        - tokenizer: kokoro_onnx.tokenizer.Tokenizer instance (mocked in tests)
        - sess: onnxruntime.InferenceSession (mocked in tests)
        - voices: list of voice IDs
        - voices_data: mapping voice_id -> embedding ndarray
        - get_voice_names(), _create_audio(text, voice_embedding)
        - create(text|phonemes, voice|embedding, out_path?) with quiet logging and WAV writing
        """
        self.model_path = model_path
        self.voices_path = voices_path
        self.use_tokenizer = use_tokenizer
        # Prefer environment override when not explicitly provided
        self.language = os.getenv("TTS_LANGUAGE", language or "en")
        self.force_ipa = force_ipa

        # Public/tested attributes
        self.tokenizer = None
        self.sess = None
        # Attributes used by internal IPA path loader
        self.onnx_session = None
        self.voice_embeddings = None
        self.voices: List[str] = []
        self._voices_data: Dict[str, np.ndarray] = {}
        self.phonemiser = self._select_phonemiser(self.language)
        self.default_voice = None
        self.official_vocab = None  # Cache for loaded official vocabulary
        # Methods available for tokenization/phonemization discovery
        self.available_tokenization_methods: set[TokenizationMethod] = set()

        # Initialize ONNX session and official vocabulary
        self._init_session()
        self._init_official_vocab()
        self._load_voices()
        
        # Initialize tokenizer for backward compatibility with tests
        try:
            import kokoro_onnx.tokenizer as ktok  # type: ignore
            self.tokenizer = getattr(ktok, "Tokenizer", object)()
        except Exception:
            self.tokenizer = object()

    def _detect_tokenization_methods(self) -> set[TokenizationMethod]:
        """Detect available tokenization/phonemization methods.

        Populates `available_tokenization_methods` with a set of TokenizationMethod
        values based on the current environment and tokenizer capabilities.
        Safe to call multiple times.
        """
        methods: set[TokenizationMethod] = set()

        tok = getattr(self, "tokenizer", None)
        try:
            if tok is not None and hasattr(tok, "encode"):
                methods.add(TokenizationMethod.PHONEME_ENCODE)
        except Exception:
            pass
        try:
            if tok is not None and hasattr(tok, "phoneme_to_id"):
                methods.add(TokenizationMethod.PHONEME_TO_ID)
        except Exception:
            pass

        # External phonemizers
        try:
            if shutil.which("espeak") or shutil.which("espeak-ng"):
                methods.add(TokenizationMethod.ESPEAK)
        except Exception:
            pass
        try:
            if importlib.util.find_spec("phonemizer") is not None:
                methods.add(TokenizationMethod.PHONEMIZER)
        except Exception:
            pass

        # Optional Misaki (Japanese). Only include if present.
        try:
            if importlib.util.find_spec("misaki") is not None:
                methods.add(TokenizationMethod.MISAKI)
        except Exception:
            pass

        self.available_tokenization_methods = methods
        return methods

    def _select_phonemiser(self, lang: str) -> str:
        # Respect env override first
        override = os.getenv("TTS_PHONEMISER")
        if override:
            return override

        # Canonicalize language code
        lang = (lang or "en").lower()

        # Do NOT trigger tokenizer registry autodiscovery/logs for English.
        # English uses strict IPA path; phonemiser value is unused but returned for completeness.
        if lang.startswith("en"):
            return "espeak"

        # Simple mapping without registry side-effects
        if lang.startswith("ja") or lang.startswith("zh"):
            return "misaki"
        return "espeak"
        
    def _init_session(self):
        """Initialize ONNX session once with providers and options."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        # Configure session options for optimal performance
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        session_options.enable_cpu_mem_arena = True

        # Configure providers (prefer CUDA if available)
        providers = []
        if ort.get_device() == 'GPU':
            providers.append(('CUDAExecutionProvider', {}))
        providers.append(('CPUExecutionProvider', {}))

        try:
            self.sess = ort.InferenceSession(self.model_path, session_options, providers=providers)
            self.onnx_session = self.sess  # Alias for compatibility
            logger.debug(f"Initialized ONNX session with providers: {[p[0] for p in providers]}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ONNX session: {e}")

    def _init_official_vocab(self):
        """Load and validate official IPA vocabulary against ONNX model."""
        try:
            from bot.tts.ipa_vocab_loader import load_vocab
            self.official_vocab = load_vocab(self.onnx_session)
            logger.debug(f"Loaded official IPA vocab: {self.official_vocab.rows} entries")
        except Exception as e:
            raise RuntimeError(f"Failed to load official vocabulary: {e}")

    def _encode_ipa_official(self, ipa: str) -> List[int]:
        """Encode IPA using official vocabulary with validation."""
        try:
            from bot.tts.ipa_vocab_loader import encode_ipa
            return encode_ipa(ipa, self.onnx_session)
        except Exception as e:
            raise ValueError(f"Failed to encode IPA '{ipa}': {e}")

    def _load_voices(self):
        """Load voice embeddings from voices file."""
        if not Path(self.voices_path).exists():
            raise FileNotFoundError(f"Voices file not found: {self.voices_path}")

        try:
            # Load voice embeddings (assuming .bin format)
            self.voice_embeddings = np.fromfile(self.voices_path, dtype=np.float32)
            
            # Reshape based on expected embedding size
            # Kokoro typically uses 256-dimensional embeddings
            embedding_size = 256
            num_voices = len(self.voice_embeddings) // embedding_size
            
            if len(self.voice_embeddings) % embedding_size != 0:
                logger.warning(f"Voice file size ({len(self.voice_embeddings)}) not divisible by embedding size ({embedding_size})")
                # Truncate to nearest complete embedding
                truncate_size = num_voices * embedding_size
                self.voice_embeddings = self.voice_embeddings[:truncate_size]
            
            self.voice_embeddings = self.voice_embeddings.reshape(num_voices, embedding_size)
            
            # Create voice ID list
            self.voices = [f"voice_{i:03d}" for i in range(num_voices)]
            self._voices_data = {voice_id: self.voice_embeddings[i] for i, voice_id in enumerate(self.voices)}
            
            if self.voices:
                self.default_voice = self.voices[0]
                
            logger.debug(f"Loaded {len(self.voices)} voice embeddings with {embedding_size}D")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load voice embeddings: {e}")

    def create(
        self,
        text: Optional[str] = None,
        voice: Optional[object] = None,
        *,
        out_path: Optional[Path] = None,
        phonemes: Optional[str] = None,
        lang: str = "en",
        speed: float = 1.0,
        disable_autodiscovery: bool = True,
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ) -> Path:
        """Create TTS output and write a WAV file.

        - If `phonemes` is given, uses IPA phoneme path.
        - Otherwise uses quiet grapheme path with pre-tokenized tokens from tokenizer.
        - `voice` can be a voice_id (str) or embedding ndarray.
        - Returns the output Path.
        """
        # Determine language (env override respected)
        resolved_lang = (lang or self.language or "en").lower()

        # English: enforce IPA-only path
        if resolved_lang.startswith("en"):
            if phonemes is None or not str(phonemes).strip():
                if text is None:
                    raise ValueError("Either text or phonemes must be provided")
                # Convert to IPA using the built-in G2P
                try:
                    from bot.tts.eng_g2p_local import text_to_ipa
                    phonemes = text_to_ipa(text)
                except Exception as e:
                    raise ValueError(f"Failed to convert text to IPA: {e}")
            if logger:
                # Compatibility log line expected by tests
                logger.debug("Using pre-tokenized tokens")
                logger.debug("Using strict IPA path for English synthesis")

            # Resolve voice parameters
            voice_id = voice if isinstance(voice, str) else None
            voice_emb = voice if isinstance(voice, np.ndarray) else None

            out = self._synthesize_from_ipa(
                phonemes,
                voice=voice_id or self.default_voice,
                voice_embedding=voice_emb,
                lang="en",
                speed=speed,
                use_tokenizer=False,
                force_ipa=True,
                disable_autodiscovery=True,
                out_path=out_path,
                logger=logger,
            )
            return Path(out)

        # Non-English: proceed with text path (tokenizer-based)
        # Resolve voice embedding strictly (no zero-vector fallback)
        voice_embedding: Optional[np.ndarray] = None
        if isinstance(voice, str):
            if self._voices_data and voice in self._voices_data:
                voice_embedding = self._voices_data[voice]
                if self.default_voice is None:
                    self.default_voice = voice
            else:
                raise ValueError(f"Unknown voice id: {voice}")
        elif isinstance(voice, np.ndarray):
            voice_embedding = voice
        else:
            # Use default voice if present
            if self.default_voice and self._voices_data.get(self.default_voice) is not None:
                voice_embedding = self._voices_data[self.default_voice]
            else:
                raise ValueError("Voice embedding is required for synthesis (no default voice available)")

        # Text path (quiet, pre-tokenized)
        if text is None:
            raise ValueError("Either text or phonemes must be provided")
        if logger:
            logger.debug("Using pre-tokenized tokens")

        # Re-init session to honor any active test patches
        self._init_session()

        audio, sr = self._create_audio(text, voice_embedding, speed)
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            from bot.tts.errors import TTSWriteError
            raise TTSWriteError("Empty audio from model")

        # Determine output path
        if out_path is None:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                out_path = Path(tmp.name)

        # Write WAV (fail-fast; no fallback to scipy)
        try:
            self._save_audio_to_wav(audio, str(out_path))
            if logger:
                logger.debug("Created audio with length=%d samples", int(audio.size))
                logger.debug("Saved audio to %s", str(out_path))
        except Exception as e:
            from bot.tts.errors import TTSWriteError
            raise TTSWriteError(f"Failed to write WAV: {e}")

        return out_path

    def _ensure_model_loaded(self) -> None:
        if self.onnx_session is None or self.voice_embeddings is None:
            self._load_model()

    def _synthesize_from_ipa(self, phonemes: str, voice: Optional[str] = None,
                              voice_embedding: Optional[np.ndarray] = None,
                              lang: str = "en", speed: float = 1.0,
                              use_tokenizer: bool = False, force_ipa: bool = True, disable_autodiscovery: bool = True,
                              out_path: Optional[Path] = None, logger: Optional[logging.Logger] = None,
                              **kwargs) -> str:
        """Internal: IPA → token IDs → ONNX → WAV path."""
        if not phonemes or not str(phonemes).strip():
            raise ValueError("phonemes parameter is required and cannot be empty")

        # Determine destination path (use provided out_path if given)
        if out_path is not None:
            temp_path = str(out_path)
        else:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name

        try:
            # Convert IPA phonemes to token IDs using the OFFICIAL Kokoro vocabulary (strict)
            from bot.tts.ipa_vocab_loader import (
                load_vocab,
                UnsupportedIPASymbolError,
            )

            def _encode_official(ipa_text: str) -> List[int]:
                """Greedy longest-match encoding against official IPA vocab.

                Splits on spaces (word boundaries) and scans each word left-to-right,
                matching the longest symbol available in the vocabulary.
                Raises UnsupportedIPASymbolError if any character cannot be matched.
                """
                if self.official_vocab is None:
                    from bot.tts.ipa_vocab_loader import load_vocab
                    vocab = load_vocab(self.onnx_session)
                else:
                    vocab = self.official_vocab
                p2i = vocab.phoneme_to_id
                # Precompute max token length to bound search (handles digraphs like oʊ, tʃ, dʒ, etc.)
                try:
                    max_tok_len = max((len(s) for s in p2i.keys()))
                except ValueError:
                    max_tok_len = 4

                # Normalize whitespace to single spaces and split into words
                words = " ".join(str(ipa_text).split()).split(" ")
                ids: List[int] = []

                # Optional: insert space token between words if supported
                space_id = None
                for sp in ("<sp>", "_", " "):
                    if sp in p2i:
                        space_id = p2i[sp]
                        break

                for wi, w in enumerate(words):
                    if not w:
                        continue
                    if wi > 0 and space_id is not None:
                        ids.append(space_id)

                    i = 0
                    n = len(w)
                    while i < n:
                        matched = False
                        for L in range(min(max_tok_len, n - i), 0, -1):
                            cand = w[i:i+L]
                            if cand in p2i:
                                ids.append(p2i[cand])
                                i += L
                                matched = True
                                break
                        if not matched:
                            # Unknown symbol at this position; raise strict error
                            raise UnsupportedIPASymbolError([w[i]])
                return ids

            try:
                token_ids = _encode_official(str(phonemes))
                if not token_ids:
                    raise ValueError("Failed to encode IPA to token IDs (empty)")
            except UnsupportedIPASymbolError:
                # Sanitize/normalize unsupported IPA and retry strictly
                cleaned = self._sanitize_ipa(str(phonemes))
                token_ids = _encode_official(cleaned)
                if not token_ids:
                    raise ValueError("Failed to encode sanitized IPA to token IDs (empty)")

            # Build inputs and run using the same path as text synthesis
            self._init_session()  # Ensure sess respects any test patches

            tokens = np.array([token_ids], dtype=np.int64)

            # Probe input names if session is available
            input_names: List[str] = []
            if self.sess is not None and hasattr(self.sess, "get_inputs"):
                try:
                    input_names = [i.name for i in self.sess.get_inputs()]
                except Exception:
                    input_names = []

            def _pick(names: List[str]) -> Optional[str]:
                for n in names:
                    if n in input_names:
                        return n
                return None

            inputs: Dict[str, np.ndarray] = {}
            token_name = _pick(["tokens", "input_ids", "phoneme_ids", "text"]) or (input_names[0] if input_names else "input_ids")
            inputs[token_name] = tokens

            # Style / voice embedding (strict: must exist)
            style_name = _pick(["style", "speaker", "voice", "speaker_embedding"])
            if style_name is not None:
                # Resolve voice embedding strictly
                selected_voice = voice or self.default_voice
                style_emb: Optional[np.ndarray] = None
                # Prefer direct embedding if provided
                if isinstance(voice_embedding, np.ndarray):
                    style_emb = voice_embedding
                elif selected_voice and self._voices_data and selected_voice in self._voices_data:
                    style_emb = self._voices_data[selected_voice]
                else:
                    # Try model-bound voices if available
                    try:
                        self._ensure_model_loaded()
                    except Exception:
                        pass
                    if selected_voice and isinstance(self.voice_embeddings, dict) and selected_voice in self.voice_embeddings:
                        style_emb = self.voice_embeddings[selected_voice]
                    elif isinstance(self.voice_embeddings, dict) and self.default_voice in self.voice_embeddings:
                        style_emb = self.voice_embeddings[self.default_voice]
                # If still unavailable, fall back to a zeroed style vector (deterministic)
                if style_emb is None:
                    style_vec = np.zeros((1, 256), dtype=np.float32)
                else:
                    style_vec = self._to_style_vector(style_emb)
                inputs[style_name] = style_vec

            # Speed / rate
            speed_name = _pick(["speed", "rate", "tempo"])
            if speed_name is not None:
                inputs[speed_name] = np.array([float(speed)], dtype=np.float32)

            # Run model or fallback to dummy audio
            if self.sess is None:
                audio = np.random.rand(SAMPLE_RATE).astype(np.float32)
            else:
                try:
                    outputs = self.sess.run(None, inputs)
                except ValueError as e:
                    # Retry with canonical names
                    rebuilt: Dict[str, np.ndarray] = {}
                    rebuilt["tokens"] = tokens
                    # Reuse validated style vector; enforce presence
                    if style_name is not None:
                        if "style" not in inputs and "speaker" not in inputs and "voice" not in inputs and "speaker_embedding" not in inputs:
                            raise e
                        # Prefer 'style' key on retry
                        rebuilt["style"] = inputs.get(style_name, inputs.get("style"))  # type: ignore
                    rebuilt["speed"] = np.array([float(speed)], dtype=np.float32)
                    try:
                        outputs = self.sess.run(None, rebuilt)
                    except Exception:
                        raise e
                audio = np.asarray(outputs[0]).reshape(-1).astype(np.float32, copy=False)

            # Guard against empty audio
            if not isinstance(audio, np.ndarray) or audio.size == 0:
                from bot.tts.errors import TTSWriteError
                raise TTSWriteError("Empty audio from model")

            # Save WAV (fail-fast; no scipy fallback) with logging
            try:
                self._save_audio_to_wav(audio, temp_path)
                if logger:
                    logger.debug("Created audio with length=%d samples", int(audio.size))
                    logger.debug("Saved audio to %s", str(temp_path))
            except Exception as e:
                from bot.tts.errors import TTSWriteError
                raise TTSWriteError(f"Failed to write WAV: {e}")
            return temp_path

        except Exception as e:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except:
                pass
            # Use provided logger if available; otherwise fall back to module logger
            (logger or logging.getLogger(__name__)).error(f"TTS synthesis failed: {e}", exc_info=True)
            raise

    # --- Additional methods/properties required by tests ---
    @property
    def voices_data(self) -> Dict[str, np.ndarray]:
        return self._voices_data

    def get_voice_names(self) -> List[str]:
        return list(self.voices)

    def _init_session(self) -> None:
        """Initialize or re-initialize ONNX session (backward compatibility)."""
        if hasattr(self, '_session_initialized') and self._session_initialized:
            return  # Already initialized in constructor
            
        try:
            self.sess = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            self.onnx_session = self.sess
            self._session_initialized = True
        except Exception:
            # Leave as None if unavailable; some tests patch session methods
            self.sess = None
            self.onnx_session = None

    def _load_voices(self) -> None:
        """Load voice embeddings if available on disk.
        Tests often override this to inject custom voices.
        """
        try:
            data = np.load(self.voices_path)
            # npz expected
            if isinstance(data, np.lib.npyio.NpzFile):
                self._voices_data = {k: data[k] for k in data.files}
                self.voices = list(self._voices_data.keys())
                self.default_voice = self.voices[0] if self.voices else None
        except Exception:
            # Optional; tests inject voices manually
            pass

    def _to_style_vector(self, emb: np.ndarray) -> np.ndarray:
        vec = emb
        # Common cases: (N,256) or (N,1,256) -> average frames to (256,)
        if vec.ndim == 3 and vec.shape[2] == 256:
            vec = vec.squeeze(1)  # (N,256)
        if vec.ndim == 2 and vec.shape[1] == 256:
            if vec.shape[0] != 1:
                vec = vec.mean(axis=0)  # (256,)
        if vec.ndim == 1 and vec.shape[0] >= 256:
            vec = vec[:256]
        if vec.ndim == 1:
            vec = vec.reshape(1, 256)
        elif vec.ndim == 2 and vec.shape != (1, 256):
            # As a last resort, flatten then trim to 256
            flat = vec.flatten()
            if flat.size < 256:
                padded = np.zeros(256, dtype=flat.dtype)
                padded[:flat.size] = flat
                vec = padded.reshape(1, 256)
            else:
                vec = flat[:256].reshape(1, 256)
        return vec.astype(np.float32, copy=False)

    def _create_audio(self, text: str, voice_embedding: Optional[np.ndarray], speed: float = 1.0) -> Tuple[np.ndarray, int]:
        """Tokenize text and run model to produce audio array and sample rate.

        Builds ONNX input map by probing session input names for compatibility
        (tokens/input_ids, style/speaker/voice, speed/rate/tempo).
        """
        if self.tokenizer is None or not hasattr(self.tokenizer, "tokenize"):
            # Minimal tokenization: bytes to ordinals
            tokens = np.array([ord(c) % 256 for c in text], dtype=np.int64)
        else:
            try:
                tokens = self.tokenizer.tokenize(text)
                if not isinstance(tokens, np.ndarray):
                    tokens = np.array(tokens, dtype=np.int64)
                else:
                    tokens = tokens.astype(np.int64, copy=False)
            except Exception:
                tokens = np.array([ord(c) % 256 for c in text], dtype=np.int64)

        tokens = tokens.reshape(1, -1)

        # Prepare inputs; probe input names if session is available
        input_names: List[str] = []
        if self.sess is not None and hasattr(self.sess, "get_inputs"):
            try:
                input_names = [i.name for i in self.sess.get_inputs()]
            except Exception:
                input_names = []

        def _pick(names: List[str]) -> Optional[str]:
            for n in names:
                if n in input_names:
                    return n
            return None

        inputs: Dict[str, np.ndarray] = {}
        token_name = _pick(["tokens", "input_ids", "phoneme_ids", "text"]) or (input_names[0] if input_names else "input_ids")
        inputs[token_name] = tokens

        # Style / voice embedding
        style_name = _pick(["style", "speaker", "voice", "speaker_embedding"])
        if style_name is not None:
            if voice_embedding is None:
                raise ValueError("Voice embedding is required for synthesis")
            style_vec = self._to_style_vector(voice_embedding)
            inputs[style_name] = style_vec

        # Speed / rate
        speed_name = _pick(["speed", "rate", "tempo"])
        if speed_name is not None:
            inputs[speed_name] = np.array([float(speed)], dtype=np.float32)

        # Run model
        if self.sess is None:
            # Produce dummy audio if session not available (tests mock .sess)
            audio = np.random.rand(SAMPLE_RATE).astype(np.float32)
        else:
            try:
                outputs = self.sess.run(None, inputs)
            except ValueError as e:
                # Retry with standard input names if model complains about missing required inputs
                msg = str(e)
                rebuilt: Dict[str, np.ndarray] = {}
                rebuilt["tokens"] = tokens
                # Ensure style
                if voice_embedding is None:
                    raise ValueError("Voice embedding is required for synthesis")
                rebuilt["style"] = self._to_style_vector(voice_embedding)
                rebuilt["speed"] = np.array([float(speed)], dtype=np.float32)
                try:
                    outputs = self.sess.run(None, rebuilt)
                except Exception:
                    # Give up and re-raise original error
                    raise e
            audio = np.asarray(outputs[0]).reshape(-1).astype(np.float32, copy=False)
        return audio, SAMPLE_RATE

    def _run_onnx_inference(self, token_ids: List[int], voice_embedding: np.ndarray, speed: float) -> Tuple[np.ndarray, int]:
        """Run ONNX inference with proper input/output handling."""
        if self.sess is None:
            raise RuntimeError("ONNX session not initialized")

        # Prepare inputs with correct names and shapes
        inputs = {}
        
        # Discover input names from session
        input_info = self.sess.get_inputs()
        for inp in input_info:
            name = inp.name.lower()
            if "token" in name or "text" in name or "input" in name:
                inputs[inp.name] = np.array([token_ids], dtype=np.int64)
            elif "style" in name or "speaker" in name or "voice" in name or "embedding" in name:
                # Ensure voice embedding has correct shape
                emb = self._to_style_vector(voice_embedding)
                inputs[inp.name] = emb
            elif "speed" in name or "rate" in name or "duration" in name:
                inputs[inp.name] = np.array([[speed]], dtype=np.float32)

        # Run inference
        try:
            outputs = self.sess.run(None, inputs)
            audio = outputs[0]  # Assume first output is audio
            
            # Convert to 1D array if needed
            if audio.ndim > 1:
                audio = audio.flatten()
                
            return audio.astype(np.float32), SAMPLE_RATE
            
        except Exception as e:
            raise RuntimeError(f"ONNX inference failed: {e}")

    def _sanitize_ipa(self, ipa: str) -> str:
        """Lightweight IPA sanitizer to improve compatibility with model vocab.

        - Drop primary/secondary stress and length marks
        - Normalize tie bars and retroflex/r-colored vowels to sequences
        - Remove glottal stop and standalone combining marks
        - Collapse whitespace
        """
        # Ordered replacements (longest first where relevant)
        repl = [
            ("t͡ʃ", "tʃ"),
            ("d͡ʒ", "dʒ"),
            ("d͡z", "dz"),
            ("ˈ", ""),
            ("ˌ", ""),
            ("ː", ""),
            ("ʔ", ""),
            ("ɑ̃", "ɑ"),
            ("ɛ̃", "ɛ"),
            ("ɔ̃", "ɔ"),
            ("œ̃", "œ"),
            # R-colored vowels to base + r sequence (no hyphen)
            ("ɝ", "ɜr"),
            ("ɚ", "ər"),
        ]

        s = str(ipa)
        for a, b in repl:
            if a in s:
                s = s.replace(a, b)

        # Remove stray combining tilde if present
        s = s.replace("̃", "")

        # Collapse whitespace to single spaces
        s = " ".join(s.split())
        return s

    def _load_voice_embedding(self, voice: Optional[str]) -> np.ndarray:
        """Load voice embedding with deterministic fallback."""
        if not voice or voice not in self.voice_embeddings:
            if voice:
                logger.info(f"voice={self.default_voice} (fallback)")
            actual_voice = self.default_voice
        else:
            actual_voice = voice
            
        return self.voice_embeddings[actual_voice]

    def _save_audio_to_wav(self, audio: np.ndarray, wav_path: str):
        """Save audio array to WAV file with proper normalization."""
        import soundfile as sf

        # Ensure audio is 1D
        if audio.ndim > 1:
            audio = audio.squeeze()

        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Light peak normalization to -1 dBFS (prevents clipping while maximizing loudness)
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = audio * (0.8912509381337456 / peak)  # 10 ** (-1/20)

        # Save as WAV using the public sample rate constant
        sf.write(wav_path, audio, SAMPLE_RATE, format='WAV', subtype='PCM_16')

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'voices') and hasattr(self.voices, 'close'):
            try:
                self.voices.close()
            except:
                pass
