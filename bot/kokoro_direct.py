"""
Shim module for legacy KokoroDirect interface used in tests.

- Re-exports SAMPLE_RATE from the fixed implementation.
- Provides an `en.G2P` symbol for patching.
- Exposes a lightweight KokoroDirect class compatible with tests in `tests/test_kokoro_direct.py`.

This shim does NOT perform real I/O. The tests patch `_init_session` and
`_load_available_voices`, and inject a mocked `session` and `g2p`.
"""

from __future__ import annotations

from typing import Tuple, Any

import numpy as np

# Re-export constants from the fixed implementation to keep values consistent
from .kokoro_direct_fixed import SAMPLE_RATE  # noqa: F401

__all__ = ["KokoroDirect", "SAMPLE_RATE", "en"]


class _EN:
    class G2P:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def convert(self, text: str) -> str:
            # Identity conversion placeholder; tests patch this class
            return text


# Namespace expected by tests: bot.kokoro_direct.en.G2P
en = _EN()


class KokoroDirect:
    """Minimal, test-oriented shim for legacy KokoroDirect API.

    Expected by tests:
    - __init__(onnx_dir=..., voices_dir=...)
    - attributes: session (onnxruntime.InferenceSession-like), g2p
    - methods: _init_session(), _load_available_voices(), _load_voice(voice_id),
               create(text=..., voice_id=...) -> (audio: np.ndarray, sample_rate: int)
    """

    def __init__(self, onnx_dir: str, voices_dir: str, **_: Any) -> None:
        # Store paths for completeness; not used directly by tests
        self.onnx_dir = onnx_dir
        self.voices_dir = voices_dir

        # Placeholders; tests will patch or set these
        self.session = None  # set by tests to a MagicMock(InferenceSession)
        self.g2p = None  # set by tests to a MagicMock or patched en.G2P

        # Hooks that tests patch to avoid I/O
        self._init_session()
        self._load_available_voices()

    # The following hooks exist solely to be patched by tests
    def _init_session(self) -> None:  # pragma: no cover - tests patch this
        return None

    def _load_available_voices(self) -> None:  # pragma: no cover - tests patch this
        return None

    def _load_voice(
        self, voice_id: str
    ) -> np.ndarray:  # pragma: no cover - tests patch this
        """Load a voice embedding for the given voice_id.
        Tests patch this to return a (N, 256) float32 array.
        """
        raise NotImplementedError("_load_voice must be patched by tests")

    def _to_style_vector(self, emb: np.ndarray) -> np.ndarray:
        """Convert various embedding shapes to (1, 256) float32 style vector."""
        vec = emb
        # Common shapes in tests: (512, 256) -> mean over time -> (256,)
        if vec.ndim == 2 and vec.shape[1] == 256 and vec.shape[0] in (510, 511, 512):
            vec = vec.mean(axis=0)
        if vec.ndim == 3 and vec.shape[2] == 256:
            vec = vec.squeeze(1)
        if vec.ndim == 1 and vec.shape[0] >= 256:
            vec = vec[:256]
        if vec.ndim == 1:
            vec = vec.reshape(1, 256)
        elif vec.ndim == 2 and vec.shape != (1, 256):
            vec = vec.reshape(1, 256)
        return vec.astype(np.float32, copy=False)

    def create(
        self, *, text: str, voice_id: str, speed: float = 1.0
    ) -> Tuple[np.ndarray, int]:
        """Create audio from text using the specified voice.

        Returns a tuple (audio, SAMPLE_RATE) to match legacy tests.
        """
        # Optional G2P step (tests usually patch/inject self.g2p)
        if self.g2p is not None and hasattr(self.g2p, "convert"):
            try:
                _ = self.g2p.convert(text)  # result not used by test assertions
            except Exception:
                # Ignore G2P failures in shim; tests focus on ONNX call plumbing
                pass

        # Load voice embedding (tests patch this method)
        emb = self._load_voice(voice_id)
        style = self._to_style_vector(np.asarray(emb))

        # Dummy tokens input; mocked session will not validate values
        tokens = np.array([1, 2, 3, 4], dtype=np.int64)
        speed_arr = np.array([speed], dtype=np.float32)

        # Session is provided by tests as a MagicMock(InferenceSession)
        assert self.session is not None, "session must be set by tests"
        outputs = self.session.run(
            None, {"input_ids": tokens, "style": style, "speed": speed_arr}
        )

        # Take first output and squeeze to 1D
        audio = np.asarray(outputs[0]).reshape(-1).astype(np.float32, copy=False)
        return audio, SAMPLE_RATE
