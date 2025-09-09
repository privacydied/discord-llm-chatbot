import numpy as np
import logging

logger = logging.getLogger(__name__)


class KokoroDirect:
    def __init__(self, model_path: str, voices_path: str, use_tokenizer: bool = True):
        self.model_path = model_path
        self.voices_path = voices_path
        self.use_tokenizer = use_tokenizer

        # Load voices with proper exception handling
        try:
            self.voices = np.load(self.voices_path)
            logger.debug(f"Loaded {len(self.voices)} voices")
        except Exception as e:
            logger.error(f"Failed to load voices: {e}", exc_info=True)
            raise
