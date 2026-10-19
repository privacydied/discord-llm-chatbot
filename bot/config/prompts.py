"""System prompt loading from files."""

import os
from datetime import UTC, datetime
from pathlib import Path

from bot.utils.logging import get_logger

logger = get_logger(__name__)

# Placeholders a prompt file may use; filled at request time, not load time,
# because prompts are cached for the life of the process. [CMV]
DATE_PLACEHOLDER = "{current_date}"
TIME_PLACEHOLDER = "{current_time_utc}"


def render_prompt_vars(text: str, now: datetime | None = None) -> str:
    """Fill date/time placeholders in a system prompt. Plain replace, so other braces are safe."""
    if not text or "{current_" not in text:
        return text
    now = now or datetime.now(UTC)
    return text.replace(DATE_PLACEHOLDER, now.strftime("%A %Y-%m-%d").lower()).replace(TIME_PLACEHOLDER, now.strftime("%H:%M"))


def load_system_prompts() -> dict[str, str]:
    """Loads system prompts from files specified in .env and returns them as a dictionary.

    Supports non-breaking synonyms:
    - TEXT_PROMPT_PATH = PROMPT_FILE
    - VL_PROMPT_PATH   = VL_PROMPT_FILE
    """
    prompts = {}
    try:
        prompt_file = os.getenv("TEXT_PROMPT_PATH") or os.getenv("PROMPT_FILE", "prompts/prompt-yoroi-super-chill.txt")
        vl_prompt_file = os.getenv("VL_PROMPT_PATH") or os.getenv("VL_PROMPT_FILE", "prompts/vl-prompt.txt")

        prompts["text_prompt"] = Path(prompt_file).read_text()
        prompts["vl_prompt"] = Path(vl_prompt_file).read_text()

        logger.info(f"[OK] Loaded system prompts: {list(prompts.keys())}")
        return prompts
    except FileNotFoundError as e:
        logger.warning(f"[WARNING] Prompt file not found at {e.filename}; using minimal fallback prompts for startup.")
        prompts.setdefault("text_prompt", "You are a helpful assistant.")
        prompts.setdefault("vl_prompt", "Describe the image succinctly.")
        return prompts
