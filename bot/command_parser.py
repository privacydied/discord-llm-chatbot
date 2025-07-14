"""
Unified command parser with mode extraction
"""
import re
from typing import Tuple

def parse_command(content: str) -> Tuple[str, str]:
    """Extract mode and cleaned content from command"""
    # Default mode is text
    mode = "text"
    
    # Check for mode flags
    mode_flag_match = re.search(r"--mode=(\w+)", content)
    if mode_flag_match:
        mode = mode_flag_match.group(1).lower()
        content = content.replace(mode_flag_match.group(0), "").strip()
    
    # Check for command-based mode indicators
    if content.startswith("!speak"):
        mode = "tts"
        content = content.replace("!speak", "", 1).strip()
    elif content.startswith("!hear"):
        mode = "stt"
        content = content.replace("!hear", "", 1).strip()
    elif content.startswith("!see"):
        mode = "vl"
        content = content.replace("!see", "", 1).strip()
    elif content.startswith("!tts"):
        # Extract subcommand and text
        parts = content.split(maxsplit=1)
        if len(parts) > 1 and not parts[1].startswith('-'):
            # If there's text after !tts (not a flag), treat as TTS input
            mode = "tts"
            content = parts[1].strip()
        else:
            # Otherwise keep mode as text for subcommands like !tts on/off
            mode = "text"
    
    # Validate mode
    valid_modes = {"text", "tts", "stt", "vl", "both"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Valid modes are: {', '.join(valid_modes)}")
    
    return content, mode