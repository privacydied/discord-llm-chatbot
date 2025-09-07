"""
VL Output Sanitizer - Removes chain-of-thought leakage and model reasoning.

Enforces clean VL output before passing to Text Flow in the 1-hop pipeline.
"""
import os
import re
from typing import Optional


def sanitize_model_output(text: str) -> str:
    """
    Sanitize VL model output to remove reasoning, rules, and thinking blocks.
    
    Args:
        text: Raw VL model response
        
    Returns:
        Cleaned text suitable for Text Flow input
    """
    # Check if sanitization is enabled
    strip_reasoning = os.getenv("VL_STRIP_REASONING", "1").lower() in ("1", "true", "yes", "on")
    if not strip_reasoning:
        return text.strip()
    
    # Start with the original text
    cleaned = text.strip()
    
    # 1. Strip any <think>...</think> blocks (case-insensitive, multiline)
    think_pattern = re.compile(r'<think>.*?</think>', re.IGNORECASE | re.DOTALL)
    cleaned = think_pattern.sub('', cleaned)
    
    # 2. Remove first block of rules/explain steps boilerplate
    rules_patterns = [
        r'^.*?Follow specific rules.*?\n',
        r'^.*?Let me break down.*?\n',
        r'^.*?Don\'t speculate.*?\n',
        r'^.*?Here are the guidelines.*?\n',
        r'^.*?I need to follow.*?\n',
        r'^\s*[-•]\s*.*?rules?.*?\n',
        r'^\s*[-•]\s*.*?guidelines?.*?\n',
        r'^\s*\d+\.\s*.*?rules?.*?\n',
    ]
    
    for pattern in rules_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
    
    # 3. Remove common intros that echo instructions
    intro_patterns = [
        r'^Got your image\s*[—-]\s*here\'s what I see:?\s*',
        r'^I can see .*? image.*?\. Let me analyze.*?\s*',
        r'^Looking at .*? image.*?\s*',
        r'^I\'ll analyze .*? image.*?\s*',
        r'^Here\'s my analysis.*?:\s*',
        r'^Based on .*? guidelines.*?\s*',
    ]
    
    for pattern in intro_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # 4. Clean up whitespace and line breaks
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)  # Collapse multiple blank lines
    cleaned = cleaned.strip()
    
    # 5. Cap length to ~1000 chars on sentence boundary
    if len(cleaned) > 1000:
        # Find last sentence boundary before 1000 chars
        truncate_at = 1000
        sentence_endings = ['.', '!', '?']
        
        # Look backward from position 1000 to find sentence ending
        for i in range(min(1000, len(cleaned) - 1), 500, -1):  # Don't go below 500 chars
            if cleaned[i] in sentence_endings and i < len(cleaned) - 1:
                # Make sure next char is whitespace or end of string
                if i + 1 >= len(cleaned) or cleaned[i + 1].isspace():
                    truncate_at = i + 1
                    break
        
        cleaned = cleaned[:truncate_at].strip()
    
    # 6. Final cleanup - remove any remaining empty lines at start/end
    cleaned = cleaned.strip()
    
    return cleaned


def has_reasoning_content(text: str) -> bool:
    """
    Check if text contains reasoning content that should be sanitized.
    
    Args:
        text: Text to check
        
    Returns:
        True if text contains <think> blocks or rule explanations
    """
    if not text:
        return False
    
    # Check for <think> blocks
    if re.search(r'<think>', text, re.IGNORECASE):
        return True
    
    # Check for common rule/guideline language
    rule_indicators = [
        'follow specific rules',
        'let me break down',
        'don\'t speculate',
        'here are the guidelines',
        'i need to follow',
        'based on the rules',
        'according to guidelines'
    ]
    
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in rule_indicators)
