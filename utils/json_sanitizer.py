import json
import re
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

class JSONSanitizer:
    @staticmethod
    def sanitize(raw_json: str) -> Dict[str, Any]:
        """
        Sanitize malformed JSON from PDF extraction with multiple fallback strategies
        """
        # Strategy 1: Try direct parsing with relaxed rules
        try:
            return json.loads(raw_json, strict=False)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Remove control characters
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', raw_json)
        try:
            return json.loads(cleaned, strict=False)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error after basic cleaning: {e}")
        
        # Strategy 3: Fix common structural issues
        repaired = JSONSanitizer._repair_json(cleaned)
        try:
            return json.loads(repaired, strict=False)
        except json.JSONDecodeError as e:
            logger.error(f"JSON repair failed: {e}")
        
        # Final fallback: Extract content directly
        return JSONSanitizer._extract_content_fallback(cleaned)

    @staticmethod
    def _repair_json(cleaned: str) -> str:
        """Fix common JSON structural issues"""
        # Add missing commas between objects
        repaired = re.sub(r'}\s*{', '},{', cleaned)
        
        # Fix unterminated strings
        repaired = re.sub(r'(?<!\\)"([^"]*)$', r'"\1"', repaired)
        
        # Balance brackets
        open_count = repaired.count('{')
        close_count = repaired.count('}')
        if open_count > close_count:
            repaired += '}' * (open_count - close_count)
        elif close_count > open_count:
            repaired = repaired.rsplit('}', close_count - open_count)[0]
        
        return repaired

    @staticmethod
    def _extract_content_fallback(text: str) -> Dict[str, Any]:
        """Fallback content extraction when JSON is unrecoverable"""
        # Extract content between quotes
        match = re.search(r'"content":\s*"((?:\\"|[^"])*)"', text, re.DOTALL)
        content = match.group(1) if match else text
        
        return {"success": bool(match), "content": content}
