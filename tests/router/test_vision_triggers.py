"""
Unit tests for vision trigger detection changes [CA][REH][IV]
Focused tests for the imagine trigger removal and media noun requirements
"""

import unittest
import re


class TestVisionTriggers(unittest.TestCase):
    """Test vision trigger detection logic"""

    def setUp(self):
        """Setup test fixtures"""
        # Test the exact regex patterns from router.py
        self.trigger_patterns = [
            # generate variants
            r"generate\s+(an?\s+)?(image|picture|pic|photo|art|drawing|render)",
            r"create\s+(an?\s+)?(image|picture|pic|photo|art|drawing|render)",
            r"make\s+(an?\s+)?(image|picture|pic|photo|art|drawing|render)",
            r"draw\s+(me\s+)?(an?\s+)?(image|picture|pic|photo|art|drawing|render)",  # "draw me a..." with media noun
            r"render\s+(an?\s+)?(image|picture|pic|photo|art|drawing|render)",  # "render something" with media noun
            r"paint\s+(an?\s+)?(image|picture|pic|photo|art|drawing|render)",  # "paint a..." with media noun
        ]

    def _should_trigger(self, content: str) -> bool:
        """Test if content should trigger T2I based on current patterns"""
        content_clean = content.lower().strip()
        content_clean = re.sub(r"[!.?]+", ".", content_clean)  # Collapse punctuation

        for pattern in self.trigger_patterns:
            if re.search(pattern, content_clean):
                return True
        return False

    def test_imagine_prose_no_trigger(self):
        """Test that 'imagine not being addicted to...' does NOT trigger T2I"""
        test_cases = [
            "imagine not being addicted to sex",
            "imagine being free from addiction",
            "I can only imagine what that feels like",
            "imagine a world without war",
            "imagine yourself succeeding",
            "imagine the possibilities",
        ]

        for content in test_cases:
            with self.subTest(content=content):
                result = self._should_trigger(content)
                self.assertFalse(result, f"'{content}' should NOT trigger T2I")

    def test_explicit_generate_still_works(self):
        """Test that explicit 'generate' commands still trigger T2I"""
        test_cases = [
            "generate an image of a puppy",
            "generate a picture of a castle",
            "generate a photo of mountains",
            "generate art of a dragon",
            "generate a drawing of flowers",
        ]

        for content in test_cases:
            with self.subTest(content=content):
                result = self._should_trigger(content)
                self.assertTrue(result, f"'{content}' should trigger T2I")

    def test_explicit_create_still_works(self):
        """Test that explicit 'create' commands still trigger T2I"""
        test_cases = [
            "create an image of a sunset",
            "create a picture of a cat",
            "create a photo of space",
            "create art of abstract shapes",
        ]

        for content in test_cases:
            with self.subTest(content=content):
                result = self._should_trigger(content)
                self.assertTrue(result, f"'{content}' should trigger T2I")

    def test_explicit_make_still_works(self):
        """Test that explicit 'make' commands still trigger T2I"""
        test_cases = [
            "make an image of a robot",
            "make a picture of pizza",
            "make a drawing of trees",
        ]

        for content in test_cases:
            with self.subTest(content=content):
                result = self._should_trigger(content)
                self.assertTrue(result, f"'{content}' should trigger T2I")

    def test_draw_requires_media_noun(self):
        """Test that 'draw' now requires a media noun"""
        # Should NOT trigger (no media noun)
        no_trigger_cases = [
            "draw me a puppy",
            "draw a circle",
            "draw something cool",
            "draw me a map",
        ]

        for content in no_trigger_cases:
            with self.subTest(content=content):
                result = self._should_trigger(content)
                self.assertFalse(
                    result, f"'{content}' should NOT trigger T2I without media noun"
                )

        # Should trigger (has media noun)
        trigger_cases = [
            "draw me a picture of a puppy",
            "draw an image of a circle",
            "draw a photo of something cool",
            "draw art of a landscape",
        ]

        for content in trigger_cases:
            with self.subTest(content=content):
                result = self._should_trigger(content)
                self.assertTrue(result, f"'{content}' should trigger T2I")

    def test_render_requires_media_noun(self):
        """Test that 'render' now requires a media noun"""
        # Should NOT trigger (no media noun)
        no_trigger_cases = [
            "render this useless",
            "render unto Caesar",
            "render a verdict",
        ]

        for content in no_trigger_cases:
            with self.subTest(content=content):
                result = self._should_trigger(content)
                self.assertFalse(
                    result, f"'{content}' should NOT trigger T2I without media noun"
                )

        # Should trigger (has media noun)
        trigger_cases = [
            "render an image of a building",
            "render a picture of space",
            "render art of futuristic city",
        ]

        for content in trigger_cases:
            with self.subTest(content=content):
                result = self._should_trigger(content)
                self.assertTrue(result, f"'{content}' should trigger T2I")

    def test_paint_requires_media_noun(self):
        """Test that 'paint' now requires a media noun"""
        # Should NOT trigger (no media noun)
        no_trigger_cases = [
            "paint the fence",
            "paint over the mistakes",
            "paint me surprised",
        ]

        for content in no_trigger_cases:
            with self.subTest(content=content):
                result = self._should_trigger(content)
                self.assertFalse(
                    result, f"'{content}' should NOT trigger T2I without media noun"
                )

        # Should trigger (has media noun)
        trigger_cases = [
            "paint a picture of mountains",
            "paint an image of flowers",
            "paint art of abstract design",
        ]

        for content in trigger_cases:
            with self.subTest(content=content):
                result = self._should_trigger(content)
                self.assertTrue(result, f"'{content}' should trigger T2I")

    def test_case_insensitive_matching(self):
        """Test that trigger detection is case-insensitive"""
        test_cases = [
            "Generate An Image Of A Dog",
            "CREATE a PICTURE of a cat",
            "Make A Photo Of Space",
            "DRAW AN IMAGE of trees",
        ]

        for content in test_cases:
            with self.subTest(content=content):
                result = self._should_trigger(content)
                self.assertTrue(
                    result, f"'{content}' should trigger T2I (case insensitive)"
                )

    def test_punctuation_tolerance(self):
        """Test that trigger detection handles punctuation correctly"""
        test_cases = [
            "generate an image of a puppy!",
            "create a picture of a castle?",
            "make a photo of mountains...",
            "draw an image of flowers!!!",
        ]

        for content in test_cases:
            with self.subTest(content=content):
                result = self._should_trigger(content)
                self.assertTrue(
                    result, f"'{content}' should trigger T2I despite punctuation"
                )

    def test_no_false_positives_common_phrases(self):
        """Test that common phrases don't accidentally trigger T2I"""
        no_trigger_cases = [
            "create a plan",
            "make a decision",
            "draw conclusions",
            "render assistance",
            "paint the town red",
            "picture this scenario",
            "art of war",
            "photo opportunity",
        ]

        for content in no_trigger_cases:
            with self.subTest(content=content):
                result = self._should_trigger(content)
                self.assertFalse(result, f"'{content}' should NOT trigger T2I")


if __name__ == "__main__":
    unittest.main()
