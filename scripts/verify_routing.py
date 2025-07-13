#!/usr/bin/env python3
"""Validate command routing and mode selection"""
import unittest
from unittest.mock import MagicMock, patch
from bot.command_parser import parse_command
from bot.controller import hybrid_pipeline

class TestRouting(unittest.TestCase):
    def setUp(self):
        self.ctx = MagicMock()
        self.ctx.send = MagicMock()

    @patch('bot.controller.hybrid_pipeline')
    def test_text_mode(self, mock_pipeline):
        content, mode = parse_command("!ask Hello --mode=text")
        self.assertEqual(mode, "text")
        mock_pipeline.assert_called_with(self.ctx, "Hello", "text")

    @patch('bot.controller.hybrid_pipeline')
    def test_tts_mode(self, mock_pipeline):
        content, mode = parse_command("!speak Hello")
        self.assertEqual(mode, "tts")
        mock_pipeline.assert_called_with(self.ctx, "Hello", "tts")

    @patch('bot.controller.hybrid_pipeline')
    def test_vl_mode(self, mock_pipeline):
        content, mode = parse_command("!see Describe this image --mode=vl")
        self.assertEqual(mode, "vl")
        mock_pipeline.assert_called_with(self.ctx, "Describe this image", "vl")

if __name__ == "__main__":
    unittest.main()