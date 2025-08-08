"""
Event handlers for Discord bot.

This module contains event handlers that enhance bot robustness and user experience.
"""
from .command_error_handler import setup_command_error_handler

__all__ = ['setup_command_error_handler']
