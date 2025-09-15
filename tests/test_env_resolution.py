"""Tests for environment variable resolution."""

import unittest
import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.env_utils import resolve_env, resolve_path, get_config_singleton


class TestEnvResolution(unittest.TestCase):
    """Test environment variable resolution."""

    def setUp(self):
        """Set up test environment."""
        # Clear any environment variables that might interfere with tests
        for var in ["TEST_NEW", "TEST_LEGACY", "TEST_PATH_NEW", "TEST_PATH_LEGACY"]:
            if var in os.environ:
                del os.environ[var]

    def test_resolve_env_new_only(self):
        """Test resolving environment variable with only new variable set."""
        with patch.dict(os.environ, {"TEST_NEW": "new_value"}):
            result = resolve_env("TEST_NEW", "TEST_LEGACY", "default_value")
            self.assertEqual(result, "new_value")

    def test_resolve_env_legacy_only(self):
        """Test resolving environment variable with only legacy variable set."""
        with patch.dict(os.environ, {"TEST_LEGACY": "legacy_value"}):
            result = resolve_env("TEST_NEW", "TEST_LEGACY", "default_value")
            self.assertEqual(result, "legacy_value")

    def test_resolve_env_both_set(self):
        """Test resolving environment variable with both variables set."""
        with patch.dict(
            os.environ, {"TEST_NEW": "new_value", "TEST_LEGACY": "legacy_value"}
        ):
            result = resolve_env("TEST_NEW", "TEST_LEGACY", "default_value")
            self.assertEqual(result, "new_value")  # New value should take precedence

    def test_resolve_env_none_set(self):
        """Test resolving environment variable with no variables set."""
        result = resolve_env("TEST_NEW", "TEST_LEGACY", "default_value")
        self.assertEqual(result, "default_value")

    def test_resolve_path_new_only(self):
        """Test resolving path with only new variable set."""
        with patch.dict(os.environ, {"TEST_PATH_NEW": "/path/to/new"}):
            result = resolve_path("TEST_PATH_NEW", "TEST_PATH_LEGACY", "/default/path")
            self.assertEqual(result, Path("/path/to/new"))

    def test_resolve_path_legacy_only(self):
        """Test resolving path with only legacy variable set."""
        with patch.dict(os.environ, {"TEST_PATH_LEGACY": "/path/to/legacy"}):
            result = resolve_path("TEST_PATH_NEW", "TEST_PATH_LEGACY", "/default/path")
            self.assertEqual(result, Path("/path/to/legacy"))

    def test_resolve_path_both_set(self):
        """Test resolving path with both variables set."""
        with patch.dict(
            os.environ,
            {"TEST_PATH_NEW": "/path/to/new", "TEST_PATH_LEGACY": "/path/to/legacy"},
        ):
            result = resolve_path("TEST_PATH_NEW", "TEST_PATH_LEGACY", "/default/path")
            self.assertEqual(
                result, Path("/path/to/new")
            )  # New path should take precedence

    def test_resolve_path_none_set(self):
        """Test resolving path with no variables set."""
        result = resolve_path("TEST_PATH_NEW", "TEST_PATH_LEGACY", "/default/path")
        self.assertEqual(result, Path("/default/path"))

    def test_config_singleton(self):
        """Test that the config singleton stores resolved paths."""
        with patch.dict(os.environ, {"TEST_PATH_NEW": "/path/to/new"}):
            # Resolve a path to store it in the singleton
            resolve_path("TEST_PATH_NEW", "TEST_PATH_LEGACY", "/default/path")

            # Get the singleton config
            config = get_config_singleton()

            # Check that the resolved path is in the config
            self.assertIn("paths", config)
            self.assertIn("TEST_PATH_NEW|TEST_PATH_LEGACY", config["paths"])
            self.assertEqual(
                config["paths"]["TEST_PATH_NEW|TEST_PATH_LEGACY"], "/path/to/new"
            )


if __name__ == "__main__":
    unittest.main()
