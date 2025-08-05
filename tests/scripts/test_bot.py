"""
Tests for the Discord LLM ChatBot.

These tests verify the core functionality of the bot without requiring
an actual Discord connection.
"""
import os
import sys
import json
import asyncio
import unittest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import bot modules
from bot.config import load_config
from bot.memory import (
    get_profile, save_profile, 
    get_server_profile, save_server_profile,
    user_profiles, server_profiles
)
from bot.ollama import OllamaClient, ollama_client
from bot.tts import generate_tts, cleanup_tts
from bot.search import search_all, SearchResult
from bot.web import get_url_preview, process_url
from bot.pdf_utils import pdf_processor, PDFProcessor

class TestMemory(unittest.IsolatedAsyncioTestCase):
    """Test memory management functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_user_id = "test_user_123"
        self.test_server_id = "test_server_456"
        
        # Create a temporary directory for test data
        self.test_dir = tempfile.TemporaryDirectory()
        os.environ['USER_PROFILE_DIR'] = os.path.join(self.test_dir.name, 'user_profiles')
        os.environ['SERVER_PROFILE_DIR'] = os.path.join(self.test_dir.name, 'server_profiles')
        os.environ['USER_LOGS_DIR'] = os.path.join(self.test_dir.name, 'user_logs')
        os.environ['TEMP_DIR'] = os.path.join(self.test_dir.name, 'temp')
        
        # Create directories if they don't exist
        os.makedirs(os.environ['USER_PROFILE_DIR'], exist_ok=True)
        os.makedirs(os.environ['SERVER_PROFILE_DIR'], exist_ok=True)
        os.makedirs(os.environ['USER_LOGS_DIR'], exist_ok=True)
        os.makedirs(os.environ['TEMP_DIR'], exist_ok=True)
        
        # Clear any existing test data
        user_profiles.clear()
        server_profiles.clear()
    
    def tearDown(self):
        """Clean up test environment."""
        self.test_dir.cleanup()
    
    def test_user_profile_creation(self):
        """Test creating and retrieving a user profile."""
        # Get a new profile
        profile = get_profile(self.test_user_id)
        
        # Check default values
        self.assertIn('user_id', profile)
        self.assertEqual(profile['user_id'], self.test_user_id)
        self.assertIn('first_seen', profile)
        self.assertIn('preferences', profile)
        self.assertIn('memories', profile)
        
        # Modify and save the profile
        profile['preferences'] = {'theme': 'dark'}
        save_profile(profile, force=True)
        
        # Retrieve it again and verify
        loaded_profile = get_profile(self.test_user_id)
        self.assertEqual(loaded_profile['preferences']['theme'], 'dark')
    
    def test_server_profile_creation(self):
        """Test creating and retrieving a server profile."""
        # Get a new server profile
        profile = get_server_profile(self.test_server_id)
        
        # Check default values
        self.assertIn('server_id', profile)
        self.assertEqual(profile['server_id'], self.test_server_id)
        self.assertIn('preferences', profile)
        self.assertIn('memories', profile)
        
        # Modify and save the profile
        profile['preferences'] = {'tts_enabled': True}
        save_server_profile(profile, force=True)
        
        # Retrieve it again and verify
        loaded_profile = get_server_profile(self.test_server_id)
        self.assertTrue(loaded_profile['preferences']['tts_enabled'])

class TestOllamaIntegration(unittest.IsolatedAsyncioTestCase):
    """Test Ollama API integration."""
    
    async def asyncSetUp(self):
        """Set up test environment."""
        self.ollama = OllamaClient(base_url="http://localhost:11434")
        await self.ollama.ensure_session()
    
    async def asyncTearDown(self):
        """Clean up test environment."""
        await self.ollama.close()
    
    @patch('aiohttp.ClientSession.post')
    async def test_generate_text(self, mock_post):
        """Test text generation with a mock response."""
        # Set up mock response
        mock_response = {
            'response': 'This is a test response.',
            'done': True,
            'model': 'llama3',
            'prompt_eval_count': 10,
            'eval_count': 20
        }
        
        # Configure the mock
        mock_response_obj = AsyncMock()
        mock_response_obj.status = 200
        mock_response_obj.json = AsyncMock(return_value=mock_response)
        mock_post.return_value.__aenter__.return_value = mock_response_obj
        
        # Call the method
        response = await self.ollama.generate(
            prompt="Test prompt",
            model="llama3",
            max_tokens=100
        )
        
        # Verify the response
        self.assertEqual(response['text'], 'This is a test response.')
        self.assertEqual(response['model'], 'llama3')
        self.assertEqual(response['usage']['prompt_tokens'], 10)
        self.assertEqual(response['usage']['completion_tokens'], 20)

class TestSearch(unittest.IsolatedAsyncioTestCase):
    """Test search functionality."""
    
    @patch('aiohttp.ClientSession.post')
    async def test_web_search(self, mock_post):
        """Test web search with mock response."""
        # Set up mock response
        mock_html = """
        <html>
            <body>
                <div class="result">
                    <h2 class="result__a">Test Result 1</h2>
                    <div class="result__snippet">This is a test snippet.</div>
                </div>
                <div class="result">
                    <h2 class="result__a">Test Result 2</h2>
                    <div class="result__snippet">Another test snippet.</div>
                </div>
            </body>
        </html>
        """
        
        # Configure the mock
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=mock_html)
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Import here to avoid patching issues
        from bot.search import web_search
        
        # Call the function
        results = await web_search("test query")
        
        # Verify the results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "Test Result 1")
        self.assertEqual(results[0].snippet, "This is a test snippet.")
        self.assertEqual(results[1].title, "Test Result 2")
        self.assertEqual(results[1].snippet, "Another test snippet.")

class TestTTS(unittest.IsolatedAsyncioTestCase):
    """Test text-to-speech functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.TemporaryDirectory()
        os.environ['TEMP_DIR'] = self.test_dir.name
    
    def tearDown(self):
        """Clean up test environment."""
        self.test_dir.cleanup()
    
    @patch('subprocess.run')
    async def test_generate_tts(self, mock_run):
        """Test TTS generation with mock subprocess."""
        # Configure the mock
        mock_run.return_value = MagicMock(returncode=0)
        
        # Call the function
        output_file = await generate_tts("Test text", "test_user_123")
        
        # Verify the output
        self.assertTrue(output_file.exists())
        self.assertEqual(output_file.suffix, ".wav")
        self.assertIn("test_user_123", str(output_file))
    
    async def test_cleanup_tts(self):
        """Test TTS cleanup."""
        # Create a test file
        test_file = Path(self.test_dir.name) / "test_tts.wav"
        test_file.touch()
        
        # Call the cleanup function
        await cleanup_tts()
        
        # Verify the file was deleted
        self.assertFalse(test_file.exists())

if __name__ == "__main__":
    unittest.main()
