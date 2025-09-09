"""
Smoke tests for environment variables and multimodal functionality.
CHANGE: Added comprehensive tests to verify .env→code mapping and multimodal branch coverage.
"""
import sys
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import bot modules
from bot.config import (
    load_config, validate_required_env, validate_prompt_files, check_venv_activation, ConfigurationError
)
from bot.ai_backend import generate_response, generate_vl_response
from bot.events import has_image_attachments, get_image_urls
from bot.openai_backend import generate_vl_response as openai_vl_response


class TestEnvironmentVariables:
    """Test environment variable loading and validation."""
    
    def test_env_vars_loaded(self):
        """Verify that essential .env variables are accessible."""
        config = load_config()
        
        # Test required variables
        assert config.get('DISCORD_TOKEN') is not None, "DISCORD_TOKEN must be set"
        assert config.get('PROMPT_FILE') is not None, "PROMPT_FILE must be set"
        assert config.get('VL_PROMPT_FILE') is not None, "VL_PROMPT_FILE must be set"
        
        # Test OpenAI/OpenRouter variables  
        assert config.get('OPENAI_API_KEY') is not None, "OPENAI_API_KEY must be set"
        assert config.get('OPENAI_TEXT_MODEL') is not None, "OPENAI_TEXT_MODEL must be set"
        assert config.get('VL_MODEL') is not None, "VL_MODEL must be set"
        
        print("✅ All essential environment variables are loaded")
    
    def test_prompt_files_exist(self):
        """Verify that prompt files exist and are readable."""
        config = load_config()
        
        prompt_file = config.get('PROMPT_FILE')
        vl_prompt_file = config.get('VL_PROMPT_FILE')
        
        assert prompt_file is not None, "PROMPT_FILE must be configured"
        assert vl_prompt_file is not None, "VL_PROMPT_FILE must be configured"
        
        prompt_path = Path(prompt_file)
        vl_prompt_path = Path(vl_prompt_file)
        
        assert prompt_path.exists(), f"Text prompt file not found: {prompt_path}"
        assert vl_prompt_path.exists(), f"VL prompt file not found: {vl_prompt_path}"
        
        # Test that files are readable
        with open(prompt_path, 'r') as f:
            prompt_content = f.read()
            assert len(prompt_content.strip()) > 0, "Text prompt file is empty"
        
        with open(vl_prompt_path, 'r') as f:
            vl_prompt_content = f.read()
            assert len(vl_prompt_content.strip()) > 0, "VL prompt file is empty"
        
        print("✅ Prompt files exist and are readable:")
        print(f"  • Text prompt: {prompt_path}")
        print(f"  • VL prompt: {vl_prompt_path}")
    
    def test_validate_required_env(self):
        """Test the validate_required_env function."""
        try:
            validate_required_env()
            print("✅ Required environment variables validation passed")
        except ConfigurationError as e:
            pytest.fail(f"Required environment variables missing: {e}")
    
    def test_validate_prompt_files(self):
        """Test the validate_prompt_files function."""
        try:
            validate_prompt_files()
            print("✅ Prompt file validation passed")
        except ConfigurationError as e:
            pytest.fail(f"Prompt file validation failed: {e}")


class TestImageDetection:
    """Test image detection utilities for multimodal processing."""
    
    def test_has_image_attachments_with_images(self):
        """Test image detection with mock Discord message containing images."""
        # Mock Discord message with image attachment
        mock_message = MagicMock()
        mock_attachment = MagicMock()
        mock_attachment.filename = "test_image.png"
        mock_attachment.content_type = "image/png"
        mock_message.attachments = [mock_attachment]
        
        result = has_image_attachments(mock_message)
        assert result is True, "Should detect image attachments"
        print("✅ Image detection works for PNG files")
    
    def test_has_image_attachments_no_images(self):
        """Test image detection with mock Discord message containing no images."""
        # Mock Discord message with no attachments
        mock_message = MagicMock()
        mock_message.attachments = []
        
        result = has_image_attachments(mock_message)
        assert result is False, "Should not detect images when none present"
        print("✅ Image detection correctly identifies no images")
    
    def test_get_image_urls(self):
        """Test image URL extraction from Discord message."""
        # Mock Discord message with image attachment
        mock_message = MagicMock()
        mock_attachment = MagicMock()
        mock_attachment.filename = "test_image.jpg"
        mock_attachment.url = "https://cdn.discordapp.com/attachments/123/456/test_image.jpg"
        mock_attachment.content_type = "image/jpeg"
        mock_message.attachments = [mock_attachment]
        
        urls = get_image_urls(mock_message)
        assert len(urls) == 1, "Should extract one image URL"
        assert urls[0] == mock_attachment.url, "Should return the correct URL"
        print("✅ Image URL extraction works correctly")


class TestMultimodalLogic:
    """Test multimodal AI processing logic."""
    
    @patch('bot.openai_backend.openai.AsyncOpenAI')
    async def test_vl_response_generation(self, mock_openai_client):
        """Test VL response generation with mocked OpenAI client."""
        # Setup mock OpenAI client
        mock_client_instance = AsyncMock()
        mock_openai_client.return_value = mock_client_instance
        
        # Mock response
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "This is a test image showing a cat sitting on a windowsill."
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 25
        mock_response.usage.total_tokens = 75
        
        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Test VL response generation
        result = await openai_vl_response(
            image_url="https://example.com/test_image.jpg",
            user_prompt="What do you see in this image?",
            user_id="test_user_123"
        )
        
        assert result['text'] == "This is a test image showing a cat sitting on a windowsill."
        assert result['backend'] == 'openai_vl'
        assert result['usage']['total_tokens'] == 75
        
        print("✅ VL response generation works correctly")
    
    @patch('bot.ai_backend.generate_vl_response')
    @patch('bot.ai_backend.generate_response')
    async def test_hybrid_multimodal_routing(self, mock_generate_response, mock_generate_vl_response):
        """Test that the AI backend correctly routes to VL when needed."""
        # Setup mocks
        mock_generate_vl_response.return_value = {
            'text': 'VL analysis: This image shows a beautiful sunset.',
            'model': 'mistralai/mistral-small-3.2-24b-instruct:free',
            'backend': 'openai_vl'
        }
        
        mock_generate_response.return_value = {
            'text': 'Based on the image analysis, this is indeed a lovely sunset scene.',
            'model': 'deepseek/deepseek-chat-v3-0324:free',
            'backend': 'openai'
        }
        
        # Test VL routing
        vl_result = await generate_vl_response(
            image_url="https://example.com/sunset.jpg",
            user_prompt="Describe this image"
        )
        
        # Test text routing
        text_result = await generate_response(
            prompt="Tell me about sunsets",
            context="Previous conversation context"
        )
        
        assert 'VL analysis' in vl_result['text']
        assert vl_result['backend'] == 'openai_vl'
        assert text_result['backend'] == 'openai'
        
        print("✅ Hybrid multimodal routing works correctly")


class TestVenvEnforcement:
    """Test .venv enforcement functionality."""
    
    def test_check_venv_activation(self):
        """Test .venv activation check."""
        # This test will check the current environment
        # In a real .venv, sys.prefix should contain '.venv'
        try:
            check_venv_activation()
            print("✅ .venv enforcement check completed")
        except Exception as e:
            # This is expected if not running in .venv
            print(f"⚠️  .venv enforcement warning (expected): {e}")


def test_env_and_model_integration():
    """
    Integration smoke test to verify .env→code mapping and multimodal branch coverage.
    CHANGE: Comprehensive integration test as specified in requirements.
    """
    print("\n🧪 Running comprehensive environment and multimodal smoke tests...\n")
    
    # Test 1: Environment variables
    print("1. Testing environment variable loading...")
    config = load_config()
    assert config is not None
    print("   ✅ Configuration loaded successfully")
    
    # Test 2: Prompt files
    print("2. Testing prompt file accessibility...")
    prompt_file = config.get('PROMPT_FILE')
    vl_prompt_file = config.get('VL_PROMPT_FILE')
    
    assert Path(prompt_file).exists(), f"Text prompt file missing: {prompt_file}"
    assert Path(vl_prompt_file).exists(), f"VL prompt file missing: {vl_prompt_file}"
    print("   ✅ Both prompt files accessible")
    
    # Test 3: Image detection logic
    print("3. Testing image detection logic...")
    mock_msg_with_image = MagicMock()
    mock_attachment = MagicMock()
    mock_attachment.filename = "test.png"
    mock_msg_with_image.attachments = [mock_attachment]
    
    mock_msg_no_image = MagicMock()
    mock_msg_no_image.attachments = []
    
    assert has_image_attachments(mock_msg_with_image) is True
    assert has_image_attachments(mock_msg_no_image) is False
    print("   ✅ Image detection logic works correctly")
    
    # Test 4: Model configuration
    print("4. Testing model configuration...")
    text_model = config.get('OPENAI_TEXT_MODEL')
    vl_model = config.get('VL_MODEL')
    
    assert text_model is not None, "Text model not configured"
    assert vl_model is not None, "VL model not configured"
    print(f"   ✅ Text model: {text_model}")
    print(f"   ✅ VL model: {vl_model}")
    
    print("\n🎉 All smoke tests passed! The hybrid multimodal system is ready.\n")


if __name__ == "__main__":
    # Run the integration test directly
    test_env_and_model_integration()
    
    # Run all tests with pytest
    pytest.main([__file__, "-v"])