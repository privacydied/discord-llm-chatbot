"""
Ollama API integration for the Discord bot.
"""
import os
import json
import logging
import aiohttp
import asyncio
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from datetime import datetime, timedelta

# Import bot modules
from .config import load_config
from .logs import log_command
from .memory import get_profile, save_profile, get_server_profile, save_server_profile

# Load configuration
config = load_config()

# Ollama API settings
OLLAMA_BASE_URL = config["OLLAMA_BASE_URL"]
OLLAMA_MODEL = config["OLLAMA_MODEL"]
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_PRESENCE_PENALTY = 0.0

# Rate limiting
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 30  # Max requests per window

# Track rate limits
user_rate_limits = {}

class OllamaAPIError(Exception):
    """Custom exception for Ollama API errors."""
    pass

class OllamaClient:
    """Client for interacting with the Ollama API."""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        """Initialize the Ollama client."""
        self.base_url = base_url or OLLAMA_BASE_URL
        self.api_key = api_key
        self.session = None
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }
        if self.api_key:
            self.headers['Authorization'] = f'Bearer {self.api_key}'
    
    async def ensure_session(self) -> None:
        """Ensure we have an active aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(headers=self.headers)
    
    async def close(self) -> None:
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def check_rate_limit(self, user_id: str) -> bool:
        """Check if a user has exceeded their rate limit."""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=RATE_LIMIT_WINDOW)
        
        # Clean up old entries
        if user_id in user_rate_limits:
            user_rate_limits[user_id] = [
                t for t in user_rate_limits[user_id] 
                if t > window_start
            ]
        else:
            user_rate_limits[user_id] = []
        
        # Check if user is over the limit
        if len(user_rate_limits[user_id]) >= RATE_LIMIT_MAX_REQUESTS:
            return False
        
        # Add current request to the rate limit counter
        user_rate_limits[user_id].append(now)
        return True
    
    async def generate(
        self,
        prompt: str,
        model: str = None,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        stop: List[str] = None,
        user_id: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using the Ollama API.
        
        Args:
            prompt: The prompt to generate text from
            model: The model to use (default: OLLAMA_MODEL from config)
            max_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (0.0 to 1.0)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            frequency_penalty: Penalize new tokens based on frequency in the text so far
            presence_penalty: Penalize new tokens based on whether they appear in the text so far
            stop: List of strings that will stop generation when encountered
            user_id: Optional user ID for rate limiting and context
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Dictionary with the generated text and metadata
        """
        # Apply defaults
        model = model or OLLAMA_MODEL
        max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        top_p = top_p if top_p is not None else DEFAULT_TOP_P
        frequency_penalty = frequency_penalty if frequency_penalty is not None else DEFAULT_FREQUENCY_PENALTY
        presence_penalty = presence_penalty if presence_penalty is not None else DEFAULT_PRESENCE_PENALTY
        stop = stop or []
        
        # Check rate limit
        if user_id and not await self.check_rate_limit(user_id):
            raise OllamaAPIError("Rate limit exceeded. Please try again later.")
        
        # Prepare the request payload
        payload = {
            'model': model,
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': max(0.0, min(1.0, temperature)),
            'top_p': max(0.0, min(1.0, top_p)),
            'frequency_penalty': max(0.0, min(1.0, frequency_penalty)),
            'presence_penalty': max(0.0, min(1.0, presence_penalty)),
            'stop': stop,
            **kwargs
        }
        
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        try:
            await self.ensure_session()
            
            # Make the API request
            url = f"{self.base_url.rstrip('/')}/api/generate"
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise OllamaAPIError(f"API request failed with status {response.status}: {error_text}")
                
                # Parse the response
                response_data = await response.json()
                
                # Extract the generated text
                generated_text = response_data.get('response', '')
                
                # Return the result
                return {
                    'text': generated_text,
                    'model': model,
                    'usage': {
                        'prompt_tokens': response_data.get('prompt_eval_count', 0),
                        'completion_tokens': response_data.get('eval_count', 0),
                        'total_tokens': response_data.get('prompt_eval_count', 0) + response_data.get('eval_count', 0),
                    },
                    'finish_reason': response_data.get('done_reason', 'unknown'),
                    'raw_response': response_data
                }
        
        except asyncio.TimeoutError:
            raise OllamaAPIError("Request timed out. The server is taking too long to respond.")
        except aiohttp.ClientError as e:
            raise OllamaAPIError(f"Network error: {str(e)}")
        except Exception as e:
            logging.error(f"Error in Ollama generate: {e}", exc_info=True)
            raise OllamaAPIError(f"An error occurred: {str(e)}")
    
    async def generate_stream(
        self,
        prompt: str,
        model: str = None,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        frequency_penalty: float = None,
        presence_penalty: float = None,
        stop: List[str] = None,
        user_id: str = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate text using the Ollama API with streaming.
        
        Yields:
            Dictionaries with partial results and metadata
        """
        # Apply defaults
        model = model or OLLAMA_MODEL
        max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        top_p = top_p if top_p is not None else DEFAULT_TOP_P
        frequency_penalty = frequency_penalty if frequency_penalty is not None else DEFAULT_FREQUENCY_PENALTY
        presence_penalty = presence_penalty if presence_penalty is not None else DEFAULT_PRESENCE_PENALTY
        stop = stop or []
        
        # Check rate limit
        if user_id and not await self.check_rate_limit(user_id):
            raise OllamaAPIError("Rate limit exceeded. Please try again later.")
        
        # Prepare the request payload
        payload = {
            'model': model,
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': max(0.0, min(1.0, temperature)),
            'top_p': max(0.0, min(1.0, top_p)),
            'frequency_penalty': max(0.0, min(1.0, frequency_penalty)),
            'presence_penalty': max(0.0, min(1.0, presence_penalty)),
            'stop': stop,
            'stream': True,
            **kwargs
        }
        
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        try:
            await self.ensure_session()
            
            # Make the streaming API request
            url = f"{self.base_url.rstrip('/')}/api/generate"
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise OllamaAPIError(f"API request failed with status {response.status}: {error_text}")
                
                # Process the streaming response
                buffer = ""
                async for chunk in response.content.iter_any():
                    if not chunk:
                        continue
                    
                    # Decode the chunk
                    try:
                        chunk_text = chunk.decode('utf-8')
                        buffer += chunk_text
                        
                        # Split by newlines (Ollama sends JSON objects separated by newlines)
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            if not line.strip():
                                continue
                            
                            try:
                                data = json.loads(line)
                                yield {
                                    'text': data.get('response', ''),
                                    'done': data.get('done', False),
                                    'model': model,
                                    'raw_response': data
                                }
                            except json.JSONDecodeError:
                                logging.warning(f"Failed to parse JSON: {line}")
                                continue
                    
                    except Exception as e:
                        logging.error(f"Error processing chunk: {e}", exc_info=True)
                        continue
        
        except asyncio.TimeoutError:
            raise OllamaAPIError("Request timed out. The server is taking too long to respond.")
        except aiohttp.ClientError as e:
            raise OllamaAPIError(f"Network error: {str(e)}")
        except Exception as e:
            logging.error(f"Error in Ollama generate_stream: {e}", exc_info=True)
            raise OllamaAPIError(f"An error occurred: {str(e)}")
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models from the Ollama API."""
        try:
            await self.ensure_session()
            
            # Make the API request
            url = f"{self.base_url.rstrip('/')}/api/tags"
            async with self.session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise OllamaAPIError(f"API request failed with status {response.status}: {error_text}")
                
                # Parse the response
                response_data = await response.json()
                return response_data.get('models', [])
        
        except Exception as e:
            logging.error(f"Error listing models: {e}", exc_info=True)
            raise OllamaAPIError(f"Failed to list models: {str(e)}")
    
    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        try:
            models = await self.list_models()
            for model in models:
                if model['name'] == model_name or model['model'] == model_name:
                    return model
            return None
        except Exception as e:
            logging.error(f"Error getting model info: {e}", exc_info=True)
            return None

# Create a global instance for convenience
ollama_client = OllamaClient()

# Helper functions for common tasks
async def generate_response(
    prompt: str,
    context: str = "",
    user_id: str = None,
    guild_id: str = None,
    temperature: float = None,
    max_tokens: int = None,
    stream: bool = False,
    **kwargs
) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
    """
    Generate a response using the Ollama API with context and user preferences.
    
    Args:
        prompt: The user's input prompt
        context: Optional context to include in the prompt
        user_id: Optional user ID for personalization and rate limiting
        guild_id: Optional guild ID for server-specific context
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        stream: Whether to stream the response
        **kwargs: Additional parameters to pass to the API
        
    Returns:
        Dictionary with the generated text and metadata, or an async generator for streaming
    """
    try:
        # Get user preferences if user_id is provided
        if user_id:
            profile = get_profile(str(user_id))
            user_prefs = profile.get('preferences', {})
            
            # Apply user preferences if not overridden
            if temperature is None:
                temperature = user_prefs.get('temperature', DEFAULT_TEMPERATURE)
            
            # Get user's preferred model if set
            model = user_prefs.get('model', OLLAMA_MODEL)
        else:
            model = OLLAMA_MODEL
        
        # Get server context if guild_id is provided
        server_context = ""
        if guild_id:
            server_profile = get_server_profile(str(guild_id))
            server_context = server_profile.get('context_notes', '')
        
        # CHANGE: Use PROMPT_FILE from environment instead of hardcoded prompt
        config = load_config()
        prompt_file_path = config.get("PROMPT_FILE")
        if not prompt_file_path:
            raise OllamaAPIError("PROMPT_FILE not configured in environment variables")
        
        try:
            with open(prompt_file_path, "r", encoding="utf-8") as pf:
                base_system_prompt = pf.read().strip()
                logging.debug(f"Loaded text prompt from {prompt_file_path}")
        except FileNotFoundError:
            raise OllamaAPIError(f"Prompt file not found: {prompt_file_path}")
        except Exception as e:
            raise OllamaAPIError(f"Error reading prompt file {prompt_file_path}: {e}")
        
        # Build the full prompt with context and server context
        full_prompt = f"""{base_system_prompt}

Context: {context}

Server Context: {server_context}

User: {prompt}
Assistant:"""
        
        # Generate the response
        if stream:
            return ollama_client.generate_stream(
                prompt=full_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                user_id=user_id,
                **kwargs
            )
        else:
            return await ollama_client.generate(
                prompt=full_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                user_id=user_id,
                **kwargs
            )
    
    except Exception as e:
        logging.error(f"Error in generate_response: {e}", exc_info=True)
        raise OllamaAPIError(f"Failed to generate response: {str(e)}")

# Cleanup function to close the client
async def cleanup():
    """Clean up resources."""
    await ollama_client.close()
