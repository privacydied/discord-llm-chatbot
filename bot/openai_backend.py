"""
OpenAI/OpenRouter Backend - Handles OpenAI API calls including OpenRouter.
"""
from typing import Dict, Any, Union, AsyncGenerator
import openai
import aiohttp
import base64
import os

from .config import load_config
from .memory import get_profile, get_server_profile
from .exceptions import APIError
from .util.logging import get_logger

logger = get_logger(__name__)


async def generate_openai_response(
    prompt: str,
    context: str = "",
    system_prompt: str = None,
    user_id: str = None,
    guild_id: str = None,
    temperature: float = None,
    max_tokens: int = None,
    stream: bool = False,
    **kwargs
) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
    """
    Generate a response using OpenAI API (including OpenRouter).
    
    Args:
        prompt: The user's input prompt
        context: Optional context to include in the prompt
        user_id: Optional user ID for personalization
        guild_id: Optional guild ID for server-specific context
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        stream: Whether to stream the response
        **kwargs: Additional parameters to pass to the API
        
    Returns:
        Dictionary with the generated text and metadata
    """
    try:
        config = load_config()
        
        # Configure OpenAI client
        client = openai.AsyncOpenAI(
            api_key=config.get('OPENAI_API_KEY'),
            base_url=config.get('OPENAI_API_BASE', 'https://api.openai.com/v1')
        )
        
        # Get model configuration
        model = config.get('OPENAI_TEXT_MODEL', 'gpt-4')
        
        # Get user preferences if user_id is provided
        if user_id:
            profile = get_profile(str(user_id))
            user_prefs = profile.get('preferences', {}) if profile else {}
            
            # Apply user preferences if not overridden
            if temperature is None:
                temperature = user_prefs.get('temperature', config.get('TEMPERATURE', 0.7))
        else:
            temperature = temperature or config.get('TEMPERATURE', 0.7)
        
        # Get server context if guild_id is provided
        server_context = ""
        if guild_id:
            server_profile = get_server_profile(str(guild_id))
            if server_profile:
                server_context = server_profile.get('context_notes', '')
        
        # Prepare the messages payload
        messages = []

        # If a specific system_prompt is provided, use it.
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        # Otherwise, construct the default system prompt.
        else:
            prompt_file_path = config.get("PROMPT_FILE")
            if not prompt_file_path:
                raise APIError("PROMPT_FILE not configured in environment variables")
            try:
                with open(prompt_file_path, "r", encoding="utf-8") as pf:
                    base_system_prompt = pf.read().strip()
            except FileNotFoundError:
                raise APIError(f"Prompt file not found: {prompt_file_path}")
            except Exception as e:
                raise APIError(f"Error reading prompt file {prompt_file_path}: {e}")
            
            # Combine base prompt, server context, and conversation history
            full_system_prompt = f"""{base_system_prompt}

Server Context: {server_context}"""
            messages.append({"role": "system", "content": full_system_prompt})

        # Add conversation history if it exists and a specific system prompt was used
        if system_prompt and context:
            messages.append({"role": "system", "content": f"PREVIOUS_CONVERSATION_HISTORY:\n{context}"})

        # Finally, add the user's actual prompt
        messages.append({"role": "user", "content": prompt})
        
        # Set default max_tokens if not provided
        if max_tokens is None:
            max_tokens = config.get('MAX_RESPONSE_TOKENS', 1000)
        
        logger.info(f"Generating OpenAI response with model: {model}")
        
        # Generate the response
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs
        )
        
        if stream:
            # Handle streaming response
            async def stream_generator():
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield {
                            'text': chunk.choices[0].delta.content,
                            'finished': False
                        }
                yield {'text': '', 'finished': True}
            
            return stream_generator()
        else:
            # Handle non-streaming response
            response_text = response.choices[0].message.content
            
            return {
                'text': response_text,
                'model': model,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                    'completion_tokens': response.usage.completion_tokens if response.usage else 0,
                    'total_tokens': response.usage.total_tokens if response.usage else 0
                },
                'backend': 'openai'
            }
    
    except openai.AuthenticationError as e:
        logger.error(f"OpenAI authentication failed: {e}")
        raise APIError(f"OpenAI authentication failed - check API key: {str(e)}")
    except openai.RateLimitError as e:
        logger.warning(f"OpenAI rate limit exceeded: {e}")
        raise APIError(f"OpenAI rate limit exceeded: {str(e)}")
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise APIError(f"OpenAI API error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in generate_openai_response: {e}", exc_info=True)
        raise APIError(f"Failed to generate OpenAI response: {str(e)}")


async def get_base64_image(image_url: str) -> str:
    """
    Process image from URL or file path and convert to base64 data URI.
    CHANGE: Enhanced to handle both URLs and file paths.
    """
    logger.debug(f"📥 Processing image from: {image_url}")
    
    # Handle file paths (file:// protocol or direct file path)
    if image_url.startswith('file://') or os.path.exists(image_url):
        try:
            # Extract actual path from file:// URL if needed
            file_path = image_url[7:] if image_url.startswith('file://') else image_url
            
            # Verify file exists
            if not os.path.exists(file_path):
                error_msg = f"File not found: {file_path}"
                logger.error(error_msg)
                raise APIError(error_msg)
                
            # Read file and encode to base64
            with open(file_path, 'rb') as f:
                data = f.read()
                
            # Determine content type from file extension
            ext = os.path.splitext(file_path)[1].lower()
            content_type = "image/jpeg" if ext in ('.jpg', '.jpeg') else \
                          "image/png" if ext == '.png' else \
                          "image/webp" if ext == '.webp' else \
                          "image/gif" if ext == '.gif' else \
                          "image/png"  # Default to PNG
                          
            base64_data = base64.b64encode(data).decode('utf-8')
            logger.debug(f"✅ Image loaded from file: size={len(data)} bytes, type={content_type}")
            return f"data:{content_type};base64,{base64_data}"
            
        except Exception as e:
            error_msg = f"Failed to process image file: {e}"
            logger.error(error_msg, exc_info=True)
            raise APIError(error_msg)
    
    # Handle HTTP/HTTPS URLs
    elif image_url.startswith(('http://', 'https://')):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        data = await response.read()
                        base64_data = base64.b64encode(data).decode('utf-8')
                        content_type = response.headers.get('Content-Type', 'image/png')
                        logger.debug(f"✅ Image downloaded: size={len(data)} bytes, type={content_type}")
                        return f"data:{content_type};base64,{base64_data}"
                    else:
                        error_msg = f"Failed to download image: status={response.status}"
                        logger.error(error_msg)
                        raise APIError(error_msg)
        except Exception as e:
            error_msg = f"Failed to download image from URL: {e}"
            logger.error(error_msg, exc_info=True)
            raise APIError(error_msg)
    
    # Already a data URL
    elif image_url.startswith('data:'):
        logger.debug("✅ Image already in data URL format")
        return image_url
    
    # Unsupported format
    else:
        error_msg = f"Unsupported image source format: {image_url[:30]}..."
        logger.error(error_msg)
        raise APIError(error_msg)


async def generate_vl_response(
    image_url: str,
    user_prompt: str = "",
    user_id: str = None,
    guild_id: str = None,
    temperature: float = None,
    max_tokens: int = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate a vision-language response using the VL model and VL_PROMPT_FILE.
    CHANGE: Enhanced VL model support with comprehensive debug logging for hybrid multimodal inference.
    
    Args:
        image_url: URL or path to the image
        user_prompt: Optional additional user prompt
        user_id: Optional user ID for personalization
        guild_id: Optional guild ID for server-specific context
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        **kwargs: Additional parameters to pass to the API
        
    Returns:
        Dictionary with the generated VL analysis and metadata
    """
    try:
        logger.info("🎨 === VL RESPONSE GENERATION STARTED ===")
        logger.debug(f"🎨 Image URL: {image_url[:100]}{'...' if len(image_url) > 100 else ''}")
        logger.debug(f"🎨 User prompt: '{user_prompt[:100]}{'...' if len(user_prompt) > 100 else ''}'")
        
        config = load_config()
        
        # Configure OpenAI client
        logger.debug("🎨 Configuring OpenAI client for VL")
        client = openai.AsyncOpenAI(
            api_key=config.get('OPENAI_API_KEY'),
            base_url=config.get('OPENAI_API_BASE', 'https://api.openai.com/v1')
        )
        logger.debug(f"🎨 Using API base: {config.get('OPENAI_API_BASE', 'https://api.openai.com/v1')}")
        
        # Get VL model configuration
        vl_model = config.get('VL_MODEL')
        if not vl_model:
            logger.error("❌ VL_MODEL not configured in environment variables")
            raise APIError("VL_MODEL not configured in environment variables")
        
        logger.info(f"🎨 Using VL model: {vl_model}")
        
        # CHANGE: Load VL prompt from VL_PROMPT_FILE with enhanced logging
        vl_prompt_file_path = config.get("VL_PROMPT_FILE")
        if not vl_prompt_file_path:
            logger.error("❌ VL_PROMPT_FILE not configured in environment variables")
            raise APIError("VL_PROMPT_FILE not configured in environment variables")
        
        logger.debug(f"🎨 Loading VL prompt from: {vl_prompt_file_path}")
        try:
            with open(vl_prompt_file_path, "r", encoding="utf-8") as vpf:
                vl_system_prompt = vpf.read().strip()
                logger.info(f"✅ VL prompt loaded: {len(vl_system_prompt)} characters")
                logger.debug(f"🎨 VL prompt preview: {vl_system_prompt[:100]}{'...' if len(vl_system_prompt) > 100 else ''}")
        except FileNotFoundError:
            logger.error(f"❌ VL prompt file not found: {vl_prompt_file_path}")
            raise APIError(f"VL prompt file not found: {vl_prompt_file_path}")
        except Exception as e:
            logger.error(f"❌ Error reading VL prompt file {vl_prompt_file_path}: {e}")
            raise APIError(f"Error reading VL prompt file {vl_prompt_file_path}: {e}")
        
        # Get user preferences if user_id is provided
        if user_id:
            profile = get_profile(str(user_id))
            user_prefs = profile.get('preferences', {}) if profile else {}
            
            # Apply user preferences if not overridden
            if temperature is None:
                temperature = user_prefs.get('temperature', config.get('TEMPERATURE', 0.7))
        else:
            temperature = temperature or config.get('TEMPERATURE', 0.7)
        
        logger.debug(f"🎨 Temperature: {temperature}")
        
        # Combine VL system prompt with user prompt if provided
        full_prompt = vl_system_prompt
        if user_prompt:
            full_prompt += f"\n\nAdditional context: {user_prompt}"
            logger.debug(f"🎨 Enhanced prompt with user context: +{len(user_prompt)} chars")
        
        # Handle data URLs directly, otherwise try to download and encode
        if isinstance(image_url, str) and image_url.startswith('data:'):
            logger.info("✅ Using provided data URL for image")
            image_content = image_url
        else:
            try:
                image_content = await get_base64_image(image_url)
                logger.info("✅ Image downloaded and converted to base64 data URI")
            except Exception as img_error:
                logger.error(f"❌ Failed to process image: {img_error}")
                raise APIError(f"Image processing failed: {str(img_error)}")
        
        logger.debug(f"🎨 Base64 data length: {len(image_content)} chars")
        logger.debug(f"🎨 Base64 preview: {image_content[:100]}...")
        
        # Prepare the messages for vision model
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": full_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_content
                        }
                    }
                ]
            }
        ]
        
        # Set default max_tokens if not provided
        if max_tokens is None:
            max_tokens = config.get('MAX_RESPONSE_TOKENS', 1000)
        
        logger.debug(f"🎨 Max tokens: {max_tokens}")
        logger.info(f"🎨 Calling OpenAI VL API with model: {vl_model}")
        
        # Generate the VL response
        logger.debug(f"🎨 Sending request with messages: {len(messages)} messages")
        response = await client.chat.completions.create(
            model=vl_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # CHANGE: Enhanced error handling with detailed response logging
        if response is None:
            logger.error("❌ VL API returned None response")
            raise APIError("VL API returned None response")
        
        # Log complete response for debugging
        logger.debug(f"🎨 Full API response: {response}")
        logger.debug(f"🎨 Response type: {type(response)}")
        logger.debug(f"🎨 Response attributes: {dir(response)}")
        
        # Try to dump the response model if possible
        try:
            if hasattr(response, 'model_dump'):
                logger.debug(f"🎨 Response model_dump: {response.model_dump()}")
        except Exception as dump_error:
            logger.warning(f"⚠️ Failed to dump response model: {dump_error}")
        
        # Validate response structure
        if not hasattr(response, 'choices'):
            logger.error("❌ VL API response has no 'choices' attribute")
            raise APIError("VL API response has no 'choices' attribute")
        
        if response.choices is None:
            logger.error("❌ VL API response.choices is None")
            raise APIError("VL API response.choices is None")
        
        if not response.choices:
            logger.error("❌ VL API response choices list is empty")
            logger.debug(f"❌ Choices: {response.choices}")
            raise APIError("VL API response choices list is empty")
        
        # Validate first choice
        choice = response.choices[0]
        if not hasattr(choice, 'message'):
            logger.error("❌ VL API response choice has no 'message' attribute")
            logger.debug(f"❌ Choice attributes: {dir(choice)}")
            raise APIError("VL API response choice has no 'message' attribute")
        
        if choice.message is None:
            logger.error("❌ VL API response choice message is None")
            raise APIError("VL API response choice message is None")
        
        logger.info("✅ VL API call completed successfully")
        
        # Extract the response text
        response_text = choice.message.content
        
        if not response_text:
            logger.warning("⚠️  VL model returned empty response")
        else:
            logger.info(f"✅ VL response received: {len(response_text)} characters")
            logger.debug(f"🎨 VL response preview: {response_text[:200]}{'...' if len(response_text) > 200 else ''}")
        
        usage_info = {
            'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
            'completion_tokens': response.usage.completion_tokens if response.usage else 0,
            'total_tokens': response.usage.total_tokens if response.usage else 0
        }
        
        logger.debug(f"🎨 Token usage: {usage_info}")
        logger.info("🎉 === VL RESPONSE GENERATION COMPLETED ===")
        
        return {
            'text': response_text,
            'model': vl_model,
            'usage': usage_info,
            'backend': 'openai_vl'
        }
    
    except Exception as e:
        logger.error(f"❌ Error in generate_vl_response: {e}", exc_info=True)
        raise APIError(f"Failed to generate VL response: {str(e)}")
