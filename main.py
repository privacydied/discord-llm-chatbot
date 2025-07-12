import os
import re
import json
import time
import random
import logging
import asyncio
import base64
import pathlib
import glob
import tempfile
import mimetypes
import subprocess
import shutil
import traceback
from typing import Optional, Tuple
from pathlib import Path

# DIA TTS imports with error handling
try:
    from TTS.api import TTS
    DIA_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    DIA_AVAILABLE = False
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Coroutine, TypeVar
from functools import wraps
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse

# Third-party imports
from dotenv import load_dotenv
from discord import Intents, Message, File
from discord.ext import commands
import discord
import httpx
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)

# PDF Processing
try:
    from pdf_processor import PDFProcessor
    PDF_PROCESSOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PDF processing disabled: {e}")
    PDF_PROCESSOR_AVAILABLE = False
from urllib.parse import urlparse
from collections import defaultdict, Counter

from datetime import datetime, timedelta, timezone
from http.client import HTTPResponse
from io import StringIO
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, urlunparse
from urllib.robotparser import RobotFileParser
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import socket
import tempfile
import hashlib
from fake_useragent import UserAgent
import cachetools
from collections import defaultdict, Counter, namedtuple

# Optional Playwright import
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Types
WebpageResult = Dict[str, Any]
CacheBackend = Any  # Type for cache backends

# Constants
DEFAULT_TIMEOUT = 10  # seconds
CACHE_TTL = 3600  # 1 hour in seconds
RATE_LIMIT_WINDOW = 60  # seconds
MAX_RETRIES = 2
MIN_CONTENT_LENGTH = 50  # Minimum characters to consider content valid

# Cache for rate limiting
rate_limits = {}

# Default cache (in-memory)
memory_cache = cachetools.TTLCache(maxsize=1000, ttl=CACHE_TTL)

# Default user agents
DEFAULT_USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
]

# Helper functions
def get_domain(url: str) -> str:
    """Extract domain from URL."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"

def normalize_url(url: str) -> str:
    """Normalize URL for consistent cache keys."""
    parsed = urlparse(url)
    return urlunparse(parsed._replace(fragment='', query=''))

def get_cache_key(url: str) -> str:
    """Generate a cache key from URL."""
    return hashlib.md5(normalize_url(url).encode()).hexdigest()

def is_rate_limited(domain: str, window: int = RATE_LIMIT_WINDOW) -> bool:
    """Check if we're rate limited for this domain."""
    now = time.time()
    last_request = rate_limits.get(domain, 0)
    
    if now - last_request < window:
        return True
    
    rate_limits[domain] = now
    return False

def check_robots_txt(url: str, user_agent: str = '*') -> Tuple[bool, str]:
    """Check if URL is allowed by robots.txt."""
    try:
        domain = get_domain(url)
        robots_url = f"{domain}/robots.txt"
        
        # Check cache first
        cache_key = f"robots:{domain}"
        if cache_key in memory_cache:
            return memory_cache[cache_key]
        
        # Fetch robots.txt
        rp = RobotFileParser()
        rp.set_url(robots_url)
        
        try:
            rp.read()
            can_fetch = rp.can_fetch(user_agent, url)
            result = (can_fetch, "" if can_fetch else f"Blocked by {robots_url}")
            memory_cache[cache_key] = result
            return result
        except Exception as e:
            # If we can't read robots.txt, assume it's allowed
            return True, f"Could not parse robots.txt: {str(e)}"
    except Exception as e:
        return True, f"Error checking robots.txt: {str(e)}"

def get_random_user_agent() -> str:
    """Get a random user agent."""
    try:
        return UserAgent().random
    except:
        return random.choice(DEFAULT_USER_AGENTS)

async def extract_with_playwright(url: str, timeout: int = 30000) -> Tuple[str, str]:
    """Extract page content using Playwright (for JS-heavy sites)."""
    if not PLAYWRIGHT_AVAILABLE:
        return "", "Playwright not available"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            await page.goto(url, timeout=timeout, wait_until="networkidle")
            title = await page.title()
            content = await page.content()
            return content, title
        except Exception as e:
            return "", f"Playwright error: {str(e)}"
        finally:
            await browser.close()

# ---- RAG (Retrieval Augmented Generation) ----
@dataclass
class KnowledgeSnippet:
    """Represents a snippet of text from the knowledge base."""
    text: str
    source_file: str
    line_number: int

class KnowledgeBase:
    def __init__(self, kb_dir: str = "kb"):
        self.kb_dir = kb_dir
        self.knowledge: List[KnowledgeSnippet] = []
        self._load_knowledge_base()
    
    def _load_knowledge_base(self) -> None:
        """Load all .txt and .md files from the knowledge base directory."""
        self.knowledge = []
        if not os.path.exists(self.kb_dir):
            logging.warning(f"Knowledge base directory '{self.kb_dir}' not found.")
            return
            
        for ext in ('*.txt', '*.md'):
            for file_path in glob.glob(os.path.join(self.kb_dir, '**', ext), recursive=True):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for i, line in enumerate(f, 1):
                            line = line.strip()
                            if line:  # Skip empty lines
                                self.knowledge.append(
                                    KnowledgeSnippet(
                                        text=line,
                                        source_file=os.path.basename(file_path),
                                        line_number=i
                                    )
                                )
                except Exception as e:
                    logging.error(f"Error reading {file_path}: {e}")
    
    def get_relevant_snippets(self, query: str, top_k: int = 3) -> List[KnowledgeSnippet]:
        """
        Retrieve the most relevant snippets from the knowledge base.
        
        Args:
            query: The user's query
            top_k: Maximum number of snippets to return
            
        Returns:
            List of relevant KnowledgeSnippet objects
        """
        if not query or not self.knowledge:
            return []
            
        # Simple keyword matching - can be replaced with more sophisticated retrieval
        query_terms = set(re.findall(r'\w+', query.lower()))
        
        scored_snippets = []
        for snippet in self.knowledge:
            snippet_terms = set(re.findall(r'\w+', snippet.text.lower()))
            common_terms = query_terms & snippet_terms
            if common_terms:
                # Simple scoring: number of matching terms divided by snippet length
                score = len(common_terms) / (len(snippet_terms) + 1)
                scored_snippets.append((score, snippet))
        
        # Sort by score (highest first) and take top_k
        scored_snippets.sort(key=lambda x: x[0], reverse=True)
        return [snippet for _, snippet in scored_snippets[:top_k]]

# Initialize the knowledge base
knowledge_base = KnowledgeBase()

T = TypeVar('T', bound=Callable[..., Coroutine[Any, Any, Any]])

# ---- Configuration and Loading ----
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_prompt_file():
    prompt_path = os.getenv('PROMPT_FILE', 'prompt.txt')
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logging.warning(f"Prompt file '{prompt_path}' not found, using fallback prompt.")
        return "You are an AI assistant on Discord."

def load_vl_prompt():
    vl_prompt_path = os.getenv('VL_PROMPT_FILE', 'prompts/vl-prompt.txt')
    try:
        with open(vl_prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logging.warning(f"VL prompt file '{vl_prompt_path}' not found, using default VL prompt.")
        return "Describe this image in detail."

async def transcribe_audio(
    audio_path: str, 
    config: dict, 
    original_filename: Optional[str] = None,
    keep_converted: bool = False
) -> Tuple[Optional[str], Optional[str]]:
    """
    Transcribe audio using the Whisper API with robust error handling and validation.
    
    Args:
        audio_path: Path to the audio file to transcribe
        config: Configuration dictionary containing Whisper settings
        original_filename: Original filename (for logging and model selection)
        keep_converted: If True, keep the converted WAV file for debugging
        
    Returns:
        Tuple of (transcript, error_message) where:
        - transcript is the transcribed text if successful, None otherwise
        - error_message is None if successful, or contains an error message
    """
    # Validate configuration
    whisper_api_key = config.get("WHISPER_API_KEY")
    whisper_api_base = config.get("WHISPER_API_BASE")
    whisper_model = str(config.get("WHISPER_MODEL", "whisper-1"))  # Ensure model is a string
    
    if not all([whisper_api_key, whisper_api_base, whisper_model]):
        error_msg = "Missing required Whisper API configuration"
        logging.error(error_msg)
        return None, error_msg
    
    # Validate input file
    if not os.path.exists(audio_path):
        error_msg = f"Audio file not found: {audio_path}"
        logging.error(error_msg)
        return None, error_msg
    
    converted_path = None
    try:
        # Always convert to WAV for consistency, even if the file already has a .wav extension
        logging.info(f"Processing audio file: {audio_path} (size: {os.path.getsize(audio_path)} bytes)")
        converted_path, conversion_error = convert_audio_to_wav(audio_path, keep_failed=keep_converted)
        
        if not converted_path or conversion_error:
            error_msg = f"Failed to convert audio file: {conversion_error or 'Unknown error'}"
            logging.error(error_msg)
            return None, error_msg
        
        # Verify the converted file
        if not os.path.exists(converted_path):
            error_msg = "Converted file not found after successful conversion"
            logging.error(error_msg)
            return None, error_msg
            
        converted_size = os.path.getsize(converted_path)
        logging.info(f"Converted file: {converted_path} (size: {converted_size} bytes)")
        
        # Prepare the API request
        url = f"{whisper_api_base}/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {whisper_api_key}",
        }
        
        # Use the original filename if available, otherwise use the WAV filename
        filename = original_filename if original_filename else os.path.basename(audio_path)
        
        try:
            with open(converted_path, "rb") as audio_file:
                # Prepare the multipart form data
                files = {
                    "file": (filename, audio_file, "audio/wav"),
                }
                
                # Prepare the form data (all non-file fields)
                data = {
                    "model": whisper_model,
                    "response_format": "text",
                }
                
                # Log the request details (without file content)
                log_data = {
                    "file": f"<{os.path.getsize(converted_path)} bytes>",
                    **data
                }
                logging.info(f"Sending request to Whisper API: {url} with data: {log_data}")
                
                # Make the API request with timeout
                timeout = config.get("TIMEOUT", 30.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    start_time = time.time()
                    response = await client.post(
                        url,
                        headers=headers,
                        files=files,
                        data=data
                    )
                    process_time = time.time() - start_time
                    
                    logging.info(f"Whisper API response in {process_time:.2f}s: {response.status_code}")
                    
                    if response.status_code != 200:
                        error_detail = response.text[:500]  # Limit error detail length
                        error_msg = (
                            f"Whisper API error {response.status_code}: {error_detail}"
                        )
                        logging.error(f"{error_msg}. Full response: {response.text}")
                        return None, error_msg
                    
                    # Success!
                    transcript = response.text.strip()
                    logging.info(f"Transcription successful. Length: {len(transcript)} characters")
                    return transcript, None
                    
        except httpx.TimeoutException:
            error_msg = f"Whisper API request timed out after {timeout} seconds"
            logging.error(error_msg)
            return None, error_msg
            
        except httpx.RequestError as e:
            error_msg = f"Request to Whisper API failed: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return None, error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error during Whisper API request: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return None, error_msg
            
    except Exception as e:
        error_msg = f"Error during audio transcription: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return None, error_msg
        
    finally:
        # Clean up the converted file if it was created and we're not keeping it
        if converted_path and os.path.exists(converted_path) and not keep_converted:
            try:
                os.unlink(converted_path)
                logging.debug(f"Cleaned up temporary file: {converted_path}")
            except Exception as e:
                logging.warning(f"Failed to clean up temporary file {converted_path}: {e}")

def convert_audio_to_wav(input_path: str, keep_failed: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """
    Convert any audio file to a Whisper-compatible WAV file using ffmpeg.
    
    Args:
        input_path: Path to the input audio file
        keep_failed: If True, keep failed conversion files for debugging
        
    Returns:
        Tuple of (output_path, error_message) where:
        - output_path: Path to the converted WAV file on success, None on failure
        - error_message: None on success, user-friendly error message on failure
    """
    if not shutil.which('ffmpeg'):
        error_msg = "Audio processing is not properly configured. Please contact support."
        logging.error("ffmpeg is not installed. Please install ffmpeg to enable audio conversion.")
        return None, error_msg
    
    # Verify input file exists and has content
    try:
        input_size = os.path.getsize(input_path)
        if input_size < 1024:  # 1KB minimum size
            logging.warning(f"Input file too small ({input_size} bytes), likely corrupted or empty")
            return None, "The audio file is too small or corrupted. Please try with a different file."
    except OSError as e:
        logging.error(f"Failed to access input file: {e}")
        return None, "Could not read the audio file. Please check the file and try again."
    
    output_path = None
    try:
        # Create a temporary file for the output with a predictable name for debugging
        temp_dir = tempfile.gettempdir()
        temp_prefix = f"discord_audio_{os.getpid()}_{int(time.time())}_"
        temp_fd, output_path = tempfile.mkstemp(prefix=temp_prefix, suffix='.wav')
        os.close(temp_fd)  # We'll let ffmpeg create the file
        
        # Build the ffmpeg command with explicit settings for Whisper compatibility
        cmd = [
            'ffmpeg',
            '-hide_banner',  # Less verbose output
            '-loglevel', 'warning',  # Only show warnings and errors
            '-y',  # Overwrite output file if it exists
            '-i', input_path,  # Input file
            '-ar', '16000',  # Sample rate: 16kHz (Whisper's expected rate)
            '-ac', '1',  # Mono audio
            '-c:a', 'pcm_s16le',  # 16-bit little-endian PCM
            '-f', 'wav',  # Force WAV format
            '-fflags', '+bitexact',  # Ensure consistent output
            output_path
        ]
        
        logging.info(f"Running ffmpeg command: {' '.join(cmd)}")
        
        # Run ffmpeg and capture both stdout and stderr separately
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        # Always log the full ffmpeg output at debug level
        if result.stderr:
            logging.debug(f"ffmpeg stderr (exit code {result.returncode}):\n{result.stderr}")
        if result.stdout:
            logging.debug(f"ffmpeg stdout:\n{result.stdout}")
        
        # Check if output file was created
        if not os.path.exists(output_path):
            logging.error(f"FFmpeg failed to create output file. Command: {' '.join(cmd)}")
            return None, "Failed to process the audio file. The format might not be supported."
            
        output_size = os.path.getsize(output_path)
        logging.info(f"Output file created: {output_path} (size: {output_size} bytes)")
        
        # Only fail on non-zero exit code
        if result.returncode != 0:
            logging.error(f"FFmpeg failed with code {result.returncode}")
            if not keep_failed:
                try:
                    os.unlink(output_path)
                except Exception as e:
                    logging.error(f"Failed to clean up output file: {e}")
            return None, "Failed to process the audio file. The format might not be supported."
        
        # Log a warning if the output file is very small but don't fail
        if output_size < 1024:  # 1KB minimum size for WAV header + some audio data
            logging.warning(f"Output file is very small ({output_size} bytes), but ffmpeg reported success.")
        
        return output_path, None
        
    except Exception as e:
        logging.error(f"Unexpected error during audio conversion: {e}", exc_info=True)
        if output_path and os.path.exists(output_path) and not keep_failed:
            try:
                os.unlink(output_path)
            except Exception as cleanup_err:
                logging.error(f"Failed to clean up output file: {cleanup_err}")
        return None, "An unexpected error occurred while processing the audio file."

def is_url(string: str) -> bool:
    """Check if a string is a valid URL."""
    try:
        result = urlparse(string)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except ValueError:
        return False


async def process_pdf(
    source: Union[str, bytes],
    is_url: bool = False,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Process a PDF file or URL and extract text and metadata.
    
    Args:
        source: Path to PDF file, URL, or PDF content as bytes
        is_url: Whether source is a URL
        debug: Enable debug mode for detailed output
        
    Returns:
        Dict containing:
        - text: Extracted text content
        - title: PDF title or filename
        - error: Error message if any
        - source: Source URL or 'uploaded_file'
        - pages: Number of pages
        - is_scanned: Whether OCR was used
        - metadata: PDF metadata
    """
    if not PDF_PROCESSOR_AVAILABLE:
        return {
            'text': '',
            'title': 'PDF Processing Unavailable',
            'error': 'PDF processing dependencies not installed',
            'source': 'pdf',
            'pages': 0,
            'is_scanned': False,
            'metadata': {}
        }
    try:
        processor = PDFProcessor()
        # Ensure we await the coroutine
        result = await processor.process_pdf(source, is_url=is_url)
        
        if debug:
            logging.info(f"PDF processing result: {result.get('pages', 0)} pages, "
                       f"scanned: {result.get('is_scanned', False)}")
        
        # Ensure we have a proper dict with all required keys
        if not isinstance(result, dict):
            raise ValueError("Invalid PDF processing result")
            
        return {
            'text': result.get('text', ''),
            'title': result.get('metadata', {}).get('title', 
                   os.path.basename(source) if isinstance(source, str) else 'PDF Document'),
            'error': result.get('error'),
            'source': 'pdf',
            'pages': result.get('pages', 0),
            'is_scanned': result.get('is_scanned', False),
            'metadata': result.get('metadata', {})
        }
    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}"
        if debug:
            logging.error(error_msg, exc_info=True)
        return {
            'text': '',
            'title': 'Error Processing PDF',
            'error': error_msg,
            'source': 'pdf',
            'pages': 0,
            'is_scanned': False,
            'metadata': {}
        }


async def read_webpage(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    use_playwright: bool = False,
    user_agents: Optional[List[str]] = None,
    proxies: Optional[Dict[str, str]] = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Fetch and extract text content from a webpage or PDF with enhanced features.
    
    Args:
        url: The URL of the webpage/PDF to read
        timeout: Request timeout in seconds
        use_playwright: Whether to try Playwright for JS-heavy sites
        user_agents: List of user agents to rotate through
        proxies: Proxy configuration (e.g., {'http': 'http://proxy:port'})
        debug: Enable debug mode for detailed output
        
    Returns:
        Dict containing:
        - text: Extracted text content (empty string if failed)
        - title: Page title (empty string if not found)
        - error: Error message if any (None if successful)
        - source: Source of the content ('cache', 'direct', 'playwright', 'pdf')
        - cached: Whether the response was served from cache
        - debug_info: Additional debug information if debug=True
    """
    # Check if this is a PDF URL
    is_pdf = url.lower().endswith('.pdf') or 'application/pdf' in url
    
    # Initialize result
    result = {
        'text': '',
        'title': '',
        'error': None,
        'source': 'pdf' if is_pdf else 'direct',
        'cached': False,
        'debug_info': {}
    }
    
    # Handle PDF URLs
    if is_pdf and PDF_PROCESSOR_AVAILABLE:
        if debug:
            logging.info(f"Processing PDF URL: {url}")
        pdf_result = await process_pdf(url, is_url=True, debug=debug)
        result.update(pdf_result)
        return result
    
    # Generate cache key
    cache_key = get_cache_key(url)
    domain = get_domain(url)
    
    # Check cache first
    if cache_key in memory_cache:
        result.update(memory_cache[cache_key])
        result['cached'] = True
        result['source'] = 'cache'
        if debug:
            result['debug_info']['cache_hit'] = True
        return result
    
    # Check rate limiting
    if is_rate_limited(domain):
        result['error'] = f"Rate limited for domain: {domain}"
        return result
    
    # Check robots.txt
    allowed, robots_msg = check_robots_txt(url)
    if not allowed:
        result['error'] = f"Blocked by robots.txt: {robots_msg}"
        return result
    
    # Prepare request
    headers = {
        'User-Agent': user_agents[0] if user_agents and len(user_agents) > 0 else get_random_user_agent(),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',  # Do Not Track
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    # Try direct fetch first
    try:
        if debug:
            result['debug_info']['fetch_attempt'] = 'direct'
        
        response = requests.get(
            url,
            headers=headers,
            timeout=timeout,
            proxies=proxies,
            allow_redirects=True,
            stream=True
        )
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            result['error'] = f"Unsupported content type: {content_type}"
            return result
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'noscript', 'nav', 'footer', 'header', 'iframe', 'svg']):
            element.decompose()
        
        # Get page title
        title = soup.title.string.strip() if soup.title else ''
        
        # Extract and clean text
        text = '\n'.join(
            line.strip() for line in 
            soup.get_text(separator='\n', strip=True).splitlines() 
            if line.strip()
        )
        
        # Check if content seems valid
        if len(text) < MIN_CONTENT_LENGTH and not title:
            if use_playwright and PLAYWRIGHT_AVAILABLE:
                if debug:
                    result['debug_info']['fallback_reason'] = 'low_content_length'
                return await _try_playwright_fallback(url, timeout, result, debug)
            result['error'] = "Page may be paywalled, empty, or require JavaScript"
            return result
            
        # Truncate if needed
        if len(text) > 8000:
            truncated = text[:8000]
            last_space = truncated.rfind(' ')
            if last_space > 7500:
                text = truncated[:last_space] + '...'
            else:
                text = truncated + '...'
        
        # Update result
        result.update({
            'text': text,
            'title': title,
            'source': 'direct',
            'cached': False
        })
        
        # Cache successful response
        memory_cache[cache_key] = result.copy()
        return result
        
    except (requests.exceptions.RequestException, Exception) as e:
        if debug:
            result['debug_info']['error'] = str(e)
            result['debug_info']['error_type'] = e.__class__.__name__
        
        # Try Playwright if enabled and this looks like a JS-heavy site
        if use_playwright and PLAYWRIGHT_AVAILABLE:
            if debug:
                result['debug_info']['fallback_reason'] = f'direct_fetch_failed: {str(e)}'
            return await _try_playwright_fallback(url, timeout, result, debug)
            
        # Handle specific errors
        if isinstance(e, requests.exceptions.Timeout):
            result['error'] = f"Request timed out after {timeout} seconds"
        elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
            if e.response.status_code == 404:
                result['error'] = "Page not found (404)"
            else:
                result['error'] = f"HTTP {e.response.status_code} error"
        else:
            result['error'] = f"Failed to fetch page: {str(e)}"
            
        return result

async def _try_playwright_fallback(url: str, timeout: int, result: Dict[str, Any], debug: bool) -> Dict[str, Any]:
    """Try to fetch content using Playwright as a fallback."""
    if not PLAYWRIGHT_AVAILABLE:
        result['error'] = "Playwright not available for JavaScript rendering"
        return result
    
    try:
        content, title = await extract_with_playwright(url, timeout * 1000)  # Convert to ms
        
        if not content and not title:
            result['error'] = "Playwright returned empty content"
            return result
            
        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script, style, and other non-content elements
        for element in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript"]):
            element.decompose()
            
        # Get text content
        text = '\n'.join(p.get_text().strip() for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'article', 'section']))
        text = '\n'.join(line for line in text.split('\n') if line.strip())
        
        if not text.strip() and not title:
            result['error'] = "No readable content found on page"
            return result
            
        # Update result with Playwright data
        result.update({
            'text': text[:8000],
            'title': title or result.get('title', ''),
            'source': 'playwright',
            'cached': False
        })
        
        # Cache successful response
        cache_key = get_cache_key(url)
        memory_cache[cache_key] = result.copy()
        
        return result
        
    except Exception as e:
        if debug:
            result['debug_info']['playwright_error'] = str(e)
        result['error'] = f"Playwright fallback failed: {str(e)}"
        return result


def is_audio_file(filename: str, content_type: Optional[str] = None) -> bool:
    """Check if a file is an audio file based on extension and/or content type."""
    audio_extensions = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.opus', '.m4b', '.aac', '.wma', '.webm'}
    audio_content_types = {
        'audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/ogg', 'audio/webm',
        'audio/flac', 'audio/opus', 'audio/x-wav', 'audio/x-m4a', 'audio/aac',
        'audio/x-m4b', 'audio/x-ms-wma', 'audio/aacp', 'audio/mp4a-latm',
        'audio/mpeg3', 'audio/x-mpeg-3', 'audio/x-m4p', 'audio/x-m4b', 'audio/x-m4r'
    }
    
    # Check by extension
    ext = os.path.splitext(filename.lower())[1]
    if ext in audio_extensions:
        return True
        
    # Check by content type if provided
    if content_type and any(ct in content_type.lower() for ct in audio_content_types):
        return True
        
    return False

def is_video_file(filename: str, content_type: Optional[str] = None) -> bool:
    """Check if a file is a video file based on extension and/or content type."""
    video_extensions = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.wmv', '.flv', '.m4v'}
    video_content_types = {
        'video/mp4', 'video/mov', 'video/mkv', 'video/webm', 'video/avi', 'video/wmv', 'video/flv', 'video/m4v'
    }
    
    # Check by extension
    ext = os.path.splitext(filename.lower())[1]
    if ext in video_extensions:
        return True
        
    # Check by content type if provided
    if content_type and any(ct in content_type.lower() for ct in video_content_types):
        return True
        
    return False

def process_media_file(input_path: str, keep_failed: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """
    Process audio or video file for transcription.
    
    Args:
        input_path: Path to the input media file
        keep_failed: If True, keep failed conversion files for debugging
        
    Returns:
        Tuple of (output_path, error_message) where:
        - output_path: Path to the processed WAV file on success, None on failure.
                      The caller is responsible for cleaning up this file after use.
        - error_message: None on success, user-friendly error message on failure
    """
    output_path = None
    temp_files = []
    error_msg = None  # Initialize error_msg at function start
    
    try:
        if not os.path.exists(input_path):
            error_msg = f"Input file not found: {input_path}"
            logging.error(error_msg)
            return None, error_msg
            
        # First, probe the input file to check for audio streams
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a',  # Only check audio streams
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            input_path
        ]
        
        probe_result = subprocess.run(
            probe_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Check if there are any audio streams
        if not probe_result.stdout.strip():
            error_msg = "No audio stream found in the media file"
            logging.error(error_msg)
            return None, error_msg
        
        # Create a temporary output file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_out:
            output_path = temp_out.name
        
        # Build ffmpeg command with explicit audio stream selection and better error handling
        # Using a single atempo filter for 1.75x speed-up
        cmd = [
            'ffmpeg',
            '-y',                    # Overwrite output file if it exists
            '-i', input_path,        # Input file
            '-map', '0:a:0',         # Select first audio stream
            '-c:a', 'pcm_s16le',     # 16-bit PCM
            '-ar', '16000',          # 16kHz sample rate
            '-ac', '1',             # Mono
            '-af', 'atempo=1.75',    # 1.75x speed
            '-f', 'wav',            # Output format
            output_path
        ]
        
        # Run ffmpeg with timeout
        logging.info(f"Running FFmpeg command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Check if conversion was successful
        if result.returncode != 0:
            error_msg = (
                f"FFmpeg failed with return code {result.returncode}. "
                f"Error: {result.stderr[-500:] if result.stderr else 'No error output'}"
            )
            logging.error(f"Media processing failed: {error_msg}")
            return None, f"Failed to process media: {error_msg}"
            
        if not os.path.exists(output_path):
            error_msg = "FFmpeg did not create the output file"
            logging.error(error_msg)
            return None, error_msg
            
        if os.path.getsize(output_path) == 0:
            error_msg = "Output file is empty"
            logging.error(error_msg)
            return None, error_msg
            
        # On success, return the path to the output file - caller is responsible for cleanup
        return output_path, None
        
    except subprocess.TimeoutExpired:
        error_msg = "Media processing timed out (took longer than 5 minutes)"
        logging.error(error_msg)
        return None, error_msg
        
    except Exception as e:
        error_msg = f"Error processing media file: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return None, error_msg
        
    finally:
        # Only clean up temporary files if there was an error and we're not keeping failed files
        try:
            if (output_path is None or error_msg is not None) and not keep_failed:
                for temp_file in [f for f in temp_files if f and os.path.exists(f)]:
                    try:
                        os.unlink(temp_file)
                    except Exception as e:
                        logging.error(f"Error cleaning up temporary file {temp_file}: {e}")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
            # Don't raise, just log the cleanup error

def load_config():
    load_dotenv(override=True)
    return {
        "DISCORD_TOKEN": os.getenv('DISCORD_TOKEN'),
        "TEXT_BACKEND": os.getenv('TEXT_BACKEND', 'ollama'),  # 'ollama' or 'openai'
        "OLLAMA_BASE_URL": os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
        "TEXT_MODEL": os.getenv('TEXT_MODEL', 'qwen3-235b-a22b'),
        "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY'),
        "OPENAI_API_BASE": os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
        "OPENAI_TEXT_MODEL": os.getenv('OPENAI_TEXT_MODEL', 'gpt-4o'),
        "VL_MODEL": os.getenv('VL_MODEL', 'gpt-4-vision-preview'),
        "VL_PROMPT": load_vl_prompt(),
        "WHISPER_API_KEY": os.getenv('WHISPER_API_KEY'),
        "WHISPER_API_BASE": os.getenv('WHISPER_API_BASE'),
        "WHISPER_MODEL": os.getenv('WHISPER_MODEL', 'whisper-1'),
        "TEMPERATURE": float(os.getenv('TEMPERATURE', '0.7')),
        "TIMEOUT": float(os.getenv('TIMEOUT', '120.0')),
        "CHANGE_NICKNAME": os.getenv('CHANGE_NICKNAME', 'True').lower() == 'true',
        "MAX_CONVERSATION_LENGTH": int(os.getenv('MAX_CONVERSATION_LENGTH', '30')),
        "MAX_FILE_SIZE": int(os.getenv('MAX_FILE_SIZE', '20')) * 1024 * 1024,  # 20MB default for general files
        "MAX_AUDIO_SIZE": int(os.getenv('MAX_AUDIO_SIZE', '50')) * 1024 * 1024,  # 50MB default for audio files
        "MAX_TEXT_ATTACHMENT_SIZE": int(os.getenv('MAX_TEXT_ATTACHMENT_SIZE', '20000')),
        "SYSTEM_PROMPT": load_prompt_file(),
        "MAX_USER_MEMORY": int(os.getenv('MAX_USER_MEMORY', '10')),
        "MEMORY_SAVE_INTERVAL": int(os.getenv('MEMORY_SAVE_INTERVAL', '30')),
    }

USER_PROFILE_DIR = pathlib.Path("user_profiles"); USER_PROFILE_DIR.mkdir(exist_ok=True)
SERVER_PROFILE_DIR = pathlib.Path("server_profiles"); SERVER_PROFILE_DIR.mkdir(exist_ok=True)
USER_LOGS_DIR = pathlib.Path("user_logs"); USER_LOGS_DIR.mkdir(exist_ok=True)
DM_LOGS_DIR = pathlib.Path("dm_logs"); DM_LOGS_DIR.mkdir(exist_ok=True)
user_cache: Dict[str, dict] = {}
server_cache: Dict[str, dict] = {}

def default_server_profile(guild_id: Optional[str] = None) -> dict:
    """Create a new server profile with default values."""
    return {
        "guild_id": guild_id if guild_id else "",
        "memories": [],
        "history": [],
        "preferences": {},
        "context_notes": [],
        "total_messages": 0,
        "last_updated": datetime.now().isoformat()
    }

def ensure_server_profile_schema(profile: dict, guild_id: Optional[str] = None) -> dict:
    """Ensure a server profile has all required fields, adding any that are missing."""
    default = default_server_profile(guild_id)
    
    # If the profile is empty, return a fresh default
    if not profile:
        return default.copy()
        
    # Ensure all required fields exist with default values if missing
    for key, default_value in default.items():
        if key not in profile:
            profile[key] = default_value.copy() if hasattr(default_value, 'copy') else default_value
    
    # Remove any extra fields that shouldn't be there
    for key in list(profile.keys()):
        if key not in default:
            del profile[key]
    
    return profile

def get_server_profile(guild_id: Optional[str] = None) -> dict:
    """
    Get or create a server profile, ensuring it has all required fields.
    
    Args:
        guild_id: The Discord guild ID (server ID)
        
    Returns:
        dict: The server profile with all required fields
    """
    if not guild_id:
        return default_server_profile()
        
    # Return cached version if available
    if guild_id in server_cache:
        return server_cache[guild_id]
        
    file_path = SERVER_PROFILE_DIR / f"{guild_id}.json"
    
    try:
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                # Load and validate the profile
                profile = json.load(f)
                server_cache[guild_id] = ensure_server_profile_schema(profile, guild_id)
        else:
            # Create a new profile if none exists
            server_cache[guild_id] = default_server_profile(guild_id)
            save_server_profile(guild_id)
    except Exception as e:
        logging.error(f"Error loading server profile {guild_id}: {e}")
        server_cache[guild_id] = default_server_profile(guild_id)
    
    return server_cache[guild_id]

def save_server_profile(guild_id: str) -> None:
    """
    Save a server profile to disk, ensuring it has all required fields.
    
    Args:
        guild_id: The Discord guild ID (server ID)
    """
    if not guild_id:
        return
        
    profile = server_cache.get(guild_id)
    if not profile:
        return
        
    # Ensure the profile has all required fields before saving
    profile = ensure_server_profile_schema(profile, guild_id)
    
    # Update the last_updated timestamp
    profile["last_updated"] = datetime.now().isoformat()
    
    file_path = SERVER_PROFILE_DIR / f"{guild_id}.json"
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Error saving server profile {guild_id}: {e}")

def default_profile(user_id=None, username=None):
    return {
        "discord_id": user_id if user_id else "",
        "username": username if username else "",
        "history": [],
        "last_active": 0,
        "tone": "neutral",
        "memories": [],
        "total_messages": 0,
        "first_seen": datetime.now().isoformat(),
        "preferences": {},
        "context_notes": [],
        "last_memory_extraction": 0,
        "message_batches_processed": 0
    }

def get_profile(user_id: str, username: Optional[str] = None) -> dict:
    file_path = USER_PROFILE_DIR / f"{user_id}.json"
    if user_id in user_cache:
        profile = user_cache[user_id]
        if username and profile.get("username") != username:
            profile["username"] = username
            save_profile(user_id)
        return profile
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in default_profile(user_id, username).items():
            data.setdefault(k, v)
        if username and data.get("username") != username:
            data["username"] = username
        if data.get("discord_id") != user_id:
            data["discord_id"] = user_id
        user_cache[user_id] = data
        save_profile(user_id)
        return data
    else:
        data = default_profile(user_id, username)
        user_cache[user_id] = data
        save_profile(user_id)
        return data

def save_profile(user_id: str):
    file_path = USER_PROFILE_DIR / f"{user_id}.json"
    profile = user_cache.get(user_id)
    if not profile:
        return
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

def save_all_profiles():
    for user_id in list(user_cache.keys()):
        save_profile(user_id)
    for guild_id in list(server_cache.keys()):
        save_server_profile(guild_id)

# ---- Memory Extraction ----
def extract_memory_from_message(message_content, user_profile):
    content = message_content.lower()
    memories = []
    name_patterns = [r"my name is (\w+)", r"call me (\w+)", r"i'm (\w+)", r"i am (\w+)"]
    for pattern in name_patterns:
        match = re.search(pattern, content)
        if match:
            name = match.group(1).capitalize()
            memories.append(f"Preferred name: {name}")
    interest_patterns = [
        r"i love (\w+)", r"i like (\w+)", r"i enjoy (\w+)", r"i'm into (\w+)",
        r"i work as (?:a |an )?(\w+)", r"i'm (?:a |an )?(\w+) by profession",
        r"i study (\w+)", r"i'm studying (\w+)"
    ]
    for pattern in interest_patterns:
        match = re.search(pattern, content)
        if match:
            interest = match.group(1)
            memories.append(f"Interest/occupation: {interest}")
    if any(word in content for word in ["hate", "dislike", "don't like"]):
        for word in ["hate", "dislike", "don't like"]:
            if word in content:
                dislike_match = re.search(f"{word} (\\w+)", content)
                if dislike_match:
                    memories.append(f"Dislikes: {dislike_match.group(1)}")
    location_patterns = [r"i'm from (\w+)", r"i live in (\w+)", r"i'm in (\w+)"]
    for pattern in location_patterns:
        match = re.search(pattern, content)
        if match:
            location = match.group(1).capitalize()
            memories.append(f"Location: {location}")
    return memories

def add_memory(user_id: str, memory_text: str, guild_id: Optional[str] = None, username: Optional[str] = None) -> None:
    """
    Add a memory for a user and optionally to a server.
    
    Args:
        user_id: Discord user ID
        memory_text: The memory text to add
        guild_id: Optional server ID to also add the memory to
        username: Optional username for server memory context
    """
    config = load_config()
    
    # Add to user's personal memories
    profile = get_profile(user_id)
    
    # Check for duplicate memories (case-insensitive partial match)
    memory_lower = memory_text.lower()
    for existing_memory in profile["memories"]:
        if memory_lower in existing_memory.lower() or existing_memory.lower() in memory_lower:
            return
            
    timestamped_memory = f"[{datetime.now().strftime('%Y-%m-%d')}] {memory_text}"
    
    # Add to user's memories with limit enforcement
    profile["memories"].append(timestamped_memory)
    if len(profile["memories"]) > config["MAX_USER_MEMORY"]:
        profile["memories"] = profile["memories"][-config["MAX_USER_MEMORY"]:]
    save_profile(user_id)
    
    # Add to server memories if guild_id is provided
    if guild_id and username:
        server_profile = get_server_profile(guild_id)
        
        # Create server memory entry with user context
        server_memory = f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] [user: {user_id} | name: {username}] {memory_text}"
        
        # Add to server memories, avoiding duplicates
        if not any(server_memory.lower() in m.lower() for m in server_profile["memories"]):
            server_profile["memories"].append(server_memory)
            
            # Enforce server memory limit
            if len(server_profile["memories"]) > config["MAX_SERVER_MEMORY"]:
                server_profile["memories"] = server_profile["memories"][-config["MAX_SERVER_MEMORY"]:]
            
            # Update server stats
            server_profile["total_messages"] = server_profile.get("total_messages", 0) + 1
            server_profile["last_updated"] = datetime.now().isoformat()
            
            # Save the updated profile
            save_server_profile(guild_id)
    
    logging.info(f"Added memory for user {user_id} in server {guild_id or 'DM'}: {memory_text}")

def get_user_context(user_id):
    profile = get_profile(user_id)
    context_parts = []
    if profile["memories"]:
        context_parts.append(f"Important memories: {' | '.join(profile['memories'])}")
    if profile["preferences"]:
        prefs = [f"{k}: {v}" for k, v in profile["preferences"].items()]
        context_parts.append(f"User preferences: {' | '.join(prefs)}")
    if profile["context_notes"]:
        context_parts.append(f"Recent context: {' | '.join(profile['context_notes'])}")
    context_parts.append(f"Total messages: {profile['total_messages']}")
    days_known = (datetime.now() - datetime.fromisoformat(profile["first_seen"])).days
    if days_known > 0:
        context_parts.append(f"Known user for {days_known} days")
    return " | ".join(context_parts) if context_parts else "New user"

def detect_tone(msg):
    msg = msg.lower()
    if any(word in msg for word in ["fuck", "idiot", "dumb", "stupid"]): return "hostile"
    elif any(word in msg for word in ["thank", "pls", "please", "appreciate"]): return "friendly"
    elif any(word in msg for word in ["lol", "lmao", "rofl", "bro", "mate"]): return "banter"
    return "neutral"

def log_dm_message(message: Message):
    if not isinstance(message.channel, discord.DMChannel):
        return
    username = str(message.author).replace("#", "_")
    filename = DM_LOGS_DIR / f"{message.author.id}_{username}.jsonl"
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": str(message.author.id),
        "username": str(message.author),
        "content": message.content,
        "attachments": [
            {
                "filename": att.filename,
                "size": att.size,
                "url": att.url
            } for att in message.attachments
        ]
    }
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

def log_user_message(message: Message):
    user_id = str(message.author.id)
    log_path = USER_LOGS_DIR / f"{user_id}.jsonl"
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "channel_id": str(message.channel.id),
        "guild_id": str(message.guild.id) if message.guild else None,
        "content": message.content,
        "attachments": [
            {
                "filename": att.filename,
                "size": att.size,
                "url": att.url
            } for att in message.attachments
        ]
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

# ---- DuckDuckGo Search ----
async def ddg_search(query: str) -> str:
    url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get(url, params=params)
            data = r.json()
            if data.get("AbstractText"):
                return data["AbstractText"] + (f"\nSource: {data.get('AbstractURL','')}" if data.get("AbstractURL") else "")
            elif data.get("Answer"):
                return data["Answer"]
            elif data.get("Definition"):
                return data["Definition"]
            elif data.get("RelatedTopics"):
                topics = data["RelatedTopics"]
                if topics and "Text" in topics[0]:
                    return topics[0]["Text"]
            return "No relevant DuckDuckGo result found."
        except Exception as e:
            return f"Search failed: {e}"

async def should_auto_search(content: str) -> Optional[str]:
    content = content.lower().strip()
    patterns = [
        r"\bwho is\b", r"\bwho was\b", r"\bwhat is\b", r"\bwhat was\b",
        r"\bwhen did\b", r"\bhow many\b", r"\bhow much\b", r"\bhow long\b",
        r"\bcurrent\b.*\b(time|date|president|prime minister|exchange rate|price)\b",
        r"\b(latest|current|recent)\b", r"\bnews\b", r"\bmeaning of\b", r"\bdefine\b",
        r"\bcapital of\b", r"\bpopulation of\b", r"\bwho invented\b", r"\bwho wrote\b",
        r"\bhow do i\b",
    ]
    ignore_patterns = [r"\b(joke|opinion|should i|recommend)\b"]
    for pat in ignore_patterns:
        if re.search(pat, content):
            return None
    for pat in patterns:
        if re.search(pat, content):
            q = re.sub(r"^<@!?[0-9]+>", "", content).strip(" ,")
            qm = q.find("?")
            if qm != -1:
                q = q[:qm+1]
            return q
    return None

# ---- Context Management ----
conversation_store = defaultdict(list)
CONTEXT_TTL = 900
CONTEXT_MAXLEN = 30
last_active = {}

def context_key(message: Message):
    if isinstance(message.channel, discord.DMChannel):
        return (str(message.author.id), "DM")
    else:
        return (str(message.guild.id), str(message.channel.id), str(message.author.id))

def reset_context(message: Message, config=None):
    key = context_key(message)
    conversation_store[key].clear()
    if not config:
        config = load_config()
    conversation_store[key].append({'role': 'system', 'content': config["SYSTEM_PROMPT"]})

def get_context(message: Message):
    key = context_key(message)
    return conversation_store[key]

def update_last_active(message: Message):
    key = context_key(message)
    last_active[key] = time.time()

def should_reset_context(message: Message):
    key = context_key(message)
    now = time.time()
    if key not in last_active:
        return True
    if now - last_active[key] > CONTEXT_TTL:
        return True
    if len(conversation_store[key]) > CONTEXT_MAXLEN:
        return True
    return False

intents = Intents.default()
intents.message_content = True
intents.dm_messages = True
bot = commands.Bot(command_prefix='!', intents=intents)

def clean_response(text):
    return re.sub(r"</?[\w_]+>", "", text)

def is_text_file(file_content):
    try:
        file_content.decode('utf-8')
        return True
    except (UnicodeDecodeError, AttributeError):
        return False

async def send_in_chunks(channel, text, reference=None, chunk_size=2000):
    text = text.strip()
    for start in range(0, len(text), chunk_size):
        await channel.send(text[start:start + chunk_size], reference=reference if start == 0 else None)

def build_system_prompt(message, config):
    user_id = str(message.author.id)
    username = str(message.author)
    profile = get_profile(user_id, username)
    last_tone = profile['tone']
    history_count = len(profile['history'])
    user_context = get_user_context(user_id)
    guild_note = f"(Server: {message.guild.name})" if message.guild else "(Direct Message)"
    custom_blurb = (
        f"{guild_note} | Current user: {username} (id: {user_id}). "
        f"Messages: {history_count}, Last detected tone: {last_tone}.\n"
        f"User context: {user_context}\n"
        f"Adjust your attitude, banter, or respect as appropriate for this user's history and what you know about them."
    )
    return f"{config['SYSTEM_PROMPT']}\n{custom_blurb}"

def should_include_mention(message: discord.Message, response: str, context_len=1) -> bool:
    if isinstance(message.channel, discord.DMChannel): return False
    if context_len > 3: return False
    if len(response.split()) < 5: return False
    bot_mention = f"<@{bot.user.id}>"
    if bot_mention in message.content.lower(): return True
    if message.content.lower().startswith(bot.user.name.lower()): return True
    if any(marker in message.content.lower() for marker in ['?', 'what', 'when', 'where', 'why', 'how', 'who', 'can you']): return True
    return False

# ---- TEXT BACKEND ROUTING ----
async def ollama_chat(messages, config, model=None):
    base_url = config["OLLAMA_BASE_URL"]
    payload = {
        "model": model or config["TEXT_MODEL"],
        "messages": messages,
        "stream": False,
        "options": {"temperature": config["TEMPERATURE"]}
    }
    try:
        async with httpx.AsyncClient(timeout=config["TIMEOUT"]) as client:
            resp = await client.post(
                f"{base_url}/api/chat",
                json=payload,
                timeout=config["TIMEOUT"]
            )
            data = resp.json()
            if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
                return data["message"]["content"]
            if "messages" in data and isinstance(data["messages"], list):
                return " ".join([m["content"] for m in data["messages"]])
            return data.get("response", "No response from Ollama model.")
    except Exception as e:
        return f"Model error: {e}"

async def openai_chat(messages, config, model=None):
    api_key = config["OPENAI_API_KEY"]
    api_base = config.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    model = model or config.get("OPENAI_TEXT_MODEL", "gpt-4o")
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": config.get("TEMPERATURE", 0.7)
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=config["TIMEOUT"]) as client:
        resp = await client.post(f"{api_base}/chat/completions", json=payload, headers=headers)
        if resp.status_code != 200:
            return f"OpenAI error: {resp.status_code} {resp.text}"
        data = resp.json()
        return data["choices"][0]["message"]["content"]

# ---- Vision-Language Inference via OpenAI API ----
async def get_vision_caption(image_path, prompt, config):
    """
    Get a caption/description of the image using the vision-language model.
    
    Args:
        image_path: Path to the image file
        prompt: Optional user prompt, falls back to VL_PROMPT from config
        config: Configuration dictionary
        
    Returns:
        str: The generated caption/description or error message
    """
    import base64
    api_key = config["OPENAI_API_KEY"]
    api_base = config.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    model = config.get("VL_MODEL", "gpt-4-vision-preview")
    
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        logging.error(f"Error reading image file {image_path}: {e}")
        return None
        
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    
    # Use the VL-specific prompt if no user prompt is provided
    user_prompt = prompt or config.get("VL_PROMPT", "Describe this image in detail.")
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"}
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.1  # Lower temperature for more factual descriptions
    }
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{api_base}/chat/completions", 
                json=payload, 
                headers=headers
            )
            
            if resp.status_code != 200:
                try:
                    data = resp.json()
                    if data.get("error", {}).get("code") in (400, "data_inspection_failed"):
                        return "[Image contains content that cannot be processed]"
                    if data.get("error", {}).get("message"):
                        logging.error(f"VL API error: {data['error']['message']}")
                except Exception as e:
                    logging.error(f"Error parsing VL API error response: {e}")
                return None
                
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
            
    except Exception as e:
        logging.error(f"Error in VL API call: {e}")
        return None

# Kept for backward compatibility
async def openai_vl_inference(image_path, prompt, config):
    return await get_vision_caption(image_path, prompt, config)


# ---- Main Hybrid Handler ----
async def process_message(message: Message, is_dm: bool):
    config = load_config()
    key = context_key(message)
    if should_reset_context(message):
        reset_context(message, config)
    update_last_active(message)
    user_id = str(message.author.id)
    username = str(message.author)
    user_mention = message.author.mention
    profile = get_profile(user_id, username)
    profile['last_active'] = time.time()
    profile['history'].append(message.content)
    profile['tone'] = detect_tone(message.content)
    profile['total_messages'] = profile.get('total_messages', 0) + 1
    save_profile(user_id)
    context = get_context(message)

    # Web search
    search_match = re.search(r"\[search:(.*?)\]", message.content, re.I)
    auto_search_query = None
    if search_match:
        query = search_match.group(1).strip()
        search_result = await ddg_search(query)
        context.append({
            "role": "system",
            "content": f"DuckDuckGo search for '{query}':\n{search_result}"
        })
    else:
        auto_search_query = await should_auto_search(message.content)
        if auto_search_query:
            search_result = await ddg_search(auto_search_query)
            if search_result and not search_result.lower().startswith("no relevant"):
                context.append({
                    "role": "system",
                    "content": f"DuckDuckGo search for '{auto_search_query}':\n{search_result}"
                })
    # Memory extraction
    new_memories = extract_memory_from_message(message.content, profile)
    for memory in new_memories:
        add_memory(user_id, memory)
    user_context = get_user_context(user_id)
    system_prompt_with_user = build_system_prompt(message, config)
    
    # Add RAG-retrieved knowledge if available
    if message.content.strip():
        relevant_snippets = knowledge_base.get_relevant_snippets(message.content)
        if relevant_snippets:
            knowledge_text = "\n".join(
                f"- {snippet.text} (from {snippet.source_file}:{snippet.line_number})"
                for snippet in relevant_snippets
            )
            system_prompt_with_user += "\n\nRetrieved knowledge:\n" + knowledge_text
    
    # Update context with user context and system prompt
    if user_context:
        context.append({"role": "system", "content": user_context.strip()})
    context[0]['content'] = system_prompt_with_user.strip()

    # Initialize total_text_content with the message content
    total_text_content = message.content.strip()
    
    # Extract and process URLs in the message
    urls = [word for word in message.content.split() if is_url(word)]
    if urls:
        for url in urls:
            try:
                # Notify user that we're processing the URL
                processing_msg = await message.channel.send(f"🌐 Reading webpage: {url}")
                
                # Read the webpage content with error handling
                result = await read_webpage(url, use_playwright=True, debug=True)
                
                if result.get('error'):
                    # If there was an error, notify the user
                    error_msg = result.get('error', 'Unknown error')
                    await processing_msg.edit(content=f"❌ Error reading {url}: {error_msg}")
                    logging.error(f"Error processing URL {url}: {error_msg}")
                    continue
                    
                # Extract content and title from the result
                content = result.get('text', '')
                title = result.get('title', 'No title')
                source = result.get('source', 'webpage')
                
                # Format the content for the context
                url_content = f"[Webpage: {title}]\n{content}"
                
                # Add to context as system message
                context.append({
                    "role": "system",
                    "content": f"User shared a {source} page. Content:\n{url_content}"
                })
                
                # Update the processing message with success
                await processing_msg.edit(content=f"✅ Processed: {title}")
                
            except Exception as e:
                logging.error(f"Error processing URL {url}", exc_info=True)
                try:
                    await processing_msg.edit(content=f"❌ Error processing {url}: {str(e)}")
                except:
                    await message.channel.send(f"❌ Error processing {url}: {str(e)}")
    
    # Separate attachments by type
    image_attachments = []
    media_attachments = []  # Combined audio and video attachments
    pdf_attachments = []
    text_attachments = []
    other_attachments = []
    
    for att in message.attachments:
        # Get content type and filename
        content_type = getattr(att, 'content_type', '')
        filename = getattr(att, 'filename', '').lower()
        
        # Check if it's a media file (audio or video)
        if is_audio_file(filename, content_type) or is_video_file(filename, content_type):
            media_attachments.append(att)
        # Check if it's a PDF
        elif (content_type == 'application/pdf' or filename.endswith('.pdf')):
            pdf_attachments.append(att)
        # Check if it's an image
        elif (content_type and content_type.startswith('image/')) or \
             (filename and any(ext in filename for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp'])):
            image_attachments.append(att)
        # Check if it's a text file
        elif (content_type and content_type.startswith('text/')) or \
             (filename and any(ext in filename for ext in ['.txt', '.md', '.log'])):
            text_attachments.append(att)
        else:
            other_attachments.append(att)
        
    # Process media attachments if any (audio or video)
    if media_attachments:
        if len(media_attachments) > 1:
            await message.channel.send("I'll process the first media file. Please send other files one at a time.")
        
        # Process the first media attachment (audio or video)
        media_attachment = media_attachments[0]
        is_video = is_video_file(media_attachment.filename, getattr(media_attachment, 'content_type', ''))
        media_type = "video" if is_video else "audio"
        temp_path = None
        
        try:
            # Create a temporary file to store the downloaded media
            with tempfile.NamedTemporaryFile(delete=False, 
                                          suffix=os.path.splitext(media_attachment.filename)[1]) as temp_media:
                temp_path = temp_media.name
                
                # Download the media file with timeout
                media_bytes = await asyncio.wait_for(media_attachment.read(), timeout=120.0)
                temp_media.write(media_bytes)
                
            # Check file size after downloading
            if media_attachment.size < 1024:  # 1KB minimum
                await message.channel.send(f"The {media_type} file is too small to process. Please try with a different file.")
                return
                
            if media_attachment.size > config["MAX_FILE_SIZE"]:
                await message.channel.send(
                    f"The {media_type} file is too large (max {config['MAX_FILE_SIZE'] // (1024*1024)}MB). "
                    "Please send a smaller file."
                )
                return
            
            # Send a processing message
            processing_msg = await message.channel.send(f"🔊 Processing {media_type}... This may take a moment...")
            
            # Process the media file (convert to WAV, speed up, etc.)
            processed_path, error_msg = process_media_file(temp_path, config.get("DEBUG", False))
            
            if not processed_path or error_msg:
                await processing_msg.edit(content=f"❌ Error processing {media_type}: {error_msg}")
                return
            
            # Transcribe the processed audio
            transcript, error_msg = await transcribe_audio(
                processed_path, 
                config, 
                original_filename=media_attachment.filename,
                keep_converted=config.get("DEBUG", False)
            )
            
            # Clean up the processed file if not in debug mode
            if not config.get("DEBUG", False) and os.path.exists(processed_path):
                os.unlink(processed_path)
            
            if transcript:
                # Add the transcription to the total text content
                transcript_prefix = f"🎥 Video transcription" if is_video else "🎤 Audio transcription"
                
                if total_text_content:
                    total_text_content = f"{total_text_content}\n\n{transcript_prefix}: {transcript}"
                else:
                    total_text_content = f"{transcript_prefix}: {transcript}"
                
                # Update the processing message to show completion
                try:
                    await processing_msg.edit(content=f"✅ {media_type.capitalize()} transcription complete!")
                except Exception as e:
                    logging.error(f"Failed to update processing message: {e}")
                
                logging.info(f"Successfully transcribed {media_type} message (length: {len(transcript)} chars)")
            else:
                user_error = f"Could not transcribe {media_type}: {error_msg or 'Unknown error'}"
                logging.error(f"Transcription failed: {user_error}")
                await processing_msg.edit(content=f"❌ {user_error}")
        
        except asyncio.TimeoutError:
            user_error = f"The {media_type} file took too long to process. Please try again with a smaller file."
            await message.channel.send(user_error)
            logging.error(f"{media_type.capitalize()} processing timed out: {media_attachment.filename}")
        
        except Exception as e:
            user_error = f"An error occurred while processing the {media_type}: {str(e)}"
            logging.error(f"Error processing {media_type}: {user_error}", exc_info=True)
            await message.channel.send(f"❌ {user_error}")
        
        finally:
            # Clean up the temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logging.error(f"Error cleaning up temporary file {temp_path}: {e}")
    
    # Process image attachments if any
    temp_img_path = None
    if image_attachments:
        if len(image_attachments) > 1:
            await message.channel.send("I'll analyze the first image you sent. Please send other images one at a time for analysis.")
        
        # Process the first image
        img_attachment = image_attachments[0]
        
        try:
            # Create a temporary file with proper cleanup
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(img_attachment.filename)[1]) as temp_img:
                temp_img_path = temp_img.name
                
                # Download the image with timeout
                img_bytes = await asyncio.wait_for(img_attachment.read(), timeout=30.0)
                temp_img.write(img_bytes)
                
            # Check file size after downloading
            if img_attachment.size > config["MAX_FILE_SIZE"]:
                await message.channel.send(
                    f"The image is too large (max {config['MAX_FILE_SIZE'] // (1024*1024)}MB). "
                    "Please send a smaller image."
                )
                return
            
            # Get MIME type for the downloaded file
            mime_type, _ = mimetypes.guess_type(img_attachment.filename)
            if not mime_type or not mime_type.startswith('image/'):
                mime_type = 'application/octet-stream'
            
            # Always use the VL model for image analysis
            try:
                # Get the VL prompt from config (loaded from file via VL_PROMPT_FILE)
                vl_prompt = config.get("VL_PROMPT", "Describe this image in detail.")
                vision_caption = await get_vision_caption(temp_img_path, vl_prompt, config)
                
                if not vision_caption:
                    logging.error("VL model returned empty caption")
                    # Continue with original message content if VL fails
                    vision_caption = "[Image processing failed]"
                
                # Format the message content based on whether there was original text
                if total_text_content:
                    total_text_content = f"{total_text_content}\n\n[Image description: {vision_caption}]"
                else:
                    total_text_content = f"[Image description: {vision_caption}]"
                
                # Add to context with MIME type information
                context.append({
                    "role": "user",
                    "content": total_text_content,
                    "mime_type": mime_type,
                    "size_bytes": img_attachment.size
                })
                
            except Exception as e:
                # Log the error but don't show it to the user
                logging.error(f"Error in vision-language model: {e}", exc_info=True)
                # Continue with original message content if VL processing fails
                if not total_text_content:
                    total_text_content = "[Image processing failed]"
        
        except asyncio.TimeoutError:
            await message.channel.send("Sorry, the image took too long to download. Please try again.")
            return
        except Exception as e:
            logging.error(f"Error downloading image: {e}")
            await message.channel.send("Sorry, I couldn't process that image. Please try another one.")
            return
        
        finally:
            # Clean up the temporary file
            if temp_img_path and os.path.exists(temp_img_path):
                try:
                    os.unlink(temp_img_path)
                except Exception as e:
                    logging.error(f"Error cleaning up temporary file {temp_img_path}: {e}")
    
    # Log warning for unprocessed attachments
    for att in other_attachments:
        logging.warning(f"Unprocessed attachment type: {att.filename} (content_type: {getattr(att, 'content_type', 'unknown')})")
    
    # Process image attachments if any
    temp_img_path = None
    if image_attachments:
        # Notify if multiple images were attached
        if len(image_attachments) > 1:
            await message.channel.send("I'll analyze the first image you sent. Please send other images one at a time for analysis.")
        
        # Process the first image
        img_attachment = image_attachments[0]
        
        # Create a temporary file with proper cleanup
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(img_attachment.filename)[1]) as temp_img:
            temp_img_path = temp_img.name
            
            try:
                # Download the image with timeout
                img_bytes = await asyncio.wait_for(img_attachment.read(), timeout=30.0)
                temp_img.write(img_bytes)
                
                # Check file size after downloading
                if img_attachment.size > config["MAX_FILE_SIZE"]:
                    await message.channel.send(f"The image is too large (max {config['MAX_FILE_SIZE'] // (1024*1024)}MB). Please send a smaller image.")
                    return
                    
                # Get MIME type for the downloaded file
                mime_type, _ = mimetypes.guess_type(img_attachment.filename)
                if not mime_type or not mime_type.startswith('image/'):
                    mime_type = 'application/octet-stream'
                
                # Always use the VL model for image analysis
                try:
                    # Get the VL prompt from config (loaded from file via VL_PROMPT_FILE)
                    vl_prompt = config.get("VL_PROMPT", "Describe this image in detail.")
                    vision_caption = await get_vision_caption(temp_img_path, vl_prompt, config)
                    
                    if not vision_caption:
                        logging.error("VL model returned empty caption")
                        # Continue with original message content if VL fails
                        vision_caption = "[Image processing failed]"
                    
                    # Format the message content based on whether there was original text
                    original_content = message.content.strip()
                    if original_content:
                        message.content = f"{original_content}\n[Image description: {vision_caption}]"
                    else:
                        message.content = f"[Image description: {vision_caption}]"
                    
                    # Add to context with MIME type information
                    context.append({
                        "role": "user",
                        "content": message.content,  # Use the formatted message content
                        "mime_type": mime_type,
                        "size_bytes": img_attachment.size
                    })
                    
                except Exception as e:
                    # Log the error but don't show it to the user
                    logging.error(f"Error in vision-language model: {e}", exc_info=True)
                    # Continue with original message content if VL processing fails
                    if not message.content.strip():
                        message.content = "[Image processing failed]"
                
            except asyncio.TimeoutError:
                await message.channel.send("Sorry, the image took too long to download. Please try again.")
                return
            except Exception as e:
                logging.error(f"Error downloading image: {e}")
                await message.channel.send("Sorry, I couldn't process that image. Please try another one.")
                return
            
    # Clean up temporary files after all processing
    def safe_remove(filepath):
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                logging.error(f"Error removing temporary file {filepath}: {e}")
    
    # Clean up image file if it exists
    if 'temp_img_path' in locals() and temp_img_path:
        safe_remove(temp_img_path)
    
    # Process PDF attachments if any
    if pdf_attachments:
        if len(pdf_attachments) > 1:
            await message.channel.send("I'll process the first PDF file. Please send other files one at a time.")
        
        # Process the first PDF attachment
        pdf_attachment = pdf_attachments[0]
        temp_pdf_path = None
        
        try:
            # Create a temporary file to store the downloaded PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                temp_pdf_path = temp_pdf.name
                
                # Download the PDF file with timeout
                pdf_bytes = await asyncio.wait_for(pdf_attachment.read(), timeout=120.0)
                temp_pdf.write(pdf_bytes)
                
            # Check file size after downloading
            if pdf_attachment.size < 1024:  # 1KB minimum
                await message.channel.send("The PDF file is too small to process. Please try with a different file.")
                return
                
            if pdf_attachment.size > config["MAX_FILE_SIZE"]:
                await message.channel.send(
                    f"The PDF file is too large (max {config['MAX_FILE_SIZE'] // (1024*1024)}MB). "
                    "Please send a smaller file."
                )
                return
            
            # Send a processing message
            processing_msg = await message.channel.send("📄 Processing PDF... This may take a moment...")
            
            # Process the PDF file
            pdf_result = await process_pdf(temp_pdf_path, debug=config.get("DEBUG", False))
            
            if pdf_result.get('error'):
                await processing_msg.edit(content=f"❌ Error processing PDF: {pdf_result['error']}")
                return
                
            # Add the PDF content to the total text
            pdf_title = pdf_result.get('title', 'PDF Document')
            pdf_pages = pdf_result.get('pages', 0)
            is_scanned = pdf_result.get('is_scanned', False)
            
            pdf_summary = (
                f"📄 **{pdf_title}** ({pdf_pages} page{'s' if pdf_pages != 1 else ''})\n"
                f"*This is a {'scanned ' if is_scanned else ''}PDF document*\n\n"
                f"{pdf_result.get('text', '')}"
            )
            
            # Truncate if too long
            if len(pdf_summary) > 1500:
                pdf_summary = pdf_summary[:1497] + "..."
                
            if total_text_content:
                total_text_content = f"{total_text_content}\n\n{'-'*20}\n{pdf_summary}"
            else:
                total_text_content = pdf_summary
                
            # Update the processing message to show completion
            try:
                await processing_msg.edit(content=f"✅ PDF processing complete! Extracted {pdf_pages} pages.")
            except Exception as e:
                logging.error(f"Failed to update processing message: {e}")
            
            logging.info(f"Successfully processed PDF: {pdf_title} ({pdf_pages} pages, {len(pdf_result.get('text', ''))} chars)")
            
        except asyncio.TimeoutError:
            await message.channel.send("The PDF file took too long to process. Please try again with a smaller file.")
            logging.error(f"PDF processing timed out: {pdf_attachment.filename}")
            
        except Exception as e:
            error_msg = f"An error occurred while processing the PDF: {str(e)}"
            logging.error(f"Error processing PDF: {error_msg}", exc_info=True)
            await message.channel.send(f"❌ {error_msg}")
            
        finally:
            # Clean up the temporary file
            if temp_pdf_path and os.path.exists(temp_pdf_path):
                try:
                    os.unlink(temp_pdf_path)
                except Exception as e:
                    logging.error(f"Error cleaning up temporary PDF file {temp_pdf_path}: {e}")
    
    # Process text attachments if any
    if text_attachments:
        for attachment in text_attachments:
            if attachment.size > config["MAX_FILE_SIZE"]:
                await message.channel.send(
                    f"File {attachment.filename} is too big. Maximum size is {config['MAX_FILE_SIZE'] // (1024 * 1024)} MB."
                )
                return
            try:
                file_content = await attachment.read()
                if not is_text_file(file_content):
                    await message.channel.send(f"{attachment.filename} isn't a text file.")
                    return
                file_text = file_content.decode('utf-8')
                total_text_content += f"\n\n{attachment.filename}\n{file_text}"
                
                if len(total_text_content) > config["MAX_TEXT_ATTACHMENT_SIZE"]:
                    await message.channel.send("❌ Total attachment size exceeds the maximum allowed.")
                    return
                    
            except Exception as e:
                logging.error(f"Error reading attachment {attachment.filename}: {e}")
                await message.channel.send(f"Error reading file {attachment.filename}")
                return
    
    # Update message content with any processed text from attachments and transcriptions
    if total_text_content:
        message.content = total_text_content.strip()
    
    # Add the final message content to the context
    if message.content.strip():
        context.append({'role': 'user', 'content': message.content.strip()})
    async with message.channel.typing():
        messages = [
            {"role": msg['role'], "content": msg['content']}
            for msg in context
            if msg['role'] in ('system', 'user', 'assistant')
        ]
        if config["TEXT_BACKEND"] == "openai":
            response = await openai_chat(messages, config)
        else:
            response = await ollama_chat(messages, config, model=config["TEXT_MODEL"])
    if response:
        response = clean_response(response).strip()
        if not is_dm and should_include_mention(message, response, len(context)):
            response = f"{user_mention} {response}"
            
        # Send the response with TTS if enabled
        if isinstance(message.channel, discord.DMChannel) or message.guild is None:
            # In DMs, just send the response directly
            await handle_tts_reply(message.channel, response, author_id=str(message.author.id))
        else:
            # In guild, send with reference to the original message
            await handle_tts_reply(
                message.channel, 
                response, 
                reference=message,
                author_id=str(message.author.id)
            )
            
        context.append({"role": "assistant", "content": response.strip()})
        if len(context) > CONTEXT_MAXLEN:
            system_prompt = context[0]
            context.clear()
            context.append(system_prompt)
        if random.random() < 0.1:
            save_all_profiles()

# ---- TTS Preferences Storage ----
tts_prefs = {
    'global_enabled': True,  # Global TTS toggle (admins can change this)
    'users': {}  # user_id -> bool (whether TTS is enabled for them)
}

# File to persist TTS preferences
TTS_PREFS_FILE = 'tts_prefs.json'

def load_tts_prefs():
    """Load TTS preferences from JSON file."""
    try:
        if os.path.exists(TTS_PREFS_FILE):
            with open(TTS_PREFS_FILE, 'r') as f:
                prefs = json.load(f)
                tts_prefs.update(prefs)
    except Exception as e:
        logging.error(f"Error loading TTS preferences: {e}")

async def save_tts_prefs():
    """Save TTS preferences to JSON file."""
    try:
        with open(TTS_PREFS_FILE, 'w') as f:
            json.dump(tts_prefs, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving TTS preferences: {e}")

def is_tts_enabled(user_id: str) -> bool:
    """Check if TTS is enabled for a user."""
    if not tts_prefs['global_enabled']:
        return False
    return tts_prefs['users'].get(str(user_id), False)

def set_tts_enabled(user_id: str, enabled: bool) -> None:
    """Set TTS preference for a user."""
    tts_prefs['users'][str(user_id)] = enabled
    asyncio.create_task(save_tts_prefs())

def set_global_tts(enabled: bool) -> None:
    """Set global TTS setting."""
    tts_prefs['global_enabled'] = enabled
    asyncio.create_task(save_tts_prefs())

# Load preferences when module loads
load_tts_prefs()

# ---- TTS Functions ----
async def synthesize_speech(text: str) -> Tuple[Optional[Path], Optional[str]]:
    """
    Synthesize speech from text using DIA TTS.
    
    Args:
        text: Text to synthesize
        
    Returns:
        Tuple of (output_path, error_message) where:
        - output_path: Path to the generated WAV file on success, None on failure
        - error_message: None on success, error message on failure
    """
    if not DIA_AVAILABLE:
        return None, "DIA TTS is not installed. Please install it with: pip install TTS"
    
    if not text or not text.strip():
        return None, "No text provided for TTS"
        
    # Limit text length to prevent abuse
    if len(text) > 1000:
        return None, "Text too long (max 1000 characters)"
    
    temp_dir = Path(tempfile.gettempdir()) / "discord_tts"
    temp_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # Generate a unique filename
        timestamp = int(time.time())
        output_path = temp_dir / f"tts_{timestamp}.wav"
        
        # Initialize TTS with DIA model
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
        
        # Run TTS in a thread pool since it's blocking
        def _synthesize():
            tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker_wav=None,  # Use default voice
            )
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _synthesize)
        
        if not output_path.exists() or output_path.stat().st_size == 0:
            return None, "Failed to generate TTS audio"
            
        return output_path, None
        
    except Exception as e:
        error_msg = f"TTS synthesis error: {str(e)}"
        logging.error(f"TTS synthesis failed: {error_msg}\n{traceback.format_exc()}")
        return None, error_msg

# ---- Commands ----

@bot.command(name='say')
async def say_tts(ctx, *, message: str):
    """
    Convert text to speech using DIA TTS and send as an audio file.
    Usage: !say [your message]
    """
    await handle_tts_reply(ctx, message)

@bot.command(name='tts')
async def tts_toggle(ctx, setting: str = None):
    """
    Toggle TTS for your messages.
    Usage: !tts [on|off]  (or just !tts to toggle)
    """
    user_id = str(ctx.author.id)
    current_setting = is_tts_enabled(user_id)
    
    # Determine new setting
    if setting is None:
        new_setting = not current_setting
    else:
        new_setting = setting.lower() in ('on', 'true', 'enable', '1')
    
    set_tts_enabled(user_id, new_setting)
    
    if new_setting:
        await ctx.reply("✅ TTS enabled for your messages! I'll read my replies to you aloud.", mention_author=False)
    else:
        await ctx.reply("🔇 TTS disabled for your messages.", mention_author=False)

@bot.command(name='tts-all')
@commands.has_permissions(administrator=True)
async def tts_global_toggle(ctx, setting: str):
    """
    [Admin] Enable/disable TTS for all users.
    Usage: !tts-all [on|off]
    """
    if setting.lower() in ('on', 'true', 'enable', '1'):
        set_global_tts(True)
        await ctx.reply("🌍 TTS enabled globally for all users who have it enabled.", mention_author=False)
    else:
        set_global_tts(False)
        await ctx.reply("🌍 TTS disabled globally for all users.", mention_author=False)

@tts_global_toggle.error
async def tts_global_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.reply("❌ You don't have permission to use this command.", mention_author=False)
    else:
        await ctx.reply(f"❌ Error: {str(error)}", mention_author=False)

async def handle_tts_reply(target, text: str, author_id: str = None, **kwargs):
    """
    Send a message with optional TTS if enabled for the user.
    
    Args:
        target: Discord channel or context object to send the message to
        text: Text to send (and potentially speak)
        author_id: ID of the user who should receive TTS (if enabled)
        **kwargs: Additional arguments to pass to the send method
    """
    # Get the author ID if not provided
    if author_id is None and hasattr(target, 'author'):
        author_id = str(target.author.id)
    
    # Send the text message
    try:
        # For channel objects (TextChannel, DMChannel, etc.)
        if hasattr(target, 'send'):
            # Handle reference if provided
            if 'reference' in kwargs and hasattr(kwargs['reference'], 'to_reference'):
                kwargs['reference'] = kwargs['reference'].to_reference()
            reply = await target.send(text, **{k: v for k, v in kwargs.items() if k != 'author_id'})
        else:
            logging.error("Target does not have a send method")
            return None
            
    except Exception as e:
        logging.error(f"Failed to send message: {e}")
        return None
    
    # Check if we should send TTS
    if not DIA_AVAILABLE or not author_id or not is_tts_enabled(author_id):
        return reply
    
    # Generate and send TTS in the background
    asyncio.create_task(send_tts_audio(target, text))
    return reply

async def send_tts_audio(target, text: str):
    """
    Generate and send TTS audio for a message.
    
    Args:
        target: Discord channel or context object to send the TTS to
        text: Text to convert to speech
    """
    output_path = None
    try:
        # Generate TTS audio
        output_path, error = await synthesize_speech(text)
        
        if error or not output_path:
            logging.warning(f"TTS synthesis failed: {error}")
            return
            
        # Send the audio file
        with open(output_path, 'rb') as audio_file:
            await target.send(
                "🔊 TTS:",
                file=discord.File(audio_file, filename="tts_message.wav"),
                mention_author=False
            )
    except Exception as e:
        logging.error(f"Error in TTS audio generation: {e}\n{traceback.format_exc()}")
    finally:
        # Clean up the temp file
        if output_path and isinstance(output_path, Path) and output_path.exists():
            try:
                output_path.unlink()
            except Exception as e:
                logging.error(f"Failed to clean up TTS temp file: {e}")

@bot.command(name='memories')
async def show_memories(ctx):
    """Show all your memories."""
    user_id = str(ctx.author.id)
    profile = get_profile(user_id)
    memories = profile["memories"]
    
    if not memories:
        await ctx.reply("You don't have any memories yet!")
        return
        
    memory_text = "\n".join(f"• {mem}" for mem in memories)
    
    # Split into chunks if too long
    if len(memory_text) > 1900:
        chunks = [memory_text[i:i+1900] for i in range(0, len(memory_text), 1900)]
        for i, chunk in enumerate(chunks, 1):
            await ctx.send(f"**Your memories (part {i}/{len(chunks)}):**\n```\n{chunk}\n```")
    else:
        await ctx.reply(f"**Your memories:**\n```\n{memory_text}\n```")

@bot.command(name='servermemories')
async def show_server_memories(ctx):
    """Show all memories in this server."""
    if not ctx.guild:
        await ctx.reply("This command can only be used in a server!")
        return
        
    guild_id = str(ctx.guild.id)
    server_profile = get_server_profile(guild_id)
    memories = server_profile["memories"]
    
    if not memories:
        await ctx.reply("This server doesn't have any memories yet!")
        return
        
    memory_text = "\n".join(f"• {mem}" for mem in memories)
    
    # Split into chunks if too long
    if len(memory_text) > 1900:
        chunks = [memory_text[i:i+1900] for i in range(0, len(memory_text), 1900)]
        for i, chunk in enumerate(chunks, 1):
            await ctx.send(f"**Server memories (part {i}/{len(chunks)}):**\n```\n{chunk}\n```")
    else:
        await ctx.reply(f"**Server memories:**\n```\n{memory_text}\n```")

@bot.command(name='clearservermemories')
@commands.has_permissions(administrator=True)
async def clear_server_memories(ctx):
    """[Admin] Clear all memories in this server."""
    if not ctx.guild:
        await ctx.reply("This command can only be used in a server!")
        return
        
    guild_id = str(ctx.guild.id)
    server_profile = get_server_profile(guild_id)
    count = len(server_profile["memories"])
    server_profile["memories"] = []
    server_profile["last_updated"] = datetime.now().isoformat()
    save_server_profile(guild_id)
    
    await ctx.reply(f"✅ Cleared {count} memories from this server.")

@bot.command(name='remember')
async def manual_memory(ctx, *, memory_text):
    """Manually add a memory."""
    user_id = str(ctx.author.id)
    username = ctx.author.display_name
    guild_id = str(ctx.guild.id) if ctx.guild else None
    
    add_memory(user_id, memory_text, guild_id, username)
    save_profile(user_id)
    await ctx.reply("✅ Memory added!", mention_author=False)

@bot.command(name='preference')
async def set_preference(ctx, key, *, value):
    user_id = str(ctx.author.id)
    profile = get_profile(user_id, str(ctx.author))
    profile["preferences"][key] = value
    save_profile(user_id)
    await ctx.send(f"✅ Set preference `{key}` to `{value}`")

@bot.command(name='forget')
async def forget_user(ctx, user_mention=None):
    if user_mention:
        if not ctx.author.guild_permissions.administrator:
            await ctx.send("Only administrators can forget other users' memories.")
            return
        user_id = user_mention.strip('<@!>').strip('<@>')
    else:
        user_id = str(ctx.author.id)
    profile = get_profile(user_id)
    profile["memories"] = []
    profile["preferences"] = {}
    profile["context_notes"] = []
    save_profile(user_id)
    await ctx.send(f"Forgotten all memories for user <@{user_id}>")

@bot.command(name='reset')
async def reset(ctx):
    reset_context(ctx.message, load_config())
    await ctx.send("Conversation context has been reset (hot config).")

@bot.command(name='search')
async def cmd_search(ctx, *, query):
    search_result = await ddg_search(query)
    await send_in_chunks(ctx.channel, search_result)

async def change_nickname(guild):
    config = load_config()
    nickname = config["TEXT_MODEL"].split('/')[-1].split(':')[0].capitalize()
    try:
        await guild.me.edit(nick=nickname)
        logging.info(f"Nickname changed to {nickname} in guild {guild.name}")
    except discord.Forbidden:
        logging.warning(f"No permission to change nickname in guild {guild.name}")
    except Exception as e:
        logging.error(f"Failed to change nickname in guild {guild.name}: {str(e)}")

@bot.event
async def on_ready():
    config = load_config()
    logging.info(f'{bot.user.name} is now running! (hot config)')
    asyncio.create_task(periodic_save())
    if config["CHANGE_NICKNAME"]:
        for guild in bot.guilds:
            await change_nickname(guild)

async def periodic_save():
    config = load_config()
    while True:
        await asyncio.sleep(config["MEMORY_SAVE_INTERVAL"])
        save_all_profiles()

@bot.event
async def on_message(message: Message):
    try:
        if message.author.bot:
            return
        log_user_message(message)
        if isinstance(message.channel, discord.DMChannel):
            log_dm_message(message)
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_guild_mention = (not is_dm and hasattr(message, 'mentions') and bot.user in message.mentions)
        # If command: process commands (even in DMs!)
        if message.content.startswith('!'):
            await bot.process_commands(message)
            return
        # If DM or mention, process as a chat message (but also process commands)
        if is_dm or is_guild_mention:
            await process_message(message, is_dm)
        # Always call process_commands (in case a command is hidden in normal text)
        await bot.process_commands(message)
    except Exception as e:
        logging.error(f"Discord event error in on_message: {e}", exc_info=True)


@bot.event
async def on_error(event, *args, **kwargs):
    logging.error(f'Discord event error in {event}: {args}')

def main():
    config = load_config()
    if not config["DISCORD_TOKEN"]:
        raise ValueError("DISCORD_TOKEN not set in .env!")
    try:
        bot.run(config["DISCORD_TOKEN"])
    except discord.LoginFailure:
        logging.error("Invalid Discord token!")
    except Exception as e:
        logging.error(f"Failed to start bot: {e}")
    finally:
        save_all_profiles()
        logging.info("Bot shutdown complete.")

if __name__ == '__main__':
    main()
