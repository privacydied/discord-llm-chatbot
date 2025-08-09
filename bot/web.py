"""
Web content extraction and processing for the Discord bot.
"""
import logging
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse, urljoin
import httpx
from bs4 import BeautifulSoup
import discord
import trafilatura
from trafilatura.settings import use_config

# Import bot modules
from .config import load_config

# Load configuration
config = load_config()

# Configure trafilatura for better content extraction
trafilatura_config = use_config()
trafilatura_config.set("DEFAULT", "EXTRACTION_TIMEOUT", "10")  # 10 second timeout

# User agent for web requests
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# Domain-specific extraction rules
DOMAIN_RULES = {
    'youtube.com': {'type': 'youtube', 'extract': False},
    'youtu.be': {'type': 'youtube', 'extract': False},
    'twitter.com': {'type': 'twitter', 'extract': True, 'smart_routing': True},
    'x.com': {'type': 'twitter', 'extract': True, 'smart_routing': True},
    'fxtwitter.com': {'type': 'twitter', 'extract': True, 'smart_routing': True},
    'reddit.com': {'type': 'reddit', 'extract': True},
    'github.com': {'type': 'github', 'extract': True},
    'wikipedia.org': {'type': 'wikipedia', 'extract': True},
}

# File extensions to ignore for full content extraction
IGNORE_EXTENSIONS = {
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.zip', '.rar', '.7z', '.tar', '.gz', '.mp3', '.mp4', '.avi',
    '.mov', '.wmv', '.flv', '.mkv', '.webm', '.gif', '.jpg', '.jpeg',
    '.png', '.webp', '.svg', '.exe', '.dmg', '.pkg', '.iso'
}

def get_domain_info(url: str) -> Dict[str, str]:
    """
    Extract domain information from a URL.
    
    Args:
        url: The URL to analyze
        
    Returns:
        Dictionary with domain information
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Get domain without subdomains
        domain_parts = domain.split('.')
        if len(domain_parts) > 2:
            main_domain = '.'.join(domain_parts[-2:])
        else:
            main_domain = domain
        
        # Check for domain-specific rules
        domain_type = 'website'
        should_extract = True
        force_screenshot = False
        smart_routing = False
        
        for key, rule in DOMAIN_RULES.items():
            if key in domain:
                domain_type = rule['type']
                should_extract = rule.get('extract', True)
                force_screenshot = rule.get('force_screenshot', False)
                smart_routing = rule.get('smart_routing', False)
                break
        
        # Check file extensions
        path = parsed.path.lower()
        if any(path.endswith(ext) for ext in IGNORE_EXTENSIONS):
            should_extract = False
        
        return {
            'domain': domain,
            'main_domain': main_domain,
            'type': domain_type,
            'should_extract': should_extract,
            'force_screenshot': force_screenshot,
            'smart_routing': smart_routing,
            'scheme': parsed.scheme,
            'path': parsed.path,
            'query': parsed.query,
            'fragment': parsed.fragment
        }
    except Exception as e:
        logging.error(f"Error parsing URL {url}: {e}")
        return {
            'domain': 'unknown',
            'main_domain': 'unknown',
            'type': 'website',
            'should_extract': False
        }

async def fetch_url_content(url: str, timeout: int = 15) -> Optional[Tuple[bytes, str]]:
    """
    Fetch the content of a URL using httpx.
    """
    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }

    try:
        async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=timeout) as client:
            response = await client.get(url)
            response.raise_for_status()  # Raise exception for 4xx/5xx responses
            content = await response.aread()
            content_type = response.headers.get('Content-Type', 'application/octet-stream')
            logging.info(f"Successfully fetched {url} with httpx.")
            return content, content_type
    except (httpx.HTTPStatusError, httpx.RequestError, httpx.TooManyRedirects, httpx.InvalidHeader) as e:
        logging.error(f"httpx fetch failed for {url}: {e}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while fetching {url}: {e}", exc_info=True)
        return None



def extract_metadata(html: str, url: str) -> Dict[str, str]:
    """
    Extract metadata from HTML content.
    
    Args:
        html: The HTML content
        url: The URL the HTML was fetched from
        
    Returns:
        Dictionary with extracted metadata
    """
    metadata = {
        'title': '',
        'description': '',
        'image': '',
        'site_name': '',
        'url': url,
        'type': 'website'
    }
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Try to get OpenGraph metadata first
        og_props = {}
        for meta in soup.find_all('meta', property=lambda x: x and x.startswith('og:')):
            prop = meta['property'][3:]  # Remove 'og:' prefix
            og_props[prop] = meta.get('content', '')
        
        # Use OpenGraph data if available
        if og_props:
            metadata.update({
                'title': og_props.get('title', ''),
                'description': og_props.get('description', ''),
                'image': og_props.get('image', ''),
                'site_name': og_props.get('site_name', ''),
                'type': og_props.get('type', 'website')
            })
        
        # Fall back to standard metadata
        if not metadata['title']:
            title_tag = soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.get_text(strip=True)
        
        if not metadata['description']:
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                metadata['description'] = meta_desc['content']
        
        # Get favicon
        favicon = soup.find('link', rel=lambda x: x and 'icon' in x.lower())
        if favicon and favicon.get('href'):
            metadata['favicon'] = urljoin(url, favicon['href'])
        
        # Clean up text
        for key in ['title', 'description', 'site_name']:
            if key in metadata:
                text = metadata[key]
                # Remove extra whitespace and newlines
                text = ' '.join(text.split())
                metadata[key] = text
        
        return metadata
    
    except Exception as e:
        logging.error(f"Error extracting metadata from {url}: {e}")
        return metadata

def extract_main_content(html: str, url: str) -> Dict[str, str]:
    """
    Extract the main content from HTML using trafilatura.
    
    Args:
        html: The HTML content
        url: The URL the HTML was fetched from
        
    Returns:
        Dictionary with extracted content and metadata
    """
    try:
        # Use trafilatura to extract main content
        result = trafilatura.extract(
            html,
            include_links=True,
            include_tables=True,
            include_images=True,
            output_format='json',
            url=url,
            config=trafilatura_config
        )
        
        if not result:
            return {'content': '', 'text': ''}
        
        # Parse the JSON result
        import json
        data = json.loads(result)
        
        # Extract relevant fields
        content = {
            'title': data.get('title', ''),
            'author': data.get('author', ''),
            'date': data.get('date', ''),
            'categories': data.get('categories', []),
            'tags': data.get('tags', []),
            'content': data.get('text', ''),
            'text': data.get('text', ''),
            'url': url,
            'language': data.get('language', ''),
            'excerpt': data.get('excerpt', ''),
            'sitename': data.get('sitename', ''),
            'image': data.get('image', ''),
            'word_count': len(data.get('text', '').split())
        }
        
        return content
    
    except Exception as e:
        logging.error(f"Error extracting text from HTML: {e}")
        return ""

def should_extract_text(content_type: str) -> Dict[str, bool]:
    """
    Determine if text extraction should be performed based on the content type.
    
    Args:
        content_type: The content type
        
    Returns:
        A dictionary with a single key 'should_extract' indicating whether text extraction should be performed
    """
    if content_type.startswith('text/html'):
        return {'should_extract': True}
    else:
        return {'should_extract': False}

def strip_boilerplate(text: str) -> str:
    """Removes common boilerplate text from web content."""
    if not text:
        return ""
    BLOCKLIST = [
        "cookie", "privacy", "gdpr", "javascript", "login", "log in",
        "browser support", "enable js", "advertisement", "subscribe",
        "sign up", "newsletter", "allow notifications"
    ]
    # A line is considered boilerplate if it's short and contains a keyword
    return "\n".join(
        line for line in text.splitlines()
        if not (len(line.split()) < 15 and any(keyword in line.lower() for keyword in BLOCKLIST))
    )

async def process_url(url: str) -> Dict[str, Any]:
    """Process a URL, using smart routing for Twitter/X.com URLs and falling back to text extraction."""
    domain_info = get_domain_info(url)
    # Screenshot fallback removed; screenshots are command-gated via !ss
    smart_routing = domain_info.get('smart_routing', False)
    screenshot_path = None

    # 1. Smart routing for Twitter/X.com URLs (detect media first)
    if smart_routing:
        logging.info(f"🧠 Smart routing enabled for {url}. Checking for media content...")
        
        try:
            # Import media capability detector
            from .media_capability import is_twitter_video_url
            
            # Check if this Twitter URL contains video/audio
            probe_result = await is_twitter_video_url(url)
            
            if probe_result.is_media_capable:
                logging.info(f"🎥 Media detected in Twitter URL {url}. Routing to yt-dlp flow.")
                # Return special indicator that this should be routed to yt-dlp
                return {'url': url, 'text': None, 'screenshot_path': None, 'error': None, 'route_to_ytdlp': True}
            else:
                logging.info(f"📝 No media in Twitter URL {url}. Proceeding with text extraction (no auto-screenshot).")
                
        except Exception as e:
            logging.error(f"❌ Smart routing failed for {url}: {e}. Continuing with text extraction.")

    # 2. Attempt screenshot if forced (legacy behavior for non-smart domains)
    # 'force_screenshot' is ignored; screenshots are gated by explicit command.

    # 2. Attempt standard text extraction (primary method for non-forced sites)
    try:
        httpx_content_data = await fetch_url_content(url)
        if httpx_content_data:
            content, content_type = httpx_content_data
            if 'text/html' in content_type:
                html_string = content.decode('utf-8')
                raw_text = trafilatura.extract(html_string, url=url, config=trafilatura_config)
                text_content = strip_boilerplate(raw_text or "")

                # Check for placeholder content indicating a JS-heavy site
                js_required_keywords = ['enable javascript', 'javascript is required', 'requires javascript']
                if text_content and not any(keyword in text_content.lower() for keyword in js_required_keywords):
                    return {'url': url, 'text': text_content, 'screenshot_path': None, 'error': None}
                else:
                    logging.warning(f"Content from {url} is empty or requires JavaScript. Attempting screenshot fallback.")

    except Exception as e:
        logging.warning(f"Initial httpx fetch failed for {url}: {e}.")

    # 3. No automatic screenshot fallback; allow caller to decide next steps.

    # 4. If all methods fail
    error_msg = f"All content extraction methods (text and screenshot) failed for {url}."
    logging.error(error_msg)
    return {'url': url, 'text': None, 'screenshot_path': None, 'error': error_msg}

async def get_url_preview(url: str) -> discord.Embed:
    """
    Create a Discord embed preview for a URL.
    
    Args:
        url: The URL to create a preview for
        
    Returns:
        A Discord Embed object
    """
    try:
        domain_info = get_domain_info(url)
        title = domain_info.get('domain', 'Link Preview')
        
        embed = discord.Embed(
            title=title,
            url=url,
            description=f"[Click to open]({url})",
            color=discord.Color.blue()
        )
        
        embed.set_footer(text=domain_info.get('domain', ''))
        
        return embed
    
    except Exception as e:
        logging.error(f"Error creating URL preview for {url}: {e}", exc_info=True)
        
        # Fallback to a simple embed
        embed = discord.Embed(
            title="Link Preview",
            url=url,
            description=f"[Click to open]({url})\n\n*Preview unavailable*",
            color=discord.Color.blue()
        )
        return embed
