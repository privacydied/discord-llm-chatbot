"""
Web content extraction and processing for the Discord bot.
"""
import logging
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse, urljoin
import aiohttp
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
    'twitter.com': {'type': 'twitter', 'extract': False},
    'x.com': {'type': 'twitter', 'extract': False},
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
        
        for key, rule in DOMAIN_RULES.items():
            if key in domain:
                domain_type = rule['type']
                should_extract = rule.get('extract', True)
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

async def fetch_url_content(url: str, timeout: int = 10) -> Optional[Tuple[bytes, str]]:
    """
    Fetch the content of a URL.
    
    Args:
        url: The URL to fetch
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (content, content_type) or None if the request failed
    """
    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
    }
    
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, timeout=timeout, allow_redirects=True) as response:
                if response.status != 200:
                    logging.warning(f"Failed to fetch {url}: HTTP {response.status}")
                    return None
                
                content_type = response.headers.get('Content-Type', '').split(';')[0].strip()
                content = await response.read()
                
                return content, content_type
    except Exception as e:
        logging.error(f"Error fetching URL {url}: {e}")
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
        logging.error(f"Error extracting main content from {url}: {e}")
        return {'content': '', 'text': ''}

async def process_url(url: str, extract_content: bool = True) -> Dict[str, Any]:
    """
    Process a URL and extract relevant information.
    
    Args:
        url: The URL to process
        extract_content: Whether to extract the main content (can be slow)
        
    Returns:
        Dictionary with extracted information
    """
    try:
        # Get domain information
        domain_info = get_domain_info(url)
        
        # Don't extract content for certain types
        if not domain_info.get('should_extract', True):
            extract_content = False
        
        # Fetch the URL content
        result = await fetch_url_content(url)
        if not result:
            return {'error': 'Failed to fetch URL', 'url': url}
        
        content, content_type = result
        
        # Handle non-HTML content
        if not content_type.startswith('text/html'):
            return {
                'url': url,
                'type': 'file',
                'content_type': content_type,
                'domain_info': domain_info,
                'metadata': {}
            }
        
        # Decode HTML content
        try:
            html = content.decode('utf-8', errors='replace')
        except UnicodeDecodeError:
            try:
                html = content.decode('latin-1', errors='replace')
            except Exception as e:
                logging.error(f"Failed to decode content from {url}: {e}")
                return {'error': 'Failed to decode content', 'url': url}
        
        # Extract metadata
        metadata = extract_metadata(html, url)
        
        # Extract main content if requested
        main_content = {}
        if extract_content:
            main_content = extract_main_content(html, url)
        
        # Prepare result
        result = {
            'url': url,
            'type': domain_info['type'],
            'content_type': content_type,
            'domain_info': domain_info,
            'metadata': metadata,
            'content': main_content
        }
        
        return result
    
    except Exception as e:
        logging.error(f"Error processing URL {url}: {e}", exc_info=True)
        return {'error': str(e), 'url': url}

async def get_url_preview(url: str) -> discord.Embed:
    """
    Create a Discord embed preview for a URL.
    
    Args:
        url: The URL to create a preview for
        
    Returns:
        A Discord Embed object
    """
    try:
        # Process the URL
        result = await process_url(url, extract_content=False)
        
        if 'error' in result:
            embed = discord.Embed(
                title="Error",
                description=f"Failed to process URL: {result['error']}",
                color=discord.Color.red()
            )
            return embed
        
        metadata = result.get('metadata', {})
        domain_info = result.get('domain_info', {})
        
        # Create embed
        title = metadata.get('title', domain_info.get('domain', 'Untitled'))
        if len(title) > 256:
            title = title[:253] + '...'
        
        description = metadata.get('description', '')
        if len(description) > 1000:
            description = description[:997] + '...'
        
        embed = discord.Embed(
            title=title,
            description=description,
            url=url,
            color=discord.Color.blue()
        )
        
        # Add image if available
        if metadata.get('image'):
            try:
                # Validate image URL
                parsed = urlparse(metadata['image'])
                if parsed.scheme in ('http', 'https') and parsed.netloc:
                    embed.set_image(url=metadata['image'])
            except Exception as e:
                logging.warning(f"Invalid image URL: {metadata['image']}: {e}")
        
        # Add footer with domain
        site_name = metadata.get('site_name', domain_info.get('domain', 'Link'))
        embed.set_footer(text=site_name, icon_url=metadata.get('favicon', ''))
        
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
