"""
Utility functions for the Discord bot.
"""
import re
import os
import mimetypes
from typing import Optional, List, Tuple, Any, Dict, Union
from pathlib import Path
import asyncio
import aiohttp
import logging

# Import config
from .config import load_config

# Load configuration
config = load_config()

def clean_response(text: str) -> str:
    """Clean up response text before sending to Discord."""
    if not text:
        return ""
    
    # Remove any code block markers if present
    text = re.sub(r'```(?:\w+)?\s*', '', text)
    
    # Remove multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip whitespace
    text = text.strip()
    
    return text

def is_text_file(file_path: Union[str, Path]) -> bool:
    """Check if a file is a text file based on its extension and content type."""
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    # Check common text file extensions
    text_extensions = {'.txt', '.md', '.markdown', '.log', '.json', '.yaml', '.yml', '.csv', '.html', '.css', '.js', '.py'}
    if file_path.suffix.lower() in text_extensions:
        return True
    
    # Check MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith('text/'):
        return True
    
    # Check file content
    try:
        with open(file_path, 'rb') as f:
            # Read first 1024 bytes to check for non-text characters
            chunk = f.read(1024)
            # Check for null bytes or too many non-printable characters
            if b'\x00' in chunk:
                return False
            
            # Count non-printable ASCII bytes
            non_printable = sum(1 for byte in chunk if byte < 32 and byte not in {9, 10, 13})
            if non_printable > len(chunk) * 0.3:  # More than 30% non-printable
                return False
                
        return True
    except (IOError, UnicodeDecodeError):
        return False

async def send_in_chunks(channel, text: str, reference=None, chunk_size: int = 2000) -> List[Any]:
    """
    Send a long message in chunks to avoid Discord's 2000 character limit.
    
    Args:
        channel: The Discord channel to send the message to
        text: The text to send
        reference: Optional message to reply to
        chunk_size: Maximum size of each chunk (default: 2000)
        
    Returns:
        List of sent message objects
    """
    if not text:
        return []
    
    # Split into chunks, trying to preserve paragraphs
    chunks = []
    current_chunk = ""
    
    # First, try to split on double newlines
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        # If adding this paragraph would exceed the chunk size, save current chunk
        if len(current_chunk) + len(paragraph) + 2 > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = ""
        
        # Add paragraph to current chunk
        if current_chunk:
            current_chunk += '\n\n' + paragraph
        else:
            current_chunk = paragraph
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    # If we still have chunks that are too long, split by lines
    new_chunks = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            new_chunks.append(chunk)
            continue
            
        # Split by lines
        lines = chunk.split('\n')
        current_line_chunk = ""
        
        for line in lines:
            if len(current_line_chunk) + len(line) + 1 > chunk_size and current_line_chunk:
                new_chunks.append(current_line_chunk)
                current_line_chunk = ""
            
            if current_line_chunk:
                current_line_chunk += '\n' + line
            else:
                current_line_chunk = line
        
        if current_line_chunk:
            new_chunks.append(current_line_chunk)
    
    chunks = new_chunks
    
    # If we still have chunks that are too long, split by character
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final_chunks.append(chunk)
        else:
            # Split by character
            for i in range(0, len(chunk), chunk_size):
                final_chunks.append(chunk[i:i+chunk_size])
    
    # Send each chunk
    sent_messages = []
    for i, chunk in enumerate(final_chunks):
        try:
            if i == 0 and reference is not None:
                msg = await channel.send(chunk, reference=reference)
            else:
                msg = await channel.send(chunk)
            sent_messages.append(msg)
        except Exception as e:
            logging.error(f"Error sending message chunk {i+1}/{len(final_chunks)}: {e}")
    
    return sent_messages

# Alias for backward compatibility with main.py import
send_chunks = send_in_chunks

def extract_mentions(text: str) -> List[Tuple[str, str]]:
    """
    Extract user and role mentions from text.
    
    Returns:
        List of tuples (mention_type, mention_id)
    """
    # User mentions: <@1234567890> or <@!1234567890>
    user_mentions = re.findall(r'<@!?(\d+)>', text)
    
    # Role mentions: <@&1234567890>
    role_mentions = re.findall(r'<@&(\d+)>', text)
    
    # Format as (type, id) tuples
    mentions = [('user', uid) for uid in user_mentions] + \
               [('role', rid) for rid in role_mentions]
    
    return mentions

def format_duration(seconds: int) -> str:
    """Format a duration in seconds to a human-readable string."""
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 and days == 0:  # Only show minutes if < 1 day
        parts.append(f"{minutes}m")
    if seconds > 0 and hours == 0:  # Only show seconds if < 1 hour
        parts.append(f"{seconds}s")
    
    return " ".join(parts) if parts else "0s"

def get_file_extension(filename: str) -> str:
    """Get the file extension from a filename, converted to lowercase."""
    return os.path.splitext(filename)[1].lower()

def is_image_file(filename: str) -> bool:
    """Check if a file is an image based on its extension."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff'}
    return get_file_extension(filename) in image_extensions

def is_audio_file(filename: str) -> bool:
    """Check if a file is an audio file based on its extension."""
    audio_extensions = {'.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac'}
    return get_file_extension(filename) in audio_extensions

def is_video_file(filename: str) -> bool:
    """Check if a file is a video file based on its extension."""
    video_extensions = {'.mp4', '.webm', '.mov', '.avi', '.mkv', '.flv'}
    return get_file_extension(filename) in video_extensions

def is_document_file(filename: str) -> bool:
    """Check if a file is a document based on its extension."""
    doc_extensions = {
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.odt', '.ods', '.odp', '.txt', '.rtf', '.md', '.csv'
    }
    return get_file_extension(filename) in doc_extensions

async def download_file(url: str, save_path: Path, session: Optional[aiohttp.ClientSession] = None) -> bool:
    """
    Download a file from a URL and save it to the specified path.
    
    Args:
        url: URL of the file to download
        save_path: Path to save the file to
        session: Optional aiohttp session to use
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True
    
    try:
        async with session.get(url) as response:
            if response.status != 200:
                logging.error(f"Failed to download {url}: HTTP {response.status}")
                return False
            
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the file
            with open(save_path, 'wb') as f:
                while True:
                    chunk = await response.content.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
            
            return True
    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")
        return False
    finally:
        if close_session and not session.closed:
            await session.close()
