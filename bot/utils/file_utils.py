"""
File utility functions for the Discord bot.
"""
import logging
import aiohttp
from pathlib import Path
from typing import Optional

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

def is_text_file(file_path: str) -> bool:
    """
    Check if a file is a text file by examining its content.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        bool: True if the file is a text file, False otherwise
    """
    try:
        with open(file_path, 'rb') as f:
            # Read the first 8000 bytes to determine if it's text
            chunk = f.read(8000)
            
            # Check for null bytes which indicate binary content
            if b'\x00' in chunk:
                return False
            
            # Try to decode as UTF-8
            try:
                chunk.decode('utf-8')
                return True
            except UnicodeDecodeError:
                return False
    except Exception as e:
        logging.error(f"Error checking if {file_path} is a text file: {e}")
        return False
