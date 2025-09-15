"""
Search functionality for the Discord bot.
"""

import logging
from typing import List, Dict
import asyncio
from datetime import datetime, timedelta

import discord
import aiohttp
from bs4 import BeautifulSoup
from pathlib import Path

# Import bot modules
from .config import load_config
from .memory import get_profile, get_server_profile

# Import from the utils.py module using absolute import to avoid package conflict
from bot.utils import download_file, is_text_file

# Load configuration
config = load_config()

# Search result cache
search_cache = {}
CACHE_EXPIRY = timedelta(minutes=30)


class SearchResult:
    """Class to represent a search result."""

    def __init__(self, title: str, url: str, snippet: str, source: str = "web"):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source
        self.timestamp = datetime.utcnow()

    def to_embed(self) -> discord.Embed:
        """Convert the search result to a Discord embed."""
        embed = discord.Embed(
            title=self.title[:256],  # Limit title length
            url=self.url,
            description=self.snippet[:2048],  # Limit description length
            color=discord.Color.blue(),
        )
        embed.set_footer(
            text=f"Source: {self.source} • {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
        )
        return embed


async def web_search(query: str, max_results: int | None = None) -> List[SearchResult]:
    """
    Perform a web search and return results.

    Args:
        query: Search query
        max_results: Optional maximum number of results to return. If None, return all parsed results.

    Returns:
        List of SearchResult objects
    """
    # Check cache first only when a specific max_results is requested
    cache_key = f"web:{query}:{max_results if max_results is not None else 'all'}"
    if max_results is not None and cache_key in search_cache:
        cached = search_cache[cache_key]
        if datetime.utcnow() - cached["timestamp"] < CACHE_EXPIRY:
            return cached["results"]

    results = []

    try:
        # Use a search API or scrape search results
        # This is a placeholder - you'd typically use an API key here
        search_url = "https://html.duckduckgo.com/html/"
        params = {
            "q": query,
            "kl": "us-en",
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                search_url, data=params, headers=headers
            ) as response:
                if response.status != 200:
                    logging.error(
                        f"Search request failed with status {response.status}"
                    )
                    return []

                html = await response.text()

        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # Extract search results (this depends on the search engine's HTML structure)
        result_elements = soup.select(".result")
        # Apply limit only if provided
        limited = (
            result_elements if max_results is None else result_elements[:max_results]
        )

        for i, result in enumerate(limited):
            try:
                title_elem = result.select_one(".result__a")
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)
                url = title_elem.get("href", "")

                # Clean up URL (DuckDuckGo adds a redirect)
                if url.startswith("//duckduckgo.com/l/"):
                    url = url.replace("//duckduckgo.com/l/?uddg=", "")
                    url = url.split("&", 1)[0]
                    import urllib.parse

                    url = urllib.parse.unquote(url)

                snippet_elem = result.select_one(".result__snippet")
                snippet = (
                    snippet_elem.get_text(strip=True)
                    if snippet_elem
                    else "No description available."
                )

                results.append(SearchResult(title, url, snippet, "DuckDuckGo"))

            except Exception as e:
                logging.error(f"Error parsing search result {i}: {e}")
                continue

        # Cache the results only when a specific max_results is requested
        if max_results is not None:
            search_cache[cache_key] = {
                "results": results,
                "timestamp": datetime.utcnow(),
            }

        return results

    except Exception as e:
        logging.error(f"Error performing web search: {e}", exc_info=True)
        return []


async def search_memories(
    query: str, user_id: str = None, guild_id: str = None
) -> List[SearchResult]:
    """
    Search through user and server memories.

    Args:
        query: Search query
        user_id: Optional user ID to search user memories
        guild_id: Optional guild ID to search server memories

    Returns:
        List of SearchResult objects
    """
    results = []

    try:
        # Search user memories if user_id is provided
        if user_id:
            profile = get_profile(str(user_id))
            if "memories" in profile:
                for memory in profile["memories"]:
                    if (
                        isinstance(memory, dict)
                        and "content" in memory
                        and query.lower() in memory["content"].lower()
                    ):
                        results.append(
                            SearchResult(
                                title="Personal Memory",
                                url="",
                                snippet=memory["content"],
                                source="Your Memories",
                            )
                        )

        # Search server memories if guild_id is provided
        if guild_id:
            server_profile = get_server_profile(str(guild_id))
            if "memories" in server_profile:
                for memory in server_profile["memories"]:
                    if (
                        isinstance(memory, dict)
                        and "content" in memory
                        and query.lower() in memory["content"].lower()
                    ):
                        results.append(
                            SearchResult(
                                title="Server Memory",
                                url="",
                                snippet=memory["content"],
                                source=f"{memory.get('added_by', 'Server')}'s Memory",
                            )
                        )

        return results

    except Exception as e:
        logging.error(f"Error searching memories: {e}", exc_info=True)
        return []


async def search_files(
    query: str, attachments: List[discord.Attachment]
) -> List[SearchResult]:
    """
    Search through message attachments.

    Args:
        query: Search query
        attachments: List of Discord attachments to search through

    Returns:
        List of SearchResult objects
    """
    results = []

    try:
        for attachment in attachments:
            # Skip non-text files
            if not is_text_file(attachment.filename):
                continue

            # Download and search the file
            try:
                temp_file = Path("temp") / attachment.filename
                await download_file(attachment.url, temp_file)

                with open(temp_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Simple text search
                if query.lower() in content.lower():
                    # Find context around the match
                    lines = content.split("\n")
                    for i, line in enumerate(lines):
                        if query.lower() in line.lower():
                            # Get surrounding lines for context
                            start = max(0, i - 2)
                            end = min(len(lines), i + 3)
                            context = "\n".join(lines[start:end])

                            results.append(
                                SearchResult(
                                    title=f"In file: {attachment.filename}",
                                    url=attachment.url,
                                    snippet=context,
                                    source="File Attachment",
                                )
                            )
                            break

                # Clean up
                if temp_file.exists():
                    temp_file.unlink()

            except Exception as e:
                logging.error(f"Error searching file {attachment.filename}: {e}")
                continue

        return results

    except Exception as e:
        logging.error(f"Error in search_files: {e}", exc_info=True)
        return []


async def search_all(
    query: str,
    user_id: str = None,
    guild_id: str = None,
    attachments: List[discord.Attachment] = None,
    max_web_results: int = 3,
    max_memory_results: int = 3,
    max_file_results: int = 3,
) -> Dict[str, List[SearchResult]]:
    """
    Perform a comprehensive search across all available sources.

    Args:
        query: Search query
        user_id: Optional user ID for user-specific searches
        guild_id: Optional guild ID for server-specific searches
        attachments: Optional list of attachments to search through
        max_web_results: Maximum number of web results to return
        max_memory_results: Maximum number of memory results to return
        max_file_results: Maximum number of file results to return

    Returns:
        Dictionary mapping source types to lists of SearchResult objects
    """
    if not query.strip():
        return {}

    # Run searches in parallel
    tasks = []

    # Web search
    tasks.append(web_search(query, max_web_results))

    # Memory search (if user_id or guild_id provided)
    if user_id or guild_id:
        tasks.append(search_memories(query, user_id, guild_id))
    else:
        tasks.append(asyncio.Future())  # Placeholder

    # File search (if attachments provided)
    if attachments:
        tasks.append(search_files(query, attachments[:10]))  # Limit to 10 attachments
    else:
        tasks.append(asyncio.Future())  # Placeholder

    # Wait for all searches to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    web_results = results[0] if not isinstance(results[0], Exception) else []
    memory_results = (
        results[1] if len(results) > 1 and not isinstance(results[1], Exception) else []
    )
    file_results = (
        results[2] if len(results) > 2 and not isinstance(results[2], Exception) else []
    )

    # Apply limits
    if max_web_results > 0 and len(web_results) > max_web_results:
        web_results = web_results[:max_web_results]

    if max_memory_results > 0 and len(memory_results) > max_memory_results:
        memory_results = memory_results[:max_memory_results]

    if max_file_results > 0 and len(file_results) > max_file_results:
        file_results = file_results[:max_file_results]

    return {"web": web_results, "memories": memory_results, "files": file_results}
