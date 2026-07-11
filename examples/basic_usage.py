"""Basic usage example for the Discord LLM ChatBot.

This script demonstrates how to use the bot's core functionality
programmatically without running the full Discord bot.
"""

import asyncio
import os

# Add the project root to the Python path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# Import bot modules
from bot.memory import (
    get_profile,
    get_server_profile,
    save_profile,
    save_server_profile,
)
from bot.ollama import generate_response, ollama_client
from bot.pdf_utils import pdf_processor
from bot.search import search_all
from bot.web import get_url_preview, process_url


def print_header(title: str) -> None:
    """Print a formatted header."""


async def demo_chat() -> None:
    """Demonstrate chat functionality."""
    print_header("CHAT DEMO")

    # Sample user ID for demonstration
    user_id = "demo_user_123"

    # Example conversation
    messages = [
        "Hello, who are you?",
        "What can you do?",
        "Tell me about artificial intelligence",
    ]

    for message in messages:
        try:
            # Generate a response
            await generate_response(prompt=message, user_id=user_id, max_tokens=200, temperature=0.7)

        except Exception:
            pass

        # Small delay between messages
        await asyncio.sleep(1)


async def demo_memory() -> None:
    """Demonstrate memory functionality."""
    print_header("MEMORY DEMO")

    # Sample user and server IDs
    user_id = "demo_user_123"
    server_id = "demo_server_456"

    # Get or create user profile
    user_profile = get_profile(user_id)

    # Add some memories
    if "memories" not in user_profile:
        user_profile["memories"] = []

    memory = "User prefers to be called 'Demo User'"
    user_profile["memories"].append({"content": memory, "timestamp": str(asyncio.get_event_loop().time())})

    # Save the profile
    save_profile(user_profile, force=True)

    # Get server profile
    server_profile = get_server_profile(server_id)

    # Add a server memory
    if "memories" not in server_profile:
        server_profile["memories"] = []

    server_memory = "This is a demo server for testing the bot"
    server_profile["memories"].append(
        {
            "content": server_memory,
            "added_by": user_id,
            "timestamp": str(asyncio.get_event_loop().time()),
        },
    )

    # Save the server profile
    save_server_profile(server_profile, force=True)


async def demo_search() -> None:
    """Demonstrate search functionality."""
    print_header("SEARCH DEMO")

    query = "latest developments in AI"

    try:
        results = await search_all(query, max_web_results=3, max_memory_results=2)

        for _i, _result in enumerate(results.get("web", [])[:3], 1):
            pass

        if results.get("memories"):
            for _i, _memory in enumerate(results["memories"][:2], 1):
                pass

    except Exception:
        pass


async def demo_web() -> None:
    """Demonstrate web content extraction."""
    print_header("WEB CONTENT EXTRACTION DEMO")

    url = "https://ollama.com/"

    try:
        # Get URL preview
        preview = await get_url_preview(url)
        if preview and hasattr(preview, "image") and preview.image:
            pass

        # Process URL for more detailed information
        result = await process_url(url, extract_content=True)

        if result and not result.get("error"):
            result.get("content", {})

    except Exception:
        pass


async def demo_pdf() -> None:
    """Demonstrate PDF processing."""
    print_header("PDF PROCESSING DEMO")

    # Create a sample PDF for demonstration
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    pdf_path = "sample_document.pdf"

    # Generate a simple PDF if it doesn't exist
    if not os.path.exists(pdf_path):
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont("Helvetica", 12)
        c.drawString(100, 750, "Sample PDF Document")
        c.drawString(
            100,
            730,
            "This is a test PDF document generated for demonstration purposes.",
        )
        c.drawString(
            100,
            710,
            "It contains some sample text that we'll extract using the PDF processor.",
        )
        c.save()

    # Process the PDF

    try:
        # Extract text from PDF
        pdf_processor.extract_text(pdf_path)

        # Get PDF metadata
        metadata = pdf_processor.get_metadata(pdf_path)
        for value in metadata.values():
            if value:  # Only show non-empty fields
                pass

    except Exception:
        pass
    finally:
        # Clean up the sample PDF
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


async def main() -> None:
    """Run all demos."""
    try:
        # Initialize the Ollama client
        await ollama_client.ensure_session()

        # Run demos
        await demo_chat()
        await demo_memory()
        await demo_search()
        await demo_web()
        await demo_pdf()

    except Exception:
        pass
    finally:
        # Clean up
        await ollama_client.close()


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("user_profiles", exist_ok=True)
    os.makedirs("server_profiles", exist_ok=True)
    os.makedirs("user_logs", exist_ok=True)
    os.makedirs("temp", exist_ok=True)

    # Run the demo
    asyncio.run(main())
