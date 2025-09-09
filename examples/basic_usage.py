"""
Basic usage example for the Discord LLM ChatBot.

This script demonstrates how to use the bot's core functionality
programmatically without running the full Discord bot.
"""
import asyncio
import os
from pathlib import Path

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import bot modules
from bot.ollama import ollama_client, generate_response
from bot.memory import get_profile, save_profile, get_server_profile, save_server_profile
from bot.pdf_utils import pdf_processor
from bot.search import search_all
from bot.web import get_url_preview, process_url

def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title}".ljust(80, "="))
    print("=" * 80)

async def demo_chat():
    """Demonstrate chat functionality."""
    print_header("CHAT DEMO")
    
    # Sample user ID for demonstration
    user_id = "demo_user_123"
    
    # Example conversation
    messages = [
        "Hello, who are you?",
        "What can you do?",
        "Tell me about artificial intelligence"
    ]
    
    for message in messages:
        print(f"\nYou: {message}")
        
        try:
            # Generate a response
            response = await generate_response(
                prompt=message,
                user_id=user_id,
                max_tokens=200,
                temperature=0.7
            )
            
            print(f"\nBot: {response['text'].strip()}")
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            
        # Small delay between messages
        await asyncio.sleep(1)

async def demo_memory():
    """Demonstrate memory functionality."""
    print_header("MEMORY DEMO")
    
    # Sample user and server IDs
    user_id = "demo_user_123"
    server_id = "demo_server_456"
    
    # Get or create user profile
    user_profile = get_profile(user_id)
    print(f"User profile created/loaded for user {user_id}")
    
    # Add some memories
    if 'memories' not in user_profile:
        user_profile['memories'] = []
    
    memory = "User prefers to be called 'Demo User'"
    user_profile['memories'].append({
        'content': memory,
        'timestamp': str(asyncio.get_event_loop().time())
    })
    
    # Save the profile
    save_profile(user_profile, force=True)
    print(f"Added memory: {memory}")
    
    # Get server profile
    server_profile = get_server_profile(server_id)
    print(f"Server profile created/loaded for server {server_id}")
    
    # Add a server memory
    if 'memories' not in server_profile:
        server_profile['memories'] = []
    
    server_memory = "This is a demo server for testing the bot"
    server_profile['memories'].append({
        'content': server_memory,
        'added_by': user_id,
        'timestamp': str(asyncio.get_event_loop().time())
    })
    
    # Save the server profile
    save_server_profile(server_profile, force=True)
    print(f"Added server memory: {server_memory}")

async def demo_search():
    """Demonstrate search functionality."""
    print_header("SEARCH DEMO")
    
    query = "latest developments in AI"
    print(f"Searching for: {query}")
    
    try:
        results = await search_all(query, max_web_results=3, max_memory_results=2)
        
        print("\nWeb Results:")
        for i, result in enumerate(results.get('web', [])[:3], 1):
            print(f"{i}. {result.title}")
            print(f"   {result.snippet[:150]}...")
            print(f"   URL: {result.url}\n")
        
        if results.get('memories'):
            print("\nRelevant Memories:")
            for i, memory in enumerate(results['memories'][:2], 1):
                print(f"{i}. {memory.snippet[:200]}...\n")
    
    except Exception as e:
        print(f"Search error: {str(e)}")

async def demo_web():
    """Demonstrate web content extraction."""
    print_header("WEB CONTENT EXTRACTION DEMO")
    
    url = "https://ollama.com/"
    print(f"Fetching URL: {url}")
    
    try:
        # Get URL preview
        preview = await get_url_preview(url)
        if preview:
            print(f"\nTitle: {preview.title}")
            print(f"Description: {preview.description[:200]}...")
            if hasattr(preview, 'image') and preview.image:
                print(f"Image: {preview.image.url}")
        
        # Process URL for more detailed information
        print("\nProcessing URL for detailed content...")
        result = await process_url(url, extract_content=True)
        
        if result and not result.get('error'):
            print("\nExtracted Content (first 300 chars):")
            content = result.get('content', {})
            print(content.get('text', '')[:300] + "...")
    
    except Exception as e:
        print(f"Web processing error: {str(e)}")

async def demo_pdf():
    """Demonstrate PDF processing."""
    print_header("PDF PROCESSING DEMO")
    
    # Create a sample PDF for demonstration
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    
    pdf_path = "sample_document.pdf"
    
    # Generate a simple PDF if it doesn't exist
    if not os.path.exists(pdf_path):
        print("Creating sample PDF...")
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont("Helvetica", 12)
        c.drawString(100, 750, "Sample PDF Document")
        c.drawString(100, 730, "This is a test PDF document generated for demonstration purposes.")
        c.drawString(100, 710, "It contains some sample text that we'll extract using the PDF processor.")
        c.save()
    
    # Process the PDF
    print(f"Processing PDF: {pdf_path}")
    
    try:
        # Extract text from PDF
        text = pdf_processor.extract_text(pdf_path)
        print("\nExtracted Text:")
        print(text[:500] + "..." if len(text) > 500 else text)
        
        # Get PDF metadata
        metadata = pdf_processor.get_metadata(pdf_path)
        print("\nPDF Metadata:")
        for key, value in metadata.items():
            if value:  # Only show non-empty fields
                print(f"{key}: {value}")
    
    except Exception as e:
        print(f"PDF processing error: {str(e)}")
    finally:
        # Clean up the sample PDF
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

async def main():
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
        
    except Exception as e:
        print(f"Error in demo: {str(e)}")
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
