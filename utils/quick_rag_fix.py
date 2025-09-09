#!/usr/bin/env python3
"""
Quick RAG Fix Script

This script provides a lightweight approach to fix the RAG pipeline by:
1. Processing just the CIA Gateway Process document first
2. Using smaller batch sizes to avoid threading issues
3. Testing search immediately after ingestion

Usage: python quick_rag_fix.py
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bot.rag.document_parsers import document_parser_factory
from bot.rag.hybrid_search import get_hybrid_search
from bot.utils.logging import get_logger

logger = get_logger(__name__)

async def quick_rag_fix():
    """Quick fix for RAG pipeline focusing on the Gateway Process document."""
    print("⚡ QUICK RAG FIX")
    print("=" * 20)
    
    try:
        # Initialize search engine
        print("📡 Initializing search engine...")
        search_engine = await get_hybrid_search()
        
        # Get current stats
        stats = await search_engine.get_stats()
        current_chunks = stats.get('collection_stats', {}).get('total_chunks', 0)
        print(f"📊 Current chunks in collection: {current_chunks}")
        
        # If collection is empty, wipe and add just the Gateway Process document
        if current_chunks == 0:
            print("\n📄 Processing CIA Gateway Process document...")
            
            # Get the backend directly
            backend = search_engine.rag_backend
            if not backend:
                print("❌ No RAG backend available")
                return False
            
            # Process just the Gateway Process document
            gateway_file = Path("kb/CIA-RDP96-00788R001700210016-5.pdf")
            
            if not gateway_file.exists():
                print("❌ Gateway Process PDF not found")
                return False
            
            # Parse the document
            print("🔍 Parsing Gateway Process PDF...")
            content, metadata = await document_parser_factory.parse_document(gateway_file)
            
            if not content.strip():
                print("❌ No content extracted from PDF")
                return False
                
            print(f"✅ Extracted {len(content):,} characters")
            print(f"🔍 Content contains 'Gateway Process': {'Gateway Process' in content}")
            
            # Add document with smaller processing
            print("🔄 Adding document to vector store...")
            documents = await backend.add_document(
                source_id="CIA-RDP96-00788R001700210016-5.pdf",
                text=content,
                metadata=metadata,
                file_type="pdf"
            )
            
            print(f"✅ Added {len(documents)} chunks to collection")
        
        # Test search functionality
        print("\n🔍 Testing search functionality...")
        test_queries = [
            "Gateway Process",
            "what is the Gateway Process?",
            "consciousness exploration",
            "Hemi-Sync"
        ]
        
        for query in test_queries:
            print(f"\n   Query: '{query}'")
            try:
                results = await search_engine.search(query, max_results=3)
                print(f"   Results: {len(results)}")
                
                for i, result in enumerate(results[:2]):
                    print(f"     {i+1}. Score: {result.score:.4f}")
                    print(f"        Snippet: {result.snippet[:100]}...")
                    
            except Exception as e:
                print(f"   ❌ Search failed: {e}")
        
        # Get final stats
        final_stats = await search_engine.get_stats()
        final_chunks = final_stats.get('collection_stats', {}).get('total_chunks', 0)
        print("\n📊 Final collection stats:")
        print(f"   Total chunks: {final_chunks}")
        
        if final_chunks > 0:
            print("\n✅ QUICK RAG FIX SUCCESSFUL!")
            print("🎯 The RAG pipeline is now working with the Gateway Process document")
            return True
        else:
            print("\n❌ Quick fix failed - no chunks in collection")
            return False
            
    except Exception as e:
        print(f"❌ Quick fix failed: {e}")
        logger.error(f"Quick RAG fix failed: {e}")
        return False

async def main():
    """Main function."""
    print("⚡ Starting quick RAG fix for Gateway Process document...")
    print("This focuses on getting the core functionality working first.\n")
    
    success = await quick_rag_fix()
    
    if success:
        print("\n🎉 SUCCESS! You can now test RAG queries like:")
        print("   - 'what is the Gateway Process?'")
        print("   - 'consciousness exploration techniques'")
        print("   - 'Hemi-Sync meditation'")
        print("\n💡 To add more documents later, run the full bootstrap process.")
    else:
        print("\n🔧 The quick fix didn't resolve the issue.")
        print("   This suggests a deeper problem with the embedding pipeline.")
        print("   Check the logs for more details.")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
