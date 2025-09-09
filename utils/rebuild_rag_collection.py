#!/usr/bin/env python3
"""
RAG Collection Rebuild Script

This script wipes the existing ChromaDB collection and rebuilds it with the fixed
embedding pipeline to resolve the 0 results issue.

Usage: python rebuild_rag_collection.py
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bot.rag.hybrid_search import get_hybrid_search
from bot.rag.document_parsers import document_parser_factory
from bot.utils.logging import get_logger

logger = get_logger(__name__)

async def _manual_rebuild_approach(search_engine):
    """Manual rebuild approach when bootstrap hangs or fails."""
    print("🔧 MANUAL REBUILD APPROACH")
    print("-" * 30)
    
    try:
        # Get the backend directly
        backend = search_engine.rag_backend
        if not backend:
            print("❌ No RAG backend available")
            return False
            
        # Process a smaller subset of files first (test with one file)
        kb_path = Path("kb")
        test_files = [
            kb_path / "faq.txt",  # Start with smallest file
            kb_path / "features.md",
            kb_path / "CIA-RDP96-00788R001700210016-5.pdf"  # Then the target file
        ]
        
        processed_count = 0
        total_chunks = 0
        
        for file_path in test_files:
            if not file_path.exists():
                print(f"⏭️ Skipping missing file: {file_path.name}")
                continue
                
            print(f"📄 Processing: {file_path.name}")
            
            try:
                # Parse document with timeout
                content, metadata = await asyncio.wait_for(
                    document_parser_factory.parse_document(file_path),
                    timeout=120.0  # 2 minute timeout per file
                )
                
                if not content.strip():
                    print(f"⚠️ Empty content: {file_path.name}")
                    continue
                    
                print(f"   📊 Content length: {len(content):,} chars")
                
                # Add document with timeout
                documents = await asyncio.wait_for(
                    backend.add_document(
                        source_id=str(file_path.relative_to(kb_path)),
                        text=content,
                        metadata=metadata,
                        file_type=file_path.suffix[1:] if file_path.suffix else "text"
                    ),
                    timeout=300.0  # 5 minute timeout per document
                )
                
                processed_count += 1
                total_chunks += len(documents)
                print(f"   ✅ Added {len(documents)} chunks")
                
            except asyncio.TimeoutError:
                print(f"   ⏰ Timeout processing {file_path.name}")
                continue
            except Exception as e:
                print(f"   ❌ Error processing {file_path.name}: {e}")
                continue
        
        print("\n📊 Manual rebuild completed:")
        print(f"   Files processed: {processed_count}")
        print(f"   Total chunks: {total_chunks}")
        
        return processed_count > 0
        
    except Exception as e:
        print(f"❌ Manual rebuild failed: {e}")
        return False

async def rebuild_rag_collection():
    """Rebuild the RAG collection from scratch."""
    print("🔄 RAG COLLECTION REBUILD")
    print("=" * 30)
    
    try:
        # Get hybrid search instance
        print("📡 Initializing hybrid search system...")
        search_engine = await get_hybrid_search()
        
        # Get current stats
        stats_before = await search_engine.get_stats()
        print("📊 Current collection stats:")
        print(f"   Total chunks: {stats_before.get('collection_stats', {}).get('total_chunks', 0)}")
        
        # Wipe the collection
        print("\n🗑️ Wiping existing collection...")
        wipe_success = await search_engine.wipe_collection()
        
        if not wipe_success:
            print("❌ Failed to wipe collection")
            return False
            
        print("✅ Collection wiped successfully")
        
        # Verify collection is empty
        stats_after_wipe = await search_engine.get_stats()
        chunks_after_wipe = stats_after_wipe.get('collection_stats', {}).get('total_chunks', -1)
        print(f"📊 Chunks after wipe: {chunks_after_wipe}")
        
        # Rebuild the collection
        print("\n🔄 Rebuilding collection with fixed embedding pipeline...")
        
        # Force re-initialization and bootstrap with timeout
        if hasattr(search_engine, 'rag_backend') and search_engine.rag_backend:
            if hasattr(search_engine, 'bootstrap') and search_engine.bootstrap:
                print("📚 Starting bootstrap process (this may take several minutes for large PDFs)...")
                try:
                    # Add timeout to prevent hanging
                    bootstrap_result = await asyncio.wait_for(
                        search_engine.bootstrap.bootstrap_knowledge_base(force_refresh=True),
                        timeout=600.0  # 10 minute timeout
                    )
                    print(f"📊 Bootstrap result: {bootstrap_result}")
                except asyncio.TimeoutError:
                    print("⏰ Bootstrap timed out after 10 minutes")
                    print("💡 Trying alternative approach: manual file processing...")
                    return await _manual_rebuild_approach(search_engine)
                except Exception as e:
                    print(f"❌ Bootstrap failed: {e}")
                    print("💡 Trying alternative approach: manual file processing...")
                    return await _manual_rebuild_approach(search_engine)
            else:
                print("⚠️ Bootstrap not available, trying manual re-initialization...")
                # Re-initialize the search engine
                search_engine._initialized = False
                await search_engine.initialize()
        
        # Get final stats
        stats_final = await search_engine.get_stats()
        chunks_final = stats_final.get('collection_stats', {}).get('total_chunks', 0)
        print("\n📊 Final collection stats:")
        print(f"   Total chunks: {chunks_final}")
        
        # Test a search query
        print("\n🔍 Testing search with rebuilt collection...")
        test_results = await search_engine.search("Gateway Process", max_results=3)
        print(f"   Test query results: {len(test_results)}")
        
        for i, result in enumerate(test_results[:3]):
            print(f"     {i+1}. Score: {result.score:.4f}")
            print(f"        Title: {result.title}")
            print(f"        Snippet: {result.snippet[:100]}...")
            print()
        
        if len(test_results) > 0:
            print("✅ RAG COLLECTION REBUILD SUCCESSFUL!")
            print("🎉 Vector search is now working correctly!")
            return True
        else:
            print("⚠️ Rebuild completed but test search still returns 0 results")
            return False
            
    except Exception as e:
        print(f"❌ Rebuild failed: {e}")
        logger.error(f"RAG collection rebuild failed: {e}")
        return False

async def main():
    """Main rebuild function."""
    success = await rebuild_rag_collection()
    
    if success:
        print("\n🎯 NEXT STEPS:")
        print("1. Test the Discord bot RAG functionality")
        print("2. Try queries like 'what is the Gateway Process?'")
        print("3. Verify that relevant chunks are now returned")
    else:
        print("\n🔧 TROUBLESHOOTING NEEDED:")
        print("1. Check ChromaDB database permissions")
        print("2. Verify embedding model is working correctly")
        print("3. Check for any remaining configuration issues")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
