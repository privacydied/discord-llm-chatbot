#!/usr/bin/env python3
"""
Search Pipeline Debug Script

This script debugs the vector search/retrieval mechanism to identify why
queries return 0 results despite having 5,558 chunks in the collection.

Usage: python debug_search_pipeline.py
"""

import asyncio
import sys
import numpy as np
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bot.rag.chroma_backend import ChromaRAGBackend
from bot.rag.vector_schema import HybridSearchConfig
from bot.rag.embedding_interface import create_embedding_model
from bot.util.logging import get_logger

logger = get_logger(__name__)

async def debug_search_pipeline():
    """Debug the vector search pipeline step by step."""
    print("🔍 SEARCH PIPELINE DEBUG")
    print("=" * 30)
    
    try:
        # Step 1: Initialize backend
        print("1️⃣ Initializing ChromaDB backend...")
        config = HybridSearchConfig()
        backend = ChromaRAGBackend(config=config)
        await backend.initialize()
        
        # Step 2: Check collection stats
        print("\n2️⃣ Checking collection stats...")
        stats = await backend.get_collection_stats()
        print(f"   Total chunks: {stats.get('total_chunks', 0)}")
        print(f"   Embedding model: {stats.get('embedding_model', 'unknown')}")
        print(f"   Embedding dimension: {stats.get('embedding_dimension', 'unknown')}")
        
        # Step 3: Test direct ChromaDB access
        print("\n3️⃣ Testing direct ChromaDB access...")
        try:
            import chromadb
            client = chromadb.PersistentClient(path="./chroma_db")
            collection = client.get_collection("knowledge_base")
            
            # Peek at stored documents
            peek_result = collection.peek(limit=3)
            print(f"   Documents in collection: {len(peek_result.get('ids', []))}")
            
            if peek_result.get('ids'):
                print("   Sample document IDs:")
                for i, doc_id in enumerate(peek_result['ids'][:3]):
                    metadata = peek_result.get('metadatas', [{}])[i]
                    document = peek_result.get('documents', [''])[i]
                    print(f"     {i+1}. ID: {doc_id}")
                    print(f"        Metadata: {metadata.get('filename', 'N/A')}")
                    print(f"        Content preview: {document[:100]}...")
                    print()
                    
        except Exception as e:
            print(f"   ❌ Direct ChromaDB access failed: {e}")
        
        # Step 4: Test embedding generation
        print("\n4️⃣ Testing query embedding generation...")
        embedding_model = create_embedding_model("sentence-transformers")
        
        test_query = "Gateway Process"
        print(f"   Query: '{test_query}'")
        
        try:
            query_embedding = await embedding_model.encode_single(test_query)
            print(f"   ✅ Query embedding generated: {len(query_embedding)} dimensions")
            print(f"   Embedding preview: {query_embedding[:5]}")
            
            # Check if embedding is valid (not all zeros)
            if np.allclose(query_embedding, 0):
                print("   ⚠️ WARNING: Query embedding is all zeros!")
            else:
                print("   ✅ Query embedding looks valid (non-zero values)")
                
        except Exception as e:
            print(f"   ❌ Query embedding failed: {e}")
            return False
        
        # Step 5: Test direct ChromaDB search
        print("\n5️⃣ Testing direct ChromaDB vector search...")
        try:
            # Perform direct vector search
            search_results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=5,
                include=['documents', 'metadatas', 'distances']
            )
            
            print(f"   Direct search results: {len(search_results.get('ids', [[]])[0])}")
            
            if search_results.get('ids') and search_results['ids'][0]:
                print("   Top results:")
                for i in range(min(3, len(search_results['ids'][0]))):
                    doc_id = search_results['ids'][0][i]
                    distance = search_results.get('distances', [[]])[0][i]
                    document = search_results.get('documents', [[]])[0][i]
                    
                    print(f"     {i+1}. ID: {doc_id}")
                    print(f"        Distance: {distance:.4f}")
                    print(f"        Content: {document[:100]}...")
                    print()
            else:
                print("   ❌ No results from direct search")
                
        except Exception as e:
            print(f"   ❌ Direct search failed: {e}")
        
        # Step 6: Test backend search method
        print("\n6️⃣ Testing backend search method...")
        try:
            backend_results = await backend.search(test_query, n_results=5)
            print(f"   Backend search results: {len(backend_results)}")
            
            for i, result in enumerate(backend_results[:3]):
                print(f"     {i+1}. Score: {result.score:.4f}")
                print(f"        Content: {result.content[:100]}...")
                print()
                
        except Exception as e:
            print(f"   ❌ Backend search failed: {e}")
        
        # Step 7: Check confidence threshold
        print("\n7️⃣ Checking confidence threshold settings...")
        print(f"   Vector confidence threshold: {config.vector_confidence_threshold}")
        print(f"   Max vector results: {config.max_vector_results}")
        print(f"   Log confidence scores: {config.log_confidence_scores}")
        
        # Step 8: Test with lower confidence threshold
        print("\n8️⃣ Testing with lowered confidence threshold...")
        try:
            # Temporarily lower the confidence threshold
            original_threshold = config.vector_confidence_threshold
            config.vector_confidence_threshold = 0.0  # Accept all results
            
            backend_low_threshold = ChromaRAGBackend(config=config)
            await backend_low_threshold.initialize()
            
            low_threshold_results = await backend_low_threshold.search(test_query, n_results=5)
            print(f"   Results with threshold=0.0: {len(low_threshold_results)}")
            
            for i, result in enumerate(low_threshold_results[:3]):
                print(f"     {i+1}. Score: {result.similarity_score:.4f}")
                print(f"        Content: {result.snippet[:100]}...")
                print()
            
            # Restore original threshold
            config.vector_confidence_threshold = original_threshold
            
            await backend_low_threshold.close()
            
        except Exception as e:
            print(f"   ❌ Low threshold test failed: {e}")
        
        await backend.close()
        
        print("\n✅ SEARCH PIPELINE DEBUG COMPLETED")
        return True
        
    except Exception as e:
        print(f"❌ Search pipeline debug failed: {e}")
        return False

async def main():
    """Main debug function."""
    success = await debug_search_pipeline()
    
    if success:
        print("\n🎯 ANALYSIS:")
        print("Check the output above to identify where the search pipeline breaks.")
        print("Common issues:")
        print("- Confidence threshold too high")
        print("- Embedding dimension mismatch")
        print("- Distance calculation problems")
        print("- ChromaDB query format issues")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
