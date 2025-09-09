#!/usr/bin/env python3
"""
RAG Pipeline Debug Script

This script performs a comprehensive audit of the RAG pipeline to identify
why queries are returning 0 results. It traces through:
1. PDF ingestion and text extraction
2. Text chunking and preprocessing  
3. Embedding generation and storage
4. Query embedding and similarity search
5. Retrieval scoring and ranking

Usage: python debug_rag_pipeline.py
"""

import asyncio
import os
import sys
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bot.rag.hybrid_search import get_hybrid_search
from bot.rag.chroma_backend import ChromaRAGBackend
from bot.rag.document_parsers import document_parser_factory
from bot.rag.text_chunker import create_chunker
from bot.rag.vector_schema import HybridSearchConfig
from bot.rag.embedding_interface import create_embedding_model
from bot.utils.logging import get_logger

logger = get_logger(__name__)

class RAGPipelineDebugger:
    """Comprehensive RAG pipeline debugger."""
    
    def __init__(self):
        self.kb_path = Path("kb")
        self.db_path = Path("./chroma_db")
        self.config = HybridSearchConfig()
        self.test_queries = [
            "what is the Gateway Process?",
            "Gateway Process",
            "consciousness",
            "meditation",
            "CIA",
            "remote viewing",
            "Hemi-Sync"
        ]
        
    async def run_full_audit(self):
        """Run complete RAG pipeline audit."""
        print("🔍 RAG PIPELINE COMPREHENSIVE AUDIT")
        print("=" * 50)
        
        # Step 1: Check knowledge base files
        await self._audit_knowledge_base()
        
        # Step 2: Test document parsing
        await self._audit_document_parsing()
        
        # Step 3: Test text chunking
        await self._audit_text_chunking()
        
        # Step 4: Audit ChromaDB collection
        await self._audit_chromadb_collection()
        
        # Step 5: Test embedding generation
        await self._audit_embedding_generation()
        
        # Step 6: Test search and retrieval
        await self._audit_search_retrieval()
        
        # Step 7: Debug specific query
        await self._debug_specific_query("what is the Gateway Process?")
        
        print("\n✅ RAG PIPELINE AUDIT COMPLETED")
        
    async def _audit_knowledge_base(self):
        """Audit knowledge base files."""
        print("\n📁 STEP 1: KNOWLEDGE BASE AUDIT")
        print("-" * 30)
        
        if not self.kb_path.exists():
            print(f"❌ Knowledge base directory not found: {self.kb_path}")
            return
            
        files = list(self.kb_path.glob("*"))
        print(f"📊 Found {len(files)} files in knowledge base:")
        
        for file_path in files:
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                extension = file_path.suffix.lower()
                supported = extension in document_parser_factory.get_supported_extensions()
                status = "✅ SUPPORTED" if supported else "❌ UNSUPPORTED"
                print(f"  {file_path.name} ({size_mb:.2f} MB) - {status}")
                
    async def _audit_document_parsing(self):
        """Test document parsing for key files."""
        print("\n📄 STEP 2: DOCUMENT PARSING AUDIT")
        print("-" * 35)
        
        # Test parsing the CIA Gateway Process document
        gateway_file = self.kb_path / "CIA-RDP96-00788R001700210016-5.pdf"
        
        if not gateway_file.exists():
            print("❌ Gateway Process PDF not found")
            return
            
        try:
            print(f"🔍 Parsing: {gateway_file.name}")
            content, metadata = await document_parser_factory.parse_document(gateway_file)
            
            print("✅ Parsing successful!")
            print(f"📊 Content length: {len(content):,} characters")
            print(f"📊 Line count: {len(content.splitlines()):,}")
            print(f"📊 Parser metadata: {json.dumps(metadata, indent=2)}")
            
            # Check for key terms
            key_terms = ["Gateway Process", "consciousness", "Hemi-Sync", "meditation"]
            found_terms = []
            for term in key_terms:
                if term.lower() in content.lower():
                    count = content.lower().count(term.lower())
                    found_terms.append(f"{term}: {count} occurrences")
                    
            print(f"🔍 Key terms found: {', '.join(found_terms) if found_terms else 'None'}")
            
            # Show first 500 characters
            print("📝 Content preview (first 500 chars):")
            print(f"   {repr(content[:500])}")
            
        except Exception as e:
            print(f"❌ Parsing failed: {e}")
            
    async def _audit_text_chunking(self):
        """Test text chunking process."""
        print("\n✂️ STEP 3: TEXT CHUNKING AUDIT")
        print("-" * 30)
        
        gateway_file = self.kb_path / "CIA-RDP96-00788R001700210016-5.pdf"
        
        if not gateway_file.exists():
            print("❌ Gateway Process PDF not found")
            return
            
        try:
            # Parse document
            content, metadata = await document_parser_factory.parse_document(gateway_file)
            
            # Create chunker
            chunker = create_chunker("pdf", self.config)
            
            # Chunk the content
            chunking_result = chunker.chunk_text(content, metadata)
            
            print("✅ Chunking successful!")
            print(f"📊 Total chunks created: {len(chunking_result.chunks)}")
            print(f"📊 Chunking method: {chunking_result.chunking_method}")
            print(f"📊 Chunk size range: {chunking_result.chunk_size_range}")
            print(f"📊 Average chunk size: {chunking_result.average_chunk_size}")
            
            # Analyze chunks for key terms
            gateway_chunks = []
            for i, chunk in enumerate(chunking_result.chunks):
                if "gateway process" in chunk.content.lower():
                    gateway_chunks.append(i)
                    
            print(f"🔍 Chunks containing 'Gateway Process': {len(gateway_chunks)}")
            
            # Show sample chunks
            print("\n📝 Sample chunks (first 3):")
            for i, chunk in enumerate(chunking_result.chunks[:3]):
                print(f"   Chunk {i}: {len(chunk.content)} chars")
                print(f"   Preview: {repr(chunk.content[:200])}")
                print(f"   Metadata: {chunk.metadata}")
                print()
                
        except Exception as e:
            print(f"❌ Chunking failed: {e}")
            
    async def _audit_chromadb_collection(self):
        """Audit ChromaDB collection state."""
        print("\n🗄️ STEP 4: CHROMADB COLLECTION AUDIT")
        print("-" * 38)
        
        try:
            # Initialize ChromaDB backend
            backend = ChromaRAGBackend(
                db_path=str(self.db_path),
                config=self.config
            )
            await backend.initialize()
            
            # Get collection stats
            stats = await backend.get_collection_stats()
            print("✅ ChromaDB connection successful!")
            print(f"📊 Collection stats: {json.dumps(stats, indent=2)}")
            
            # Try to peek at stored data
            if stats.get("total_chunks", 0) > 0:
                print("\n🔍 Attempting to peek at stored documents...")
                
                # Get a small sample of documents
                try:
                    import chromadb
                    client = chromadb.PersistentClient(path=str(self.db_path))
                    collection = client.get_collection("knowledge_base")
                    
                    # Get first 5 documents
                    result = collection.peek(limit=5)
                    
                    print("📊 Sample documents in collection:")
                    if result.get('ids'):
                        for i, doc_id in enumerate(result['ids']):
                            metadata = result.get('metadatas', [{}])[i] if i < len(result.get('metadatas', [])) else {}
                            document = result.get('documents', [''])[i] if i < len(result.get('documents', [])) else ''
                            
                            print(f"   Doc {i+1}: ID={doc_id}")
                            print(f"   Metadata: {metadata}")
                            print(f"   Content preview: {repr(document[:200])}")
                            print()
                            
                except Exception as e:
                    print(f"⚠️ Could not peek at documents: {e}")
            else:
                print("⚠️ Collection is empty - no documents stored")
                
            await backend.close()
            
        except Exception as e:
            print(f"❌ ChromaDB audit failed: {e}")
            
    async def _audit_embedding_generation(self):
        """Test embedding generation."""
        print("\n🧮 STEP 5: EMBEDDING GENERATION AUDIT")
        print("-" * 40)
        
        try:
            # Create embedding model
            model_type = os.getenv("RAG_EMBEDDING_MODEL_TYPE", "sentence-transformers")
            model_name = os.getenv("RAG_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
            
            embedding_model = create_embedding_model(model_type, model_name=model_name)
            
            print(f"✅ Embedding model loaded: {embedding_model.model_name}")
            print(f"📊 Embedding dimension: {await embedding_model.get_embedding_dimension()}")
            
            # Test embedding generation
            test_texts = [
                "Gateway Process consciousness exploration",
                "what is the Gateway Process?",
                "CIA remote viewing techniques"
            ]
            
            print("\n🔍 Testing embedding generation:")
            for text in test_texts:
                try:
                    embedding = await embedding_model.encode_single(text)
                    print(f"   Text: '{text}'")
                    print(f"   Embedding shape: {len(embedding)}")
                    print(f"   Embedding preview: {embedding[:5]}...")
                    print()
                except Exception as e:
                    print(f"   ❌ Failed to embed '{text}': {e}")
                    
        except Exception as e:
            print(f"❌ Embedding audit failed: {e}")
            
    async def _audit_search_retrieval(self):
        """Test search and retrieval functionality."""
        print("\n🔍 STEP 6: SEARCH & RETRIEVAL AUDIT")
        print("-" * 38)
        
        try:
            # Get hybrid search instance
            search_engine = await get_hybrid_search()
            
            # Get search statistics
            stats = await search_engine.get_stats()
            print("✅ Hybrid search engine initialized!")
            print(f"📊 Search engine stats: {json.dumps(stats, indent=2)}")
            
            # Test each query
            print("\n🔍 Testing search queries:")
            for query in self.test_queries:
                try:
                    print(f"\n   Query: '{query}'")
                    
                    # Test vector search specifically
                    results = await search_engine._vector_search(
                        query=query,
                        user_id=None,
                        guild_id=None,
                        max_results=5
                    )
                    
                    print(f"   Vector results: {len(results)}")
                    for i, result in enumerate(results):
                        print(f"     {i+1}. Score: {result.score:.4f}")
                        print(f"        Title: {result.title}")
                        print(f"        Source: {result.source}")
                        print(f"        Snippet: {result.snippet[:100]}...")
                        print()
                        
                except Exception as e:
                    print(f"   ❌ Search failed: {e}")
                    
        except Exception as e:
            print(f"❌ Search audit failed: {e}")
            
    async def _debug_specific_query(self, query: str):
        """Deep debug of a specific query."""
        print(f"\n🐛 STEP 7: DEEP DEBUG - '{query}'")
        print("-" * 50)
        
        try:
            # Initialize components manually
            backend = ChromaRAGBackend(
                db_path=str(self.db_path),
                config=self.config
            )
            await backend.initialize()
            
            # Get embedding model
            model_type = os.getenv("RAG_EMBEDDING_MODEL_TYPE", "sentence-transformers")
            model_name = os.getenv("RAG_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
            embedding_model = create_embedding_model(model_type, model_name=model_name)
            
            print(f"🔍 Debugging query: '{query}'")
            
            # Step 1: Generate query embedding
            print("\n1️⃣ Generating query embedding...")
            query_embedding = await embedding_model.encode_single(query)
            print(f"   ✅ Query embedding generated: {len(query_embedding)} dimensions")
            print(f"   Embedding preview: {query_embedding[:10]}")
            
            # Step 2: Direct ChromaDB search
            print("\n2️⃣ Direct ChromaDB search...")
            try:
                import chromadb
                client = chromadb.PersistentClient(path=str(self.db_path))
                collection = client.get_collection("knowledge_base")
                
                # Perform direct search
                search_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=10,
                    include=['documents', 'metadatas', 'distances']
                )
                
                print("   ✅ Direct search completed")
                print(f"   Results found: {len(search_results.get('ids', [[]])[0])}")
                
                if search_results.get('ids') and search_results['ids'][0]:
                    print("   Top results:")
                    for i in range(min(3, len(search_results['ids'][0]))):
                        doc_id = search_results['ids'][0][i]
                        distance = search_results.get('distances', [[]])[0][i] if search_results.get('distances') else 'N/A'
                        document = search_results.get('documents', [[]])[0][i] if search_results.get('documents') else ''
                        metadata = search_results.get('metadatas', [[]])[0][i] if search_results.get('metadatas') else {}
                        
                        print(f"     {i+1}. ID: {doc_id}")
                        print(f"        Distance: {distance}")
                        print(f"        Metadata: {metadata}")
                        print(f"        Content: {repr(document[:200])}")
                        print()
                else:
                    print("   ⚠️ No results returned from direct search")
                    
            except Exception as e:
                print(f"   ❌ Direct search failed: {e}")
            
            # Step 3: Backend search
            print("\n3️⃣ Backend search test...")
            try:
                backend_results = await backend.search(
                    query=query,
                    n_results=5
                )
                
                print("   ✅ Backend search completed")
                print(f"   Results: {len(backend_results)}")
                
                for i, result in enumerate(backend_results):
                    print(f"     {i+1}. Score: {result.score:.4f}")
                    print(f"        Content: {repr(result.content[:200])}")
                    print()
                    
            except Exception as e:
                print(f"   ❌ Backend search failed: {e}")
            
            await backend.close()
            
        except Exception as e:
            print(f"❌ Deep debug failed: {e}")

async def main():
    """Main debug function."""
    debugger = RAGPipelineDebugger()
    await debugger.run_full_audit()

if __name__ == "__main__":
    asyncio.run(main())
