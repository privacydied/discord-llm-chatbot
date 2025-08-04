"""
ChromaDB backend implementation for RAG vector storage.
"""
import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

from .embedding_interface import EmbeddingInterface, create_embedding_model
from .vector_schema import VectorDocument, HybridSearchConfig, SearchResult
from .text_chunker import TextChunker, create_chunker
from ..util.logging import get_logger

logger = get_logger(__name__)


class ChromaRAGBackend:
    """ChromaDB-based RAG backend with hybrid search capabilities."""
    
    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection_name: str = "knowledge_base",
        embedding_model: Optional[EmbeddingInterface] = None,
        config: Optional[HybridSearchConfig] = None
    ):
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.config = config or HybridSearchConfig()
        self.config.validate()
        
        # Initialize embedding model
        if embedding_model is None:
            model_type = os.getenv("RAG_EMBEDDING_MODEL_TYPE", "sentence-transformers")
            model_name = os.getenv("RAG_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
            self.embedding_model = create_embedding_model(model_type, model_name=model_name)
        else:
            self.embedding_model = embedding_model
        
        # ChromaDB components (lazy initialization)
        self.client = None
        self.collection = None
        self._initialized = False
        
        # Text chunker
        self.chunker = TextChunker(self.config)
        
    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        if self._initialized:
            return
            
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create database directory
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "RAG knowledge base collection"}
            )
            
            self._initialized = True
            logger.info(f"✔ ChromaDB initialized [path={self.db_path}, collection={self.collection_name}]")
            
        except ImportError:
            logger.error("chromadb not installed. Install with: pip install chromadb")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    async def add_document(
        self,
        source_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        file_type: str = "text"
    ) -> List[VectorDocument]:
        """
        Add a document to the vector store with chunking and embedding.
        
        Args:
            source_id: Unique identifier for the source document
            text: Document text content
            metadata: Optional metadata dictionary
            file_type: Type of file for appropriate chunking
            
        Returns:
            List of created VectorDocument objects
        """
        await self.initialize()
        
        if not text.strip():
            logger.warning(f"[RAG] Empty document provided: {source_id}")
            return []
        
        try:
            # Chunk the document
            chunker = create_chunker(file_type, self.config)
            chunking_result = chunker.chunk_text(text, metadata)
            
            if not chunking_result.chunks:
                logger.warning(f"[RAG] No chunks generated for document: {source_id}")
                return []
            
            # Generate embeddings for all chunks
            embeddings = await self.embedding_model.encode(chunking_result.chunks)
            
            # Create VectorDocument objects
            documents = []
            chunk_metadata = metadata or {}
            chunk_metadata.update(chunking_result.metadata)
            
            for i, (chunk_text, embedding) in enumerate(zip(chunking_result.chunks, embeddings)):
                doc = VectorDocument.create(
                    source_id=source_id,
                    chunk_text=chunk_text,
                    embedding=embedding,
                    chunk_index=i,
                    metadata=chunk_metadata.copy()
                )
                documents.append(doc)
            
            # Store in ChromaDB
            await self._store_documents(documents)
            
            logger.info(f"[RAG] Added document: {source_id} → {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logger.error(f"[RAG] Failed to add document {source_id}: {e}")
            raise
    
    async def _store_documents(self, documents: List[VectorDocument]) -> None:
        """Store documents in ChromaDB collection."""
        if not documents:
            return
        
        # Prepare data for ChromaDB
        ids = [doc.id for doc in documents]
        embeddings = [doc.embedding for doc in documents]
        
        # Create sanitized metadata dictionaries (exclude embedding vector and non-serializable fields)
        metadatas = []
        for doc in documents:
            # Only include primitive types in metadata (str, int, float, bool, None)
            sanitized_metadata = {
                'id': doc.id,
                'source_id': doc.source_id,
                'version_hash': doc.version_hash,
                'chunk_index': doc.chunk_index,
                'confidence_score': float(doc.confidence_score),
                'created_at': doc.created_at.isoformat(),
                'updated_at': doc.updated_at.isoformat()
            }
            
            # Add any additional metadata fields that are primitive types
            for key, value in doc.metadata.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    sanitized_metadata[key] = value
            
            metadatas.append(sanitized_metadata)
        
        documents_text = [doc.chunk_text for doc in documents]
        
        # Store in ChromaDB (run in thread pool to avoid blocking)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_text
            )
        )
        
        logger.debug(f"[RAG] Stored {len(documents)} documents in ChromaDB")
    
    async def search(
        self,
        query: str,
        n_results: int = 5,
        user_id: Optional[str] = None,
        guild_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform vector similarity search.
        
        Args:
            query: Search query text
            n_results: Maximum number of results to return
            user_id: Optional user ID for scoped search
            guild_id: Optional guild ID for scoped search
            filters: Optional metadata filters
            
        Returns:
            List of SearchResult objects
        """
        await self.initialize()
        
        try:
            # Generate query embedding
            query_embedding = await self.embedding_model.encode_single(query)
            
            # Build metadata filters
            where_clause = self._build_where_clause(user_id, guild_id, filters)
            
            # Perform vector search (run in thread pool)
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=min(n_results, self.config.max_vector_results),
                    where=where_clause,
                    include=["metadatas", "documents", "distances"]
                )
            )
            
            # Convert to SearchResult objects
            search_results = []
            if results['ids'] and results['ids'][0]:  # Check if we have results
                for i, (doc_id, metadata, document_text, distance) in enumerate(zip(
                    results['ids'][0],
                    results['metadatas'][0],
                    results['documents'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (ChromaDB uses L2 distance)
                    similarity_score = max(0.0, 1.0 - distance)
                    
                    # Skip results below confidence threshold
                    if similarity_score < self.config.vector_confidence_threshold:
                        if self.config.log_confidence_scores:
                            logger.debug(f"[RAG] Skipping low confidence result: {similarity_score:.3f}")
                        continue
                    
                    # Reconstruct VectorDocument from metadata
                    vector_doc = VectorDocument.from_dict(metadata)
                    
                    search_result = SearchResult(
                        document=vector_doc,
                        similarity_score=similarity_score,
                        search_type="vector",
                        rank=i + 1,
                        explanation=f"Vector similarity: {similarity_score:.3f}"
                    )
                    search_results.append(search_result)
            
            if self.config.log_retrieval_paths:
                logger.debug(f"[RAG] Vector search: '{query}' → {len(search_results)} results")
            
            return search_results
            
        except Exception as e:
            logger.error(f"[RAG] Vector search failed: {e}")
            if self.config.fallback_to_keyword_on_failure:
                logger.warning("[RAG] Falling back to keyword search")
                return await self._keyword_search_fallback(query, n_results, user_id, guild_id)
            raise
    
    async def _keyword_search_fallback(
        self,
        query: str,
        n_results: int,
        user_id: Optional[str] = None,
        guild_id: Optional[str] = None
    ) -> List[SearchResult]:
        """Fallback keyword search when vector search fails."""
        # This is a simple implementation - could be enhanced with proper text search
        try:
            where_clause = self._build_where_clause(user_id, guild_id)
            
            # Get all documents and filter by keyword (not efficient, but works for small datasets)
            loop = asyncio.get_event_loop()
            all_results = await loop.run_in_executor(
                None,
                lambda: self.collection.get(
                    where=where_clause,
                    include=["metadatas", "documents"]
                )
            )
            
            # Simple keyword matching
            query_words = set(query.lower().split())
            matches = []
            
            if all_results['ids']:
                for i, (doc_id, metadata, document_text) in enumerate(zip(
                    all_results['ids'],
                    all_results['metadatas'],
                    all_results['documents']
                )):
                    doc_words = set(document_text.lower().split())
                    overlap = len(query_words.intersection(doc_words))
                    
                    if overlap > 0:
                        score = overlap / len(query_words)  # Simple scoring
                        vector_doc = VectorDocument.from_dict(metadata)
                        
                        matches.append(SearchResult(
                            document=vector_doc,
                            similarity_score=score,
                            search_type="keyword_fallback",
                            rank=i + 1,
                            explanation=f"Keyword overlap: {overlap}/{len(query_words)} words"
                        ))
            
            # Sort by score and limit results
            matches.sort(key=lambda x: x.similarity_score, reverse=True)
            return matches[:n_results]
            
        except Exception as e:
            logger.error(f"[RAG] Keyword fallback search failed: {e}")
            return []
    
    def _build_where_clause(
        self,
        user_id: Optional[str] = None,
        guild_id: Optional[str] = None,
        additional_filters: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Build ChromaDB where clause for filtering."""
        where_conditions = []
        
        # Add user scoping if enabled and provided
        if self.config.enforce_user_scoping and user_id:
            where_conditions.append({"metadata.user_id": {"$eq": user_id}})
        
        # Add guild scoping if enabled and provided
        if self.config.enforce_guild_scoping and guild_id:
            where_conditions.append({"metadata.guild_id": {"$eq": guild_id}})
        
        # Add additional filters
        if additional_filters:
            for key, value in additional_filters.items():
                where_conditions.append({f"metadata.{key}": {"$eq": value}})
        
        # Combine conditions with AND
        if len(where_conditions) == 0:
            return None
        elif len(where_conditions) == 1:
            return where_conditions[0]
        else:
            return {"$and": where_conditions}
    
    async def update_document(
        self,
        source_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        file_type: str = "text"
    ) -> List[VectorDocument]:
        """
        Update an existing document by removing old chunks and adding new ones.
        
        Args:
            source_id: Source document identifier
            text: Updated document text
            metadata: Optional updated metadata
            file_type: File type for chunking
            
        Returns:
            List of new VectorDocument objects
        """
        await self.initialize()
        
        try:
            # Remove existing documents with this source_id
            await self.remove_document(source_id)
            
            # Add updated document
            return await self.add_document(source_id, text, metadata, file_type)
            
        except Exception as e:
            logger.error(f"[RAG] Failed to update document {source_id}: {e}")
            raise
    
    async def remove_document(self, source_id: str) -> int:
        """
        Remove all chunks for a source document.
        
        Args:
            source_id: Source document identifier
            
        Returns:
            Number of chunks removed
        """
        await self.initialize()
        
        try:
            # Find all chunks for this source
            loop = asyncio.get_event_loop()
            existing = await loop.run_in_executor(
                None,
                lambda: self.collection.get(
                    where={"metadata.source_id": {"$eq": source_id}},
                    include=["metadatas"]
                )
            )
            
            if existing['ids']:
                # Delete the chunks
                await loop.run_in_executor(
                    None,
                    lambda: self.collection.delete(ids=existing['ids'])
                )
                
                logger.info(f"[RAG] Removed {len(existing['ids'])} chunks for document: {source_id}")
                return len(existing['ids'])
            
            return 0
            
        except Exception as e:
            logger.error(f"[RAG] Failed to remove document {source_id}: {e}")
            raise
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        await self.initialize()
        
        try:
            loop = asyncio.get_event_loop()
            count = await loop.run_in_executor(
                None,
                lambda: self.collection.count()
            )
            
            return {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "db_path": str(self.db_path),
                "embedding_model": self.embedding_model.model_name,
                "embedding_dimension": await self.embedding_model.get_embedding_dimension()
            }
            
        except Exception as e:
            logger.error(f"[RAG] Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    async def wipe_collection(self) -> None:
        """Completely wipe the collection, removing all documents and embeddings."""
        await self.initialize()
        
        try:
            loop = asyncio.get_event_loop()
            
            # Get count before wiping for logging (with timeout)
            try:
                count_before = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.collection.count()
                    ),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.warning("[RAG] Count operation timed out, proceeding with wipe")
                count_before = "unknown"
            
            # Delete the entire collection (with timeout)
            await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.client.delete_collection(name=self.collection_name)
                ),
                timeout=60.0
            )
            
            # Small delay to allow cleanup
            await asyncio.sleep(0.1)
            
            # Recreate the collection (with timeout)
            self.collection = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.client.create_collection(
                        name=self.collection_name,
                        metadata={"description": "RAG knowledge base collection"}
                    )
                ),
                timeout=30.0
            )
            
            logger.info(f"[RAG] Successfully wiped collection '{self.collection_name}' ({count_before} chunks removed)")
            
        except asyncio.TimeoutError:
            logger.error(f"[RAG] Wipe operation timed out for collection '{self.collection_name}'")
            raise
        except Exception as e:
            logger.error(f"[RAG] Failed to wipe collection: {e}")
            raise
    
    async def close(self) -> None:
        """Clean up ChromaDB resources during shutdown."""
        try:
            if self.client:
                # ChromaDB doesn't have an explicit close method, but we should
                # clear references and log the cleanup
                logger.debug("[RAG] Closing ChromaDB backend")
                self.collection = None
                self.client = None
                self._initialized = False
                logger.info("[RAG] ✔ ChromaDB backend closed successfully")
        except Exception as e:
            logger.warning(f"[RAG] Error closing ChromaDB backend: {e}")
