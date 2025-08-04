"""
Bootstrap utility for ingesting existing knowledge base files into the RAG system.
"""
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio
import logging

from .chroma_backend import ChromaRAGBackend
from .vector_schema import HybridSearchConfig
from .document_parsers import document_parser_factory
from ..util.logging import get_logger

logger = get_logger(__name__)

async def main():
    # Configure logging
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.setLevel(logging.DEBUG)

    # ... existing code ...

class RAGBootstrap:
    """Bootstrap utility for initial RAG system setup and data ingestion."""
    
    def __init__(self, rag_backend: ChromaRAGBackend, kb_path: str = "kb"):
        self.rag_backend = rag_backend
        self.kb_path = Path(kb_path)
        self.version_file = Path("rag_versions.json")
        
    async def bootstrap_knowledge_base(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Bootstrap the RAG system by ingesting all knowledge base files.
        
        Args:
            force_refresh: If True, re-ingest all files regardless of version
            
        Returns:
            Dictionary with ingestion results and statistics
        """
        logger.info("[RAG Bootstrap] Starting knowledge base ingestion...")
        
        if not self.kb_path.exists():
            logger.warning(f"[RAG Bootstrap] Knowledge base path not found: {self.kb_path}")
            return {"error": "Knowledge base path not found", "files_processed": 0}
        
        # Load existing version tracking
        existing_versions = self._load_version_tracking()
        
        # Find all supported files in KB directory
        supported_files = self._find_supported_files()
        
        if not supported_files:
            logger.warning("[RAG Bootstrap] No supported files found in knowledge base")
            return {"warning": "No supported files found", "files_processed": 0}
        
        # Process each file
        results = {
            "files_processed": 0,
            "files_skipped": 0,
            "total_chunks": 0,
            "errors": [],
            "processed_files": []
        }
        
        for file_path in supported_files:
            try:
                file_result = await self._process_file(
                    file_path, 
                    existing_versions, 
                    force_refresh
                )
                
                if file_result["processed"]:
                    results["files_processed"] += 1
                    results["total_chunks"] += file_result["chunks_created"]
                    results["processed_files"].append({
                        "file": str(file_path.relative_to(self.kb_path)),
                        "chunks": file_result["chunks_created"],
                        "status": "processed"
                    })
                else:
                    results["files_skipped"] += 1
                    results["processed_files"].append({
                        "file": str(file_path.relative_to(self.kb_path)),
                        "chunks": 0,
                        "status": "skipped (no changes)"
                    })
                    
            except Exception as e:
                error_msg = f"Failed to process {file_path}: {e}"
                logger.error(f"[RAG Bootstrap] {error_msg}")
                results["errors"].append(error_msg)
        
        # Save updated version tracking
        self._save_version_tracking(existing_versions)
        
        # Get collection statistics
        stats = await self.rag_backend.get_collection_stats()
        results["collection_stats"] = stats
        
        logger.info(f"[RAG Bootstrap] Completed: {results['files_processed']} processed, "
                   f"{results['files_skipped']} skipped, {results['total_chunks']} total chunks")
        
        return results
    
    async def _process_file(
        self, 
        file_path: Path, 
        version_tracking: Dict[str, str], 
        force_refresh: bool
    ) -> Dict[str, Any]:
        """Process a single knowledge base file."""
        relative_path = str(file_path.relative_to(self.kb_path))
        logger.debug(f"[RAG Bootstrap] Processing file: {relative_path}")
        try:
            # Parse document content using appropriate parser
            try:
                # Parse the document
                content, doc_metadata = await document_parser_factory.parse_document(file_path)
                
            except Exception as e:
                logger.error(f"[RAG Bootstrap] Failed to parse {relative_path}: {e}")
                return {"processed": False, "error": str(e)}
            
            if not content.strip():
                logger.warning(f"[RAG Bootstrap] Empty content after parsing: {relative_path}")
                return {"processed": False, "error": "Empty content after parsing"}
            
            # Calculate content hash
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            # Check if file has changed (incremental scanning logic)
            if not force_refresh:
                if relative_path in version_tracking:
                    existing_hash = version_tracking[relative_path]
                    if existing_hash == content_hash:
                        logger.info(f"[RAG Bootstrap] ⏭️ Skipping unchanged file: {relative_path} (hash: {content_hash[:8]}...)")
                        return {"processed": False, "chunks_created": 0}
                    else:
                        logger.info(f"[RAG Bootstrap] 🔄 File changed, re-processing: {relative_path} (old: {existing_hash[:8]}..., new: {content_hash[:8]}...)")
                else:
                    logger.info(f"[RAG Bootstrap] 🆕 New file detected: {relative_path} (hash: {content_hash[:8]}...)")
            else:
                logger.info(f"[RAG Bootstrap] 🔄 Force refresh enabled, processing: {relative_path}")
            
            # Determine file type for chunking based on parser metadata
            file_type = self._determine_file_type(file_path, doc_metadata)
            
            # Prepare metadata (combine base metadata with parser metadata)
            metadata = {
                "filename": file_path.name,
                "filepath": relative_path,
                "file_type": file_type,
                "source_type": "knowledge_base",
                "ingestion_method": "bootstrap"
            }
            # Add parser-specific metadata
            metadata.update(doc_metadata)
            
            # Ingest the document
            logger.info(f"[RAG Bootstrap] Processing: {relative_path}")
            documents = await self.rag_backend.add_document(
                source_id=relative_path,
                text=content,
                metadata=metadata,
                file_type=file_type
            )
            
            # Update version tracking
            version_tracking[relative_path] = content_hash
            
            return {
                "processed": True,
                "chunks_created": len(documents),
                "content_hash": content_hash
            }
        
        finally:
            logger.debug(f"[RAG Bootstrap] Finished processing file: {relative_path}")
    
    def _find_supported_files(self) -> List[Path]:
        """Find all supported files in the knowledge base directory."""
        supported_extensions = document_parser_factory.get_supported_extensions()
        supported_files = []
        
        for file_path in self.kb_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                supported_files.append(file_path)
        
        # Sort for consistent processing order
        supported_files.sort()
        
        logger.debug(f"[RAG Bootstrap] Found {len(supported_files)} supported files")
        return supported_files
    
    def _determine_file_type(self, file_path: Path, doc_metadata: dict) -> str:
        """Determine the file type for chunking strategy based on parser metadata."""
        content_type = doc_metadata.get('content_type', '')
        extension = file_path.suffix.lower()
        
        # Map content types to chunking strategies
        if 'markdown' in content_type or extension in {'.md', '.markdown'}:
            return "markdown"
        elif 'html' in content_type or extension in {'.html', '.htm'}:
            return "html"
        elif 'pdf' in content_type or extension == '.pdf':
            return "pdf"
        elif 'epub' in content_type or extension == '.epub':
            return "epub"
        elif 'docx' in content_type or extension in {'.docx', '.docm'}:
            return "docx"
        else:
            return "text"
    
    def _load_version_tracking(self) -> Dict[str, str]:
        """Load file version tracking from disk."""
        if not self.version_file.exists():
            logger.info(f"[RAG Bootstrap] 🆕 No existing version tracking found at: {self.version_file}")
            return {}
        
        try:
            with open(self.version_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            # Check if file is empty
            if not content:
                logger.info(f"[RAG Bootstrap] 📄 Empty version tracking file, starting fresh")
                return {}
            
            # Attempt to parse JSON
            try:
                versions = json.loads(content)
            except json.JSONDecodeError as json_err:
                logger.error(f"[RAG Bootstrap] ❌ JSON parsing error in {self.version_file}: {json_err}")
                logger.error(f"[RAG Bootstrap] Error at line {json_err.lineno}, column {json_err.colno}: {json_err.msg}")
                
                # Try to recover by backing up the corrupted file and starting fresh
                backup_file = self.version_file.with_suffix('.json.backup')
                try:
                    import shutil
                    shutil.copy2(self.version_file, backup_file)
                    logger.warning(f"[RAG Bootstrap] 💾 Backed up corrupted file to: {backup_file}")
                except Exception as backup_err:
                    logger.error(f"[RAG Bootstrap] Failed to backup corrupted file: {backup_err}")
                
                logger.warning(f"[RAG Bootstrap] 🔄 Starting with empty version tracking due to JSON corruption")
                return {}
            
            # Validate the loaded data
            if not isinstance(versions, dict):
                logger.warning(f"[RAG Bootstrap] ⚠️ Invalid version tracking format (not a dict), starting fresh")
                return {}
            
            # Sanitize loaded data
            sanitized_versions = {}
            for file_path, version_hash in versions.items():
                if isinstance(file_path, str) and isinstance(version_hash, str):
                    sanitized_path = file_path.strip()
                    sanitized_hash = version_hash.strip()
                    
                    # Validate hash format
                    if sanitized_hash and all(c in '0123456789abcdefABCDEF' for c in sanitized_hash):
                        sanitized_versions[sanitized_path] = sanitized_hash
                    else:
                        logger.warning(f"[RAG Bootstrap] Skipping invalid hash entry: {sanitized_path} -> {sanitized_hash}")
                else:
                    logger.warning(f"[RAG Bootstrap] Skipping invalid entry type: {type(file_path)} -> {type(version_hash)}")
            
            logger.info(f"[RAG Bootstrap] 📁 Loaded version tracking for {len(sanitized_versions)} files from: {self.version_file}")
            if sanitized_versions:
                # Show a few examples for debugging
                sample_files = list(sanitized_versions.keys())[:3]
                logger.debug(f"[RAG Bootstrap] Sample tracked files: {sample_files}")
            
            return sanitized_versions
            
        except Exception as e:
            logger.error(f"[RAG Bootstrap] ❌ Failed to load version tracking from {self.version_file}: {e}", exc_info=True)
            return {}
    
    def _save_version_tracking(self, versions: Dict[str, str]) -> None:
        """Save file version tracking to disk."""
        try:
            # Sanitize the versions dictionary to ensure valid JSON
            sanitized_versions = {}
            for file_path, version_hash in versions.items():
                # Ensure file paths are properly escaped and valid
                sanitized_path = str(file_path).replace('\\', '/').strip()
                sanitized_hash = str(version_hash).strip()
                
                # Validate that the hash is a valid hex string
                if sanitized_hash and all(c in '0123456789abcdefABCDEF' for c in sanitized_hash):
                    sanitized_versions[sanitized_path] = sanitized_hash
                else:
                    logger.warning(f"[RAG Bootstrap] Skipping invalid hash for {sanitized_path}: {sanitized_hash}")
            
            with open(self.version_file, 'w', encoding='utf-8') as f:
                json.dump(
                    sanitized_versions, 
                    f, 
                    indent=2, 
                    ensure_ascii=False,
                    separators=(',', ': '),
                    sort_keys=True
                )
            logger.info(f"[RAG Bootstrap] 💾 Saved version tracking for {len(sanitized_versions)} files to: {self.version_file}")
            if sanitized_versions:
                # Show a few examples for debugging
                sample_files = list(sanitized_versions.keys())[:3]
                logger.debug(f"[RAG Bootstrap] Sample saved files: {sample_files}")
        except Exception as e:
            logger.error(f"[RAG Bootstrap] ❌ Failed to save version tracking to {self.version_file}: {e}", exc_info=True)
    
    async def reset_knowledge_base(self) -> Dict[str, Any]:
        """
        Reset the entire knowledge base by removing all documents.
        
        Returns:
            Dictionary with reset results
        """
        logger.warning("[RAG Bootstrap] Resetting entire knowledge base...")
        
        try:
            # Get all documents
            await self.rag_backend.initialize()
            
            # This is a bit brute force - ChromaDB doesn't have a clear collection method
            # So we'll delete the collection and recreate it
            if self.rag_backend.client and self.rag_backend.collection:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self.rag_backend.client.delete_collection(
                        self.rag_backend.collection_name
                    )
                )
                
                # Recreate collection
                self.rag_backend.collection = self.rag_backend.client.get_or_create_collection(
                    name=self.rag_backend.collection_name,
                    metadata={"description": "RAG knowledge base collection"}
                )
            
            # Clear version tracking
            if self.version_file.exists():
                self.version_file.unlink()
            
            logger.info("[RAG Bootstrap] Knowledge base reset completed")
            
            return {
                "status": "success",
                "message": "Knowledge base reset successfully"
            }
            
        except Exception as e:
            logger.error(f"[RAG Bootstrap] Failed to reset knowledge base: {e}")
            return {
                "status": "error",
                "message": f"Reset failed: {e}"
            }
    
    async def incremental_update(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform incremental update of changed files.
        
        Args:
            file_path: Optional specific file to update. If None, checks all files.
            
        Returns:
            Dictionary with update results
        """
        logger.info("[RAG Bootstrap] Starting incremental update...")
        
        if file_path:
            # Update specific file
            target_path = self.kb_path / file_path
            if not target_path.exists():
                return {"error": f"File not found: {file_path}"}
            
            files_to_check = [target_path]
        else:
            # Check all files
            files_to_check = self._find_supported_files()
        
        existing_versions = self._load_version_tracking()
        
        results = {
            "files_updated": 0,
            "files_checked": len(files_to_check),
            "total_chunks": 0,
            "errors": []
        }
        
        for file_path in files_to_check:
            try:
                file_result = await self._process_file(
                    file_path, 
                    existing_versions, 
                    force_refresh=False
                )
                
                if file_result["processed"]:
                    results["files_updated"] += 1
                    results["total_chunks"] += file_result["chunks_created"]
                    
            except Exception as e:
                error_msg = f"Failed to update {file_path}: {e}"
                logger.error(f"[RAG Bootstrap] {error_msg}")
                results["errors"].append(error_msg)
        
        # Save updated version tracking
        self._save_version_tracking(existing_versions)
        
        logger.info(f"[RAG Bootstrap] Incremental update completed: "
                   f"{results['files_updated']}/{results['files_checked']} files updated")
        
        return results


async def create_rag_system(
    kb_path: str = "kb",
    db_path: str = "./chroma_db",
    config: Optional[HybridSearchConfig] = None
) -> tuple[ChromaRAGBackend, RAGBootstrap]:
    """
    Factory function to create and initialize a complete RAG system.
    
    Args:
        kb_path: Path to knowledge base directory
        db_path: Path to ChromaDB storage
        config: Optional configuration
        
    Returns:
        Tuple of (ChromaRAGBackend, RAGBootstrap)
    """
    if config is None:
        config = HybridSearchConfig()
    
    # Create RAG backend
    rag_backend = ChromaRAGBackend(
        db_path=db_path,
        config=config
    )
    
    # Create bootstrap utility
    bootstrap = RAGBootstrap(rag_backend, kb_path)
    
    # Initialize the backend
    await rag_backend.initialize()
    
    logger.info("✔ RAG system created and initialized")
    
    return rag_backend, bootstrap

if __name__ == "__main__":
    asyncio.run(main())
