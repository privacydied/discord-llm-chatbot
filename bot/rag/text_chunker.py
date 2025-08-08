"""
Text chunking utilities for RAG document processing.
"""
import re
from typing import List, Dict, Any, Optional
from .vector_schema import ChunkingResult, HybridSearchConfig
from ..util.logging import get_logger

logger = get_logger(__name__)


class TextChunker:
    """Handles intelligent text chunking for RAG ingestion."""
    
    def __init__(self, config: HybridSearchConfig):
        self.config = config
        
    def chunk_text(
        self, 
        text: str, 
        source_metadata: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """
        Chunk text into smaller pieces based on configuration.
        
        Args:
            text: Input text to be chunked
            source_metadata: Additional metadata to include
            
        Returns:
            ChunkingResult with chunks and metadata
        """
        if not text.strip():
            logger.warning("[RAG] Empty text provided for chunking")
            return ChunkingResult.create(text, [], source_metadata)
        
        # Store original text length before cleaning [CSD]
        original_text = text
        
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        # Try semantic chunking first (paragraph-aware)
        chunks = self._semantic_chunk(cleaned_text)
        
        # If chunks are too large, apply sliding window
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= self.config.chunk_size:
                final_chunks.append(chunk)
            else:
                # Split large chunks with sliding window
                sub_chunks = self._sliding_window_chunk(chunk)
                final_chunks.extend(sub_chunks)
        
        # Filter out chunks that are too small, but be more lenient [REH]
        # If we have very little text overall, allow smaller chunks to avoid losing content
        original_text_length = len(original_text.strip())
        effective_min_size = min(
            self.config.min_chunk_size,
            max(50, original_text_length // 2)  # Allow chunks as small as half the original text, min 50 chars
        )
        
        final_chunks = [
            chunk for chunk in final_chunks 
            if len(chunk.strip()) >= effective_min_size
        ]
        
        logger.debug(f"[RAG] Chunked text: {len(original_text)} chars → {len(final_chunks)} chunks")
        
        return ChunkingResult.create(original_text, final_chunks, source_metadata)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for chunking."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove markdown artifacts that might interfere with chunking
        text = re.sub(r'```[^`]*```', '', text)  # Remove code blocks
        text = re.sub(r'`[^`]+`', '', text)      # Remove inline code
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text.strip()
    
    def _semantic_chunk(self, text: str) -> List[str]:
        """
        Chunk text semantically by preserving paragraph boundaries.
        """
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            # Fallback to sentence splitting
            return self._sentence_chunk(text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) + 2 > self.config.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                # Start new chunk with overlap from previous
                if chunks and self.config.chunk_overlap > 0:
                    overlap_text = self._get_overlap_text(chunks[-1])
                    current_chunk = overlap_text + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _sentence_chunk(self, text: str) -> List[str]:
        """
        Chunk text by sentences when paragraph chunking fails.
        """
        # Simple sentence splitting (could be enhanced with NLTK/spaCy)
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > self.config.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                # Start new chunk with overlap
                if chunks and self.config.chunk_overlap > 0:
                    overlap_text = self._get_overlap_text(chunks[-1])
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _sliding_window_chunk(self, text: str) -> List[str]:
        """
        Apply sliding window chunking for large text segments.
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.config.chunk_size
            
            # Try to end at a word boundary
            if end < len(text):
                # Look for the last space within the chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if len(chunk) >= self.config.min_chunk_size:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.config.chunk_overlap
            if start <= 0:
                start = end
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """
        Get overlap text from the end of a chunk.
        """
        if len(text) <= self.config.chunk_overlap:
            return text
        
        # Get last N characters, but try to end at word boundary
        overlap_start = len(text) - self.config.chunk_overlap
        last_space = text.rfind(' ', overlap_start)
        
        if last_space > overlap_start:
            return text[last_space:].strip()
        else:
            return text[overlap_start:].strip()


class MarkdownChunker(TextChunker):
    """Specialized chunker for Markdown documents."""
    
    def _semantic_chunk(self, text: str) -> List[str]:
        """
        Chunk markdown by headers and sections.
        """
        # Split by headers (# ## ### etc.)
        header_pattern = r'\n(#{1,6}\s+[^\n]+)\n'
        sections = re.split(header_pattern, text)
        
        if len(sections) <= 1:
            # No headers found, use parent implementation
            return super()._semantic_chunk(text)
        
        chunks = []
        current_chunk = ""
        current_header = ""
        
        i = 0
        while i < len(sections):
            section = sections[i].strip()
            
            if not section:
                i += 1
                continue
            
            # Check if this is a header
            if re.match(r'^#{1,6}\s+', section):
                # This is a header
                if current_chunk and len(current_chunk) > self.config.min_chunk_size:
                    chunks.append(current_chunk.strip())
                
                current_header = section
                current_chunk = section
                
                # Get the content after this header
                if i + 1 < len(sections):
                    content = sections[i + 1].strip()
                    if content:
                        if len(current_chunk) + len(content) + 2 <= self.config.chunk_size:
                            current_chunk += "\n\n" + content
                        else:
                            # Content is too large, chunk it separately
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                            
                            # Chunk the large content
                            sub_chunks = self._sliding_window_chunk(content)
                            for j, sub_chunk in enumerate(sub_chunks):
                                if j == 0:
                                    # First sub-chunk gets the header
                                    chunks.append(current_header + "\n\n" + sub_chunk)
                                else:
                                    chunks.append(sub_chunk)
                            current_chunk = ""
                    i += 2  # Skip both header and content
                else:
                    i += 1
            else:
                # This is content without a header
                if len(current_chunk) + len(section) + 2 <= self.config.chunk_size:
                    if current_chunk:
                        current_chunk += "\n\n" + section
                    else:
                        current_chunk = section
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = section
                i += 1
        
        # Add final chunk
        if current_chunk.strip() and len(current_chunk) >= self.config.min_chunk_size:
            chunks.append(current_chunk.strip())
        
        return chunks


class HTMLChunker(TextChunker):
    """Specialized chunker for HTML documents."""
    
    def chunk_text(self, text: str, source_metadata: Optional[Dict[str, Any]] = None) -> ChunkingResult:
        """Chunk HTML text with awareness of structure."""
        # Remove excessive whitespace and normalize
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Try to split on common HTML-derived patterns
        # Look for section breaks, headers, etc.
        section_patterns = [
            r'\n\n(?=[A-Z][^\n]{10,}\n)',  # Likely headers
            r'\n\n(?=\d+\.\s)',  # Numbered lists
            r'\n\n(?=•\s|\*\s|-\s)',  # Bullet points
        ]
        
        chunks = []
        current_text = text
        
        for pattern in section_patterns:
            if len(current_text) > self.config.chunk_size:
                parts = re.split(pattern, current_text)
                if len(parts) > 1:
                    chunks.extend(self._process_parts(parts))
                    break
        
        if not chunks:
            # Fall back to sliding window
            chunks = self._sliding_window_chunk(text)
        
        # Combine source metadata with chunking metadata
        metadata = {
            "chunking_strategy": "html_aware",
            "original_length": len(text),
            "chunk_count": len(chunks)
        }
        
        if source_metadata:
            metadata.update(source_metadata)
            
        return ChunkingResult.create(text, chunks, metadata)
        
    def _process_parts(self, parts: List[str]) -> List[str]:
        """Process parts of HTML content into chunks."""
        chunks = []
        current_chunk = ""
        
        for part in parts:
            if len(current_chunk) + len(part) > self.config.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start a new chunk
                if len(part) > self.config.chunk_size:
                    # Part is too large, use sliding window
                    sub_chunks = self._sliding_window_chunk(part)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part
            else:
                if current_chunk:
                    current_chunk += "\n\n" + part
                else:
                    current_chunk = part
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks


class PDFChunker(TextChunker):
    """Specialized chunker for PDF documents."""
    
    def chunk_text(self, text: str, source_metadata: Optional[Dict[str, Any]] = None) -> ChunkingResult:
        """Chunk PDF text with page awareness."""
        # PDFs often have page markers like [Page N]
        page_pattern = r'\[Page \d+\]'
        
        # Split by pages first if page markers exist
        if re.search(page_pattern, text):
            page_sections = re.split(page_pattern, text)
            chunks = []
            
            for i, section in enumerate(page_sections):
                if section.strip():
                    # Chunk each page section
                    page_chunks = self._sliding_window_chunk(section.strip())
                    # Add page context to chunks
                    for j, chunk in enumerate(page_chunks):
                        if i > 0:  # Skip first empty split
                            chunks.append(f"[Page {i}] {chunk}")
                        else:
                            chunks.append(chunk)
        else:
            # Fall back to paragraph-based chunking
            chunks = self._paragraph_chunk(text)
        
        # Combine source metadata with chunking metadata
        metadata = {
            "chunking_strategy": "pdf_aware",
            "original_length": len(text),
            "chunk_count": len(chunks),
            "has_page_markers": bool(re.search(page_pattern, text))
        }
        
        if source_metadata:
            metadata.update(source_metadata)
            
        return ChunkingResult.create(text, chunks, metadata)
    
    def _paragraph_chunk(self, text: str) -> List[str]:
        """Chunk text by paragraphs for PDF content."""
        # Split by paragraphs (double newlines)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            # Fallback to sentence splitting if no paragraphs
            return self._sentence_chunk(text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) + 2 > self.config.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
        
                # Start new chunk with overlap from previous
                if chunks and self.config.chunk_overlap > 0:
                    overlap_text = self._get_overlap_text(chunks[-1])
                    current_chunk = overlap_text + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


class StructuredDocumentChunker(TextChunker):
    """Chunker for structured documents like EPUB and DOCX."""
    
    def chunk_text(self, text: str, source_metadata: Optional[Dict[str, Any]] = None) -> ChunkingResult:
        """Chunk structured document text with chapter/section awareness."""
        # Look for chapter markers
        chapter_pattern = r'\[Chapter \d+\]'
        
        if re.search(chapter_pattern, text):
            # Split by chapters
            chapter_sections = re.split(chapter_pattern, text)
            chunks = []
            
            for i, section in enumerate(chapter_sections):
                if section.strip():
                    # Chunk each chapter
                    chapter_chunks = self._paragraph_chunk(section.strip())
                    # Add chapter context
                    for chunk in chapter_chunks:
                        if i > 0:  # Skip first empty split
                            chunks.append(f"[Chapter {i}] {chunk}")
                        else:
                            chunks.append(chunk)
        else:
            # Look for other structural elements
            # Tables, lists, etc.
            chunks = self._structure_aware_chunk(text)
        
        # Combine source metadata with chunking metadata
        metadata = {
            "chunking_strategy": "structured_document",
            "original_length": len(text),
            "chunk_count": len(chunks),
            "has_chapters": bool(re.search(chapter_pattern, text))
        }
        
        if source_metadata:
            metadata.update(source_metadata)
            
        return ChunkingResult.create(text, chunks, metadata)
    
    def _paragraph_chunk(self, text: str) -> List[str]:
        """Chunk text by paragraphs for structured documents."""
        # Split by paragraphs (double newlines)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if not paragraphs:
            # Fallback to sentence splitting if no paragraphs
            return self._sentence_chunk(text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) + 2 > self.config.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                # Start new chunk with overlap from previous
                if chunks and self.config.chunk_overlap > 0:
                    overlap_text = self._get_overlap_text(chunks[-1])
                    current_chunk = overlap_text + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _structure_aware_chunk(self, text: str) -> List[str]:
        """Chunk text with awareness of document structure."""
        # Look for table-like structures (| separated)
        table_pattern = r'\n[^\n]*\|[^\n]*\|[^\n]*\n'
        
        # Split on major structural breaks
        structural_breaks = [
            r'\n\n(?=\d+\.\s)',  # Numbered sections
            r'\n\n(?=[A-Z][A-Z\s]{5,}\n)',  # ALL CAPS headers
            r'\n\n(?=Table \d+)',  # Table captions
            r'\n\n(?=Figure \d+)',  # Figure captions
        ]
        
        chunks = []
        current_text = text
        
        for pattern in structural_breaks:
            if len(current_text) > self.config.chunk_size:
                parts = re.split(pattern, current_text)
                if len(parts) > 1:
                    chunks.extend(self._process_parts(parts))
                    break
        
        if not chunks:
            chunks = self._sliding_window_chunk(text)
        
        return chunks


def create_chunker(file_type: str, config: HybridSearchConfig) -> TextChunker:
    """Factory function to create appropriate chunker based on file type."""
    file_type = file_type.lower()
    
    if file_type == "markdown":
        return MarkdownChunker(config)
    elif file_type == "html":
        return HTMLChunker(config)
    elif file_type == "pdf":
        return PDFChunker(config)
    elif file_type in ["epub", "docx"]:
        return StructuredDocumentChunker(config)
    else:
        return TextChunker(config)
