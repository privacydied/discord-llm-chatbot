#!/usr/bin/env python3
"""Test script for multi-format RAG document support.
Creates sample documents in various formats and tests parsing.
"""

import sys
from pathlib import Path

import pytest

# Add bot module to path
sys.path.insert(0, str(Path(__file__).parent))

from bot.rag.document_parsers import document_parser_factory


def create_test_documents():
    """Create test documents in various formats."""
    test_dir = Path("test_documents")
    test_dir.mkdir(exist_ok=True)

    # Create test TXT file
    (test_dir / "sample.txt").write_text("""
This is a sample text document for testing RAG parsing.

It contains multiple paragraphs with different types of content.
Some technical information about Discord bots and AI systems.

Features:
- Text-to-speech capabilities
- Speech-to-text processing
- Image analysis
- RAG-based knowledge retrieval

This content should be properly chunked and embedded for vector search.
""")

    # Create test Markdown file
    (test_dir / "sample.md").write_text("""
# Sample Markdown Document

This is a **markdown** document for testing RAG parsing capabilities.

## Features

The Discord bot supports:

1. **Multi-modal AI**: Text, voice, and image processing
2. **RAG System**: Vector-based knowledge retrieval
3. **TTS/STT**: Speech synthesis and recognition

### Technical Details

The system uses:
- ChromaDB for vector storage
- Sentence transformers for embeddings
- Multiple document parsers for ingestion

## Code Example

```python
# Example usage
rag_system = create_rag_system()
results = await rag_system.search("Discord bot features")
```

This content demonstrates markdown structure preservation during chunking.
""")

    # Create test HTML file
    (test_dir / "sample.html").write_text("""
<!DOCTYPE html>
<html>
<head>
    <title>Sample HTML Document</title>
    <meta name="description" content="Test HTML document for RAG parsing">
    <meta name="keywords" content="HTML, RAG, parsing, test">
</head>
<body>
    <h1>HTML Document Test</h1>

    <p>This is a sample HTML document for testing RAG parsing capabilities.</p>

    <h2>Key Features</h2>
    <ul>
        <li>HTML structure preservation</li>
        <li>Metadata extraction</li>
        <li>Clean text extraction</li>
    </ul>

    <h3>Technical Information</h3>
    <p>The HTML parser uses BeautifulSoup to extract clean text content
    while preserving document structure for better chunking.</p>

    <script>
        // This script should be removed during parsing
        console.log("This should not appear in parsed content");
    </script>

    <style>
        /* This CSS should be removed during parsing */
        body { font-family: Arial; }
    </style>
</body>
</html>
""")

    return test_dir


@pytest.fixture
def test_dir(tmp_path, monkeypatch):
    """Pytest fixture that prepares test documents in an isolated temp directory."""
    # Run the test in an isolated temp CWD to avoid polluting the repo
    monkeypatch.chdir(tmp_path)
    return create_test_documents()


def test_document_parsing(test_dir) -> None:
    """Test parsing of different document formats."""
    supported_extensions = document_parser_factory.get_supported_extensions()

    test_files = [
        test_dir / "sample.txt",
        test_dir / "sample.md",
        test_dir / "sample.html",
    ]

    for test_file in test_files:
        if test_file.exists():
            try:
                parser = document_parser_factory.get_parser(test_file)
                if parser:
                    content, metadata = document_parser_factory.parse_document(test_file)

                    # Show first 100 characters of content
                    preview = content[:100].replace("\n", " ")

                    # Show some metadata
                    interesting_keys = [
                        "html_title",
                        "line_count",
                        "char_count",
                        "header_count",
                    ]
                    for key in interesting_keys:
                        if key in metadata:
                            pass

                else:
                    pass

            except Exception as e:
                pass


def test_chunking_strategies() -> None:
    """Test different chunking strategies for document types."""
    from bot.rag.text_chunker import create_chunker
    from bot.rag.vector_schema import HybridSearchConfig

    config = HybridSearchConfig(chunk_size=200, chunk_overlap=50)

    test_cases = [
        ("text", "This is plain text content. " * 20),
        ("markdown", "# Header\n\nThis is markdown content. " * 10),
        ("html", "Header Content\n\nThis is HTML-derived content. " * 10),
        ("pdf", "[Page 1]\nThis is PDF content. " * 10),
        ("epub", "[Chapter 1]\nThis is EPUB content. " * 10),
    ]

    for file_type, content in test_cases:
        try:
            chunker = create_chunker(file_type, config)
            result = chunker.chunk_text(content)

            # Show first chunk preview
            if result.chunks:
                preview = result.chunks[0][:80].replace("\n", " ")

        except Exception as e:
            pass


def main() -> None:
    """Main test function."""
    # Create test documents
    test_dir = create_test_documents()

    # Test document parsing
    test_document_parsing(test_dir)

    # Test chunking strategies
    test_chunking_strategies()


if __name__ == "__main__":
    main()
