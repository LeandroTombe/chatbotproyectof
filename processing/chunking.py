"""
processing.chunking — backward-compatibility shim.
Use `ingestion.chunking` for new code.
"""
from ingestion.chunking import (  # noqa: F401
    TextChunker,
    ChunkingException,
    chunk_text_simple,
)

__all__ = ["TextChunker", "ChunkingException", "chunk_text_simple"]
