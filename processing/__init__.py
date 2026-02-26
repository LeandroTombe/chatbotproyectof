"""
processing — backward-compatibility shim.
Use `ingestion.chunking` for new code.
"""
from ingestion.chunking import TextChunker, ChunkingException, chunk_text_simple  # noqa: F401

__all__ = ["TextChunker", "ChunkingException", "chunk_text_simple"]
