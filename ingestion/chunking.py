"""
Text chunking module.
Handles document splitting with overlap for context preservation.
"""
from typing import List, Optional
import logging
import uuid

from domain.models import Chunk, ChunkingConfig

logger = logging.getLogger(__name__)


class ChunkingException(Exception):
    """Exception raised for chunking errors"""
    pass


class TextChunker:
    """
    Splits long text into overlapping chunks for better context preservation.
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.config.validate()
        logger.info(
            f"TextChunker initialized — "
            f"chunk_size={self.config.chunk_size}, "
            f"overlap={self.config.chunk_overlap}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_text(
        self,
        text: str,
        document_id: str,
        metadata: Optional[dict] = None,
    ) -> List[Chunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to split
            document_id: ID of the originating document
            metadata: Extra metadata attached to every chunk

        Returns:
            List of Chunk objects

        Raises:
            ChunkingException: On processing error
        """
        if not text or not text.strip():
            logger.warning(f"Empty text for document {document_id}")
            return []

        try:
            raw_chunks = self._split_by_separator(text)
            if not raw_chunks:
                raw_chunks = self._split_by_characters(text)

            chunk_objects: List[Chunk] = []
            for idx, content in enumerate(raw_chunks):
                chunk_meta = (metadata or {}).copy()
                chunk_meta.update(
                    {
                        "chunk_method": (
                            "separator"
                            if self.config.separator in text
                            else "character"
                        ),
                        "original_length": len(text),
                    }
                )
                chunk_objects.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        document_id=document_id,
                        content=content.strip(),
                        chunk_index=idx,
                        metadata=chunk_meta,
                    )
                )

            logger.info(f"Created {len(chunk_objects)} chunks for {document_id}")
            return chunk_objects

        except Exception as e:
            msg = f"Error chunking text for document {document_id}: {str(e)}"
            logger.error(msg)
            raise ChunkingException(msg) from e

    def get_chunk_count_estimate(self, text: str) -> int:
        """Rough estimate of the number of chunks that would be produced."""
        if not text:
            return 0
        effective = self.config.chunk_size - self.config.chunk_overlap
        return max(1, (len(text) + effective - 1) // effective)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_by_separator(self, text: str) -> List[str]:
        segments = text.split(self.config.separator)
        chunks: List[str] = []
        current = ""

        for seg in segments:
            addition = (
                (self.config.separator + seg) if current else seg
            )
            if len(current) + len(addition) <= self.config.chunk_size:
                current += addition
            else:
                if current:
                    chunks.append(current)
                if len(seg) > self.config.chunk_size:
                    sub = self._split_large_segment(seg)
                    chunks.extend(sub[:-1])
                    current = sub[-1] if sub else ""
                else:
                    current = seg

        if current:
            chunks.append(current)

        return self._apply_overlap(chunks)

    def _split_by_characters(self, text: str) -> List[str]:
        chunks: List[str] = []
        start = 0
        while start < len(text):
            chunks.append(text[start : start + self.config.chunk_size])
            start += self.config.chunk_size - self.config.chunk_overlap
        return chunks

    def _split_large_segment(self, segment: str) -> List[str]:
        chunks: List[str] = []
        start = 0
        while start < len(segment):
            chunks.append(segment[start : start + self.config.chunk_size])
            start += self.config.chunk_size - self.config.chunk_overlap
        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        if len(chunks) <= 1 or self.config.chunk_overlap <= 0:
            return chunks

        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            tail = (
                prev[-self.config.chunk_overlap :]
                if len(prev) > self.config.chunk_overlap
                else prev
            )
            overlapped.append(tail + " " + chunks[i])
        return overlapped


# ------------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------------

def chunk_text_simple(
    text: str,
    document_id: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Chunk]:
    """Quick chunking helper with default settings."""
    config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return TextChunker(config).chunk_text(text, document_id)
