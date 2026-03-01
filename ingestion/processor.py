"""
Document processor module.
Orchestrates the full document ingestion pipeline:
  load → chunk → embed → store
"""
from typing import Optional
from pathlib import Path
import logging

from domain.models import Document
from ingestion.loaders.base_loader import BaseLoader
from ingestion.loaders.loader_factory import get_loader
from ingestion.chunking import TextChunker
from embeddings.base import BaseEmbedding
from vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class ProcessorException(Exception):
    """Exception raised for document processing errors"""
    pass


class DocumentProcessor:
    """
    Orchestrates the complete document processing pipeline:
    load → chunk → embed → store.

    Uses dependency injection so each component is replaceable.
    """

    def __init__(
        self,
        chunker: TextChunker,
        embedder: BaseEmbedding,
        vector_store: BaseVectorStore,
        loader: Optional[BaseLoader] = None,
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.loader = loader

        logger.info(
            f"DocumentProcessor initialized — "
            f"chunker={chunker.__class__.__name__}, "
            f"embedder={embedder.__class__.__name__}, "
            f"vector_store={vector_store.__class__.__name__}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_document(self, file_path: str | Path) -> Document:
        """
        Process a document through the full pipeline.

        Stages:
          1. Load document from file
          2. Mark as processing
          3. Chunk text into smaller segments
          4. Generate embeddings for each chunk
          5. Store embeddings in vector store
          6. Mark as completed / failed

        Returns:
            Processed Document with updated status.

        Raises:
            ProcessorException: If any stage fails.
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        logger.info(f"Processing document: {file_path}")

        document = None
        try:
            document = self._load_document(file_path)
            document.mark_processing()

            text = self._extract_text(document)
            # Ensure file_name is always present in chunk metadata so the
            # vector store can resolve a human-readable document_name.
            chunk_meta = {**document.metadata, "file_name": document.file_name}
            chunks = self.chunker.chunk_text(
                text=text,
                document_id=document.id,
                metadata=chunk_meta,
            )

            if not chunks:
                raise ProcessorException(
                    f"No chunks created for document {document.id}"
                )

            for chunk in chunks:
                chunk.embedding = self.embedder.embed_text(chunk.content)

            self.vector_store.add_chunks(chunks)
            document.mark_completed(chunks_count=len(chunks))

            logger.info(
                f"Document {document.id} processed — {len(chunks)} chunks stored"
            )
            return document

        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if document:
                document.mark_failed(error=str(e))
            raise ProcessorException(error_msg) from e

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_document(self, file_path: Path) -> Document:
        try:
            if self.loader:
                return self.loader.load(str(file_path))
            return get_loader(str(file_path)).load(str(file_path))
        except Exception as e:
            raise ProcessorException(f"Failed to load document: {str(e)}") from e

    def _extract_text(self, document: Document) -> str:
        if "content" not in document.metadata:
            raise ProcessorException(
                f"Document {document.id} has no 'content' in metadata"
            )
        content = document.metadata["content"]
        if not content or not content.strip():
            raise ProcessorException(
                f"Document {document.id} has empty content"
            )
        return content
