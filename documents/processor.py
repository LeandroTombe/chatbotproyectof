"""
Document processor module for orchestrating the document ingestion pipeline.
Coordinates loading, chunking, embedding, and storing documents.
"""
from typing import Optional
from pathlib import Path
import logging

from domain.models import Document
from documents.loaders.base_loader import BaseLoader
from documents.loaders.loader_factory import get_loader
from processing.chunking import TextChunker
from embeddings.base import BaseEmbedding
from vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class ProcessorException(Exception):
    """Exception raised for document processing errors"""
    pass


class DocumentProcessor:
    """
    Orchestrates the complete document processing pipeline:
    load → chunk → embed → store
    
    This service coordinates multiple components via dependency injection
    to transform raw documents into searchable vector embeddings.
    """
    
    def __init__(
        self,
        chunker: TextChunker,
        embedder: BaseEmbedding,
        vector_store: BaseVectorStore,
        loader: Optional[BaseLoader] = None
    ):
        """
        Initialize the document processor with required dependencies.
        
        Args:
            chunker: Service for splitting documents into chunks
            embedder: Provider for generating text embeddings
            vector_store: Storage for vector embeddings
            loader: Optional document loader (if None, factory is used)
        """
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.loader = loader
        
        logger.info(
            f"DocumentProcessor initialized with "
            f"chunker={chunker.__class__.__name__}, "
            f"embedder={embedder.__class__.__name__}, "
            f"vector_store={vector_store.__class__.__name__}"
        )
    
    def process_document(self, file_path: str | Path) -> Document:
        """
        Process a document through the complete pipeline.
        
        Pipeline stages:
        1. Load document from file
        2. Mark as processing
        3. Chunk text into smaller segments
        4. Generate embeddings for each chunk
        5. Store embeddings in vector store
        6. Mark as completed or failed
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Processed Document with updated status
            
        Raises:
            ProcessorException: If any stage of processing fails
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        logger.info(f"Starting document processing: {file_path}")
        
        document = None
        
        try:
            # Stage 1: Load document
            document = self._load_document(file_path)
            
            # Stage 2: Mark as processing
            document.mark_processing()
            logger.info(f"Document {document.id} marked as processing")
            
            # Stage 3: Extract text content
            text = self._extract_text(document)
            
            # Stage 4: Chunk text
            chunks = self.chunker.chunk_text(
                text=text,
                document_id=document.id,
                metadata=document.metadata
            )
            logger.info(f"Created {len(chunks)} chunks for document {document.id}")
            
            if not chunks:
                raise ProcessorException(
                    f"No chunks created for document {document.id}"
                )
            
            # Stage 5: Generate embeddings
            for chunk in chunks:
                chunk.embedding = self.embedder.embed_text(chunk.content)
            logger.info(f"Generated embeddings for {len(chunks)} chunks")
            
            # Stage 6: Store in vector store
            self.vector_store.add_chunks(chunks)
            logger.info(f"Stored {len(chunks)} chunks in vector store")
            
            # Stage 7: Mark as completed
            document.mark_completed(chunks_count=len(chunks))
            logger.info(
                f"Document {document.id} processed successfully: "
                f"{len(chunks)} chunks stored"
            )
            
            return document
            
        except Exception as e:
            error_msg = f"Error processing document {file_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Mark document as failed if it exists
            if document:
                document.mark_failed(error=str(e))
            
            raise ProcessorException(error_msg) from e
    
    def _load_document(self, file_path: Path) -> Document:
        """
        Load document using loader or factory.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Loaded Document
            
        Raises:
            ProcessorException: If loading fails
        """
        try:
            if self.loader:
                logger.debug(f"Loading with provided loader: {self.loader.__class__.__name__}")
                return self.loader.load(str(file_path))
            else:
                logger.debug("Loading with factory loader")
                return get_loader(str(file_path)).load(str(file_path))
        except Exception as e:
            raise ProcessorException(f"Failed to load document: {str(e)}") from e
    
    def _extract_text(self, document: Document) -> str:
        """
        Extract text content from document metadata.
        
        Args:
            document: Document to extract text from
            
        Returns:
            Extracted text content
            
        Raises:
            ProcessorException: If text extraction fails
        """
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
