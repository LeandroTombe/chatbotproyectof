"""
Ingestion module.

Centralises all document loading, chunking, processing and batch ingestion:

  ingestion.loaders     — file loaders (PDF, …)
  ingestion.chunking    — text splitting with overlap
  ingestion.processor   — end-to-end single-document pipeline
  ingestion.pipeline    — batch / directory ingestion pipeline
"""
from ingestion.loaders import (
    BaseLoader,
    LoaderException,
    PDFLoader,
    get_loader,
    load_document,
)
from ingestion.chunking import TextChunker, ChunkingException, chunk_text_simple
from ingestion.processor import DocumentProcessor, ProcessorException
from ingestion.pipeline import (
    IngestionPipeline,
    PipelineException,
    ProcessingResult,
    BatchResult,
)

__all__ = [
    # Loaders
    "BaseLoader",
    "LoaderException",
    "PDFLoader",
    "get_loader",
    "load_document",
    # Chunking
    "TextChunker",
    "ChunkingException",
    "chunk_text_simple",
    # Processor
    "DocumentProcessor",
    "ProcessorException",
    # Pipeline
    "IngestionPipeline",
    "PipelineException",
    "ProcessingResult",
    "BatchResult",
]
