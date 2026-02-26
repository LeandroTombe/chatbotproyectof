"""
documents.processor — backward-compatibility shim.
Use `ingestion.processor` for new code.
"""
from ingestion.processor import DocumentProcessor, ProcessorException  # noqa: F401

__all__ = ["DocumentProcessor", "ProcessorException"]
