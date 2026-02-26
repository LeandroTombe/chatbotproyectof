"""
documents — backward-compatibility shim.
All symbols are now canonical under `ingestion.*`.
"""
from ingestion.loaders import (
    BaseLoader,
    LoaderException,
    PDFLoader,
    get_loader,
    load_document,
)
from ingestion.processor import DocumentProcessor, ProcessorException

__all__ = [
    "BaseLoader",
    "LoaderException",
    "PDFLoader",
    "get_loader",
    "load_document",
    "DocumentProcessor",
    "ProcessorException",
]
