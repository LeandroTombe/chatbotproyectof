"""
Documents module.
Provides functionality for loading and processing documents.
"""
from documents.loaders import (
    BaseLoader,
    LoaderException,
    PDFLoader,
    get_loader,
    load_document
)
from documents.processor import (
    DocumentProcessor,
    ProcessorException
)

__all__ = [
    # Loaders
    "BaseLoader",
    "LoaderException",
    "PDFLoader",
    "get_loader",
    "load_document",
    # Processor
    "DocumentProcessor",
    "ProcessorException"
]
