"""
documents.loaders — backward-compatibility shim.
Use `ingestion.loaders` for new code.
"""
from ingestion.loaders import (
    BaseLoader,
    LoaderException,
    PDFLoader,
    get_loader,
    load_document,
)

__all__ = [
    "BaseLoader",
    "LoaderException",
    "PDFLoader",
    "get_loader",
    "load_document",
]
