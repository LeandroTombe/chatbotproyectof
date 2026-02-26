"""
documents.loaders.pdf_loader — backward-compatibility shim.
Use `ingestion.loaders.pdf_loader` for new code.
"""
from ingestion.loaders.pdf_loader import PDFLoader  # noqa: F401

__all__ = ["PDFLoader"]
