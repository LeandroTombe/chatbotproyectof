"""
documents.loaders.loader_factory — backward-compatibility shim.
Use `ingestion.loaders.loader_factory` for new code.
"""
from ingestion.loaders.loader_factory import get_loader, load_document  # noqa: F401

__all__ = ["get_loader", "load_document"]
