"""
documents.loaders.base_loader — backward-compatibility shim.
Use `ingestion.loaders.base_loader` for new code.
"""
from ingestion.loaders.base_loader import BaseLoader, LoaderException  # noqa: F401

__all__ = ["BaseLoader", "LoaderException"]
