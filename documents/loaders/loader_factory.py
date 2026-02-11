"""
Loader factory module.
Provides factory functions for creating document loaders.
"""
from pathlib import Path
import logging

from documents.loaders.base_loader import BaseLoader, LoaderException
from documents.loaders.pdf_loader import PDFLoader
from domain.models import Document

logger = logging.getLogger(__name__)


def get_loader(file_path: str, **kwargs) -> BaseLoader:
    """
    Factory function to get the appropriate loader for a file.
    
    Args:
        file_path: Path to the file
        **kwargs: Additional configuration for the loader
        
    Returns:
        Appropriate BaseLoader instance
        
    Raises:
        LoaderException: If no suitable loader is found
    """
    path = Path(file_path)
    extension = path.suffix.lower()
    
    # Map extensions to loaders
    if extension == '.pdf':
        return PDFLoader(**kwargs)
    
    # Add more loaders here as they are implemented
    # elif extension in ['.txt', '.md']:
    #     return TextLoader(**kwargs)
    # elif extension == '.docx':
    #     return DocxLoader(**kwargs)
    
    raise LoaderException(
        f"No loader available for file type: {extension}. "
        f"Supported types: .pdf"
    )


def load_document(file_path: str, **kwargs) -> Document:
    """
    Convenience function to load a document with automatic loader selection.
    
    Args:
        file_path: Path to the document
        **kwargs: Additional configuration for the loader
        
    Returns:
        Loaded Document
        
        Raises:
        LoaderException: If loading fails
        FileNotFoundError: If file doesn't exist
    """
    loader = get_loader(file_path, **kwargs)
    return loader.load(file_path)
