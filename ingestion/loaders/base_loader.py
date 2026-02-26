"""
Base loader module.
Defines abstract interface for document loaders.
"""
from abc import ABC, abstractmethod
from pathlib import Path
import logging

from domain.models import Document

logger = logging.getLogger(__name__)


class LoaderException(Exception):
    """Exception raised for document loading errors"""
    pass


class BaseLoader(ABC):
    """
    Abstract base class for document loaders.
    Defines the interface for loading documents from files.
    """

    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    def load(self, file_path: str) -> Document:
        """
        Load a document from a file.

        Args:
            file_path: Path to the file to load

        Returns:
            Document object with extracted content

        Raises:
            LoaderException: If loading fails
            FileNotFoundError: If file doesn't exist
        """
        pass

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """
        Check if this loader supports the given file.

        Args:
            file_path: Path to check

        Returns:
            True if this loader can handle the file
        """
        pass

    def _validate_file(self, file_path: str) -> Path:
        """
        Validate that the file exists and is readable.

        Raises:
            FileNotFoundError: If file doesn't exist
            LoaderException: If file is not readable
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise LoaderException(f"Path is not a file: {file_path}")
        if not path.stat().st_size > 0:
            raise LoaderException(f"File is empty: {file_path}")

        return path
