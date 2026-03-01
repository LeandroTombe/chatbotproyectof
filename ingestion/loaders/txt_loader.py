"""
Plain-text loader implementation (.txt, .md).
Reads the file directly — no external library needed.
"""
from pathlib import Path
import logging

from ingestion.loaders.base_loader import BaseLoader, LoaderException
from domain.models import Document, ProcessingStatus

logger = logging.getLogger(__name__)


class TxtLoader(BaseLoader):
    """
    Loader for plain-text documents (.txt, .md).
    """

    SUPPORTED_EXTENSIONS = [".txt", ".md"]

    def __init__(self, encoding: str = "utf-8", **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, file_path: str) -> Document:
        """
        Load a plain-text document.

        Raises:
            LoaderException: If the file cannot be read or is empty.
            FileNotFoundError: If the file does not exist.
        """
        path = self._validate_file(file_path)

        if not self.supports(file_path):
            raise LoaderException(
                f"TxtLoader does not support file type: {path.suffix}"
            )

        try:
            content = path.read_text(encoding=self.encoding, errors="replace")
        except Exception as exc:
            raise LoaderException(f"Failed to read text file '{file_path}': {exc}") from exc

        content = content.strip()
        if not content:
            raise LoaderException(f"No text content found in file: {file_path}")

        logger.info("TxtLoader: loaded '%s' (%d chars)", path.name, len(content))

        return Document(
            id=self._generate_document_id(path),
            file_path=path,
            file_name=path.name,
            status=ProcessingStatus.COMPLETED,
            content=content,
            metadata={
                "file_type": path.suffix.lstrip("."),
                "char_count": len(content),
                "file_size": path.stat().st_size,
            },
        )
