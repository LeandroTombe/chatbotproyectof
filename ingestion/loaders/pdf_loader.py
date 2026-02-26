"""
PDF loader implementation.
Loads and extracts text from PDF documents.
"""
from pathlib import Path
import logging
import hashlib
from datetime import datetime

from ingestion.loaders.base_loader import BaseLoader, LoaderException
from domain.models import Document, ProcessingStatus

logger = logging.getLogger(__name__)


class PDFLoader(BaseLoader):
    """
    Loader for PDF documents.
    Uses PyPDF2 for text extraction.
    """

    SUPPORTED_EXTENSIONS = [".pdf"]

    def __init__(self, extract_images: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.extract_images = extract_images
        try:
            import PyPDF2
            self.PyPDF2 = PyPDF2
        except ImportError:
            raise LoaderException(
                "PyPDF2 is not installed. Install with: pip install PyPDF2"
            )

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, file_path: str) -> Document:
        """
        Load a PDF document and extract text.

        Raises:
            LoaderException: If PDF loading fails
            FileNotFoundError: If file doesn't exist
        """
        path = self._validate_file(file_path)

        if not self.supports(file_path):
            raise LoaderException(
                f"PDFLoader does not support file type: {path.suffix}"
            )

        try:
            content_parts: list[str] = []
            page_count = 0

            with open(path, "rb") as file:
                pdf_reader = self.PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)
                logger.info(f"Loading PDF: {path.name} ({page_count} pages)")

                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            content_parts.append(text)
                    except Exception as e:
                        logger.warning(f"Skipping page {page_num}: {e}")

            content = "\n\n".join(content_parts)
            if not content.strip():
                raise LoaderException(
                    f"No text content extracted from PDF: {file_path}"
                )

            document = Document(
                id=self._generate_document_id(path),
                file_path=path,
                file_name=path.name,
                status=ProcessingStatus.COMPLETED,
                created_at=datetime.now(),
                processed_at=datetime.now(),
                metadata={
                    "content": content,
                    "file_type": "pdf",
                    "file_size": path.stat().st_size,
                    "page_count": page_count,
                    "content_length": len(content),
                    "loader": "PDFLoader",
                },
            )

            logger.info(
                f"Loaded PDF: {path.name} ({len(content)} chars, {page_count} pages)"
            )
            return document

        except LoaderException:
            raise
        except Exception as e:
            error_msg = f"Error loading PDF {file_path}: {str(e)}"
            logger.error(error_msg)
            raise LoaderException(error_msg) from e

    def _generate_document_id(self, path: Path) -> str:
        unique_string = f"{path.absolute()}_{path.name}"
        doc_id = hashlib.md5(unique_string.encode()).hexdigest()
        return f"doc_{doc_id[:16]}"
