"""
CSV loader implementation.
Reads tabular data and converts each row into a readable text block
so it can be chunked and embedded like any other document.
"""
from pathlib import Path
import csv
import logging

from ingestion.loaders.base_loader import BaseLoader, LoaderException
from domain.models import Document, ProcessingStatus

logger = logging.getLogger(__name__)


class CsvLoader(BaseLoader):
    """
    Loader for CSV / TSV documents.

    Each row is serialised as:
        Column A: value | Column B: value | ...
    All rows are joined with newlines to produce the document content.
    """

    SUPPORTED_EXTENSIONS = [".csv", ".tsv"]

    def __init__(self, encoding: str = "utf-8", delimiter: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.encoding = encoding
        self.delimiter = delimiter  # None → auto-detect via csv.Sniffer

    def supports(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, file_path: str) -> Document:
        """
        Load a CSV/TSV document and return its content as readable text.

        Raises:
            LoaderException: If the file cannot be read or contains no data.
            FileNotFoundError: If the file does not exist.
        """
        path = self._validate_file(file_path)

        if not self.supports(file_path):
            raise LoaderException(
                f"CsvLoader does not support file type: {path.suffix}"
            )

        try:
            raw = path.read_text(encoding=self.encoding, errors="replace")
        except Exception as exc:
            raise LoaderException(f"Failed to read CSV file '{file_path}': {exc}") from exc

        if not raw.strip():
            raise LoaderException(f"CSV file is empty: {file_path}")

        # Detect delimiter if not provided
        delimiter = self.delimiter
        if delimiter is None:
            try:
                dialect = csv.Sniffer().sniff(raw[:4096])
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ","  # fallback

        rows = list(csv.DictReader(raw.splitlines(), delimiter=delimiter))

        if not rows:
            raise LoaderException(f"No data rows found in CSV: {file_path}")

        # Convert rows → human-readable text blocks
        lines: list[str] = []
        for i, row in enumerate(rows, start=1):
            # Build "Column: value" pairs, skip empty values
            parts = [f"{k}: {v}" for k, v in row.items() if v and str(v).strip()]
            if parts:
                lines.append(f"Fila {i}: " + " | ".join(parts))

        content = "\n".join(lines)
        if not content.strip():
            raise LoaderException(f"CSV file has no readable content: {file_path}")

        logger.info(
            "CsvLoader: loaded '%s' — %d rows → %d chars",
            path.name, len(rows), len(content),
        )

        return Document(
            id=self._generate_document_id(path),
            file_path=path,
            file_name=path.name,
            status=ProcessingStatus.COMPLETED,
            content=content,
            metadata={
                "file_name": path.name,
                "file_type": "csv",
                "row_count": len(rows),
                "column_count": len(rows[0]) if rows else 0,
                "char_count": len(content),
                "file_size": path.stat().st_size,
            },
        )
