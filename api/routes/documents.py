"""
Documents routes — REST endpoints for ingesting PDF files.

Endpoints
---------
POST /api/documents/upload  — Upload one or more PDF files and ingest them
GET  /api/documents         — List ingested documents (placeholder)
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from api.dependencies import get_ingestion_pipeline, get_vector_store
from ingestion.pipeline import IngestionPipeline
from vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "application/octet-stream",  # some clients send this for .txt files
}
MAX_FILE_SIZE_MB = 50


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class UploadResult(BaseModel):
    filename: str
    success: bool
    document_id: str | None = None
    chunks: int = 0
    error: str | None = None


class UploadResponse(BaseModel):
    uploaded: int
    failed: int
    results: List[UploadResult]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload PDF files and ingest them into the vector store",
)
async def upload_documents(
    files: List[UploadFile] = File(..., description="One or more PDF files"),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> UploadResponse:
    """Upload PDF files, process them into chunks, generate embeddings, and
    store them in the vector store so they can be retrieved during chat.
    """
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided.")

    results: List[UploadResult] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for upload in files:
            filename = upload.filename or "unknown.pdf"

            # Basic content-type validation
            if upload.content_type not in ALLOWED_CONTENT_TYPES:
                results.append(
                    UploadResult(filename=filename, success=False, error="Only PDF files are allowed.")
                )
                continue

            # Save to temp file
            dest = tmp_path / filename
            try:
                with dest.open("wb") as f:
                    shutil.copyfileobj(upload.file, f)
            except Exception as exc:
                results.append(UploadResult(filename=filename, success=False, error=str(exc)))
                continue

            # Ingest
            try:
                batch = pipeline.process_files([str(dest)])
                proc_result = batch.results[0] if batch.results else None
                if proc_result and proc_result.success:
                    results.append(
                        UploadResult(
                            filename=filename,
                            success=True,
                            document_id=proc_result.document_id,
                            chunks=proc_result.chunks_count,
                        )
                    )
                else:
                    error_msg = proc_result.error_message if proc_result else "Unknown error"
                    results.append(UploadResult(filename=filename, success=False, error=error_msg))
            except Exception as exc:
                logger.exception("Failed to ingest file: %s", filename)
                results.append(UploadResult(filename=filename, success=False, error=str(exc)))

    successful = sum(1 for r in results if r.success)
    return UploadResponse(
        uploaded=successful,
        failed=len(results) - successful,
        results=results,
    )


class DeleteDocumentResponse(BaseModel):
    document_id: str
    chunks_deleted: int
    success: bool


@router.delete(
    "/{document_id}",
    response_model=DeleteDocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete a document and all its chunks from the vector store",
)
def delete_document(
    document_id: str,
    vector_store: BaseVectorStore = Depends(get_vector_store),
) -> DeleteDocumentResponse:
    """Remove all chunks for *document_id* from the vector store so the
    document can be re-ingested with updated metadata."""
    try:
        count = vector_store.delete_chunks_by_document(document_id)
        logger.info("Deleted document %s (%d chunks)", document_id, count)
        return DeleteDocumentResponse(
            document_id=document_id,
            chunks_deleted=count,
            success=True,
        )
    except Exception as exc:
        logger.exception("Error deleting document %s", document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {exc}",
        ) from exc
