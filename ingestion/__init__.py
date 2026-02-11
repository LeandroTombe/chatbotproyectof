"""
Ingestion module for batch document processing.
"""
from ingestion.pipeline import (
    IngestionPipeline,
    PipelineException,
    ProcessingResult,
    BatchResult
)

__all__ = [
    "IngestionPipeline",
    "PipelineException",
    "ProcessingResult",
    "BatchResult"
]
