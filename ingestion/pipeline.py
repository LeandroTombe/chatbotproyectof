"""
Ingestion pipeline for batch processing multiple documents.
Orchestrates document processing across multiple files or directories.
"""
from typing import List, Optional, Dict, Any, Sequence
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import logging

from domain.models import Document, ProcessingStatus
from ingestion.processor import DocumentProcessor

logger = logging.getLogger(__name__)


class PipelineException(Exception):
    """Exception raised for pipeline errors"""
    pass


@dataclass
class ProcessingResult:
    """Result of processing a single file"""
    file_path: str
    success: bool
    document_id: Optional[str] = None
    chunks_count: int = 0
    error_message: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class BatchResult:
    """Result of batch processing multiple files"""
    total_files: int
    successful: int
    failed: int
    results: List[ProcessingResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_files == 0:
            return 0.0
        return (self.successful / self.total_files) * 100
    
    @property
    def total_chunks(self) -> int:
        """Total chunks processed across all documents"""
        return sum(r.chunks_count for r in self.results if r.success)
    
    def get_failed_files(self) -> List[str]:
        """Get list of files that failed processing"""
        return [r.file_path for r in self.results if not r.success]
    
    def get_successful_files(self) -> List[str]:
        """Get list of files that were processed successfully"""
        return [r.file_path for r in self.results if r.success]


class IngestionPipeline:
    """
    Batch processing pipeline for ingesting multiple documents.
    
    Orchestrates DocumentProcessor to handle multiple files,
    providing progress tracking and error handling.
    """
    
    def __init__(
        self,
        processor: DocumentProcessor,
        supported_extensions: Optional[List[str]] = None
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            processor: DocumentProcessor for handling individual files
            supported_extensions: List of supported file extensions (default: ['.pdf'])
        """
        self.processor = processor
        self.supported_extensions = supported_extensions or ['.pdf']
        
        logger.info(
            f"IngestionPipeline initialized with "
            f"processor={processor.__class__.__name__}, "
            f"supported_extensions={self.supported_extensions}"
        )
    
    def process_files(
        self,
        file_paths: Sequence[str | Path],
        continue_on_error: bool = True
    ) -> BatchResult:
        """
        Process a list of files.
        
        Args:
            file_paths: Sequence of file paths to process
            continue_on_error: If True, continue processing after individual file errors
            
        Returns:
            BatchResult with processing statistics and individual results
            
        Raises:
            PipelineException: If continue_on_error=False and a file fails
        """
        logger.info(f"Starting batch processing of {len(file_paths)} files")
        
        batch_result = BatchResult(
            total_files=len(file_paths),
            successful=0,
            failed=0
        )
        
        for file_path in file_paths:
            result = self._process_single_file(file_path)
            batch_result.results.append(result)
            
            if result.success:
                batch_result.successful += 1
            else:
                batch_result.failed += 1
                if not continue_on_error:
                    batch_result.completed_at = datetime.now()
                    error_msg = f"Processing failed for {file_path}: {result.error_message}"
                    logger.error(error_msg)
                    raise PipelineException(error_msg)
        
        batch_result.completed_at = datetime.now()
        
        logger.info(
            f"Batch processing completed: {batch_result.successful}/{batch_result.total_files} successful, "
            f"{batch_result.total_chunks} total chunks, "
            f"success rate: {batch_result.success_rate:.1f}%"
        )
        
        return batch_result
    
    def process_directory(
        self,
        directory_path: str | Path,
        recursive: bool = False,
        continue_on_error: bool = True
    ) -> BatchResult:
        """
        Process all supported files in a directory.
        
        Args:
            directory_path: Path to directory containing files
            recursive: If True, process subdirectories recursively
            continue_on_error: If True, continue processing after individual file errors
            
        Returns:
            BatchResult with processing statistics
            
        Raises:
            PipelineException: If directory doesn't exist or is not a directory
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise PipelineException(f"Directory does not exist: {directory}")
        
        if not directory.is_dir():
            raise PipelineException(f"Path is not a directory: {directory}")
        
        # Find all supported files
        file_paths = self._find_supported_files(directory, recursive)
        
        logger.info(
            f"Found {len(file_paths)} supported files in {directory} "
            f"(recursive={recursive})"
        )
        
        if not file_paths:
            logger.warning(f"No supported files found in {directory}")
            return BatchResult(total_files=0, successful=0, failed=0)
        
        # Process all found files
        return self.process_files(file_paths, continue_on_error)
    
    def _process_single_file(self, file_path: str | Path) -> ProcessingResult:
        """
        Process a single file and return result.
        
        Args:
            file_path: Path to file to process
            
        Returns:
            ProcessingResult with processing outcome
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        start_time = datetime.now()
        
        logger.info(f"Processing file: {file_path}")
        
        try:
            # Check if file is supported
            if not self._is_supported_file(file_path):
                error_msg = f"Unsupported file type: {file_path.suffix}"
                logger.warning(error_msg)
                return ProcessingResult(
                    file_path=str(file_path),
                    success=False,
                    error_message=error_msg,
                    processing_time=0.0
                )
            
            # Process the file
            document = self.processor.process_document(str(file_path))
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(
                f"Successfully processed {file_path}: "
                f"{document.total_chunks} chunks in {processing_time:.2f}s"
            )
            
            return ProcessingResult(
                file_path=str(file_path),
                success=True,
                document_id=document.id,
                chunks_count=document.total_chunks,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            
            logger.error(
                f"Failed to process {file_path}: {error_msg}",
                exc_info=True
            )
            
            return ProcessingResult(
                file_path=str(file_path),
                success=False,
                error_message=error_msg,
                processing_time=processing_time
            )
    
    def _find_supported_files(
        self,
        directory: Path,
        recursive: bool
    ) -> Sequence[Path]:
        """
        Find all supported files in directory.
        
        Args:
            directory: Directory to search
            recursive: If True, search subdirectories
            
        Returns:
            Sequence of file paths with supported extensions
        """
        files = []
        
        if recursive:
            # Search recursively
            for ext in self.supported_extensions:
                files.extend(directory.rglob(f"*{ext}"))
        else:
            # Search only in current directory
            for ext in self.supported_extensions:
                files.extend(directory.glob(f"*{ext}"))
        
        # Sort for consistent ordering
        return sorted(files)
    
    def _is_supported_file(self, file_path: Path) -> bool:
        """
        Check if file extension is supported.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file extension is in supported_extensions
        """
        return file_path.suffix.lower() in self.supported_extensions
