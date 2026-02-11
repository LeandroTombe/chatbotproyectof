"""
Unit tests for IngestionPipeline.
"""
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import pytest

from ingestion.pipeline import (
    IngestionPipeline,
    PipelineException,
    ProcessingResult,
    BatchResult
)
from domain.models import Document, ProcessingStatus
from documents.processor import ProcessorException


class TestBatchResult:
    """Tests para BatchResult dataclass"""
    
    def test_batch_result_success_rate_empty(self):
        """Prueba success_rate con 0 archivos"""
        result = BatchResult(total_files=0, successful=0, failed=0)
        assert result.success_rate == 0.0
    
    def test_batch_result_success_rate_full(self):
        """Prueba success_rate con 100% éxito"""
        result = BatchResult(total_files=10, successful=10, failed=0)
        assert result.success_rate == 100.0
    
    def test_batch_result_success_rate_partial(self):
        """Prueba success_rate parcial"""
        result = BatchResult(total_files=10, successful=7, failed=3)
        assert result.success_rate == 70.0
    
    def test_batch_result_total_chunks(self):
        """Prueba total_chunks calculation"""
        result = BatchResult(total_files=3, successful=2, failed=1)
        result.results = [
            ProcessingResult(file_path="file1.pdf", success=True, chunks_count=10),
            ProcessingResult(file_path="file2.pdf", success=True, chunks_count=15),
            ProcessingResult(file_path="file3.pdf", success=False, chunks_count=0)
        ]
        assert result.total_chunks == 25
    
    def test_batch_result_get_failed_files(self):
        """Prueba obtención de archivos fallidos"""
        result = BatchResult(total_files=3, successful=2, failed=1)
        result.results = [
            ProcessingResult(file_path="file1.pdf", success=True),
            ProcessingResult(file_path="file2.pdf", success=False),
            ProcessingResult(file_path="file3.pdf", success=True)
        ]
        failed = result.get_failed_files()
        assert failed == ["file2.pdf"]
    
    def test_batch_result_get_successful_files(self):
        """Prueba obtención de archivos exitosos"""
        result = BatchResult(total_files=3, successful=2, failed=1)
        result.results = [
            ProcessingResult(file_path="file1.pdf", success=True),
            ProcessingResult(file_path="file2.pdf", success=False),
            ProcessingResult(file_path="file3.pdf", success=True)
        ]
        successful = result.get_successful_files()
        assert successful == ["file1.pdf", "file3.pdf"]


class TestIngestionPipelineInitialization:
    """Tests para inicialización de IngestionPipeline"""
    
    def test_initialization_default_extensions(self):
        """Prueba inicialización con extensiones por defecto"""
        processor = Mock()
        
        pipeline = IngestionPipeline(processor=processor)
        
        assert pipeline.processor == processor
        assert pipeline.supported_extensions == ['.pdf']
    
    def test_initialization_custom_extensions(self):
        """Prueba inicialización con extensiones personalizadas"""
        processor = Mock()
        
        pipeline = IngestionPipeline(
            processor=processor,
            supported_extensions=['.pdf', '.txt', '.docx']
        )
        
        assert pipeline.supported_extensions == ['.pdf', '.txt', '.docx']


class TestIngestionPipelineProcessFiles:
    """Tests para método process_files"""
    
    def test_process_files_all_successful(self):
        """Prueba procesamiento exitoso de todos los archivos"""
        processor = Mock()
        
        # Mock successful processing
        doc1 = Document(
            id="doc1",
            file_path=Path("file1.pdf"),
            file_name="file1.pdf",
            status=ProcessingStatus.COMPLETED,
            total_chunks=10
        )
        doc2 = Document(
            id="doc2",
            file_path=Path("file2.pdf"),
            file_name="file2.pdf",
            status=ProcessingStatus.COMPLETED,
            total_chunks=15
        )
        
        processor.process_document.side_effect = [doc1, doc2]
        
        pipeline = IngestionPipeline(processor=processor)
        
        result = pipeline.process_files(["file1.pdf", "file2.pdf"])
        
        assert result.total_files == 2
        assert result.successful == 2
        assert result.failed == 0
        assert result.success_rate == 100.0
        assert result.total_chunks == 25
        assert len(result.results) == 2
        
        # Verify processor called for each file
        assert processor.process_document.call_count == 2
    
    def test_process_files_with_failures_continue(self):
        """Prueba procesamiento con fallos y continue_on_error=True"""
        processor = Mock()
        
        doc1 = Document(
            id="doc1",
            file_path=Path("file1.pdf"),
            file_name="file1.pdf",
            status=ProcessingStatus.COMPLETED,
            total_chunks=10
        )
        
        # First succeeds, second fails
        processor.process_document.side_effect = [
            doc1,
            ProcessorException("Processing failed")
        ]
        
        pipeline = IngestionPipeline(processor=processor)
        
        result = pipeline.process_files(
            ["file1.pdf", "file2.pdf"],
            continue_on_error=True
        )
        
        assert result.total_files == 2
        assert result.successful == 1
        assert result.failed == 1
        assert result.success_rate == 50.0
        
        # Check individual results
        assert result.results[0].success is True
        assert result.results[1].success is False
        assert result.results[1].error_message is not None
        assert "Processing failed" in result.results[1].error_message
    
    def test_process_files_with_failures_stop(self):
        """Prueba procesamiento con fallos y continue_on_error=False"""
        processor = Mock()
        
        processor.process_document.side_effect = ProcessorException("Processing failed")
        
        pipeline = IngestionPipeline(processor=processor)
        
        with pytest.raises(PipelineException, match="Processing failed"):
            pipeline.process_files(
                ["file1.pdf", "file2.pdf"],
                continue_on_error=False
            )
    
    def test_process_files_empty_list(self):
        """Prueba procesamiento con lista vacía"""
        processor = Mock()
        
        pipeline = IngestionPipeline(processor=processor)
        
        result = pipeline.process_files([])
        
        assert result.total_files == 0
        assert result.successful == 0
        assert result.failed == 0
        
        # Processor should not be called
        processor.process_document.assert_not_called()
    
    def test_process_files_unsupported_extension(self):
        """Prueba procesamiento de archivo con extensión no soportada"""
        processor = Mock()
        
        pipeline = IngestionPipeline(processor=processor)
        
        result = pipeline.process_files(["file.txt"])
        
        assert result.total_files == 1
        assert result.successful == 0
        assert result.failed == 1
        assert result.results[0].error_message is not None
        assert "Unsupported file type" in result.results[0].error_message
        
        # Processor should not be called for unsupported files
        processor.process_document.assert_not_called()


class TestIngestionPipelineProcessDirectory:
    """Tests para método process_directory"""
    
    def test_process_directory_not_exists(self):
        """Prueba error cuando directorio no existe"""
        processor = Mock()
        
        pipeline = IngestionPipeline(processor=processor)
        
        with pytest.raises(PipelineException, match="Directory does not exist"):
            pipeline.process_directory("/nonexistent/directory")
    
    def test_process_directory_is_file(self, tmp_path):
        """Prueba error cuando path es un archivo, no directorio"""
        processor = Mock()
        
        # Create a file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        
        pipeline = IngestionPipeline(processor=processor)
        
        with pytest.raises(PipelineException, match="Path is not a directory"):
            pipeline.process_directory(test_file)
    
    def test_process_directory_no_files(self, tmp_path):
        """Prueba procesamiento de directorio vacío"""
        processor = Mock()
        
        pipeline = IngestionPipeline(processor=processor)
        
        result = pipeline.process_directory(tmp_path)
        
        assert result.total_files == 0
        assert result.successful == 0
        assert result.failed == 0
    
    def test_process_directory_with_pdf_files(self, tmp_path):
        """Prueba procesamiento de directorio con archivos PDF"""
        processor = Mock()
        
        # Create PDF files
        pdf1 = tmp_path / "file1.pdf"
        pdf2 = tmp_path / "file2.pdf"
        pdf1.write_text("content1")
        pdf2.write_text("content2")
        
        doc1 = Document(
            id="doc1",
            file_path=pdf1,
            file_name="file1.pdf",
            status=ProcessingStatus.COMPLETED,
            total_chunks=10
        )
        doc2 = Document(
            id="doc2",
            file_path=pdf2,
            file_name="file2.pdf",
            status=ProcessingStatus.COMPLETED,
            total_chunks=15
        )
        
        processor.process_document.side_effect = [doc1, doc2]
        
        pipeline = IngestionPipeline(processor=processor)
        
        result = pipeline.process_directory(tmp_path, recursive=False)
        
        assert result.total_files == 2
        assert result.successful == 2
        assert result.failed == 0
        
        # Verify files were processed
        assert processor.process_document.call_count == 2
    
    def test_process_directory_recursive(self, tmp_path):
        """Prueba procesamiento recursivo de directorios"""
        processor = Mock()
        
        # Create subdirectory with PDF
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        
        pdf1 = tmp_path / "file1.pdf"
        pdf2 = subdir / "file2.pdf"
        pdf1.write_text("content1")
        pdf2.write_text("content2")
        
        doc1 = Document(
            id="doc1",
            file_path=pdf1,
            file_name="file1.pdf",
            status=ProcessingStatus.COMPLETED,
            total_chunks=10
        )
        doc2 = Document(
            id="doc2",
            file_path=pdf2,
            file_name="file2.pdf",
            status=ProcessingStatus.COMPLETED,
            total_chunks=15
        )
        
        processor.process_document.side_effect = [doc1, doc2]
        
        pipeline = IngestionPipeline(processor=processor)
        
        # Non-recursive should find only 1 file
        result = pipeline.process_directory(tmp_path, recursive=False)
        assert result.total_files == 1
        
        # Reset mock
        processor.process_document.reset_mock()
        processor.process_document.side_effect = [doc1, doc2]
        
        # Recursive should find both files
        result = pipeline.process_directory(tmp_path, recursive=True)
        assert result.total_files == 2
    
    def test_process_directory_mixed_extensions(self, tmp_path):
        """Prueba directorio con múltiples tipos de archivos"""
        processor = Mock()
        
        # Create files with different extensions
        pdf = tmp_path / "file.pdf"
        txt = tmp_path / "file.txt"
        docx = tmp_path / "file.docx"
        
        pdf.write_text("pdf content")
        txt.write_text("txt content")
        docx.write_text("docx content")
        
        doc = Document(
            id="doc1",
            file_path=pdf,
            file_name="file.pdf",
            status=ProcessingStatus.COMPLETED,
            total_chunks=10
        )
        
        processor.process_document.return_value = doc
        
        pipeline = IngestionPipeline(processor=processor)
        
        result = pipeline.process_directory(tmp_path)
        
        # Only PDF should be processed
        assert result.total_files == 1
        assert result.successful == 1
        
        # Verify only PDF was processed
        processor.process_document.assert_called_once()


class TestIngestionPipelineHelperMethods:
    """Tests para métodos helper privados"""
    
    def test_is_supported_file_pdf(self):
        """Prueba verificación de archivo PDF soportado"""
        processor = Mock()
        pipeline = IngestionPipeline(processor=processor)
        
        assert pipeline._is_supported_file(Path("test.pdf")) is True
        assert pipeline._is_supported_file(Path("test.PDF")) is True
    
    def test_is_supported_file_unsupported(self):
        """Prueba verificación de archivo no soportado"""
        processor = Mock()
        pipeline = IngestionPipeline(processor=processor)
        
        assert pipeline._is_supported_file(Path("test.txt")) is False
        assert pipeline._is_supported_file(Path("test.docx")) is False
    
    def test_is_supported_file_custom_extensions(self):
        """Prueba verificación con extensiones personalizadas"""
        processor = Mock()
        pipeline = IngestionPipeline(
            processor=processor,
            supported_extensions=['.pdf', '.txt']
        )
        
        assert pipeline._is_supported_file(Path("test.pdf")) is True
        assert pipeline._is_supported_file(Path("test.txt")) is True
        assert pipeline._is_supported_file(Path("test.docx")) is False
    
    def test_find_supported_files_non_recursive(self, tmp_path):
        """Prueba búsqueda no recursiva de archivos"""
        processor = Mock()
        pipeline = IngestionPipeline(processor=processor)
        
        # Create files
        pdf1 = tmp_path / "file1.pdf"
        pdf2 = tmp_path / "file2.pdf"
        pdf1.write_text("content")
        pdf2.write_text("content")
        
        # Create subdirectory with file (should not be found)
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        pdf3 = subdir / "file3.pdf"
        pdf3.write_text("content")
        
        files = pipeline._find_supported_files(tmp_path, recursive=False)
        
        assert len(files) == 2
        assert all(f.parent == tmp_path for f in files)
    
    def test_find_supported_files_recursive(self, tmp_path):
        """Prueba búsqueda recursiva de archivos"""
        processor = Mock()
        pipeline = IngestionPipeline(processor=processor)
        
        # Create files
        pdf1 = tmp_path / "file1.pdf"
        pdf1.write_text("content")
        
        # Create subdirectory with file
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        pdf2 = subdir / "file2.pdf"
        pdf2.write_text("content")
        
        files = pipeline._find_supported_files(tmp_path, recursive=True)
        
        assert len(files) == 2
    
    def test_find_supported_files_sorted(self, tmp_path):
        """Prueba que archivos se retornan ordenados"""
        processor = Mock()
        pipeline = IngestionPipeline(processor=processor)
        
        # Create files in random order
        (tmp_path / "c.pdf").write_text("content")
        (tmp_path / "a.pdf").write_text("content")
        (tmp_path / "b.pdf").write_text("content")
        
        files = pipeline._find_supported_files(tmp_path, recursive=False)
        
        # Should be sorted
        assert [f.name for f in files] == ["a.pdf", "b.pdf", "c.pdf"]
