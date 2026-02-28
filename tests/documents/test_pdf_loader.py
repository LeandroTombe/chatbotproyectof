"""
Unit tests for PDFLoader.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import os

from documents.loaders.base_loader import LoaderException
from documents.loaders.pdf_loader import PDFLoader
from domain.models import Document, ProcessingStatus


class TestPDFLoaderInitialization:
    """Tests para inicialización de PDFLoader"""
    
    def test_initialization_default(self):
        """Prueba inicialización por defecto"""
        loader = PDFLoader()
        assert loader is not None
        assert loader.extract_images is False
    
    def test_pypdf2_not_installed(self):
        """Prueba error cuando PyPDF2 no está instalado"""
        with patch.dict('sys.modules', {'PyPDF2': None}):
            with pytest.raises(LoaderException, match="PyPDF2 is not installed"):
                # Force reimport
                import importlib
                import documents.loaders.pdf_loader
                importlib.reload(documents.loaders.pdf_loader)
                documents.loaders.pdf_loader.PDFLoader()


class TestPDFLoaderSupports:
    """Tests para método supports"""
    
    def test_supports_pdf_extension(self):
        """Prueba que soporta archivos .pdf"""
        loader = PDFLoader()
        assert loader.supports("test.pdf") is True
        assert loader.supports("test.PDF") is True
        assert loader.supports("/path/to/document.pdf") is True
    
    def test_does_not_support_other_extensions(self):
        """Prueba que no soporta otros formatos"""
        loader = PDFLoader()
        assert loader.supports("test.txt") is False
        assert loader.supports("test.docx") is False
        assert loader.supports("test.html") is False
        assert loader.supports("test") is False


class TestPDFLoaderLoad:
    """Tests para carga de PDFs"""
    
    def test_load_unsupported_file_type(self):
        """Prueba error al cargar tipo no soportado"""
        loader = PDFLoader()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w') as tmp:
            tmp.write("test content")
            tmp_path = tmp.name
        
        try:
            with pytest.raises(LoaderException, match="does not support file type"):
                loader.load(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    @patch('PyPDF2.PdfReader')
    def test_load_pdf_success(self, mock_pdf_reader):
        """Prueba carga exitosa de PDF"""
        # Mock PDF reader
        mock_page = Mock()
        mock_page.extract_text.return_value = "This is a test PDF content."
        
        mock_reader = Mock()
        mock_reader.pages = [mock_page, mock_page]
        
        mock_pdf_reader.return_value = mock_reader
        
        loader = PDFLoader()
        
        # Create temporary PDF file (just needs to exist and have content)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', mode='wb') as tmp:
            tmp.write(b"dummy content")
            tmp_path = tmp.name
        
        try:
            doc = loader.load(tmp_path)
            
            assert isinstance(doc, Document)
            assert "This is a test PDF content." in doc.metadata.get("content", "")
            assert doc.metadata.get("file_type") == "pdf"
            assert doc.metadata.get("page_count") == 2
            assert doc.metadata.get("loader") == "PDFLoader"
            assert doc.file_name == Path(tmp_path).name
            assert doc.status == ProcessingStatus.COMPLETED
            
        finally:
            os.unlink(tmp_path)
    
    @patch('PyPDF2.PdfReader')
    def test_load_pdf_no_text_extracted(self, mock_pdf_reader):
        """Prueba error cuando no se extrae texto"""
        # Mock PDF reader with empty text
        mock_page = Mock()
        mock_page.extract_text.return_value = ""
        
        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        
        mock_pdf_reader.return_value = mock_reader
        
        loader = PDFLoader()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', mode='wb') as tmp:
            tmp.write(b"dummy content")
            tmp_path = tmp.name
        
        try:
            with pytest.raises(LoaderException, match="No text content extracted"):
                loader.load(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    @patch('PyPDF2.PdfReader')
    def test_load_pdf_corrupted(self, mock_pdf_reader):
        """Prueba error con PDF corrupto"""
        # Mock PDF reader to raise exception
        mock_pdf_reader.side_effect = Exception("PDF processing error")
        
        loader = PDFLoader()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', mode='wb') as tmp:
            tmp.write(b"dummy content")
            tmp_path = tmp.name
        
        try:
            with pytest.raises(LoaderException, match="Error loading PDF"):
                loader.load(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    @patch('PyPDF2.PdfReader')
    def test_load_pdf_skip_failed_pages(self, mock_pdf_reader):
        """Prueba que continúa si falla extracción de una página"""
        # First page succeeds, second fails, third succeeds
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"
        
        mock_page2 = Mock()
        mock_page2.extract_text.side_effect = Exception("Extraction failed")
        
        mock_page3 = Mock()
        mock_page3.extract_text.return_value = "Page 3 content"
        
        mock_reader = Mock()
        mock_reader.pages = [mock_page1, mock_page2, mock_page3]
        
        mock_pdf_reader.return_value = mock_reader
        
        loader = PDFLoader()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', mode='wb') as tmp:
            tmp.write(b"dummy content")
            tmp_path = tmp.name
        
        try:
            doc = loader.load(tmp_path)
            
            assert "Page 1 content" in doc.metadata.get("content", "")
            assert "Page 3 content" in doc.metadata.get("content", "")
            assert doc.metadata.get("page_count") == 3
            
        finally:
            os.unlink(tmp_path)


class TestPDFLoaderGenerateDocumentId:
    """Tests para generación de IDs de documentos"""
    
    def test_generate_document_id(self):
        """Prueba generación de ID único"""
        loader = PDFLoader()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(b"some pdf content")
            tmp_path = tmp.name
        try:
            doc_id = loader._generate_document_id(Path(tmp_path))
            assert doc_id.startswith("doc_")
            assert len(doc_id) == 20  # "doc_" + 16 hex chars
        finally:
            os.unlink(tmp_path)
    
    def test_generate_document_id_consistent(self):
        """Prueba que el mismo archivo genera el mismo ID"""
        loader = PDFLoader()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(b"consistent content")
            tmp_path = tmp.name
        try:
            id1 = loader._generate_document_id(Path(tmp_path))
            id2 = loader._generate_document_id(Path(tmp_path))
            assert id1 == id2
        finally:
            os.unlink(tmp_path)
    
    def test_generate_document_id_different_paths(self):
        """Prueba que archivos con diferente contenido generan IDs diferentes"""
        loader = PDFLoader()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp1:
            tmp1.write(b"content of file one")
            tmp_path1 = tmp1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp2:
            tmp2.write(b"content of file two")
            tmp_path2 = tmp2.name
        try:
            id1 = loader._generate_document_id(Path(tmp_path1))
            id2 = loader._generate_document_id(Path(tmp_path2))
            assert id1 != id2
        finally:
            os.unlink(tmp_path1)
            os.unlink(tmp_path2)
