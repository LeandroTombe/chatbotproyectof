"""
Unit tests for loader factory functions.
"""
import pytest
from unittest.mock import Mock, patch

from documents.loaders.base_loader import LoaderException
from documents.loaders.pdf_loader import PDFLoader
from documents.loaders.loader_factory import get_loader, load_document
from domain.models import Document


class TestGetLoader:
    """Tests para función factory get_loader"""
    
    def test_get_loader_pdf(self):
        """Prueba obtención de PDFLoader"""
        loader = get_loader("test.pdf")
        assert isinstance(loader, PDFLoader)
    
    def test_get_loader_unsupported(self):
        """Prueba error con tipo no soportado"""
        with pytest.raises(LoaderException, match="No loader available"):
            get_loader("test.txt")
    
    def test_get_loader_with_kwargs(self):
        """Prueba pasar kwargs al loader"""
        loader = get_loader("test.pdf", extract_images=True)
        assert isinstance(loader, PDFLoader)
        assert loader.extract_images is True


class TestLoadDocument:
    """Tests para función de conveniencia load_document"""
    
    @patch('ingestion.loaders.loader_factory.PDFLoader.load')
    def test_load_document_success(self, mock_load):
        """Prueba carga exitosa de documento"""
        mock_doc = Mock(spec=Document)
        mock_load.return_value = mock_doc
        
        doc = load_document("test.pdf")
        assert doc == mock_doc
        mock_load.assert_called_once_with("test.pdf")
    
    def test_load_document_unsupported_type(self):
        """Prueba error con tipo no soportado"""
        with pytest.raises(LoaderException):
            load_document("test.unknown")
