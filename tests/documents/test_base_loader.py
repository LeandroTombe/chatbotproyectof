"""
Unit tests for BaseLoader.
"""
import pytest
from pathlib import Path
import tempfile
import os
from unittest.mock import Mock

from documents.loaders.base_loader import BaseLoader, LoaderException
from domain.models import Document


class TestBaseLoaderAbstract:
    """Tests para verificar que BaseLoader es abstracta"""
    
    def test_cannot_instantiate_base_loader(self):
        """Prueba que no se puede instanciar BaseLoader directamente"""
        with pytest.raises(TypeError):
            BaseLoader()  # type: ignore[abstract]


class TestBaseLoaderValidateFile:
    """Tests para validación de archivos en BaseLoader"""
    
    def test_validate_file_not_exists(self):
        """Prueba error cuando archivo no existe"""
        # Crear una clase concreta para poder testear _validate_file
        class ConcreteLoader(BaseLoader):
            def load(self, file_path: str) -> Document:
                return Mock(spec=Document)  # type: ignore[return-value]
            
            def supports(self, file_path: str) -> bool:
                return True
        
        loader = ConcreteLoader()
        with pytest.raises(FileNotFoundError, match="File not found"):
            loader._validate_file("nonexistent.txt")
    
    def test_validate_file_is_directory(self):
        """Prueba error cuando path es un directorio"""
        class ConcreteLoader(BaseLoader):
            def load(self, file_path: str) -> Document:
                return Mock(spec=Document)  # type: ignore[return-value]
            
            def supports(self, file_path: str) -> bool:
                return True
        
        loader = ConcreteLoader()
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(LoaderException, match="Path is not a file"):
                loader._validate_file(tmpdir)
    
    def test_validate_file_empty(self):
        """Prueba error cuando archivo está vacío"""
        class ConcreteLoader(BaseLoader):
            def load(self, file_path: str) -> Document:
                return Mock(spec=Document)  # type: ignore[return-value]
            
            def supports(self, file_path: str) -> bool:
                return True
        
        loader = ConcreteLoader()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with pytest.raises(LoaderException, match="File is empty"):
                loader._validate_file(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    def test_validate_file_valid(self):
        """Prueba validación exitosa de archivo"""
        class ConcreteLoader(BaseLoader):
            def load(self, file_path: str) -> Document:
                return Mock(spec=Document)  # type: ignore[return-value]
            
            def supports(self, file_path: str) -> bool:
                return True
        
        loader = ConcreteLoader()
        with tempfile.NamedTemporaryFile(delete=False, mode='wb') as tmp:
            tmp.write(b"dummy content")
            tmp_path = tmp.name
        
        try:
            result = loader._validate_file(tmp_path)
            assert isinstance(result, Path)
            assert result.exists()
        finally:
            os.unlink(tmp_path)
