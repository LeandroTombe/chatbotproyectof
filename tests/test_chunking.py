"""
Unit tests for the chunking module.
"""
import pytest
from typing import List

from domain.models import Chunk, ChunkingConfig
from processing.chunking import (
    TextChunker,
    ChunkingException,
    chunk_text_simple
)


class TestChunkingConfig:
    """Tests para la configuración de chunking"""
    
    def test_default_config(self):
        """Prueba configuración por defecto"""
        config = ChunkingConfig()
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.separator == "\n\n"
    
    def test_custom_config(self):
        """Prueba configuración personalizada"""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=100,
            separator="\n"
        )
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.separator == "\n"
    
    def test_validate_invalid_chunk_size(self):
        """Prueba validación de chunk_size inválido"""
        config = ChunkingConfig(chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size debe ser mayor a 0"):
            config.validate()
    
    def test_validate_invalid_overlap(self):
        """Prueba validación de overlap >= chunk_size"""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError, match="chunk_overlap debe ser menor que chunk_size"):
            config.validate()


class TestTextChunker:
    """Tests para la clase TextChunker"""
    
    def test_initialization_default(self):
        """Prueba inicialización con valores por defecto"""
        chunker = TextChunker()
        assert chunker.config.chunk_size == 1000
        assert chunker.config.chunk_overlap == 200
    
    def test_initialization_custom(self):
        """Prueba inicialización con config personalizada"""
        config = ChunkingConfig(chunk_size=500, chunk_overlap=50)
        chunker = TextChunker(config)
        assert chunker.config.chunk_size == 500
        assert chunker.config.chunk_overlap == 50
    
    def test_chunk_empty_text(self):
        """Prueba chunking de texto vacío"""
        chunker = TextChunker()
        chunks = chunker.chunk_text("", "doc1")
        assert chunks == []
    
    def test_chunk_whitespace_only(self):
        """Prueba chunking de texto solo con espacios"""
        chunker = TextChunker()
        chunks = chunker.chunk_text("   \n  \t  ", "doc1")
        assert chunks == []
    
    def test_chunk_short_text(self):
        """Prueba chunking de texto corto (menor que chunk_size)"""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        chunker = TextChunker(config)
        text = "Este es un texto corto."
        
        chunks = chunker.chunk_text(text, "doc1")
        
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].document_id == "doc1"
        assert chunks[0].chunk_index == 0
        assert chunks[0].id is not None
    
    def test_chunk_long_text_by_characters(self):
        """Prueba chunking de texto largo sin separadores"""
        config = ChunkingConfig(chunk_size=50, chunk_overlap=10, separator="\n\n")
        chunker = TextChunker(config)
        text = "A" * 120  # 120 caracteres sin separador
        
        chunks = chunker.chunk_text(text, "doc1")
        
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.document_id == "doc1" for c in chunks)
        # Verificar índices consecutivos
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
    
    def test_chunk_text_with_separator(self):
        """Prueba chunking de texto con separadores"""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20, separator="\n\n")
        chunker = TextChunker(config)
        text = "Párrafo 1.\n\nPárrafo 2.\n\nPárrafo 3.\n\nPárrafo 4."
        
        chunks = chunker.chunk_text(text, "doc1")
        
        assert len(chunks) > 0
        assert all(c.document_id == "doc1" for c in chunks)
    
    def test_chunk_with_metadata(self):
        """Prueba que los chunks incluyan metadata"""
        chunker = TextChunker()
        text = "Texto de prueba."
        metadata = {"author": "Test", "source": "test.pdf"}
        
        chunks = chunker.chunk_text(text, "doc1", metadata)
        
        assert len(chunks) == 1
        assert chunks[0].metadata["author"] == "Test"
        assert chunks[0].metadata["source"] == "test.pdf"
        assert "chunk_method" in chunks[0].metadata
        assert "original_length" in chunks[0].metadata
    
    def test_chunk_overlap_applied(self):
        """Prueba que se aplique el overlap correctamente"""
        config = ChunkingConfig(chunk_size=30, chunk_overlap=10, separator="|")
        chunker = TextChunker(config)
        text = "AAAAAAAAAA|BBBBBBBBBB|CCCCCCCCCC"  # 3 segmentos de 10 chars
        
        chunks = chunker.chunk_text(text, "doc1")
        
        assert len(chunks) >= 1
        # Verificar que hay contenido en los chunks
        assert all(len(c.content) > 0 for c in chunks)
    
    def test_chunk_unique_ids(self):
        """Prueba que cada chunk tenga un ID único"""
        config = ChunkingConfig(chunk_size=50, chunk_overlap=10)
        chunker = TextChunker(config)
        text = "A" * 200
        
        chunks = chunker.chunk_text(text, "doc1")
        
        chunk_ids = [c.id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))  # Todos únicos
    
    def test_get_chunk_count_estimate_empty(self):
        """Prueba estimación con texto vacío"""
        chunker = TextChunker()
        assert chunker.get_chunk_count_estimate("") == 0
    
    def test_get_chunk_count_estimate_short(self):
        """Prueba estimación con texto corto"""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        chunker = TextChunker(config)
        text = "Short text."
        
        estimate = chunker.get_chunk_count_estimate(text)
        assert estimate == 1
    
    def test_get_chunk_count_estimate_long(self):
        """Prueba estimación con texto largo"""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20)
        chunker = TextChunker(config)
        text = "A" * 500
        
        estimate = chunker.get_chunk_count_estimate(text)
        assert estimate > 1
    
    def test_large_segment_splitting(self):
        """Prueba división de segmentos grandes"""
        config = ChunkingConfig(chunk_size=50, chunk_overlap=10, separator="\n\n")
        chunker = TextChunker(config)
        # Un párrafo muy largo que excede chunk_size
        text = "A" * 150 + "\n\n" + "B" * 30
        
        chunks = chunker.chunk_text(text, "doc1")
        
        assert len(chunks) > 1
        assert all(c.document_id == "doc1" for c in chunks)
    
    def test_chunk_indices_sequential(self):
        """Prueba que los índices de chunks sean secuenciales"""
        config = ChunkingConfig(chunk_size=30, chunk_overlap=5)
        chunker = TextChunker(config)
        text = "A" * 100
        
        chunks = chunker.chunk_text(text, "doc1")
        
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


class TestChunkTextSimple:
    """Tests para la función de conveniencia chunk_text_simple"""
    
    def test_simple_chunking(self):
        """Prueba chunking simple"""
        text = "Este es un texto de prueba."
        chunks = chunk_text_simple(text, "doc1")
        
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].document_id == "doc1"
    
    def test_simple_chunking_custom_size(self):
        """Prueba chunking simple con tamaño personalizado"""
        text = "A" * 200
        chunks = chunk_text_simple(text, "doc1", chunk_size=50, chunk_overlap=10)
        
        assert len(chunks) > 1
        assert all(c.document_id == "doc1" for c in chunks)
    
    def test_simple_chunking_no_overlap(self):
        """Prueba chunking sin overlap"""
        text = "A" * 100
        chunks = chunk_text_simple(text, "doc1", chunk_size=50, chunk_overlap=0)
        
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)


class TestChunkModel:
    """Tests adicionales para verificar el modelo Chunk"""
    
    def test_chunk_has_embedding_property(self):
        """Prueba la propiedad has_embedding"""
        chunker = TextChunker()
        text = "Test text"
        chunks = chunker.chunk_text(text, "doc1")
        
        chunk = chunks[0]
        assert chunk.has_embedding is False
        
        # Simular agregar embedding
        chunk.embedding = [0.1, 0.2, 0.3]
        assert chunk.has_embedding is True


class TestEdgeCases:
    """Tests para casos extremos"""
    
    def test_very_small_chunk_size(self):
        """Prueba con chunk_size muy pequeño"""
        config = ChunkingConfig(chunk_size=10, chunk_overlap=2)
        chunker = TextChunker(config)
        text = "Este es un texto mediano para probar."
        
        chunks = chunker.chunk_text(text, "doc1")
        
        assert len(chunks) > 1
        assert all(c.document_id == "doc1" for c in chunks)
    
    def test_special_characters(self):
        """Prueba con caracteres especiales"""
        chunker = TextChunker()
        text = "Texto con símbolos: @#$%^&*(){}[]|\\/<>?~`"
        
        chunks = chunker.chunk_text(text, "doc1")
        
        assert len(chunks) == 1
        assert chunks[0].content == text
    
    def test_unicode_characters(self):
        """Prueba con caracteres unicode"""
        chunker = TextChunker()
        text = "Texto con emojis 😀🎉 y acentos: áéíóú ñ"
        
        chunks = chunker.chunk_text(text, "doc1")
        
        assert len(chunks) == 1
        assert chunks[0].content == text
    
    def test_mixed_separators(self):
        """Prueba con múltiples tipos de separadores en el texto"""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20, separator="\n")
        chunker = TextChunker(config)
        text = "Línea 1\nLínea 2\n\nLínea 3\n\n\nLínea 4"
        
        chunks = chunker.chunk_text(text, "doc1")
        
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)
