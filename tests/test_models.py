"""
Unit tests for domain models.
"""
import sys
import pytest
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any



from domain.models import (
    Document, ProcessingStatus, ChunkingConfig, Chunk,
    SearchResult, RetrievalConfig
)


class TestDocument:
    """Tests para el modelo Document"""
    
    def test_create_document(self):
        """Prueba creación de documento"""
        doc = Document(
            id="doc1",
            file_path=Path("test.pdf"),
            file_name="test.pdf",
            status=ProcessingStatus.PENDING
        )
        
        assert doc.id == "doc1"
        assert doc.status == ProcessingStatus.PENDING
        assert doc.total_chunks == 0
        assert doc.error_message is None
    
    def test_mark_processing(self):
        """Prueba cambio de estado a processing"""
        doc = Document(
            id="doc1",
            file_path=Path("test.pdf"),
            file_name="test.pdf",
            status=ProcessingStatus.PENDING
        )
        
        doc.mark_processing()
        assert doc.status == ProcessingStatus.PROCESSING
    
    def test_mark_completed(self):
        """Prueba completar procesamiento"""
        doc = Document(
            id="doc1",
            file_path=Path("test.pdf"),
            file_name="test.pdf",
            status=ProcessingStatus.PROCESSING
        )
        
        doc.mark_completed(chunks_count=10)
        
        assert doc.status == ProcessingStatus.COMPLETED
        assert doc.total_chunks == 10
        assert doc.processed_at is not None
    
    def test_mark_failed(self):
        """Prueba marcar como fallido"""
        doc = Document(
            id="doc1",
            file_path=Path("test.pdf"),
            file_name="test.pdf",
            status=ProcessingStatus.PROCESSING
        )
        
        doc.mark_failed("Error al procesar")
        
        assert doc.status == ProcessingStatus.FAILED
        assert doc.error_message == "Error al procesar"
        assert doc.processed_at is not None


class TestChunkingConfig:
    """Tests para ChunkingConfig"""
    
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
    
    def test_validate_invalid_chunk_size(self):
        """Prueba validación con chunk_size inválido"""
        config = ChunkingConfig(chunk_size=0)
        
        with pytest.raises(ValueError, match="chunk_size debe ser mayor a 0"):
            config.validate()
    
    def test_validate_invalid_overlap(self):
        """Prueba validación con overlap mayor que chunk_size"""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=200)
        
        with pytest.raises(ValueError, match="chunk_overlap debe ser menor"):
            config.validate()


class TestChunk:
    """Tests para Chunk"""
    
    def test_create_chunk(self):
        """Prueba creación de chunk"""
        chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            content="Este es un texto de prueba",
            chunk_index=0
        )
        
        assert chunk.id == "chunk1"
        assert chunk.has_embedding is False
    
    def test_chunk_with_embedding(self):
        """Prueba chunk con embedding"""
        chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            content="Texto",
            chunk_index=0,
            embedding=[0.1, 0.2, 0.3]
        )
        
        assert chunk.has_embedding is True
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 3


class TestRetrievalConfig:
    """Tests para RetrievalConfig"""
    
    def test_default_config(self):
        """Prueba configuración por defecto"""
        config = RetrievalConfig()
        
        assert config.top_k == 5
        assert config.min_score == 0.7
    
    def test_validate_invalid_top_k(self):
        """Prueba validación con top_k inválido"""
        config = RetrievalConfig(top_k=0)
        
        with pytest.raises(ValueError, match="top_k debe ser mayor a 0"):
            config.validate()
    
    def test_validate_invalid_min_score(self):
        """Prueba validación con min_score fuera de rango"""
        config = RetrievalConfig(min_score=1.5)
        
        with pytest.raises(ValueError, match="min_score debe estar entre 0 y 1"):
            config.validate()

