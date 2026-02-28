"""
Domain models for the ChatBot RAG system.
Defines the core entities and their behaviors.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pathlib import Path


class ProcessingStatus(Enum):
    """Estados del procesamiento de documentos"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Document:
    """Representa un documento PDF ingresado al sistema"""
    id: str
    file_path: Path
    file_name: str
    status: ProcessingStatus
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    total_chunks: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mark_processing(self):
        """Marca el documento como en procesamiento"""
        self.status = ProcessingStatus.PROCESSING

    def mark_completed(self, chunks_count: int):
        """Marca el documento como completado"""
        self.status = ProcessingStatus.COMPLETED
        self.processed_at = datetime.now()
        self.total_chunks = chunks_count

    def mark_failed(self, error: str):
        """Marca el documento como fallido"""
        self.status = ProcessingStatus.FAILED
        self.processed_at = datetime.now()
        self.error_message = error


@dataclass
class ChunkingConfig:
    """Configuración para el chunking de documentos"""
    chunk_size: int = 1000  # Caracteres por chunk
    chunk_overlap: int = 200  # Overlap entre chunks
    separator: str = "\n\n"  # Separador de chunks
    
    def validate(self):
        """Valida la configuración"""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size debe ser mayor a 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap debe ser menor que chunk_size")


@dataclass
class Chunk:
    """Representa un fragmento de texto procesado"""
    id: str
    document_id: str
    content: str
    chunk_index: int
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_embedding(self) -> bool:
        """Verifica si el chunk tiene embedding"""
        return bool(self.embedding and len(self.embedding) > 0)


@dataclass
class SearchResult:
    """Resultado de una búsqueda en el vector store"""
    chunk: Chunk
    score: float
    document_name: str

    def __lt__(self, other):
        """Permite ordenar por score (mayor a menor)"""
        return self.score > other.score


@dataclass
class RetrievalConfig:
    """Configuración para el retrieval de chunks"""
    top_k: int = 5  # Número de chunks a recuperar
    min_score: float = 0.7  # Score mínimo de similitud (0-1)
    
    def validate(self):
        """Valida la configuración"""
        if self.top_k <= 0:
            raise ValueError("top_k debe ser mayor a 0")
        if not 0 <= self.min_score <= 1:
            raise ValueError("min_score debe estar entre 0 y 1")