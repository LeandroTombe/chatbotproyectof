"""
Base module for vector store implementations.
Defines the abstract interface for storing and retrieving embeddings.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
import logging
import math

from domain.models import Chunk, SearchResult, RetrievalConfig

logger = logging.getLogger(__name__)


class VectorStoreException(Exception):
    """Exception raised for vector store errors"""
    pass


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calcula la similitud coseno entre dos vectores.
    
    Args:
        vec1: Primer vector
        vec2: Segundo vector
        
    Returns:
        Similitud coseno entre -1 y 1 (1 = idénticos, 0 = ortogonales, -1 = opuestos)
        
    Raises:
        ValueError: Si los vectores tienen diferentes dimensiones
    """
    if len(vec1) != len(vec2):
        raise ValueError(
            f"Vector dimension mismatch: {len(vec1)} vs {len(vec2)}"
        )
    
    if not vec1 or not vec2:
        raise ValueError("Vectors cannot be empty")
    
    # Producto punto
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # Magnitudes
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    Calcula la distancia euclidiana entre dos vectores.
    
    Args:
        vec1: Primer vector
        vec2: Segundo vector
        
    Returns:
        Distancia euclidiana (0 = idénticos, mayor = más diferentes)
        
    Raises:
        ValueError: Si los vectores tienen diferentes dimensiones
    """
    if len(vec1) != len(vec2):
        raise ValueError(
            f"Vector dimension mismatch: {len(vec1)} vs {len(vec2)}"
        )
    
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))


class BaseVectorStore(ABC):
    """
    Clase base abstracta para almacenes de vectores.
    Define la interfaz que deben implementar todos los vector stores.
    """
    
    def __init__(self, dimension: int, config: Optional[RetrievalConfig] = None):
        """
        Inicializa el vector store.
        
        Args:
            dimension: Dimensión de los vectores a almacenar
            config: Configuración de retrieval. Si es None, usa valores por defecto.
            
        Raises:
            ValueError: Si dimension es inválida
        """
        if dimension <= 0:
            raise ValueError("dimension debe ser mayor a 0")
        
        self.dimension = dimension
        self.config = config or RetrievalConfig()
        self.config.validate()
        logger.info(
            f"{self.__class__.__name__} initialized with dimension={dimension}, "
            f"top_k={self.config.top_k}"
        )
    
    @abstractmethod
    def add_chunk(self, chunk: Chunk) -> None:
        """
        Agrega un chunk con su embedding al vector store.
        
        Args:
            chunk: Chunk a agregar (debe tener embedding)
            
        Raises:
            VectorStoreException: Si hay error al agregar
            ValueError: Si el chunk no tiene embedding o dimensión incorrecta
        """
        pass
    
    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Agrega múltiples chunks al vector store.
        
        Args:
            chunks: Lista de chunks a agregar (deben tener embeddings)
            
        Raises:
            VectorStoreException: Si hay error al agregar
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        document_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Busca los chunks más similares a un embedding de consulta.
        
        Args:
            query_embedding: Vector de embedding de la consulta
            top_k: Número de resultados a retornar (usa config por defecto si None)
            min_score: Score mínimo de similitud (usa config por defecto si None)
            document_id: Filtrar por documento específico (opcional)
            
        Returns:
            Lista de SearchResult ordenados por score (mayor a menor)
            
        Raises:
            VectorStoreException: Si hay error en la búsqueda
            ValueError: Si el embedding tiene dimensión incorrecta
        """
        pass
    
    @abstractmethod
    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Elimina un chunk del vector store.
        
        Args:
            chunk_id: ID del chunk a eliminar
            
        Returns:
            True si se eliminó, False si no existía
            
        Raises:
            VectorStoreException: Si hay error al eliminar
        """
        pass
    
    @abstractmethod
    def delete_chunks_by_document(self, document_id: str) -> int:
        """
        Elimina todos los chunks de un documento.
        
        Args:
            document_id: ID del documento
            
        Returns:
            Número de chunks eliminados
            
        Raises:
            VectorStoreException: Si hay error al eliminar
        """
        pass
    
    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """
        Obtiene un chunk por su ID.
        
        Args:
            chunk_id: ID del chunk
            
        Returns:
            Chunk si existe, None si no
            
        Raises:
            VectorStoreException: Si hay error al obtener
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """
        Retorna el número total de chunks almacenados.
        
        Returns:
            Número de chunks
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """
        Elimina todos los chunks del vector store.
        
        Raises:
            VectorStoreException: Si hay error al limpiar
        """
        pass
    
    def validate_embedding(self, embedding: List[float]) -> None:
        """
        Valida que un embedding tenga la dimensión correcta.
        
        Args:
            embedding: Vector a validar
            
        Raises:
            ValueError: Si el embedding es inválido
        """
        if not embedding:
            raise ValueError("Embedding cannot be empty")
        
        if len(embedding) != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {len(embedding)}"
            )


class InMemoryVectorStore(BaseVectorStore):
    """
    Implementación en memoria del vector store.
    Útil para desarrollo, testing y prototipos.
    No persistente - los datos se pierden al terminar el proceso.
    """
    
    def __init__(self, dimension: int, config: Optional[RetrievalConfig] = None):
        """
        Inicializa el vector store en memoria.
        
        Args:
            dimension: Dimensión de los vectores
            config: Configuración de retrieval
        """
        super().__init__(dimension, config)
        self._chunks: Dict[str, Chunk] = {}
        logger.info("InMemoryVectorStore initialized")
    
    def add_chunk(self, chunk: Chunk) -> None:
        """
        Agrega un chunk al almacén en memoria.
        
        Args:
            chunk: Chunk a agregar
            
        Raises:
            ValueError: Si el chunk no tiene embedding o dimensión incorrecta
        """
        if not chunk.has_embedding:
            raise ValueError(f"Chunk {chunk.id} does not have embedding")
        
        if chunk.embedding is None:
            raise ValueError(f"Chunk {chunk.id} embedding is None")
        
        self.validate_embedding(chunk.embedding)
        
        self._chunks[chunk.id] = chunk
        logger.debug(f"Added chunk {chunk.id} to vector store")
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Agrega múltiples chunks al almacén.
        
        Args:
            chunks: Lista de chunks
            
        Raises:
            VectorStoreException: Si hay error al agregar
        """
        if not chunks:
            logger.warning("Empty chunks list provided")
            return
        
        try:
            for chunk in chunks:
                self.add_chunk(chunk)
            
            logger.info(f"Added {len(chunks)} chunks to vector store")
        except Exception as e:
            error_msg = f"Error adding chunks: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreException(error_msg) from e
    
    def search(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        document_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Busca chunks similares usando similitud coseno.
        
        Args:
            query_embedding: Vector de consulta
            top_k: Número de resultados
            min_score: Score mínimo
            document_id: Filtrar por documento
            
        Returns:
            Lista de SearchResult ordenados por score
            
        Raises:
            ValueError: Si el embedding es inválido
        """
        self.validate_embedding(query_embedding)
        
        k = top_k if top_k is not None else self.config.top_k
        min_s = min_score if min_score is not None else self.config.min_score
        
        # Filtrar chunks por documento si se especifica
        chunks_to_search = self._chunks.values()
        if document_id:
            chunks_to_search = [
                c for c in chunks_to_search if c.document_id == document_id
            ]
        
        # Calcular similitudes
        results: List[Tuple[Chunk, float]] = []
        for chunk in chunks_to_search:
            if chunk.embedding is None:
                continue
            
            try:
                score = cosine_similarity(query_embedding, chunk.embedding)
                if score >= min_s:
                    results.append((chunk, score))
            except Exception as e:
                logger.warning(f"Error calculating similarity for chunk {chunk.id}: {e}")
                continue
        
        # Ordenar por score (mayor a menor) y limitar a top_k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:k]
        
        # Crear SearchResults
        search_results = [
            SearchResult(
                chunk=chunk,
                score=score,
                document_name=chunk.metadata.get("file_name", "unknown")
            )
            for chunk, score in results
        ]
        
        logger.debug(
            f"Search returned {len(search_results)} results "
            f"(min_score={min_s}, top_k={k})"
        )
        
        return search_results
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Elimina un chunk por ID.
        
        Args:
            chunk_id: ID del chunk
            
        Returns:
            True si se eliminó, False si no existía
        """
        if chunk_id in self._chunks:
            del self._chunks[chunk_id]
            logger.debug(f"Deleted chunk {chunk_id}")
            return True
        
        logger.debug(f"Chunk {chunk_id} not found for deletion")
        return False
    
    def delete_chunks_by_document(self, document_id: str) -> int:
        """
        Elimina todos los chunks de un documento.
        
        Args:
            document_id: ID del documento
            
        Returns:
            Número de chunks eliminados
        """
        chunks_to_delete = [
            chunk_id for chunk_id, chunk in self._chunks.items()
            if chunk.document_id == document_id
        ]
        
        for chunk_id in chunks_to_delete:
            del self._chunks[chunk_id]
        
        logger.info(f"Deleted {len(chunks_to_delete)} chunks from document {document_id}")
        return len(chunks_to_delete)
    
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """
        Obtiene un chunk por ID.
        
        Args:
            chunk_id: ID del chunk
            
        Returns:
            Chunk si existe, None si no
        """
        return self._chunks.get(chunk_id)
    
    def count(self) -> int:
        """
        Retorna el número de chunks almacenados.
        
        Returns:
            Número de chunks
        """
        return len(self._chunks)
    
    def clear(self) -> None:
        """
        Elimina todos los chunks del almacén.
        """
        count = len(self._chunks)
        self._chunks.clear()
        logger.info(f"Cleared vector store ({count} chunks removed)")
    
    def get_all_chunks(self) -> List[Chunk]:
        """
        Obtiene todos los chunks almacenados.
        Útil para debugging y testing.
        
        Returns:
            Lista de todos los chunks
        """
        return list(self._chunks.values())
    
    def get_chunks_by_document(self, document_id: str) -> List[Chunk]:
        """
        Obtiene todos los chunks de un documento.
        
        Args:
            document_id: ID del documento
            
        Returns:
            Lista de chunks del documento
        """
        return [
            chunk for chunk in self._chunks.values()
            if chunk.document_id == document_id
        ]
