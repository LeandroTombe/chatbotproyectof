"""
Base module for embeddings generation.
Defines the abstract interface for embedding providers.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class EmbeddingException(Exception):
    """Exception raised for embedding generation errors"""
    pass


# =========================
# CONFIG
# =========================

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model_name: str = "default-embedding-model"
    dimension: Optional[int] = None  # ← NO obligatoria
    batch_size: int = 32
    max_retries: int = 3
    timeout: int = 30
    normalize: bool = True
    
    def validate(self) -> None:
        if self.dimension is not None and self.dimension <= 0:
            raise ValueError("dimension debe ser mayor a 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size debe ser mayor a 0")
        if self.max_retries < 0:
            raise ValueError("max_retries no puede ser negativo")
        if self.timeout <= 0:
            raise ValueError("timeout debe ser mayor a 0")


# =========================
# RESULT
# =========================

@dataclass
class EmbeddingResult:
    """Resultado de una generación de embedding"""
    embedding: List[float]
    text: str
    model: str
    dimension: Optional[int]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        if not self.embedding:
            raise ValueError("embedding no puede estar vacío")
        if self.dimension is not None and len(self.embedding) != self.dimension:
            raise ValueError(
                f"dimension mismatch: esperado {self.dimension}, "
                f"obtenido {len(self.embedding)}"
            )


# =========================
# BASE PROVIDER
# =========================

class BaseEmbedding(ABC):
    """
    Clase base abstracta para proveedores de embeddings.
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.config.validate()
        self._validate_provider()
        logger.info(
            f"{self.__class__.__name__} initialized with "
            f"model={self.config.model_name}, "
            f"dimension={self.config.dimension}"
        )
    
    @abstractmethod
    def _validate_provider(self) -> None:
        pass
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            logger.warning("Empty text list provided")
            return []
        
        embeddings: List[List[float]] = []
        batch_size = self.config.batch_size
        
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = [self.embed_text(text) for text in batch]
                embeddings.extend(batch_embeddings)
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
        
        except Exception as e:
            raise EmbeddingException(
                f"Error generating batch embeddings: {e}"
            ) from e
    
    def embed_with_metadata(self, text: str) -> EmbeddingResult:
        try:
            embedding = self.embed_text(text)
            
            return EmbeddingResult(
                embedding=embedding,
                text=text,
                model=self.config.model_name,
                dimension=self.config.dimension,
                metadata={
                    "text_length": len(text),
                    "normalized": self.config.normalize,
                },
            )
        except Exception as e:
            raise EmbeddingException(
                f"Error generating embedding with metadata: {e}"
            ) from e
    
    def get_dimension(self) -> int:
        if self.config.dimension is None:
            raise EmbeddingException("Embedding dimension not initialized")
        return self.config.dimension
    
    def get_model_name(self) -> str:
        return self.config.model_name
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        if not embedding:
            return False
        if self.config.dimension is not None and len(embedding) != self.config.dimension:
            logger.warning(
                f"Invalid embedding dimension: expected {self.config.dimension}, "
                f"got {len(embedding)}"
            )
            return False
        if not all(isinstance(x, (int, float)) for x in embedding):
            logger.warning("Embedding contains non-numeric values")
            return False
        return True
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.config.model_name}, "
            f"dimension={self.config.dimension})"
        )


# =========================
# DUMMY PROVIDER
# =========================

class DummyEmbedding(BaseEmbedding):
    """
    Implementación dummy para testing.
    """
    
    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        use_zeros: bool = False
    ):
        self.use_zeros = use_zeros
        super().__init__(config)
    
    def _validate_provider(self) -> None:
        logger.debug("DummyEmbedding provider validated")
    
    def embed_text(self, text: str) -> List[float]:
        import random
        
        if not text or not text.strip():
            raise EmbeddingException("Cannot embed empty text")
        
        # Definir dimensión si todavía no está
        dimension = self.config.dimension or 768
        self.config.dimension = dimension
        
        if self.use_zeros:
            embedding = [0.0] * dimension
        else:
            random.seed(hash(text))
            embedding = [random.random() for _ in range(dimension)]
        
        if self.config.normalize:
            magnitude = sum(x ** 2 for x in embedding) ** 0.5
            if magnitude > 0:
                embedding = [x / magnitude for x in embedding]
        
        return embedding
