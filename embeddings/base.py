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


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model_name: str = "default-embedding-model"
    dimension: int = 768  # Dimensión del vector de embedding
    batch_size: int = 32  # Tamaño de lote para procesamiento batch
    max_retries: int = 3  # Reintentos en caso de error
    timeout: int = 30  # Timeout en segundos
    normalize: bool = True  # Normalizar vectores
    
    def validate(self):
        """Valida la configuración"""
        if self.dimension <= 0:
            raise ValueError("dimension debe ser mayor a 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size debe ser mayor a 0")
        if self.max_retries < 0:
            raise ValueError("max_retries no puede ser negativo")
        if self.timeout <= 0:
            raise ValueError("timeout debe ser mayor a 0")


@dataclass
class EmbeddingResult:
    """Resultado de una generación de embedding"""
    embedding: List[float]
    text: str
    model: str
    dimension: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Valida el resultado después de la inicialización"""
        if not self.embedding:
            raise ValueError("embedding no puede estar vacío")
        if len(self.embedding) != self.dimension:
            raise ValueError(
                f"dimension mismatch: esperado {self.dimension}, "
                f"obtenido {len(self.embedding)}"
            )


class BaseEmbedding(ABC):
    """
    Clase base abstracta para proveedores de embeddings.
    Define la interfaz que deben implementar todos los proveedores.
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Inicializa el proveedor de embeddings.
        
        Args:
            config: Configuración del embedding. Si es None, usa valores por defecto.
        """
        self.config = config or EmbeddingConfig()
        self.config.validate()
        self._validate_provider()
        logger.info(
            f"{self.__class__.__name__} initialized with model={self.config.model_name}, "
            f"dimension={self.config.dimension}"
        )
    
    @abstractmethod
    def _validate_provider(self):
        """
        Valida que el proveedor esté correctamente configurado.
        Debe verificar dependencias, API keys, etc.
        
        Raises:
            EmbeddingException: Si la validación falla
        """
        pass
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Genera el embedding para un texto.
        
        Args:
            text: Texto a convertir en embedding
            
        Returns:
            Vector de embedding como lista de floats
            
        Raises:
            EmbeddingException: Si hay error en la generación
        """
        pass
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para múltiples textos.
        Implementación por defecto que procesa en batch.
        Las subclases pueden sobrescribir para optimización.
        
        Args:
            texts: Lista de textos a convertir en embeddings
            
        Returns:
            Lista de vectores de embedding
            
        Raises:
            EmbeddingException: Si hay error en la generación
        """
        if not texts:
            logger.warning("Empty text list provided")
            return []
        
        embeddings = []
        batch_size = self.config.batch_size
        
        try:
            # Procesar en batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = [self.embed_text(text) for text in batch]
                embeddings.extend(batch_embeddings)
                
                logger.debug(
                    f"Processed batch {i // batch_size + 1}, "
                    f"texts {i + 1}-{min(i + batch_size, len(texts))}"
                )
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            error_msg = f"Error generating batch embeddings: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingException(error_msg) from e
    
    def embed_with_metadata(self, text: str) -> EmbeddingResult:
        """
        Genera embedding con metadata completa.
        
        Args:
            text: Texto a convertir en embedding
            
        Returns:
            EmbeddingResult con embedding y metadata
            
        Raises:
            EmbeddingException: Si hay error en la generación
        """
        try:
            embedding = self.embed_text(text)
            
            return EmbeddingResult(
                embedding=embedding,
                text=text,
                model=self.config.model_name,
                dimension=self.config.dimension,
                metadata={
                    "text_length": len(text),
                    "normalized": self.config.normalize
                }
            )
        except Exception as e:
            error_msg = f"Error generating embedding with metadata: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingException(error_msg) from e
    
    def get_dimension(self) -> int:
        """
        Retorna la dimensión de los embeddings generados.
        
        Returns:
            Dimensión del vector de embedding
        """
        return self.config.dimension
    
    def get_model_name(self) -> str:
        """
        Retorna el nombre del modelo usado.
        
        Returns:
            Nombre del modelo
        """
        return self.config.model_name
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        Valida que un embedding tenga el formato correcto.
        
        Args:
            embedding: Vector de embedding a validar
            
        Returns:
            True si es válido, False en caso contrario
        """
        if not embedding:
            return False
        if len(embedding) != self.config.dimension:
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
        """Representación en string del proveedor"""
        return (
            f"{self.__class__.__name__}("
            f"model={self.config.model_name}, "
            f"dimension={self.config.dimension})"
        )


class DummyEmbedding(BaseEmbedding):
    """
    Implementación dummy para testing.
    Genera embeddings aleatorios o zeros.
    """
    
    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
        use_zeros: bool = False
    ):
        """
        Inicializa el proveedor dummy.
        
        Args:
            config: Configuración del embedding
            use_zeros: Si True, genera vectores de zeros; si False, aleatorios
        """
        self.use_zeros = use_zeros
        super().__init__(config)
    
    def _validate_provider(self):
        """Validación dummy - siempre pasa"""
        logger.debug("DummyEmbedding provider validated")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Genera un embedding dummy.
        
        Args:
            text: Texto (ignorado en dummy)
            
        Returns:
            Vector de embedding dummy
        """
        import random
        
        if not text or not text.strip():
            raise EmbeddingException("Cannot embed empty text")
        
        if self.use_zeros:
            embedding = [0.0] * self.config.dimension
        else:
            # Embedding pseudo-aleatorio basado en el hash del texto
            random.seed(hash(text))
            embedding = [random.random() for _ in range(self.config.dimension)]
        
        if self.config.normalize:
            # Normalización L2
            magnitude = sum(x ** 2 for x in embedding) ** 0.5
            if magnitude > 0:
                embedding = [x / magnitude for x in embedding]
        
        return embedding
