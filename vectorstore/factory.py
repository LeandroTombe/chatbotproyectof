"""
Factory Pattern for Vector Store providers.
Allows registration and creation of different vector store implementations.
"""
from typing import Dict, Type, Optional, List
import logging

from vectorstore.base import BaseVectorStore
from domain.models import RetrievalConfig

logger = logging.getLogger(__name__)

# Global registry of vector store providers
_VECTOR_STORE_REGISTRY: Dict[str, Type[BaseVectorStore]] = {}


def register_vector_store(name: str):
    """
    Decorator to register a vector store provider.
    
    Usage:
        @register_vector_store("chroma")
        class ChromaVectorStore(BaseVectorStore):
            ...
    
    Args:
        name: Unique identifier for the vector store provider
        
    Returns:
        Decorator function
    """
    def decorator(cls: Type[BaseVectorStore]) -> Type[BaseVectorStore]:
        if name in _VECTOR_STORE_REGISTRY:
            logger.warning(
                f"Vector store provider '{name}' is already registered. "
                f"Overwriting with {cls.__name__}"
            )
        
        _VECTOR_STORE_REGISTRY[name] = cls
        logger.debug(f"Registered vector store provider: {name} -> {cls.__name__}")
        return cls
    
    return decorator


def create_vector_store(
    provider: str,
    dimension: int,
    config: Optional[RetrievalConfig] = None,
    **kwargs
) -> BaseVectorStore:
    """
    Factory function to create a vector store by provider name.
    
    Usage:
        # Create ChromaDB vector store
        vector_store = create_vector_store(
            provider="chroma",
            dimension=384,
            collection_name="my_collection"
        )
        
        # Create in-memory vector store
        vector_store = create_vector_store(
            provider="memory",
            dimension=768
        )
    
    Args:
        provider: Name of the vector store provider (e.g., "chroma", "memory")
        dimension: Dimension of embeddings to store
        config: Optional retrieval configuration
        **kwargs: Additional provider-specific arguments
        
    Returns:
        Instance of the requested vector store
        
    Raises:
        ValueError: If provider is not registered
    """
    if provider not in _VECTOR_STORE_REGISTRY:
        available = list_vector_stores()
        raise ValueError(
            f"Vector store provider '{provider}' not found. "
            f"Available providers: {available}"
        )
    
    provider_class = _VECTOR_STORE_REGISTRY[provider]
    
    try:
        # Create instance with dimension and config as base parameters
        instance = provider_class(dimension=dimension, config=config, **kwargs)
        logger.info(f"Created vector store: {provider} ({provider_class.__name__})")
        return instance
    except Exception as e:
        logger.error(f"Error creating vector store '{provider}': {str(e)}")
        raise


def list_vector_stores() -> List[str]:
    """
    Get list of all registered vector store providers.
    
    Returns:
        List of provider names
    """
    return sorted(_VECTOR_STORE_REGISTRY.keys())


def is_provider_available(provider: str) -> bool:
    """
    Check if a vector store provider is registered.
    
    Args:
        provider: Provider name to check
        
    Returns:
        True if provider is available
    """
    return provider in _VECTOR_STORE_REGISTRY
