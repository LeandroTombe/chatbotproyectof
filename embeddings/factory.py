"""
Factory for creating embedding providers dynamically.
Auto-registers providers and instantiates them based on configuration.
"""
from typing import Dict, Type, Callable
from embeddings.base import BaseEmbedding, EmbeddingConfig

# Registry of embedding providers
_EMBEDDING_PROVIDERS: Dict[str, Type[BaseEmbedding]] = {}


def register_provider(name: str):
    """
    Decorator to register an embedding provider.
    
    Usage:
        @register_provider("my-provider")
        class MyEmbedding(BaseEmbedding):
            ...
    
    Args:
        name: Provider name (e.g., "dummy", "hf-e5", "openai")
    """
    def decorator(cls: Type[BaseEmbedding]) -> Type[BaseEmbedding]:
        _EMBEDDING_PROVIDERS[name] = cls
        return cls
    return decorator


def create_embedder(provider: str, config: EmbeddingConfig) -> BaseEmbedding:
    """
    Create an embedder based on the provider name.
    
    Args:
        provider: Provider name (dummy, hf-e5, openai, etc.)
        config: Embedding configuration
        
    Returns:
        Instance of the embedder
        
    Raises:
        ValueError: If the provider is not registered
        
    Example:
        config = EmbeddingConfig(model_name="dummy", dimension=768)
        embedder = create_embedder("dummy", config)
    """
    if provider not in _EMBEDDING_PROVIDERS:
        available = ', '.join(_EMBEDDING_PROVIDERS.keys())
        raise ValueError(
            f"Unknown embedding provider: '{provider}'. "
            f"Available providers: {available}"
        )
    
    embedder_class = _EMBEDDING_PROVIDERS[provider]
    return embedder_class(config)


def list_providers() -> list[str]:
    """
    List all registered embedding providers.
    
    Returns:
        List of provider names
    """
    return list(_EMBEDDING_PROVIDERS.keys())


def get_provider_class(provider: str) -> Type[BaseEmbedding]:
    """
    Get the class for a specific provider.
    
    Args:
        provider: Provider name
        
    Returns:
        Provider class
        
    Raises:
        ValueError: If provider not found
    """
    if provider not in _EMBEDDING_PROVIDERS:
        available = ', '.join(_EMBEDDING_PROVIDERS.keys())
        raise ValueError(
            f"Unknown embedding provider: '{provider}'. "
            f"Available providers: {available}"
        )
    
    return _EMBEDDING_PROVIDERS[provider]
