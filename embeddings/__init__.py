"""
Embeddings module for text vectorization.
"""
from embeddings.base import (
    BaseEmbedding,
    DummyEmbedding,
    EmbeddingConfig,
    EmbeddingResult,
    EmbeddingException
)
from embeddings.factory import (
    create_embedder,
    list_providers,
    register_provider
)

# Import providers to auto-register them
import embeddings.providers  # noqa: F401

__all__ = [
    "BaseEmbedding",
    "DummyEmbedding",
    "EmbeddingConfig",
    "EmbeddingResult",
    "EmbeddingException",
    "create_embedder",
    "list_providers",
    "register_provider",
]
