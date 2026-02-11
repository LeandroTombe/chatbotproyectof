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

__all__ = [
    "BaseEmbedding",
    "DummyEmbedding",
    "EmbeddingConfig",
    "EmbeddingResult",
    "EmbeddingException"
]
