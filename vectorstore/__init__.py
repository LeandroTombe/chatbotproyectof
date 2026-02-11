"""
Vector store module for embedding storage and similarity search.
"""
from vectorstore.base import (
    BaseVectorStore,
    InMemoryVectorStore,
    VectorStoreException,
    cosine_similarity,
    euclidean_distance
)

__all__ = [
    "BaseVectorStore",
    "InMemoryVectorStore",
    "VectorStoreException",
    "cosine_similarity",
    "euclidean_distance"
]
