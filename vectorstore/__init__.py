"""
Vector store module for embedding storage and similarity search.
"""
# Import factory first
from vectorstore.factory import (
    create_vector_store,
    list_vector_stores,
    register_vector_store,
    is_provider_available
)

# Import base classes and utilities
from vectorstore.base import (
    BaseVectorStore,
    InMemoryVectorStore,
    VectorStoreException,
    cosine_similarity,
    euclidean_distance
)

# Register InMemoryVectorStore (now factory is available)
register_vector_store("memory")(InMemoryVectorStore)

# Import implementations to trigger registration
try:
    from vectorstore.implementations.chroma import ChromaVectorStore, CHROMADB_AVAILABLE
    CHROMA_AVAILABLE = CHROMADB_AVAILABLE
    if CHROMA_AVAILABLE:
        # Register ChromaDB only if chromadb package is available
        register_vector_store("chroma")(ChromaVectorStore)
except ImportError:
    ChromaVectorStore = None  # type: ignore
    CHROMA_AVAILABLE = False

__all__ = [
    # Factory
    "create_vector_store",
    "list_vector_stores",
    "is_provider_available",
    # Base classes
    "BaseVectorStore",
    "InMemoryVectorStore",
    "ChromaVectorStore",
    "VectorStoreException",
    # Utilities
    "cosine_similarity",
    "euclidean_distance",
    "CHROMA_AVAILABLE"
]
