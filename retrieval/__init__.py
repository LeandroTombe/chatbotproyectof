"""
Retrieval module.
Provides functionality for searching and retrieving document chunks.
"""
from retrieval.retriever import (
    DocumentRetriever,
    RetrieverException
)

__all__ = [
    "DocumentRetriever",
    "RetrieverException"
]
