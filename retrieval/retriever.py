"""
Retrieval module for searching relevant document chunks.
Orchestrates query embedding and vector store search.
"""
from typing import List, Optional, Dict, Any
import logging

from domain.models import SearchResult
from embeddings.base import BaseEmbedding
from vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class RetrieverException(Exception):
    """Exception raised for retrieval errors"""
    pass


class DocumentRetriever:
    """
    Orchestrates document retrieval using semantic search.
    
    Workflow:
    1. Convert text query to embedding vector
    2. Search vector store for similar chunks
    3. Return ranked results
    
    This service provides the 'R' in RAG (Retrieval-Augmented Generation).
    """
    
    def __init__(
        self,
        embedder: BaseEmbedding,
        vector_store: BaseVectorStore
    ):
        """
        Initialize the document retriever.
        
        Args:
            embedder: Provider for generating query embeddings
            vector_store: Storage containing document chunk embeddings
        """
        self.embedder = embedder
        self.vector_store = vector_store
        
        logger.info(
            f"DocumentRetriever initialized with "
            f"embedder={embedder.__class__.__name__}, "
            f"vector_store={vector_store.__class__.__name__}"
        )
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for document chunks relevant to the query.
        
        Args:
            query: Text query to search for
            top_k: Maximum number of results to return (uses vector store default if None)
            min_score: Minimum similarity score threshold (uses vector store default if None)
            where: Optional filters (e.g., {"document_id": "doc1", "chunk_index": 0})
            
        Returns:
            List of SearchResult ordered by relevance (highest score first)
            
        Raises:
            RetrieverException: If search fails
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        logger.info(
            f"Searching for query: '{query[:50]}...' "
            f"(top_k={top_k}, min_score={min_score}, filters={where})"
        )
        
        try:
            # Stage 1: Generate query embedding
            query_embedding = self._embed_query(query)
            
            # Stage 2: Search vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                min_score=min_score,
                where=where
            )
            
            logger.info(f"Found {len(results)} results for query")
            
            return results
            
        except Exception as e:
            error_msg = f"Error searching for query '{query[:50]}...': {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RetrieverException(error_msg) from e
    
    def search_by_document(
        self,
        query: str,
        document_id: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search for relevant chunks within a specific document.
        
        Args:
            query: Text query to search for
            document_id: ID of document to search within
            top_k: Maximum number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of SearchResult from the specified document
            
        Raises:
            RetrieverException: If search fails
        """
        logger.info(f"Searching in document '{document_id}' for query: '{query[:50]}...'")
        
        return self.search(
            query=query,
            top_k=top_k,
            min_score=min_score,
            where={"document_id": document_id}
        )
    
    def get_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        separator: str = "\n\n---\n\n"
    ) -> str:
        """
        Get formatted context text from search results.
        
        Useful for RAG: retrieve relevant context to pass to LLM.
        
        Args:
            query: Text query to search for
            top_k: Maximum number of chunks to include
            separator: String to join chunk contents
            
        Returns:
            Formatted context string with relevant chunk contents
            
        Raises:
            RetrieverException: If retrieval fails
        """
        results = self.search(query=query, top_k=top_k)
        
        if not results:
            logger.warning(f"No context found for query: '{query[:50]}...'")
            return ""
        
        context_parts = []
        for result in results:
            context_parts.append(result.chunk.content)
        
        context = separator.join(context_parts)
        
        logger.info(
            f"Generated context with {len(results)} chunks "
            f"({len(context)} characters)"
        )
        
        return context
    
    def _embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for query text.
        
        Args:
            query: Text query to embed
            
        Returns:
            Embedding vector
            
        Raises:
            RetrieverException: If embedding generation fails
        """
        try:
            embedding = self.embedder.embed_text(query)
            logger.debug(f"Generated query embedding with dimension {len(embedding)}")
            return embedding
        except Exception as e:
            raise RetrieverException(f"Failed to embed query: {str(e)}") from e
