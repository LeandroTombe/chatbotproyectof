"""
Unit tests for DocumentRetriever.
"""
from unittest.mock import Mock, MagicMock
import pytest

from retrieval.retriever import DocumentRetriever, RetrieverException
from domain.models import Chunk, SearchResult
from embeddings.base import EmbeddingException
from vectorstore.base import VectorStoreException


class TestDocumentRetrieverInitialization:
    """Tests para inicialización de DocumentRetriever"""
    
    def test_initialization_success(self):
        """Prueba inicialización exitosa"""
        embedder = Mock()
        vector_store = Mock()
        
        retriever = DocumentRetriever(
            embedder=embedder,
            vector_store=vector_store
        )
        
        assert retriever.embedder == embedder
        assert retriever.vector_store == vector_store


class TestDocumentRetrieverSearch:
    """Tests para método search"""
    
    def test_search_basic(self):
        """Prueba búsqueda básica exitosa"""
        # Setup mocks
        embedder = Mock()
        vector_store = Mock()
        
        # Mock embedding
        query_embedding = [0.1, 0.2, 0.3]
        embedder.embed_text.return_value = query_embedding
        
        # Mock search results
        chunk1 = Chunk(
            id="chunk1",
            document_id="doc1",
            content="Relevant content",
            chunk_index=0,
            metadata={},
            embedding=[0.1, 0.2, 0.3]
        )
        chunk2 = Chunk(
            id="chunk2",
            document_id="doc1",
            content="More relevant content",
            chunk_index=1,
            metadata={},
            embedding=[0.15, 0.25, 0.35]
        )
        
        search_results = [
            SearchResult(chunk=chunk1, score=0.95, document_name="doc1"),
            SearchResult(chunk=chunk2, score=0.87, document_name="doc1")
        ]
        vector_store.search.return_value = search_results
        
        # Create retriever and search
        retriever = DocumentRetriever(
            embedder=embedder,
            vector_store=vector_store
        )
        
        results = retriever.search("test query")
        
        # Verify embedding was generated
        embedder.embed_text.assert_called_once_with("test query")
        
        # Verify search was called with correct parameters
        vector_store.search.assert_called_once_with(
            query_embedding=query_embedding,
            top_k=None,
            min_score=None,
            where=None
        )
        
        # Verify results
        assert len(results) == 2
        assert results[0].score == 0.95
        assert results[1].score == 0.87
    
    def test_search_with_parameters(self):
        """Prueba búsqueda con parámetros opcionales"""
        embedder = Mock()
        vector_store = Mock()
        
        embedder.embed_text.return_value = [0.1, 0.2, 0.3]
        vector_store.search.return_value = []
        
        retriever = DocumentRetriever(
            embedder=embedder,
            vector_store=vector_store
        )
        
        # Search with all parameters
        retriever.search(
            query="test query",
            top_k=5,
            min_score=0.7,
            where={"document_id": "doc1"}
        )
        
        # Verify parameters passed correctly
        vector_store.search.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            top_k=5,
            min_score=0.7,
            where={"document_id": "doc1"}
        )
    
    def test_search_empty_query(self):
        """Prueba búsqueda con query vacío"""
        embedder = Mock()
        vector_store = Mock()
        
        retriever = DocumentRetriever(
            embedder=embedder,
            vector_store=vector_store
        )
        
        # Empty query should return empty results
        results = retriever.search("")
        assert results == []
        
        results = retriever.search("   ")
        assert results == []
        
        # Embedder should not be called
        embedder.embed_text.assert_not_called()
    
    def test_search_no_results(self):
        """Prueba búsqueda sin resultados"""
        embedder = Mock()
        vector_store = Mock()
        
        embedder.embed_text.return_value = [0.1, 0.2, 0.3]
        vector_store.search.return_value = []
        
        retriever = DocumentRetriever(
            embedder=embedder,
            vector_store=vector_store
        )
        
        results = retriever.search("no matching query")
        
        assert results == []
        assert len(results) == 0


class TestDocumentRetrieverSearchByDocument:
    """Tests para método search_by_document"""
    
    def test_search_by_document(self):
        """Prueba búsqueda en documento específico"""
        embedder = Mock()
        vector_store = Mock()
        
        embedder.embed_text.return_value = [0.1, 0.2, 0.3]
        
        chunk = Chunk(
            id="chunk1",
            document_id="doc123",
            content="Content from doc123",
            chunk_index=0,
            metadata={},
            embedding=[0.1, 0.2, 0.3]
        )
        vector_store.search.return_value = [
            SearchResult(chunk=chunk, score=0.9, document_name="doc123")
        ]
        
        retriever = DocumentRetriever(
            embedder=embedder,
            vector_store=vector_store
        )
        
        results = retriever.search_by_document(
            query="test query",
            document_id="doc123"
        )
        
        # Verify search called with document filter
        vector_store.search.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            top_k=None,
            min_score=None,
            where={"document_id": "doc123"}
        )
        
        assert len(results) == 1
        assert results[0].chunk.document_id == "doc123"
    
    def test_search_by_document_with_parameters(self):
        """Prueba búsqueda por documento con parámetros adicionales"""
        embedder = Mock()
        vector_store = Mock()
        
        embedder.embed_text.return_value = [0.1, 0.2, 0.3]
        vector_store.search.return_value = []
        
        retriever = DocumentRetriever(
            embedder=embedder,
            vector_store=vector_store
        )
        
        retriever.search_by_document(
            query="test",
            document_id="doc456",
            top_k=3,
            min_score=0.8
        )
        
        vector_store.search.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            top_k=3,
            min_score=0.8,
            where={"document_id": "doc456"}
        )


class TestDocumentRetrieverGetContext:
    """Tests para método get_context"""
    
    def test_get_context_basic(self):
        """Prueba obtención de contexto básico"""
        embedder = Mock()
        vector_store = Mock()
        
        embedder.embed_text.return_value = [0.1, 0.2, 0.3]
        
        chunk1 = Chunk(
            id="chunk1",
            document_id="doc1",
            content="First chunk content",
            chunk_index=0,
            metadata={},
            embedding=[0.1, 0.2, 0.3]
        )
        chunk2 = Chunk(
            id="chunk2",
            document_id="doc1",
            content="Second chunk content",
            chunk_index=1,
            metadata={},
            embedding=[0.15, 0.25, 0.35]
        )
        
        vector_store.search.return_value = [
            SearchResult(chunk=chunk1, score=0.9, document_name="doc1"),
            SearchResult(chunk=chunk2, score=0.8, document_name="doc1")
        ]
        
        retriever = DocumentRetriever(
            embedder=embedder,
            vector_store=vector_store
        )
        
        context = retriever.get_context("test query")
        
        # Default separator is "\n\n---\n\n"
        expected = "First chunk content\n\n---\n\nSecond chunk content"
        assert context == expected
    
    def test_get_context_custom_separator(self):
        """Prueba contexto con separador personalizado"""
        embedder = Mock()
        vector_store = Mock()
        
        embedder.embed_text.return_value = [0.1, 0.2, 0.3]
        
        chunk1 = Chunk(
            id="chunk1",
            document_id="doc1",
            content="Content A",
            chunk_index=0,
            metadata={},
            embedding=[0.1, 0.2, 0.3]
        )
        chunk2 = Chunk(
            id="chunk2",
            document_id="doc1",
            content="Content B",
            chunk_index=1,
            metadata={},
            embedding=[0.15, 0.25, 0.35]
        )
        
        vector_store.search.return_value = [
            SearchResult(chunk=chunk1, score=0.9, document_name="doc1"),
            SearchResult(chunk=chunk2, score=0.8, document_name="doc1")
        ]
        
        retriever = DocumentRetriever(
            embedder=embedder,
            vector_store=vector_store
        )
        
        context = retriever.get_context("test query", separator=" | ")
        
        assert context == "Content A | Content B"
    
    def test_get_context_no_results(self):
        """Prueba contexto cuando no hay resultados"""
        embedder = Mock()
        vector_store = Mock()
        
        embedder.embed_text.return_value = [0.1, 0.2, 0.3]
        vector_store.search.return_value = []
        
        retriever = DocumentRetriever(
            embedder=embedder,
            vector_store=vector_store
        )
        
        context = retriever.get_context("no results query")
        
        assert context == ""
    
    def test_get_context_with_top_k(self):
        """Prueba contexto con límite top_k"""
        embedder = Mock()
        vector_store = Mock()
        
        embedder.embed_text.return_value = [0.1, 0.2, 0.3]
        
        chunk1 = Chunk(
            id="chunk1",
            document_id="doc1",
            content="Content 1",
            chunk_index=0,
            metadata={},
            embedding=[0.1, 0.2, 0.3]
        )
        
        vector_store.search.return_value = [
            SearchResult(chunk=chunk1, score=0.9, document_name="doc123")
        ]
        
        retriever = DocumentRetriever(
            embedder=embedder,
            vector_store=vector_store
        )
        
        context = retriever.get_context("test", top_k=1)
        
        # Verify top_k passed to search
        vector_store.search.assert_called_once()
        call_kwargs = vector_store.search.call_args[1]
        assert call_kwargs['top_k'] == 1
        
        assert context == "Content 1"


class TestDocumentRetrieverErrors:
    """Tests para manejo de errores"""
    
    def test_search_embedding_fails(self):
        """Prueba error al generar embedding"""
        embedder = Mock()
        vector_store = Mock()
        
        embedder.embed_text.side_effect = EmbeddingException("Embedding failed")
        
        retriever = DocumentRetriever(
            embedder=embedder,
            vector_store=vector_store
        )
        
        with pytest.raises(RetrieverException, match="Failed to embed query"):
            retriever.search("test query")
    
    def test_search_vector_store_fails(self):
        """Prueba error al buscar en vector store"""
        embedder = Mock()
        vector_store = Mock()
        
        embedder.embed_text.return_value = [0.1, 0.2, 0.3]
        vector_store.search.side_effect = VectorStoreException("Search failed")
        
        retriever = DocumentRetriever(
            embedder=embedder,
            vector_store=vector_store
        )
        
        with pytest.raises(RetrieverException, match="Error searching for query"):
            retriever.search("test query")
    
    def test_search_generic_error(self):
        """Prueba error genérico durante búsqueda"""
        embedder = Mock()
        vector_store = Mock()
        
        embedder.embed_text.return_value = [0.1, 0.2, 0.3]
        vector_store.search.side_effect = RuntimeError("Unexpected error")
        
        retriever = DocumentRetriever(
            embedder=embedder,
            vector_store=vector_store
        )
        
        with pytest.raises(RetrieverException, match="Error searching for query"):
            retriever.search("test query")
    
    def test_get_context_raises_retriever_exception(self):
        """Prueba que get_context propaga RetrieverException"""
        embedder = Mock()
        vector_store = Mock()
        
        embedder.embed_text.side_effect = EmbeddingException("Failed")
        
        retriever = DocumentRetriever(
            embedder=embedder,
            vector_store=vector_store
        )
        
        with pytest.raises(RetrieverException):
            retriever.get_context("test query")


class TestDocumentRetrieverHelperMethods:
    """Tests para métodos helper privados"""
    
    def test_embed_query_success(self):
        """Prueba generación exitosa de embedding de query"""
        embedder = Mock()
        vector_store = Mock()
        
        embedder.embed_text.return_value = [0.1, 0.2, 0.3]
        
        retriever = DocumentRetriever(
            embedder=embedder,
            vector_store=vector_store
        )
        
        embedding = retriever._embed_query("test query")
        
        assert embedding == [0.1, 0.2, 0.3]
        embedder.embed_text.assert_called_once_with("test query")
    
    def test_embed_query_fails(self):
        """Prueba error al generar embedding de query"""
        embedder = Mock()
        vector_store = Mock()
        
        embedder.embed_text.side_effect = Exception("Embedding error")
        
        retriever = DocumentRetriever(
            embedder=embedder,
            vector_store=vector_store
        )
        
        with pytest.raises(RetrieverException, match="Failed to embed query"):
            retriever._embed_query("test query")
