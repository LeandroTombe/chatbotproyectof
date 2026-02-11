"""
Unit tests for the retrieval service module.
"""
import pytest
from typing import List

from domain.models import Chunk, SearchResult, RetrievalConfig, Query
from embeddings.base import DummyEmbedding, EmbeddingConfig
from vectorstore.base import InMemoryVectorStore
from retrieval.service import (
    RetrievalService,
    RetrievalException,
    create_retrieval_service
)


class TestRetrievalServiceInitialization:
    """Tests para inicialización de RetrievalService"""
    
    def test_initialization_valid(self):
        """Prueba inicialización válida"""
        vector_store = InMemoryVectorStore(dimension=10)
        embedding_provider = DummyEmbedding(
            config=EmbeddingConfig(dimension=10)
        )
        
        service = RetrievalService(
            vector_store=vector_store,
            embedding_provider=embedding_provider
        )
        
        assert service.vector_store is vector_store
        assert service.embedding_provider is embedding_provider
        assert service.config.top_k == 5
        assert service.config.min_score == 0.7
    
    def test_initialization_with_config(self):
        """Prueba inicialización con config personalizada"""
        vector_store = InMemoryVectorStore(dimension=10)
        embedding_provider = DummyEmbedding(
            config=EmbeddingConfig(dimension=10)
        )
        config = RetrievalConfig(top_k=10, min_score=0.8)
        
        service = RetrievalService(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            config=config
        )
        
        assert service.config.top_k == 10
        assert service.config.min_score == 0.8
    
    def test_initialization_none_vector_store(self):
        """Prueba que falle con vector_store None"""
        embedding_provider = DummyEmbedding()
        
        with pytest.raises(ValueError, match="vector_store cannot be None"):
            RetrievalService(
                vector_store=None,  # type: ignore[arg-type]
                embedding_provider=embedding_provider
            )
    
    def test_initialization_none_embedding_provider(self):
        """Prueba que falle con embedding_provider None"""
        vector_store = InMemoryVectorStore(dimension=10)
        
        with pytest.raises(ValueError, match="embedding_provider cannot be None"):
            RetrievalService(
                vector_store=vector_store,
                embedding_provider=None  # type: ignore[arg-type]
            )
    
    def test_initialization_dimension_mismatch(self):
        """Prueba que falle con dimensiones diferentes"""
        vector_store = InMemoryVectorStore(dimension=10)
        embedding_provider = DummyEmbedding(
            config=EmbeddingConfig(dimension=20)
        )
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            RetrievalService(
                vector_store=vector_store,
                embedding_provider=embedding_provider
            )


class TestRetrievalServiceRetrieve:
    """Tests para el método retrieve"""
    
    def setup_method(self):
        """Setup para cada test"""
        self.vector_store = InMemoryVectorStore(dimension=10)
        self.embedding_provider = DummyEmbedding(
            config=EmbeddingConfig(dimension=10, normalize=True)
        )
        self.service = RetrievalService(
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider
        )
        
        # Agregar algunos chunks de prueba
        chunks = [
            Chunk(
                id=f"chunk{i}",
                document_id="doc1",
                content=f"This is test content {i}",
                chunk_index=i,
                embedding=self.embedding_provider.embed_text(f"content {i}"),
                metadata={"file_name": "test.pdf"}
            )
            for i in range(5)
        ]
        self.vector_store.add_chunks(chunks)
    
    def test_retrieve_basic(self):
        """Prueba retrieve básico"""
        query_text = "test query"
        
        result = self.service.retrieve(query_text)
        
        assert isinstance(result, Query)
        assert result.text == query_text
        assert result.embedding is not None
        assert len(result.embedding) == 10
        assert isinstance(result.results, list)
    
    def test_retrieve_returns_results(self):
        """Prueba que retrieve retorne resultados"""
        result = self.service.retrieve("content 0")
        
        assert len(result.results) > 0
        assert all(isinstance(r, SearchResult) for r in result.results)
    
    def test_retrieve_with_custom_top_k(self):
        """Prueba retrieve con top_k personalizado"""
        result = self.service.retrieve("query", top_k=2)
        
        assert len(result.results) <= 2
    
    def test_retrieve_with_custom_min_score(self):
        """Prueba retrieve con min_score personalizado"""
        result = self.service.retrieve("query", min_score=0.95)
        
        # Todos los resultados deben tener score >= 0.95
        assert all(r.score >= 0.95 for r in result.results)
    
    def test_retrieve_with_document_filter(self):
        """Prueba retrieve filtrando por documento"""
        # Agregar chunks de otro documento
        chunks_doc2 = [
            Chunk(
                id="chunk_doc2",
                document_id="doc2",
                content="doc2 content",
                chunk_index=0,
                embedding=self.embedding_provider.embed_text("doc2"),
                metadata={"file_name": "doc2.pdf"}
            )
        ]
        self.vector_store.add_chunks(chunks_doc2)
        
        result = self.service.retrieve("query", document_id="doc1")
        
        # Todos los resultados deben ser de doc1
        assert all(r.chunk.document_id == "doc1" for r in result.results)
    
    def test_retrieve_empty_query(self):
        """Prueba que falle con query vacío"""
        with pytest.raises(ValueError, match="query_text cannot be empty"):
            self.service.retrieve("")
    
    def test_retrieve_whitespace_query(self):
        """Prueba que falle con query solo espacios"""
        with pytest.raises(ValueError, match="query_text cannot be empty"):
            self.service.retrieve("   \n  \t  ")
    
    def test_retrieve_query_id_unique(self):
        """Prueba que cada query tenga ID único"""
        result1 = self.service.retrieve("query 1")
        result2 = self.service.retrieve("query 2")
        
        assert result1.id != result2.id


class TestRetrievalServiceRetrieveSimple:
    """Tests para retrieve_simple"""
    
    def setup_method(self):
        """Setup para cada test"""
        self.vector_store = InMemoryVectorStore(dimension=10)
        self.embedding_provider = DummyEmbedding(
            config=EmbeddingConfig(dimension=10)
        )
        self.service = RetrievalService(
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider
        )
        
        # Agregar chunks
        chunks = [
            Chunk(
                id=f"chunk{i}",
                document_id="doc1",
                content=f"content {i}",
                chunk_index=i,
                embedding=self.embedding_provider.embed_text(f"content {i}")
            )
            for i in range(3)
        ]
        self.vector_store.add_chunks(chunks)
    
    def test_retrieve_simple_returns_list(self):
        """Prueba que retrieve_simple retorne lista"""
        results = self.service.retrieve_simple("query")
        
        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)
    
    def test_retrieve_simple_no_query_object(self):
        """Prueba que no retorne objeto Query"""
        results = self.service.retrieve_simple("query")
        
        assert not isinstance(results, Query)


class TestRetrievalServiceRetrieveContext:
    """Tests para retrieve_context"""
    
    def setup_method(self):
        """Setup para cada test"""
        self.vector_store = InMemoryVectorStore(dimension=10)
        self.embedding_provider = DummyEmbedding(
            config=EmbeddingConfig(dimension=10)
        )
        self.service = RetrievalService(
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider
        )
        
        # Agregar chunks con contenido conocido
        chunks = [
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="First chunk content.",
                chunk_index=0,
                embedding=self.embedding_provider.embed_text("first")
            ),
            Chunk(
                id="chunk2",
                document_id="doc1",
                content="Second chunk content.",
                chunk_index=1,
                embedding=self.embedding_provider.embed_text("second")
            ),
            Chunk(
                id="chunk3",
                document_id="doc1",
                content="Third chunk content.",
                chunk_index=2,
                embedding=self.embedding_provider.embed_text("third")
            )
        ]
        self.vector_store.add_chunks(chunks)
    
    def test_retrieve_context_basic(self):
        """Prueba retrieve_context básico"""
        context = self.service.retrieve_context("query")
        
        assert isinstance(context, str)
        assert len(context) > 0
    
    def test_retrieve_context_contains_separator(self):
        """Prueba que el contexto use el separador"""
        context = self.service.retrieve_context("query", separator="\n---\n")
        
        # Si hay más de un resultado, debe contener el separador
        if self.service.retrieve_simple("query"):
            assert "\n---\n" in context or len(context.split(".")) == 1
    
    def test_retrieve_context_custom_separator(self):
        """Prueba con separador personalizado"""
        separator = " [SEP] "
        context = self.service.retrieve_context("query", separator=separator)
        
        assert isinstance(context, str)
    
    def test_retrieve_context_max_length(self):
        """Prueba límite de longitud"""
        max_length = 50
        context = self.service.retrieve_context(
            "query",
            max_context_length=max_length
        )
        
        assert len(context) <= max_length
    
    def test_retrieve_context_no_results(self):
        """Prueba con query que no retorna resultados"""
        # Crear servicio con min_score muy alto
        config = RetrievalConfig(top_k=5, min_score=0.99999)
        service = RetrievalService(
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
            config=config
        )
        
        context = service.retrieve_context("query")
        assert context == ""


class TestRetrievalServiceRetrieveWithMetadata:
    """Tests para retrieve_with_metadata"""
    
    def setup_method(self):
        """Setup para cada test"""
        self.vector_store = InMemoryVectorStore(dimension=10)
        self.embedding_provider = DummyEmbedding(
            config=EmbeddingConfig(dimension=10)
        )
        self.service = RetrievalService(
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider
        )
        
        # Agregar chunks con metadata
        chunks = [
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="content",
                chunk_index=0,
                embedding=self.embedding_provider.embed_text("content"),
                metadata={"author": "test", "page": 1}
            )
        ]
        self.vector_store.add_chunks(chunks)
    
    def test_retrieve_with_metadata_structure(self):
        """Prueba estructura de resultados con metadata"""
        results = self.service.retrieve_with_metadata("query")
        
        assert isinstance(results, list)
        if results:
            result = results[0]
            assert "content" in result
            assert "score" in result
            assert "document_name" in result
            assert "document_id" in result
            assert "chunk_id" in result
            assert "chunk_index" in result
            assert "metadata" in result
    
    def test_retrieve_with_metadata_includes_custom_metadata(self):
        """Prueba que incluya metadata personalizada"""
        results = self.service.retrieve_with_metadata("query")
        
        if results:
            assert results[0]["metadata"].get("author") == "test"
            assert results[0]["metadata"].get("page") == 1


class TestRetrievalServiceStats:
    """Tests para get_stats"""
    
    def test_get_stats_structure(self):
        """Prueba estructura de stats"""
        vector_store = InMemoryVectorStore(dimension=128)
        embedding_provider = DummyEmbedding(
            config=EmbeddingConfig(dimension=128, model_name="test-model")
        )
        service = RetrievalService(
            vector_store=vector_store,
            embedding_provider=embedding_provider
        )
        
        stats = service.get_stats()
        
        assert "dimension" in stats
        assert "total_chunks" in stats
        assert "embedding_model" in stats
        assert "config" in stats
        assert stats["dimension"] == 128
        assert stats["embedding_model"] == "test-model"
    
    def test_get_stats_chunk_count(self):
        """Prueba que stats refleje número de chunks"""
        vector_store = InMemoryVectorStore(dimension=10)
        embedding_provider = DummyEmbedding(
            config=EmbeddingConfig(dimension=10)
        )
        service = RetrievalService(
            vector_store=vector_store,
            embedding_provider=embedding_provider
        )
        
        # Agregar chunks
        chunks = [
            Chunk(
                id=f"chunk{i}",
                document_id="doc1",
                content=f"content {i}",
                chunk_index=i,
                embedding=embedding_provider.embed_text(f"content {i}")
            )
            for i in range(10)
        ]
        vector_store.add_chunks(chunks)
        
        stats = service.get_stats()
        assert stats["total_chunks"] == 10


class TestRetrievalServiceHealthCheck:
    """Tests para check_health"""
    
    def test_check_health_healthy(self):
        """Prueba health check con servicio saludable"""
        vector_store = InMemoryVectorStore(dimension=10)
        embedding_provider = DummyEmbedding(
            config=EmbeddingConfig(dimension=10)
        )
        service = RetrievalService(
            vector_store=vector_store,
            embedding_provider=embedding_provider
        )
        
        assert service.check_health() is True
    
    def test_check_health_with_chunks(self):
        """Prueba health check con chunks almacenados"""
        vector_store = InMemoryVectorStore(dimension=10)
        embedding_provider = DummyEmbedding(
            config=EmbeddingConfig(dimension=10)
        )
        service = RetrievalService(
            vector_store=vector_store,
            embedding_provider=embedding_provider
        )
        
        # Agregar chunks
        chunks = [
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="content",
                chunk_index=0,
                embedding=embedding_provider.embed_text("content")
            )
        ]
        vector_store.add_chunks(chunks)
        
        assert service.check_health() is True


class TestCreateRetrievalService:
    """Tests para la función create_retrieval_service"""
    
    def test_create_retrieval_service_default(self):
        """Prueba creación con valores por defecto"""
        vector_store = InMemoryVectorStore(dimension=10)
        embedding_provider = DummyEmbedding(
            config=EmbeddingConfig(dimension=10)
        )
        
        service = create_retrieval_service(
            vector_store=vector_store,
            embedding_provider=embedding_provider
        )
        
        assert isinstance(service, RetrievalService)
        assert service.config.top_k == 5
        assert service.config.min_score == 0.7
    
    def test_create_retrieval_service_custom(self):
        """Prueba creación con valores personalizados"""
        vector_store = InMemoryVectorStore(dimension=10)
        embedding_provider = DummyEmbedding(
            config=EmbeddingConfig(dimension=10)
        )
        
        service = create_retrieval_service(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            top_k=10,
            min_score=0.8
        )
        
        assert service.config.top_k == 10
        assert service.config.min_score == 0.8


class TestRetrievalServiceIntegration:
    """Tests de integración completos"""
    
    def test_full_retrieval_flow(self):
        """Prueba flujo completo de retrieval"""
        # Setup
        vector_store = InMemoryVectorStore(dimension=10)
        embedding_provider = DummyEmbedding(
            config=EmbeddingConfig(dimension=10)
        )
        service = RetrievalService(
            vector_store=vector_store,
            embedding_provider=embedding_provider
        )
        
        # Agregar chunks
        chunks = [
            Chunk(
                id=f"chunk{i}",
                document_id="doc1",
                content=f"Python is a programming language. Chunk {i}",
                chunk_index=i,
                embedding=embedding_provider.embed_text(f"Python programming {i}"),
                metadata={"file_name": "python.pdf", "page": i}
            )
            for i in range(5)
        ]
        vector_store.add_chunks(chunks)
        
        # Realizar búsqueda
        query = service.retrieve("What is Python?")
        
        # Verificaciones
        assert isinstance(query, Query)
        assert query.text == "What is Python?"
        assert len(query.results) > 0
        
        # Verificar que los resultados tengan scores válidos
        assert all(0 <= r.score <= 1 for r in query.results)
        
        # Verificar que estén ordenados
        scores = [r.score for r in query.results]
        assert scores == sorted(scores, reverse=True)
    
    def test_retrieve_and_build_context(self):
        """Prueba recuperar y construir contexto"""
        vector_store = InMemoryVectorStore(dimension=10)
        embedding_provider = DummyEmbedding(
            config=EmbeddingConfig(dimension=10)
        )
        service = RetrievalService(
            vector_store=vector_store,
            embedding_provider=embedding_provider
        )
        
        # Agregar chunks
        chunks = [
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="Python is a high-level programming language.",
                chunk_index=0,
                embedding=embedding_provider.embed_text("Python language")
            ),
            Chunk(
                id="chunk2",
                document_id="doc1",
                content="It is widely used for web development.",
                chunk_index=1,
                embedding=embedding_provider.embed_text("web development")
            )
        ]
        vector_store.add_chunks(chunks)
        
        # Construir contexto
        context = service.retrieve_context("Python programming")
        
        assert isinstance(context, str)
        assert len(context) > 0
    
    def test_empty_vector_store(self):
        """Prueba con vector store vacío"""
        vector_store = InMemoryVectorStore(dimension=10)
        embedding_provider = DummyEmbedding(
            config=EmbeddingConfig(dimension=10)
        )
        service = RetrievalService(
            vector_store=vector_store,
            embedding_provider=embedding_provider
        )
        
        result = service.retrieve("query")
        
        assert isinstance(result, Query)
        assert len(result.results) == 0
