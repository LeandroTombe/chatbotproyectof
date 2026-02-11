"""
Unit tests for the vector store module.
"""
import pytest
import math
from typing import List

from domain.models import Chunk, SearchResult, RetrievalConfig
from vectorstore.base import (
    BaseVectorStore,
    InMemoryVectorStore,
    VectorStoreException,
    cosine_similarity,
    euclidean_distance
)


class TestCosineSimilarity:
    """Tests para la función cosine_similarity"""
    
    def test_identical_vectors(self):
        """Prueba con vectores idénticos"""
        vec = [1.0, 2.0, 3.0]
        similarity = cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 1e-6
    
    def test_orthogonal_vectors(self):
        """Prueba con vectores ortogonales"""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - 0.0) < 1e-6
    
    def test_opposite_vectors(self):
        """Prueba con vectores opuestos"""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = cosine_similarity(vec1, vec2)
        assert abs(similarity - (-1.0)) < 1e-6
    
    def test_similar_vectors(self):
        """Prueba con vectores similares"""
        vec1 = [1.0, 1.0, 1.0]
        vec2 = [1.0, 1.0, 0.9]
        similarity = cosine_similarity(vec1, vec2)
        assert 0.9 < similarity < 1.0
    
    def test_dimension_mismatch(self):
        """Prueba con vectores de diferentes dimensiones"""
        vec1 = [1.0, 2.0]
        vec2 = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            cosine_similarity(vec1, vec2)
    
    def test_empty_vectors(self):
        """Prueba con vectores vacíos"""
        with pytest.raises(ValueError, match="Vectors cannot be empty"):
            cosine_similarity([], [])
    
    def test_zero_magnitude(self):
        """Prueba con vector de magnitud cero"""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]
        similarity = cosine_similarity(vec1, vec2)
        assert similarity == 0.0


class TestEuclideanDistance:
    """Tests para la función euclidean_distance"""
    
    def test_identical_vectors(self):
        """Prueba con vectores idénticos"""
        vec = [1.0, 2.0, 3.0]
        distance = euclidean_distance(vec, vec)
        assert abs(distance - 0.0) < 1e-6
    
    def test_unit_distance(self):
        """Prueba con distancia unitaria"""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        distance = euclidean_distance(vec1, vec2)
        assert abs(distance - 1.0) < 1e-6
    
    def test_known_distance(self):
        """Prueba con distancia conocida"""
        vec1 = [0.0, 0.0]
        vec2 = [3.0, 4.0]
        distance = euclidean_distance(vec1, vec2)
        assert abs(distance - 5.0) < 1e-6  # 3-4-5 triangle
    
    def test_dimension_mismatch(self):
        """Prueba con vectores de diferentes dimensiones"""
        vec1 = [1.0, 2.0]
        vec2 = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            euclidean_distance(vec1, vec2)


class TestBaseVectorStoreAbstract:
    """Tests para verificar que BaseVectorStore es abstracta"""
    
    def test_cannot_instantiate_base_vectorstore(self):
        """Prueba que no se puede instanciar BaseVectorStore directamente"""
        with pytest.raises(TypeError):
            BaseVectorStore(dimension=128)  # type: ignore[abstract]


class TestInMemoryVectorStoreInitialization:
    """Tests para inicialización de InMemoryVectorStore"""
    
    def test_initialization_valid(self):
        """Prueba inicialización válida"""
        store = InMemoryVectorStore(dimension=128)
        assert store.dimension == 128
        assert store.count() == 0
    
    def test_initialization_with_config(self):
        """Prueba inicialización con config personalizada"""
        config = RetrievalConfig(top_k=10, min_score=0.8)
        store = InMemoryVectorStore(dimension=256, config=config)
        assert store.dimension == 256
        assert store.config.top_k == 10
        assert store.config.min_score == 0.8
    
    def test_initialization_invalid_dimension(self):
        """Prueba que falle con dimensión inválida"""
        with pytest.raises(ValueError, match="dimension debe ser mayor a 0"):
            InMemoryVectorStore(dimension=0)
        
        with pytest.raises(ValueError, match="dimension debe ser mayor a 0"):
            InMemoryVectorStore(dimension=-10)


class TestInMemoryVectorStoreAddChunk:
    """Tests para agregar chunks"""
    
    def test_add_chunk_valid(self):
        """Prueba agregar chunk válido"""
        store = InMemoryVectorStore(dimension=3)
        chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            content="test",
            chunk_index=0,
            embedding=[0.1, 0.2, 0.3]
        )
        
        store.add_chunk(chunk)
        assert store.count() == 1
    
    def test_add_chunk_without_embedding(self):
        """Prueba que falle al agregar chunk sin embedding"""
        store = InMemoryVectorStore(dimension=3)
        chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            content="test",
            chunk_index=0,
            embedding=None
        )
        
        with pytest.raises(ValueError, match="does not have embedding"):
            store.add_chunk(chunk)
    
    def test_add_chunk_wrong_dimension(self):
        """Prueba que falle con dimensión incorrecta"""
        store = InMemoryVectorStore(dimension=3)
        chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            content="test",
            chunk_index=0,
            embedding=[0.1, 0.2]  # Solo 2 dimensiones
        )
        
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            store.add_chunk(chunk)
    
    def test_add_chunk_overwrites_existing(self):
        """Prueba que sobrescribir un chunk funcione"""
        store = InMemoryVectorStore(dimension=3)
        chunk1 = Chunk(
            id="chunk1",
            document_id="doc1",
            content="test 1",
            chunk_index=0,
            embedding=[0.1, 0.2, 0.3]
        )
        chunk2 = Chunk(
            id="chunk1",  # Mismo ID
            document_id="doc1",
            content="test 2",
            chunk_index=0,
            embedding=[0.4, 0.5, 0.6]
        )
        
        store.add_chunk(chunk1)
        store.add_chunk(chunk2)
        
        assert store.count() == 1
        retrieved = store.get_chunk("chunk1")
        assert retrieved is not None
        assert retrieved.content == "test 2"


class TestInMemoryVectorStoreAddChunks:
    """Tests para agregar múltiples chunks"""
    
    def test_add_chunks_multiple(self):
        """Prueba agregar múltiples chunks"""
        store = InMemoryVectorStore(dimension=2)
        chunks = [
            Chunk(
                id=f"chunk{i}",
                document_id="doc1",
                content=f"test {i}",
                chunk_index=i,
                embedding=[float(i), float(i+1)]
            )
            for i in range(5)
        ]
        
        store.add_chunks(chunks)
        assert store.count() == 5
    
    def test_add_chunks_empty_list(self):
        """Prueba con lista vacía"""
        store = InMemoryVectorStore(dimension=2)
        store.add_chunks([])
        assert store.count() == 0
    
    def test_add_chunks_with_invalid(self):
        """Prueba que falle si alguno es inválido"""
        store = InMemoryVectorStore(dimension=2)
        chunks = [
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="test 1",
                chunk_index=0,
                embedding=[0.1, 0.2]
            ),
            Chunk(
                id="chunk2",
                document_id="doc1",
                content="test 2",
                chunk_index=1,
                embedding=None  # Inválido
            )
        ]
        
        with pytest.raises(VectorStoreException):
            store.add_chunks(chunks)


class TestInMemoryVectorStoreSearch:
    """Tests para búsqueda de chunks"""
    
    def test_search_basic(self):
        """Prueba búsqueda básica"""
        store = InMemoryVectorStore(dimension=3)
        
        # Agregar chunks
        chunks = [
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="similar",
                chunk_index=0,
                embedding=[1.0, 0.0, 0.0]
            ),
            Chunk(
                id="chunk2",
                document_id="doc1",
                content="different",
                chunk_index=1,
                embedding=[0.0, 1.0, 0.0]
            )
        ]
        store.add_chunks(chunks)
        
        # Buscar con vector similar al primero
        results = store.search([0.9, 0.1, 0.0], top_k=1)
        
        assert len(results) == 1
        assert results[0].chunk.id == "chunk1"
    
    def test_search_top_k(self):
        """Prueba límite top_k"""
        store = InMemoryVectorStore(dimension=2)
        
        chunks = [
            Chunk(
                id=f"chunk{i}",
                document_id="doc1",
                content=f"test {i}",
                chunk_index=i,
                embedding=[float(i), float(i)]
            )
            for i in range(10)
        ]
        store.add_chunks(chunks)
        
        results = store.search([5.0, 5.0], top_k=3)
        assert len(results) == 3
    
    def test_search_min_score(self):
        """Prueba filtro de score mínimo"""
        config = RetrievalConfig(top_k=10, min_score=0.9)
        store = InMemoryVectorStore(dimension=3, config=config)
        
        chunks = [
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="very similar",
                chunk_index=0,
                embedding=[1.0, 0.0, 0.0]
            ),
            Chunk(
                id="chunk2",
                document_id="doc1",
                content="very different",
                chunk_index=1,
                embedding=[0.0, 0.0, 1.0]
            )
        ]
        store.add_chunks(chunks)
        
        # Buscar con vector muy similar al primero
        results = store.search([0.99, 0.01, 0.0])
        
        # Solo debe retornar el muy similar
        assert len(results) >= 1
        assert results[0].chunk.id == "chunk1"
    
    def test_search_by_document(self):
        """Prueba filtro por documento usando where"""
        store = InMemoryVectorStore(dimension=2)
        
        chunks = [
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="doc1 chunk",
                chunk_index=0,
                embedding=[1.0, 0.0]
            ),
            Chunk(
                id="chunk2",
                document_id="doc2",
                content="doc2 chunk",
                chunk_index=0,
                embedding=[1.0, 0.0]
            )
        ]
        store.add_chunks(chunks)
        
        results = store.search([1.0, 0.0], where={"document_id": "doc1"})
        
        assert all(r.chunk.document_id == "doc1" for r in results)
    
    def test_search_empty_store(self):
        """Prueba búsqueda en store vacío"""
        store = InMemoryVectorStore(dimension=3)
        results = store.search([1.0, 0.0, 0.0])
        assert results == []
    
    def test_search_invalid_dimension(self):
        """Prueba búsqueda con dimensión incorrecta"""
        store = InMemoryVectorStore(dimension=3)
        
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            store.search([1.0, 0.0])  # Solo 2 dimensiones
    
    def test_search_results_sorted(self):
        """Prueba que los resultados estén ordenados por score"""
        store = InMemoryVectorStore(dimension=2)
        
        chunks = [
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="far",
                chunk_index=0,
                embedding=[0.0, 1.0]
            ),
            Chunk(
                id="chunk2",
                document_id="doc1",
                content="close",
                chunk_index=1,
                embedding=[1.0, 0.0]
            ),
            Chunk(
                id="chunk3",
                document_id="doc1",
                content="medium",
                chunk_index=2,
                embedding=[0.5, 0.5]
            )
        ]
        store.add_chunks(chunks)
        
        results = store.search([1.0, 0.0], top_k=3)
        
        # Verificar que están ordenados de mayor a menor score
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score


class TestInMemoryVectorStoreDelete:
    """Tests para eliminación de chunks"""
    
    def test_delete_chunk_exists(self):
        """Prueba eliminar chunk existente"""
        store = InMemoryVectorStore(dimension=2)
        chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            content="test",
            chunk_index=0,
            embedding=[1.0, 0.0]
        )
        store.add_chunk(chunk)
        
        result = store.delete_chunk("chunk1")
        
        assert result is True
        assert store.count() == 0
    
    def test_delete_chunk_not_exists(self):
        """Prueba eliminar chunk que no existe"""
        store = InMemoryVectorStore(dimension=2)
        result = store.delete_chunk("nonexistent")
        assert result is False
    
    def test_delete_chunks_by_document(self):
        """Prueba eliminar todos los chunks de un documento"""
        store = InMemoryVectorStore(dimension=2)
        
        chunks = [
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="doc1 chunk 1",
                chunk_index=0,
                embedding=[1.0, 0.0]
            ),
            Chunk(
                id="chunk2",
                document_id="doc1",
                content="doc1 chunk 2",
                chunk_index=1,
                embedding=[0.0, 1.0]
            ),
            Chunk(
                id="chunk3",
                document_id="doc2",
                content="doc2 chunk",
                chunk_index=0,
                embedding=[0.5, 0.5]
            )
        ]
        store.add_chunks(chunks)
        
        deleted = store.delete_chunks_by_document("doc1")
        
        assert deleted == 2
        assert store.count() == 1
        assert store.get_chunk("chunk3") is not None
    
    def test_delete_chunks_by_document_not_exists(self):
        """Prueba eliminar documento que no existe"""
        store = InMemoryVectorStore(dimension=2)
        deleted = store.delete_chunks_by_document("nonexistent")
        assert deleted == 0


class TestInMemoryVectorStoreGet:
    """Tests para obtención de chunks"""
    
    def test_get_chunk_exists(self):
        """Prueba obtener chunk existente"""
        store = InMemoryVectorStore(dimension=2)
        chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            content="test",
            chunk_index=0,
            embedding=[1.0, 0.0]
        )
        store.add_chunk(chunk)
        
        retrieved = store.get_chunk("chunk1")
        
        assert retrieved is not None
        assert retrieved.id == "chunk1"
        assert retrieved.content == "test"
    
    def test_get_chunk_not_exists(self):
        """Prueba obtener chunk que no existe"""
        store = InMemoryVectorStore(dimension=2)
        retrieved = store.get_chunk("nonexistent")
        assert retrieved is None
    
    def test_get_all_chunks(self):
        """Prueba obtener todos los chunks"""
        store = InMemoryVectorStore(dimension=2)
        
        chunks = [
            Chunk(
                id=f"chunk{i}",
                document_id="doc1",
                content=f"test {i}",
                chunk_index=i,
                embedding=[float(i), 0.0]
            )
            for i in range(3)
        ]
        store.add_chunks(chunks)
        
        all_chunks = store.get_all_chunks()
        assert len(all_chunks) == 3
    
    def test_get_chunks_by_document(self):
        """Prueba obtener chunks de un documento"""
        store = InMemoryVectorStore(dimension=2)
        
        chunks = [
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="doc1",
                chunk_index=0,
                embedding=[1.0, 0.0]
            ),
            Chunk(
                id="chunk2",
                document_id="doc2",
                content="doc2",
                chunk_index=0,
                embedding=[0.0, 1.0]
            )
        ]
        store.add_chunks(chunks)
        
        doc1_chunks = store.get_chunks_by_document("doc1")
        
        assert len(doc1_chunks) == 1
        assert doc1_chunks[0].document_id == "doc1"


class TestInMemoryVectorStoreClear:
    """Tests para limpiar el store"""
    
    def test_clear(self):
        """Prueba limpiar el store"""
        store = InMemoryVectorStore(dimension=2)
        
        chunks = [
            Chunk(
                id=f"chunk{i}",
                document_id="doc1",
                content=f"test {i}",
                chunk_index=i,
                embedding=[float(i), 0.0]
            )
            for i in range(5)
        ]
        store.add_chunks(chunks)
        
        assert store.count() == 5
        
        store.clear()
        
        assert store.count() == 0
    
    def test_clear_empty(self):
        """Prueba limpiar store vacío"""
        store = InMemoryVectorStore(dimension=2)
        store.clear()
        assert store.count() == 0


class TestInMemoryVectorStoreCount:
    """Tests para contar chunks"""
    
    def test_count_empty(self):
        """Prueba count en store vacío"""
        store = InMemoryVectorStore(dimension=2)
        assert store.count() == 0
    
    def test_count_after_add(self):
        """Prueba count después de agregar"""
        store = InMemoryVectorStore(dimension=2)
        
        for i in range(10):
            chunk = Chunk(
                id=f"chunk{i}",
                document_id="doc1",
                content=f"test {i}",
                chunk_index=i,
                embedding=[float(i), 0.0]
            )
            store.add_chunk(chunk)
        
        assert store.count() == 10
    
    def test_count_after_delete(self):
        """Prueba count después de eliminar"""
        store = InMemoryVectorStore(dimension=2)
        
        chunks = [
            Chunk(
                id=f"chunk{i}",
                document_id="doc1",
                content=f"test {i}",
                chunk_index=i,
                embedding=[float(i), 0.0]
            )
            for i in range(5)
        ]
        store.add_chunks(chunks)
        
        store.delete_chunk("chunk0")
        assert store.count() == 4


    def test_search_by_chunk_index(self):
        """Prueba filtro por chunk_index usando where"""
        store = InMemoryVectorStore(dimension=2)
        
        chunks = [
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="first chunk",
                chunk_index=0,
                embedding=[1.0, 0.0]
            ),
            Chunk(
                id="chunk2",
                document_id="doc1",
                content="second chunk",
                chunk_index=1,
                embedding=[1.0, 0.0]
            )
        ]
        store.add_chunks(chunks)
        
        results = store.search([1.0, 0.0], where={"chunk_index": 0})
        
        assert len(results) == 1
        assert results[0].chunk.chunk_index == 0
    
    def test_search_by_metadata(self):
        """Prueba filtro por metadatos personalizados usando where"""
        store = InMemoryVectorStore(dimension=2)
        
        chunks = [
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="premium content",
                chunk_index=0,
                embedding=[1.0, 0.0],
                metadata={"tier": "premium"}
            ),
            Chunk(
                id="chunk2",
                document_id="doc1",
                content="free content",
                chunk_index=1,
                embedding=[1.0, 0.0],
                metadata={"tier": "free"}
            )
        ]
        store.add_chunks(chunks)
        
        results = store.search([1.0, 0.0], where={"tier": "premium"})
        
        assert len(results) == 1
        assert results[0].chunk.metadata.get("tier") == "premium"
    
    def test_search_with_multiple_filters(self):
        """Prueba con múltiples filtros en where"""
        store = InMemoryVectorStore(dimension=2)
        
        chunks = [
            Chunk(
                id="chunk1",
                document_id="doc1",
                content="chunk 1",
                chunk_index=0,
                embedding=[1.0, 0.0],
                metadata={"category": "tech"}
            ),
            Chunk(
                id="chunk2",
                document_id="doc1",
                content="chunk 2",
                chunk_index=1,
                embedding=[1.0, 0.0],
                metadata={"category": "science"}
            ),
            Chunk(
                id="chunk3",
                document_id="doc2",
                content="chunk 3",
                chunk_index=0,
                embedding=[1.0, 0.0],
                metadata={"category": "tech"}
            )
        ]
        store.add_chunks(chunks)
        
        # Filtrar por documento Y categoría
        results = store.search([1.0, 0.0], where={"document_id": "doc1", "category": "tech"})
        
        assert len(results) == 1
        assert results[0].chunk.id == "chunk1"


class TestEdgeCases:
    """Tests para casos extremos"""
    
    def test_high_dimensional_vectors(self):
        """Prueba con vectores de alta dimensión"""
        store = InMemoryVectorStore(dimension=1024)
        
        chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            content="test",
            chunk_index=0,
            embedding=[float(i) for i in range(1024)]
        )
        store.add_chunk(chunk)
        
        assert store.count() == 1
    
    def test_many_chunks(self):
        """Prueba con muchos chunks"""
        store = InMemoryVectorStore(dimension=10)
        
        chunks = [
            Chunk(
                id=f"chunk{i}",
                document_id=f"doc{i % 10}",
                content=f"test {i}",
                chunk_index=i,
                embedding=[float(j) for j in range(10)]
            )
            for i in range(1000)
        ]
        
        store.add_chunks(chunks)
        assert store.count() == 1000
    
    def test_search_with_all_zeros(self):
        """Prueba búsqueda con vector de ceros"""
        store = InMemoryVectorStore(dimension=3)
        
        chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            content="test",
            chunk_index=0,
            embedding=[1.0, 0.0, 0.0]
        )
        store.add_chunk(chunk)
        
        results = store.search([0.0, 0.0, 0.0])
        # Debe retornar resultados aunque el score sea 0
        assert isinstance(results, list)
