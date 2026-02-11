"""
Tests for Vector Store Factory Pattern.
"""
import pytest

from vectorstore import (
    create_vector_store,
    list_vector_stores,
    is_provider_available,
    InMemoryVectorStore,
    CHROMA_AVAILABLE
)
from domain.models import RetrievalConfig


def test_list_vector_stores():
    """Verifica que lista al menos el provider 'memory'"""
    providers = list_vector_stores()
    
    assert isinstance(providers, list)
    assert len(providers) > 0
    assert "memory" in providers
    
    # Si ChromaDB está disponible, debe estar registrado
    if CHROMA_AVAILABLE:
        assert "chroma" in providers


def test_is_provider_available():
    """Verifica la función de disponibilidad de providers"""
    assert is_provider_available("memory") is True
    assert is_provider_available("nonexistent") is False
    
    if CHROMA_AVAILABLE:
        assert is_provider_available("chroma") is True


def test_create_memory_vector_store():
    """Verifica que se puede crear un vector store en memoria"""
    vector_store = create_vector_store(
        provider="memory",
        dimension=384
    )
    
    assert vector_store is not None
    assert isinstance(vector_store, InMemoryVectorStore)
    assert vector_store.dimension == 384
    assert vector_store.count() == 0


def test_create_memory_vector_store_with_config():
    """Verifica creación con configuración personalizada"""
    config = RetrievalConfig(top_k=5, min_score=0.5)
    
    vector_store = create_vector_store(
        provider="memory",
        dimension=768,
        config=config
    )
    
    assert vector_store.dimension == 768
    assert vector_store.config.top_k == 5
    assert vector_store.config.min_score == 0.5


@pytest.mark.skipif(not CHROMA_AVAILABLE, reason="ChromaDB not installed")
def test_create_chroma_vector_store():
    """Verifica que se puede crear un vector store ChromaDB"""
    from vectorstore import ChromaVectorStore
    
    vector_store = create_vector_store(
        provider="chroma",
        dimension=384,
        collection_name="test_collection"
    )
    
    assert vector_store is not None
    assert isinstance(vector_store, ChromaVectorStore)
    assert vector_store.dimension == 384
    
    # Limpiar
    vector_store.clear()


def test_create_invalid_provider():
    """Verifica que falla con provider inválido"""
    with pytest.raises(ValueError, match="Vector store provider 'invalid' not found"):
        create_vector_store(
            provider="invalid",
            dimension=384
        )


def test_factory_error_message_shows_available():
    """Verifica que el error muestre los providers disponibles"""
    try:
        create_vector_store(provider="doesnotexist", dimension=384)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        error_msg = str(e)
        assert "doesnotexist" in error_msg
        assert "Available providers:" in error_msg
        assert "memory" in error_msg


def test_multiple_instances_independent():
    """Verifica que múltiples instancias son independientes"""
    store1 = create_vector_store(provider="memory", dimension=384)
    store2 = create_vector_store(provider="memory", dimension=768)
    
    assert store1.dimension == 384
    assert store2.dimension == 768
    assert store1 is not store2


def test_provider_registration_is_case_sensitive():
    """Verifica que los nombres de providers son case-sensitive"""
    assert is_provider_available("memory") is True
    assert is_provider_available("Memory") is False
    assert is_provider_available("MEMORY") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
