"""
Unit tests for the embeddings base module.
"""
import pytest
from typing import List

from embeddings.base import (
    BaseEmbedding,
    DummyEmbedding,
    EmbeddingConfig,
    EmbeddingResult,
    EmbeddingException
)


class TestEmbeddingConfig:
    """Tests para la configuración de embeddings"""
    
    def test_default_config(self):
        """Prueba configuración por defecto"""
        config = EmbeddingConfig()
        assert config.model_name == "default-embedding-model"
        assert config.dimension is None
        assert config.batch_size == 32
        assert config.max_retries == 3
        assert config.timeout == 30
        assert config.normalize is True
    
    def test_custom_config(self):
        """Prueba configuración personalizada"""
        config = EmbeddingConfig(
            model_name="custom-model",
            dimension=384,
            batch_size=16,
            max_retries=5,
            timeout=60,
            normalize=False
        )
        assert config.model_name == "custom-model"
        assert config.dimension == 384
        assert config.batch_size == 16
        assert config.max_retries == 5
        assert config.timeout == 60
        assert config.normalize is False
    
    def test_validate_invalid_dimension(self):
        """Prueba validación de dimensión inválida"""
        config = EmbeddingConfig(dimension=0)
        with pytest.raises(ValueError, match="dimension debe ser mayor a 0"):
            config.validate()
        
        config = EmbeddingConfig(dimension=-10)
        with pytest.raises(ValueError, match="dimension debe ser mayor a 0"):
            config.validate()
    
    def test_validate_invalid_batch_size(self):
        """Prueba validación de batch_size inválido"""
        config = EmbeddingConfig(batch_size=0)
        with pytest.raises(ValueError, match="batch_size debe ser mayor a 0"):
            config.validate()
    
    def test_validate_invalid_max_retries(self):
        """Prueba validación de max_retries negativo"""
        config = EmbeddingConfig(max_retries=-1)
        with pytest.raises(ValueError, match="max_retries no puede ser negativo"):
            config.validate()
    
    def test_validate_invalid_timeout(self):
        """Prueba validación de timeout inválido"""
        config = EmbeddingConfig(timeout=0)
        with pytest.raises(ValueError, match="timeout debe ser mayor a 0"):
            config.validate()


class TestEmbeddingResult:
    """Tests para EmbeddingResult"""
    
    def test_create_embedding_result(self):
        """Prueba creación de resultado válido"""
        embedding = [0.1, 0.2, 0.3]
        result = EmbeddingResult(
            embedding=embedding,
            text="test text",
            model="test-model",
            dimension=3
        )
        
        assert result.embedding == embedding
        assert result.text == "test text"
        assert result.model == "test-model"
        assert result.dimension == 3
        assert result.metadata == {}
    
    def test_embedding_result_with_metadata(self):
        """Prueba resultado con metadata"""
        result = EmbeddingResult(
            embedding=[0.1, 0.2],
            text="test",
            model="model",
            dimension=2,
            metadata={"source": "test"}
        )
        
        assert result.metadata["source"] == "test"
    
    def test_embedding_result_empty_embedding(self):
        """Prueba que falle con embedding vacío"""
        with pytest.raises(ValueError, match="embedding no puede estar vacío"):
            EmbeddingResult(
                embedding=[],
                text="test",
                model="model",
                dimension=0
            )
    
    def test_embedding_result_dimension_mismatch(self):
        """Prueba que falle con dimensión incorrecta"""
        with pytest.raises(ValueError, match="dimension mismatch"):
            EmbeddingResult(
                embedding=[0.1, 0.2, 0.3],
                text="test",
                model="model",
                dimension=5  # No coincide con len(embedding)
            )


class TestDummyEmbedding:
    """Tests para DummyEmbedding"""
    
    def test_initialization_default(self):
        """Prueba inicialización con valores por defecto"""
        embedder: DummyEmbedding = DummyEmbedding()
        assert embedder.config.dimension is None
        assert embedder.use_zeros is False  # type: ignore[attr-defined]
    
    def test_initialization_custom(self):
        """Prueba inicialización con config personalizada"""
        config = EmbeddingConfig(dimension=128, model_name="dummy")
        embedder = DummyEmbedding(config)
        assert embedder.config.dimension == 128
        assert embedder.config.model_name == "dummy"
    
    def test_initialization_with_zeros(self):
        """Prueba inicialización con use_zeros=True"""
        embedder: DummyEmbedding = DummyEmbedding(use_zeros=True)  # type: ignore[call-arg]
        assert embedder.use_zeros is True  # type: ignore[attr-defined]
    
    def test_embed_text_random(self):
        """Prueba generación de embedding aleatorio"""
        config = EmbeddingConfig(dimension=10)
        embedder: DummyEmbedding = DummyEmbedding(config, use_zeros=False)  # type: ignore[call-arg]
        
        embedding = embedder.embed_text("test text")
        
        assert len(embedding) == 10
        assert all(isinstance(x, float) for x in embedding)
    
    def test_embed_text_zeros(self):
        """Prueba generación de embedding con zeros"""
        config = EmbeddingConfig(dimension=10, normalize=False)
        embedder: DummyEmbedding = DummyEmbedding(config, use_zeros=True)  # type: ignore[call-arg]
        
        embedding = embedder.embed_text("test text")
        
        assert len(embedding) == 10
        assert all(x == 0.0 for x in embedding)
    
    def test_embed_text_empty(self):
        """Prueba que falle con texto vacío"""
        embedder = DummyEmbedding()
        
        with pytest.raises(EmbeddingException, match="Cannot embed empty text"):
            embedder.embed_text("")
        
        with pytest.raises(EmbeddingException, match="Cannot embed empty text"):
            embedder.embed_text("   ")
    
    def test_embed_text_normalized(self):
        """Prueba que los embeddings estén normalizados"""
        config = EmbeddingConfig(dimension=10, normalize=True)
        embedder: DummyEmbedding = DummyEmbedding(config, use_zeros=False)  # type: ignore[call-arg]
        
        embedding = embedder.embed_text("test text")
        
        # Calcular magnitud L2
        magnitude = sum(x ** 2 for x in embedding) ** 0.5
        assert abs(magnitude - 1.0) < 1e-6  # Debe estar normalizado
    
    def test_embed_text_deterministic(self):
        """Prueba que el mismo texto produzca el mismo embedding"""
        embedder = DummyEmbedding()
        
        embedding1 = embedder.embed_text("same text")
        embedding2 = embedder.embed_text("same text")
        
        assert embedding1 == embedding2
    
    def test_embed_text_different_texts(self):
        """Prueba que textos diferentes produzcan embeddings diferentes"""
        embedder = DummyEmbedding()
        
        embedding1 = embedder.embed_text("text one")
        embedding2 = embedder.embed_text("text two")
        
        assert embedding1 != embedding2
    
    def test_embed_texts_multiple(self):
        """Prueba generación de múltiples embeddings"""
        config = EmbeddingConfig(dimension=10, batch_size=2)
        embedder = DummyEmbedding(config)
        
        texts = ["text 1", "text 2", "text 3"]
        embeddings = embedder.embed_texts(texts)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 10 for emb in embeddings)
    
    def test_embed_texts_empty_list(self):
        """Prueba con lista vacía"""
        embedder = DummyEmbedding()
        embeddings = embedder.embed_texts([])
        assert embeddings == []
    
    def test_embed_texts_batching(self):
        """Prueba que el batching funcione correctamente"""
        config = EmbeddingConfig(dimension=5, batch_size=2)
        embedder = DummyEmbedding(config)
        
        texts = ["a", "b", "c", "d", "e"]
        embeddings = embedder.embed_texts(texts)
        
        assert len(embeddings) == 5
        # Cada embedding debe ser consistente
        for text, embedding in zip(texts, embeddings):
            assert embedding == embedder.embed_text(text)
    
    def test_embed_with_metadata(self):
        """Prueba generación con metadata"""
        config = EmbeddingConfig(dimension=10)
        embedder = DummyEmbedding(config)
        
        result = embedder.embed_with_metadata("test text")
        
        assert isinstance(result, EmbeddingResult)
        assert result.text == "test text"
        assert result.model == config.model_name
        assert result.dimension == 10
        assert len(result.embedding) == 10
        assert "text_length" in result.metadata
        assert result.metadata["text_length"] == len("test text")
    
    def test_get_dimension(self):
        """Prueba obtención de dimensión"""
        config = EmbeddingConfig(dimension=256)
        embedder = DummyEmbedding(config)
        assert embedder.get_dimension() == 256
    
    def test_get_model_name(self):
        """Prueba obtención de nombre del modelo"""
        config = EmbeddingConfig(model_name="test-model")
        embedder = DummyEmbedding(config)
        assert embedder.get_model_name() == "test-model"
    
    def test_validate_embedding_valid(self):
        """Prueba validación de embedding válido"""
        config = EmbeddingConfig(dimension=5)
        embedder = DummyEmbedding(config)
        
        valid_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert embedder.validate_embedding(valid_embedding) is True
    
    def test_validate_embedding_wrong_dimension(self):
        """Prueba validación con dimensión incorrecta"""
        config = EmbeddingConfig(dimension=5)
        embedder = DummyEmbedding(config)
        
        wrong_embedding = [0.1, 0.2, 0.3]  # Solo 3 elementos
        assert embedder.validate_embedding(wrong_embedding) is False
    
    def test_validate_embedding_empty(self):
        """Prueba validación con embedding vacío"""
        embedder = DummyEmbedding()
        assert embedder.validate_embedding([]) is False
    
    def test_validate_embedding_non_numeric(self):
        """Prueba validación con valores no numéricos"""
        config = EmbeddingConfig(dimension=3)
        embedder = DummyEmbedding(config)
        
        invalid_embedding = [0.1, "invalid", 0.3]
        assert embedder.validate_embedding(invalid_embedding) is False
    
    def test_repr(self):
        """Prueba representación en string"""
        config = EmbeddingConfig(model_name="test-model", dimension=128)
        embedder = DummyEmbedding(config)
        
        repr_str = repr(embedder)
        assert "DummyEmbedding" in repr_str
        assert "test-model" in repr_str
        assert "128" in repr_str


class TestBaseEmbeddingAbstract:
    """Tests para verificar que BaseEmbedding es abstracta"""
    
    def test_cannot_instantiate_base_embedding(self):
        """Prueba que no se puede instanciar BaseEmbedding directamente"""
        with pytest.raises(TypeError):
            BaseEmbedding()  # type: ignore[abstract]


class TestEdgeCases:
    """Tests para casos extremos"""
    
    def test_very_long_text(self):
        """Prueba con texto muy largo"""
        embedder = DummyEmbedding()
        long_text = "A" * 10000
        
        embedding = embedder.embed_text(long_text)
        assert len(embedding) == embedder.config.dimension
    
    def test_special_characters(self):
        """Prueba con caracteres especiales"""
        embedder = DummyEmbedding()
        special_text = "!@#$%^&*(){}[]|\\/<>?~`"
        
        embedding = embedder.embed_text(special_text)
        assert len(embedding) == embedder.config.dimension
    
    def test_unicode_characters(self):
        """Prueba con caracteres unicode"""
        embedder = DummyEmbedding()
        unicode_text = "Texto con emojis 😀🎉 y acentos: áéíóú ñ"
        
        embedding = embedder.embed_text(unicode_text)
        assert len(embedding) == embedder.config.dimension
    
    def test_newlines_and_tabs(self):
        """Prueba con saltos de línea y tabs"""
        embedder = DummyEmbedding()
        text_with_whitespace = "Line 1\nLine 2\tTabbed"
        
        embedding = embedder.embed_text(text_with_whitespace)
        assert len(embedding) == embedder.config.dimension
    
    def test_large_batch(self):
        """Prueba con batch grande"""
        config = EmbeddingConfig(dimension=10, batch_size=10)
        embedder = DummyEmbedding(config)
        
        texts = [f"text {i}" for i in range(100)]
        embeddings = embedder.embed_texts(texts)
        
        assert len(embeddings) == 100
        assert all(len(emb) == 10 for emb in embeddings)
    
    def test_small_dimension(self):
        """Prueba con dimensión muy pequeña"""
        config = EmbeddingConfig(dimension=1)
        embedder = DummyEmbedding(config)
        
        embedding = embedder.embed_text("test")
        assert len(embedding) == 1
    
    def test_no_normalization(self):
        """Prueba sin normalización"""
        config = EmbeddingConfig(dimension=10, normalize=False)
        embedder: DummyEmbedding = DummyEmbedding(config, use_zeros=False)  # type: ignore[call-arg]
        
        embedding = embedder.embed_text("test")
        
        magnitude = sum(x ** 2 for x in embedding) ** 0.5
        # Sin normalización, la magnitud puede no ser 1
        assert len(embedding) == 10
    
    def test_single_text_batch(self):
        """Prueba batch de un solo texto"""
        embedder = DummyEmbedding()
        embeddings = embedder.embed_texts(["single text"])
        
        assert len(embeddings) == 1
        assert len(embeddings[0]) == embedder.config.dimension


class TestEmbeddingConsistency:
    """Tests para verificar consistencia"""
    
    def test_embed_text_vs_embed_texts(self):
        """Prueba que embed_text y embed_texts den el mismo resultado"""
        embedder = DummyEmbedding()
        text = "consistency test"
        
        single = embedder.embed_text(text)
        batch = embedder.embed_texts([text])
        
        assert single == batch[0]
    
    def test_multiple_embedders_same_config(self):
        """Prueba que embedders con la misma config den mismos resultados"""
        config = EmbeddingConfig(dimension=10)
        embedder1 = DummyEmbedding(config)
        embedder2 = DummyEmbedding(config)
        
        text = "same config test"
        emb1 = embedder1.embed_text(text)
        emb2 = embedder2.embed_text(text)
        
        assert emb1 == emb2
