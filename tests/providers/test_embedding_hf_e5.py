from embeddings.base import EmbeddingConfig  
from embeddings.providers.hf_e5_embedding import HFMultilingualE5Embedding

def main():
    """Test básico para el proveedor de embeddings HF E5."""
    try:
        config = EmbeddingConfig(
            model_name="intfloat/multilingual-e5-small",
            dimension=None,   # importante - se inferirá automáticamente
            normalize=True
        )

        embedding_provider = HFMultilingualE5Embedding(config)

        text = "Este es un texto de prueba para embeddings."
        vec = embedding_provider.embed_text(text)

        print("✓ Embedding generado exitosamente")
        print(f"Dimensión: {len(vec)}")
        print(f"Primeros 5 valores: {vec[:5]}")
        
        # Test adicional para queries
        query = "¿Qué es un embedding?"
        query_vec = embedding_provider.embed_query(query)
        print(f"\n✓ Embedding de query generado")
        print(f"Dimensión query: {len(query_vec)}")
        
        # Verificar que las dimensiones coinciden
        assert len(vec) == len(query_vec), "Las dimensiones no coinciden"
        print("✓ Test completado exitosamente")
        
    except Exception as e:
        print(f"✗ Error en el test: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
