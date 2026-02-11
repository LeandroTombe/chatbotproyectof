"""
Compara Dummy vs HF-E5 embeddings con ejemplos reales.
"""
from embeddings.base import DummyEmbedding, EmbeddingConfig
from embeddings.providers.hf_e5_embedding import HFMultilingualE5Embedding
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compare_embeddings():
    print("🔬 COMPARACIÓN: Dummy vs HF-E5 Embeddings\n")
    
    # Documentos
    docs = [
        "Python es un lenguaje de programación usado en ciencia de datos y machine learning",
        "Java es un lenguaje orientado a objetos usado en aplicaciones empresariales",
        "HTML es un lenguaje de marcado para crear páginas web",
        "JavaScript se utiliza para desarrollo web frontend y backend"
    ]
    
    # Query
    query = "¿Qué lenguaje usar para inteligencia artificial?"
    
    print(f"📝 Query: '{query}'\n")
    print("📚 Documentos:")
    for i, doc in enumerate(docs, 1):
        print(f"  {i}. {doc}")
    
    print("\n" + "="*80 + "\n")
    
    # ===== DUMMY EMBEDDINGS =====
    print("🎲 DUMMY EMBEDDINGS (Aleatorio)\n")
    dummy = DummyEmbedding(EmbeddingConfig(dimension=384))
    
    query_vec_dummy = dummy.embed_text(query)
    doc_vecs_dummy = [dummy.embed_text(doc) for doc in docs]
    
    similarities_dummy = [
        cosine_similarity(query_vec_dummy, doc_vec) 
        for doc_vec in doc_vecs_dummy
    ]
    
    print("Similitudes:")
    for i, (doc, sim) in enumerate(zip(docs, similarities_dummy), 1):
        print(f"  Doc {i}: {sim:.4f} - {doc[:50]}...")
    
    best_dummy = np.argmax(similarities_dummy)
    print(f"\n✓ Mejor match: Doc {best_dummy + 1} (score: {similarities_dummy[best_dummy]:.4f})")
    print(f"  '{docs[best_dummy][:60]}...'\n")
    
    print("="*80 + "\n")
    
    # ===== HF-E5 EMBEDDINGS =====
    print("🧠 HF-E5 EMBEDDINGS (Semántico)\n")
    
    try:
        hf = HFMultilingualE5Embedding(
            EmbeddingConfig(
                model_name="intfloat/multilingual-e5-small",
                dimension=384
            )
        )
        
        query_vec_hf = hf.embed_query(query)
        doc_vecs_hf = [hf.embed_text(doc) for doc in docs]
        
        similarities_hf = [
            cosine_similarity(query_vec_hf, doc_vec) 
            for doc_vec in doc_vecs_hf
        ]
        
        print("Similitudes:")
        for i, (doc, sim) in enumerate(zip(docs, similarities_hf), 1):
            print(f"  Doc {i}: {sim:.4f} - {doc[:50]}...")
        
        best_hf = np.argmax(similarities_hf)
        print(f"\n✓ Mejor match: Doc {best_hf + 1} (score: {similarities_hf[best_hf]:.4f})")
        print(f"  '{docs[best_hf][:60]}...'\n")
        
        # ===== COMPARACIÓN =====
        print("="*80 + "\n")
        print("📊 ANÁLISIS:\n")
        
        if best_dummy == 0 and best_hf == 0:
            print("✅ Ambos encontraron Python (correcto)")
            print(f"   Pero HF-E5 tiene mayor confianza: {similarities_hf[0]:.4f} vs {similarities_dummy[0]:.4f}")
        elif best_hf == 0 and best_dummy != 0:
            print("✅ HF-E5 encontró Python (CORRECTO)")
            print(f"❌ Dummy encontró {docs[best_dummy][:40]}... (INCORRECTO)")
            print("\n💡 HF-E5 entiende que Python se usa para IA/ML")
            print("   Dummy solo hace matching aleatorio")
        else:
            print(f"HF-E5: Doc {best_hf + 1}")
            print(f"Dummy: Doc {best_dummy + 1}")
        
        print("\n📈 Mejora de precisión:")
        improvement = (similarities_hf[0] - similarities_dummy[0]) / similarities_dummy[0] * 100
        print(f"   HF-E5 es {abs(improvement):.1f}% {'más' if improvement > 0 else 'menos'} preciso para el mejor resultado")
        
    except ImportError:
        print("⚠️  HF-E5 no instalado. Instala con:")
        print("   pip install torch transformers sentence-transformers")

if __name__ == "__main__":
    compare_embeddings()