# Quick Start Guide for main.py

## Para probar rápidamente SIN PDFs reales:

```python
# test_without_pdf.py - Script de prueba sin archivos
from main import setup_components
from domain.models import Document, ProcessingStatus
from pathlib import Path

# Setup
processor, retriever, pipeline, vector_store = setup_components()

# Crear documento de prueba manualmente (simulando carga de PDF)
test_document = Document(
    id="test-doc-1",
    file_path=Path("test.pdf"),
    file_name="test.pdf",
    status=ProcessingStatus.PENDING,
    metadata={
        "content": """
        Python es un lenguaje de programación de alto nivel.
        Python es interpretado, orientado a objetos y con tipado dinámico.
        
        Python se usa en:
        - Desarrollo web con Django y Flask
        - Ciencia de datos con NumPy, Pandas, Matplotlib
        - Machine Learning con TensorFlow, PyTorch, scikit-learn
        - Automatización y scripting
        
        Características principales:
        1. Sintaxis simple y legible
        2. Gran ecosistema de librerías
        3. Comunidad activa
        4. Multiplataforma
        """
    }
)

# Procesar el documento de prueba
print("Procesando documento de prueba...")
doc = processor.process_document.__wrapped__(processor, test_document.file_path)

# Buscar información
print("\n🔍 Búsqueda: '¿Qué es Python?'")
results = retriever.search("¿Qué es Python?", top_k=3)

for idx, result in enumerate(results, 1):
    print(f"\n{idx}. Score: {result.score:.3f}")
    print(f"   {result.chunk.content[:100]}...")

# Obtener contexto
print("\n📄 Contexto para RAG:")
context = retriever.get_context("usos de Python", top_k=2)
print(context[:300] + "...")
```

## Para usar con PDFs reales:

1. **Coloca tus PDFs:**
   ```powershell
   # Copia tus PDFs aquí
   data/pdfs/documento1.pdf
   data/pdfs/documento2.pdf
   ```

2. **Ejecuta el demo:**
   ```powershell
   python main.py
   ```

3. **O usa programáticamente:**
   ```python
   from main import setup_components, demo_process_single_file
   
   processor, retriever, pipeline, vector_store = setup_components()
   
   # Procesar archivo
   doc = demo_process_single_file(processor, "data/pdfs/documento1.pdf")
   
   # Buscar
   results = retriever.search("tu pregunta aquí")
   ```

## Funciones principales disponibles:

```python
# 1. Procesar archivo individual
document = processor.process_document("archivo.pdf")

# 2. Procesar directorio completo
result = pipeline.process_directory("data/pdfs/", recursive=True)

# 3. Buscar información
results = retriever.search("pregunta", top_k=5)

# 4. Obtener contexto para LLM
context = retriever.get_context("pregunta", top_k=3)

# 5. Ver estadísticas
from main import show_vector_store_stats
show_vector_store_stats(vector_store)
```

## Ejemplos rápidos:

### Ejemplo 1: Pipeline completo
```python
from main import setup_components

# Setup
processor, retriever, pipeline, vector_store = setup_components()

# Procesar todos los PDFs de un directorio
result = pipeline.process_directory("data/pdfs/")
print(f"Procesados: {result.successful}/{result.total_files}")

# Buscar
results = retriever.search("¿Qué dice el documento sobre X?")
for r in results:
    print(f"[{r.score:.2f}] {r.chunk.content[:100]}...")
```

### Ejemplo 2: Solo búsqueda (si ya procesaste archivos)
```python
from main import setup_components

_, retriever, _, _ = setup_components()

results = retriever.search("tu pregunta")
for r in results:
    print(r.chunk.content)
```

### Ejemplo 3: Obtener contexto para RAG
```python
from main import setup_components

_, retriever, _, _ = setup_components()

# Obtener contexto formateado
context = retriever.get_context(
    query="Explicame sobre X",
    top_k=5,
    separator="\n\n---\n\n"
)

# Usar con LLM (ejemplo conceptual)
prompt = f"""
Contexto:
{context}

Pregunta: Explicame sobre X

Respuesta:
"""
```
