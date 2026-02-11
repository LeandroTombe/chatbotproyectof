# 🤖 Guía de Uso - Chat RAG con Ollama

## 🚀 Inicio Rápido

### 1. Preparar tus PDFs

Coloca tus archivos PDF en la carpeta `data/pdfs/`:

```bash
data/
  pdfs/
    documento1.pdf
    documento2.pdf
    documento3.pdf
```

### 2. Verificar que Ollama esté corriendo

```bash
# Ver modelos disponibles
ollama list

# Si Ollama no está corriendo, inícialo:
ollama serve
```

### 3. Ejecutar el sistema

```bash
python main.py
```

## 📋 Proceso del Demo

El script `main.py` te guiará a través de estos pasos:

### **Demo 1: Procesar un PDF individual**
- Carga un archivo PDF
- Lo divide en chunks
- Genera embeddings
- Lo almacena en el vector store

### **Demo 2: Procesamiento en lote**
- Procesa todos los PDFs de la carpeta
- Muestra estadísticas de éxito/fallo

### **Demo 3: Búsqueda semántica**
- Busca información en los documentos procesados
- Muestra los chunks más relevantes

### **Demo 4: Extracción de contexto**
- Obtiene contexto formateado para RAG

### **Demo 5: Chat Setup**
- Inicializa el servicio de chat con Ollama
- Verifica disponibilidad del modelo

### **Demo 6: Chat RAG**
- Pregunta de ejemplo con contexto de documentos
- Muestra fuentes utilizadas

### **Demo 7: Chat Interactivo** 💬
- **Aquí puedes hacer tus preguntas!**
- El sistema busca en tus PDFs
- El LLM responde usando el contexto
- Cita las fuentes utilizadas

## 💬 Uso del Chat Interactivo

### Comandos disponibles:

```
👤 You: tu pregunta aquí          → Hacer una pregunta
👤 You: clear                      → Limpiar historial de conversación
👤 You: history                    → Ver historial de mensajes
👤 You: quit / exit / q            → Salir
```

### Ejemplos de preguntas:

```
👤 You: ¿Cuál es el tema principal del documento?
👤 You: Resume los puntos clave
👤 You: ¿Qué dice sobre [tema específico]?
👤 You: Dame más detalles sobre [concepto]
```

## ⚙️ Configuración Avanzada

### Cambiar el modelo de Ollama

Edita `main.py` en la función `setup_chat_service()`:

```python
rag_service = setup_chat_service(
    retriever=retriever,
    model_name="llama3.2",  # ← Cambia aquí el modelo
    base_url="http://localhost:11434"
)
```

### Modelos recomendados:

- **llama3.2** (2GB) - Rápido, buena calidad
- **llama2** (3.8GB) - Muy popular, balanceado
- **mistral** (4.1GB) - Excelente rendimiento
- **codellama** - Especializado en código

Para descargar un modelo:
```bash
ollama pull llama3.2
ollama pull mistral
```

### Ajustar parámetros RAG

En `main.py`, modifica `RAGConfig`:

```python
rag_config = RAGConfig(
    top_k=3,                    # Número de documentos a recuperar
    min_relevance=0.3,          # Score mínimo de relevancia (0-1)
    max_context_length=2000,    # Máximo de caracteres de contexto
    include_sources=True,       # Mostrar fuentes en respuesta
    system_prompt="..."         # Prompt del sistema
)
```

### Ajustar parámetros del LLM

En `main.py`, modifica `LLMConfig`:

```python
llm_config = LLMConfig(
    model_name="llama3.2",
    temperature=0.7,       # Creatividad (0=conservador, 1=creativo)
    max_tokens=512,        # Longitud máxima de respuesta
    timeout=60            # Timeout en segundos
)
```

## 🛠️ Solución de Problemas

### ❌ "Ollama is not running"

```bash
# Iniciar Ollama
ollama serve

# En otra terminal, verificar
ollama list
```

### ❌ "Model 'llama2' is not available"

```bash
# Descargar el modelo
ollama pull llama2

# O usar el que ya tienes
ollama pull llama3.2
```

### ❌ "No results found"

- Asegúrate de que los PDFs se hayan procesado correctamente
- Verifica que haya archivos en `data/pdfs/`
- Revisa que los documentos contengan texto (no solo imágenes)

### ❌ Error de conexión timeout

**Síntoma**: `Request to Ollama timed out after XXs`

**Causas**:
- Primera consulta (modelo cargándose en memoria)
- Contexto muy largo enviado al LLM
- Procesador lento o recursos limitados

**Soluciones**:

1. **Espera un poco más en la primera consulta** - El sistema ahora precalienta el modelo automáticamente

2. **Si sigue ocurriendo**, los parámetros ya están optimizados en `main.py`:
   - Timeout: 120 segundos (fue aumentado)
   - max_tokens: 300 (reducido para respuestas más rápidas)
   - max_context_length: 1000 (reducido para menos texto)
   - top_k: 2 (menos documentos de contexto)

3. **Para ajustes adicionales**, edita en `main.py`:
   ```python
   llm_config = LLMConfig(
       timeout=180  # Aumenta aún más si es necesario
   )
   
   rag_config = RAGConfig(
       top_k=1,              # Usa solo 1 documento
       max_context_length=500  # Reduce más el contexto
   )
   ```

4. **Verifica Ollama**:
   ```bash
   # Ver si el modelo está cargado
   ollama ps
   
   # Si ves 100% CPU, está trabajando
   ```

5. **Usa un modelo más rápido**:
   ```bash
   # Descargar un modelo más pequeño
   ollama pull phi
   
   # Luego en main.py cambia a model_name="phi"
   ```

### ❌ Error de conexión timeout

- Aumenta el timeout en `LLMConfig`
- Usa un modelo más pequeño (llama3.2 en lugar de llama2)

## 📊 Flujo Completo

```
1. PDFs en data/pdfs/
        ↓
2. Procesamiento (main.py Demo 1-2)
   - Extracción de texto
   - División en chunks
   - Generación de embeddings
   - Almacenamiento en vector store
        ↓
3. Chat Interactivo (Demo 7)
   - Tu pregunta
        ↓
   - Búsqueda semántica en vector store
        ↓
   - Recuperación de chunks relevantes
        ↓
   - Construcción de contexto
        ↓
   - Generación de respuesta con Ollama
        ↓
   - Respuesta + Fuentes citadas
```

## 🎯 Ejemplo Completo

```bash
# 1. Colocar PDFs
cp mis_documentos/*.pdf data/pdfs/

# 2. Ejecutar el sistema
python main.py

# 3. Seguir los demos (presionar ENTER)
# ...

# 4. En el chat interactivo:
👤 You: ¿De qué trata este documento?

🤔 Thinking...

🤖 Assistant: Este documento trata principalmente sobre [respuesta basada en el contenido]...

📚 [2 sources used]

👤 You: Dame más detalles sobre [tema]

🤖 Assistant: [Respuesta detallada]...

👤 You: quit
```

## 📝 Notas Importantes

- **Embeddings**: Por defecto usa DummyEmbedding (para demo). Para producción, considera usar HuggingFace E5.
- **Memoria**: El chat mantiene historial de conversación (configurable)
- **Privacidad**: Todo se ejecuta localmente, no se envían datos a APIs externas
- **Rendimiento**: Primera pregunta puede tardar más (carga del modelo)

## 🔗 Recursos Adicionales

- [Ollama Documentation](https://ollama.ai)
- [Available Models](https://ollama.ai/library)
- README_DEMO.md - Documentación técnica completa
- QUICKSTART.md - Guía rápida de instalación

---

¡Disfruta conversando con tus documentos! 🚀
