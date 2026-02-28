# 🤖 ChatBot RAG - Sistema de Chat con Recuperación de Documentos

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Ollama](https://img.shields.io/badge/Ollama-LLM-green.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Sistema de chat inteligente que responde preguntas basándose exclusivamente en tus documentos**

[Características](#-características) •
[Instalación](#-instalación) •
[Uso](#-uso) •
[Docker](#-docker) •
[Documentación](#-documentación)

</div>

---

## 📋 Tabla de Contenidos

- [Descripción Ejecutiva](#-descripción-ejecutiva)
- [Descripción Técnica](#-descripción-técnica)
- [Cómo Funciona en Simple](#-cómo-funciona-en-simple)
- [Qué Problema Resuelve](#-qué-problema-resuelve)
- [Por Qué Es una Solución Profesional](#-por-qué-es-una-solución-profesional-y-segura)
- [Características](#-características)
- [Cómo Funciona](#-cómo-funciona)
- [Arquitectura](#-arquitectura)
- [Tecnologías](#-tecnologías)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Configuración](#-configuración)
- [Uso](#-uso)
- [Docker](#-docker)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Testing](#-testing)
- [Documentación Adicional](#-documentación-adicional)
- [Contribución](#-contribución)
- [Licencia](#-licencia)

---

## � Descripción Ejecutiva

**ChatBot RAG** es un asistente de inteligencia artificial que responde preguntas usando **únicamente la información de los documentos internos de la organización**. A diferencia de herramientas como ChatGPT, no tiene acceso a internet ni inventa respuestas: todo lo que dice proviene textualmente de los archivos que se le proporcionan, citando siempre la fuente exacta. Funciona de forma completamente privada dentro de la red corporativa, sin enviar ningún dato al exterior. Es configurable, escalable y está construido con estándares de ingeniería de software profesional, garantizando mantenibilidad a largo plazo. Puede desplegarse en cualquier servidor interno o nube privada, y su arquitectura modular permite incorporar nuevos tipos de documentos o modelos de IA sin reescribir el sistema.

✅ **Solo responde con información de tus documentos**  
✅ **Cita las fuentes** de cada respuesta  
✅ **Detecta preguntas maliciosas** con filtros de seguridad  
✅ **Funciona 100% offline** — sin dependencias externas  
✅ **Mantiene contexto** de la conversación  
✅ **Se actualiza automáticamente** al agregar nuevos PDFs

**Casos de uso ideales**: Atención al cliente, documentación interna, base de conocimientos empresarial, asistentes educativos, soporte técnico automatizado.

---

## 🛠️ Descripción Técnica

El sistema implementa una arquitectura **RAG (Retrieval-Augmented Generation)** con separación estricta de responsabilidades:

**Pipeline de ingesta (ETL):**
- Carga PDFs mediante loaders intercambiables (`Factory Pattern`)
- Divide el texto en fragmentos con overlap configurable (`TextChunker`)
- Genera embeddings semánticos con `intfloat/multilingual-e5` vía HuggingFace
- Persiste vectores en **ChromaDB** con detección de duplicados por hash determinístico
- Vigila carpetas automáticamente con `watchdog` para ingesta en tiempo real
- Al arrancar, indexa solo los documentos nuevos — salta los ya procesados

**Pipeline de consulta:**
- Búsqueda por similitud coseno con soporte a **MMR** (máxima marginal relevancia) y búsqueda expandida
- Filtro de relevancia configurable (`min_score`)
- Validación de seguridad en doble capa: 44+ palabras clave + 7 patrones regex contra prompt injection

**Capa LLM:**
- Cliente Ollama con interfaz abstracta (`Strategy Pattern`) — soporta `llama3.2`, `mistral`, `phi`, `codellama`
- Modo estricto: el modelo **no puede responder fuera del contexto recuperado**
- Historial de conversación con ventana de contexto gestionada

**Calidad de código:**
- `BaseSettings` Pydantic para configuración tipada vía `.env`
- 290+ tests con pytest (unitarios + integración)
- Type hints completos, validados con mypy
- Containerización completa con Docker Compose

---

## 💡 Cómo Funciona en Simple

Imaginá que tenés un empleado nuevo muy inteligente. El primer día le das a leer todos los manuales, reglamentos y documentos de la empresa. Él los lee, los memoriza y los organiza internamente de una forma que le permite encontrar información en segundos.

Cuando alguien le hace una pregunta, **no inventa nada**: busca en su memoria qué parte de qué documento responde mejor esa pregunta y te da la respuesta citando exactamente de dónde la sacó. Si no sabe algo porque no está en ningún documento, lo dice directamente.

Además, este empleado **nunca sale de la oficina**: toda su memoria y todo su conocimiento está guardado dentro de la empresa, sin depender de internet ni de servidores externos. Y si alguien intenta confundirlo con preguntas maliciosas o engañosas, tiene entrenamiento para detectarlas y no responderlas.

---

## 🎯 Qué Problema Resuelve

Las empresas acumulan enormes volúmenes de documentación interna (manuales, políticas, contratos, FAQs, reglamentos) que el personal no puede consultar fácilmente. Buscar información relevante toma tiempo, genera errores y depende de que la persona correcta esté disponible.

| Chatbot común | Este sistema |
|---|---|
| Responde con conocimiento general de internet | Responde **solo** con los documentos de la empresa |
| Puede inventar información (*alucinaciones*) | Solo habla si la información está en los documentos |
| Envía datos a servidores externos | Funciona **completamente offline** en red interna |
| No cita fuentes | Indica exactamente de qué documento viene cada respuesta |
| Sin control de seguridad específico | Detecta y bloquea intentos de manipulación |
| Base de conocimiento fija | Se actualiza automáticamente al agregar nuevos PDFs |

---

## 🔐 Por Qué Es una Solución Profesional y Segura

**Privacidad garantizada:** El sistema corre 100% dentro de la infraestructura propia. Ningún dato, pregunta ni documento sale de la red corporativa. Es apto para entornos con restricciones de confidencialidad o cumplimiento normativo (GDPR, ISO 27001, etc.).

**Confiabilidad de las respuestas:** El modo estricto impide que el modelo genere contenido fuera de los documentos cargados. Cada respuesta viene acompañada de su fuente, lo que permite auditar y verificar la información en segundos.

**Seguridad activa:** El sistema incluye un validador con doble capa de protección contra intentos de manipulación (*prompt injection*), un riesgo real en sistemas de IA expuestos a usuarios finales.

**Mantenibilidad a largo plazo:** La arquitectura modular basada en principios SOLID significa que agregar un nuevo tipo de documento, cambiar el modelo de IA o migrar la base de datos vectorial son tareas de horas, no de semanas. Los 290+ tests automatizados garantizan que cada cambio no rompe el comportamiento existente.

**Autonomía tecnológica:** Al usar modelos de código abierto (Ollama + HuggingFace), la empresa no depende de ningún proveedor externo, no paga por consulta y puede cambiar de modelo cuando aparezca una mejor alternativa, sin tocar el resto del sistema.

---

## ✨ Características

### 🎯 RAG (Retrieval-Augmented Generation)
- **Modo estricto**: Solo responde con información de documentos
- **Recuperación semántica**: Encuentra documentos relevantes por significado, no solo palabras clave
- **Citación de fuentes**: Muestra de qué documento viene cada respuesta
- **Control de relevancia**: Configurable con threshold de similitud

### 🔒 Seguridad
- **Validación de queries**: Detecta 44+ palabras clave maliciosas
- **Filtros regex**: 7 patrones para detectar inyecciones
- **Límite de longitud**: Protección contra queries excesivamente largas
- **Modo estricto RAG**: Previene "jailbreaking"

### 🧠 Embeddings Inteligentes
- **Factory Pattern**: Soporte para múltiples proveedores
- **HuggingFace E5**: Embeddings multilingües de alta calidad
- **Dummy Provider**: Para testing sin dependencias pesadas
- **Fácil extensión**: Agrega nuevos proveedores sin modificar código

### 💾 Vector Store
- **ChromaDB**: Persistencia en disco (recomendado producción)
- **InMemory**: Rápido para desarrollo y testing
- **Factory Pattern**: Cambia entre stores sin tocar código

### 🔧 Modularidad
- **Configuración centralizada**: Todo en `.env` o `settings.py`
- **Type hints completos**: Type safety con mypy
- **Logging estructurado**: Debug fácil
- **Testing exhaustivo**: 296+ tests con pytest

### 🌐 LLM Local
- **Ollama**: Modelos LLM corriendo localmente
- **Soporte GPU**: NVIDIA CUDA para inferencia rápida
- **Múltiples modelos**: llama3.2, phi, mistral, codellama

---

## 🔍 Cómo Funciona

### Flujo de Trabajo

```
┌─────────────────────────────────────────────────────────────────┐
│                     1. INGESTA DE DOCUMENTOS                    │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌──────────────────┐            ┌──────────────────┐
    │  Cargar PDFs     │            │  Extraer Texto   │
    │  (pdfplumber)    │───────────▶│  (metadata)      │
    └──────────────────┘            └──────────────────┘
                                             │
                              ┌──────────────┴──────────────┐
                              ▼                             ▼
                    ┌──────────────────┐         ┌──────────────────┐
                    │  Chunking        │         │  Embeddings      │
                    │  (1000 chars)    │────────▶│  (E5 384-dim)    │
                    └──────────────────┘         └──────────────────┘
                                                          │
                                                          ▼
                                              ┌──────────────────────┐
                                              │  Vector Store        │
                                              │  (ChromaDB/Memory)   │
                                              └──────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     2. QUERY (PREGUNTA DEL USUARIO)             │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌──────────────────┐            ┌──────────────────┐
    │  Validación      │            │  Embedding       │
    │  Seguridad       │───────────▶│  de Query        │
    └──────────────────┘            └──────────────────┘
                                             │
                              ┌──────────────┴──────────────┐
                              ▼                             ▼
                    ┌──────────────────┐         ┌──────────────────┐
                    │  Búsqueda        │         │  Filtro por      │
                    │  Similitud       │────────▶│  Relevancia      │
                    │  (cosine)        │         │  (min_score)     │
                    └──────────────────┘         └──────────────────┘
                                                          │
                                                          ▼
                                              ┌──────────────────────┐
                                              │  Top K Documentos    │
                                              │  Más Relevantes      │
                                              └──────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     3. GENERACIÓN DE RESPUESTA                  │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌──────────────────┐            ┌──────────────────┐
    │  Formatear       │            │  Prompt con      │
    │  Contexto        │───────────▶│  Contexto        │
    └──────────────────┘            └──────────────────┘
                                             │
                              ┌──────────────┴──────────────┐
                              ▼                             ▼
                    ┌──────────────────┐         ┌──────────────────┐
                    │  Enviar a LLM    │         │  Generar         │
                    │  (Ollama)        │────────▶│  Respuesta       │
                    └──────────────────┘         └──────────────────┘
                                                          │
                                                          ▼
                                              ┌──────────────────────┐
                                              │  Respuesta +         │
                                              │  Fuentes Citadas     │
                                              └──────────────────────┘
```

### Ejemplo Práctico

1. **Usuario pregunta**: "¿Cuál es el horario de atención?"

2. **Sistema busca** en documentos usando embeddings semánticos

3. **Encuentra chunks relevantes**:
   ```
   Documento: FAQ.pdf, Página 3
   Texto: "Nuestro horario de atención es de Lunes a Viernes..."
   Relevancia: 0.87
   ```

4. **Genera respuesta**:
   ```
   El horario de atención es de Lunes a Viernes de 9:00 a 18:00 horas.
   
   Fuente: FAQ.pdf, página 3
   ```

---

## 🏗️ Arquitectura

### Diagrama de Componentes

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│                          MAIN APPLICATION                          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│   INGESTION   │         │   RETRIEVAL   │         │     CHAT      │
│   PIPELINE    │         │   SYSTEM      │         │    SERVICE    │
└───────────────┘         └───────────────┘         └───────────────┘
        │                          │                          │
        ├─────────────┬────────────┤                          │
        ▼             ▼            ▼                          ▼
┌─────────────┐ ┌──────────┐ ┌──────────┐         ┌──────────────────┐
│  Document   │ │Embeddings│ │  Vector  │         │   LLM Client     │
│  Processor  │ │ Provider │ │  Store   │         │   (Ollama)       │
└─────────────┘ └──────────┘ └──────────┘         └──────────────────┘
        │             │            │                          │
        ▼             ▼            ▼                          ▼
┌─────────────┐ ┌──────────┐ ┌──────────┐         ┌──────────────────┐
│   Chunking  │ │ HF E5 /  │ │ ChromaDB │         │  Security        │
│   Strategy  │ │  Dummy   │ │ / Memory │         │  Validator       │
└─────────────┘ └──────────┘ └──────────┘         └──────────────────┘
```

### Patrones de Diseño Implementados

- **Factory Pattern**: Creación de embeddings y vector stores
- **Strategy Pattern**: Diferentes proveedores de embeddings
- **Repository Pattern**: Abstracción de vector store
- **Dependency Injection**: Configuración centralizada

---

## 🛠️ Tecnologías

### Core
- **Python 3.12** - Lenguaje principal
- **Pydantic 2.12** - Validación y configuración
- **python-dotenv** - Variables de entorno

### LLM & Embeddings
- **Ollama 0.1.x** - Servidor LLM local
- **HuggingFace Transformers 5.1** - Modelos de embeddings
- **PyTorch 2.10** - Backend para embeddings
- **sentence-transformers** - E5 multilingual embeddings

### Vector Database
- **ChromaDB 1.5** - Vector store persistente
- **numpy 2.4** - Operaciones vectoriales

### Document Processing
- **pdfplumber 0.11** - Extracción de PDFs
- **PyPDF2 3.0** - Procesamiento de PDFs

### Testing & Quality
- **pytest 9.0** - Framework de testing
- **mypy 1.19** - Type checking
- **coverage** - Cobertura de tests

### Docker
- **Docker 20.10+** - Containerización
- **Docker Compose 2.0+** - Orquestación

---

## 📦 Requisitos

### Requisitos del Sistema

#### Instalación Local
- **Python**: 3.12 o superior
- **RAM**: Mínimo 8GB (recomendado 16GB)
- **Disco**: 10GB libres (para modelos)
- **GPU**: Opcional pero recomendada (NVIDIA con CUDA)
- **OS**: Windows 10/11, Linux, macOS

#### Docker
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **RAM**: Mínimo 8GB
- **Disco**: 15GB libres

### Dependencias Python

Ver `requirements.txt` para lista completa. Principales:
```
ollama>=0.1.0
transformers>=5.0.0
torch>=2.10.0
chromadb>=1.5.0
pydantic>=2.12.0
pydantic-settings>=2.7.0
pytest>=9.0.0
```

### Ollama

Necesitas Ollama instalado y corriendo:

**Linux/Mac:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve  # Inicia el servidor
ollama pull llama3.2  # Descarga modelo
```

**Windows:**
```powershell
# Descargar desde: https://ollama.com/download
# Instalar e iniciar desde el menú
```

---

## 🚀 Instalación

### Opción 1: Instalación Local (Desarrollo)

#### 1. Clonar el Repositorio

```bash
git clone https://github.com/LeandroTombe/chatbotproyectof.git
cd chatbotproyectof
```

#### 2. Crear Entorno Virtual

**Linux/Mac:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### 3. Instalar Dependencias

```bash
# Actualizar pip
pip install --upgrade pip

# Instalar dependencias
pip install -r requirements.txt
```

#### 4. Configurar Ollama

```bash
# Verificar que Ollama esté corriendo
ollama list

# Descargar modelo (primera vez)
ollama pull llama3.2

# Verificar
ollama list
```

#### 5. Configurar Variables de Entorno

```bash
# Copiar template
cp .env.example .env

# Editar .env con tu editor favorito
nano .env  # o code .env, vim .env, etc.
```

Configuración mínima en `.env`:
```bash
OLLAMA_MODEL=llama3.2
EMBEDDING_PROVIDER=hf-e5
EMBEDDING_MODEL=intfloat/multilingual-e5-small
EMBEDDING_DIMENSION=384
VECTOR_STORE_TYPE=chroma
```

#### 6. Preparar Documentos

La carpeta `data/pdfs/` ya existe en el proyecto. Simplemente copiá tus PDFs ahí:

```bash
# Copiar tus PDFs a data/pdfs/
cp /ruta/a/tus/pdfs/*.pdf data/pdfs/
```

```powershell
# Windows
Copy-Item "C:\ruta\a\tus\pdfs\*.pdf" "data\pdfs\"
```

> Al iniciar la aplicación, los documentos se indexan automáticamente. No hace falta ningún paso adicional.

#### 7. ¡Listo para Usar!

```bash
python main.py
```

### Opción 2: Docker (Producción/Testing)

Ver [sección Docker](#-docker) más abajo o consultar [README.docker.md](README.docker.md).

**Quick Start:**
```powershell
# Windows
.\scripts\docker-setup.ps1

# Linux/Mac
chmod +x scripts/docker-setup.sh
./scripts/docker-setup.sh
```

---

## ⚙️ Configuración

### Archivo .env

El proyecto usa variables de entorno para configuración. Ejemplo completo:

```bash
# ========================================
# Ollama LLM Configuration
# ========================================
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_TIMEOUT=180
OLLAMA_MAX_TOKENS=300
OLLAMA_TEMPERATURE=0.7

# ========================================
# RAG Configuration
# ========================================
RAG_TOP_K=2                      # Documentos a recuperar
RAG_MIN_RELEVANCE=0.3            # Score mínimo (0-1)
RAG_MAX_CONTEXT_LENGTH=1000      # Caracteres máximos
RAG_INCLUDE_SOURCES=True         # Incluir fuentes en respuesta
RAG_STRICT_MODE=True             # Solo responder con docs
RAG_ENABLE_SECURITY=True         # Validación de seguridad

# ========================================
# Embeddings Configuration
# ========================================
EMBEDDING_PROVIDER=hf-e5
EMBEDDING_MODEL=intfloat/multilingual-e5-small
EMBEDDING_DIMENSION=384
EMBEDDING_BATCH_SIZE=32

# Alternativa: Dummy (para testing)
# EMBEDDING_PROVIDER=dummy
# EMBEDDING_MODEL=dummy-embeddings
# EMBEDDING_DIMENSION=768

# ========================================
# Chunking Configuration
# ========================================
CHUNK_SIZE=1000                  # Tamaño de chunks
CHUNK_OVERLAP=200                # Overlap entre chunks
CHUNK_SEPARATOR=\n\n             # Separador de chunks

# ========================================
# Vector Store Configuration
# ========================================
VECTOR_STORE_TYPE=chroma         # chroma o memory
CHROMA_PERSIST_DIRECTORY=./vectorstore_data
CHROMA_COLLECTION_NAME=documents
```

### Modelos Disponibles

#### Ollama (LLM)
```bash
# Pequeños y rápidos
ollama pull phi              # 1.3GB
ollama pull llama3.2         # 2GB (recomendado)

# Modelos grandes (mejor calidad)
ollama pull mistral          # 4GB
ollama pull llama2           # 4GB
ollama pull codellama        # 4GB
```

#### HuggingFace (Embeddings)
```python
# En .env, cambiar EMBEDDING_MODEL:

# Pequeño y rápido (recomendado)
intfloat/multilingual-e5-small     # 384 dim

# Mejor calidad
intfloat/multilingual-e5-base      # 768 dim
intfloat/multilingual-e5-large     # 1024 dim
```

---

## 💻 Uso

### Comando Principal

```bash
# Ejecutar aplicación principal
python main.py
```

### Flujo de Uso

1. **Al iniciar**, el sistema:
   - Carga configuración
   - Inicializa embeddings
   - Crea/conecta vector store
   - Configura LLM client

2. **Carga documentos** (si es primera vez):
   - Lee PDFs de `./documents/`
   - Extrae y divide texto en chunks
   - Genera embeddings
   - Almacena en vector store

3. **Chat interactivo**:
   ```
   Usuario: ¿Cuál es el horario de atención?
   Bot: El horario de atención es...
        Fuente: FAQ.pdf, página 3
   
   Usuario: ¿Aceptan tarjetas?
   Bot: Sí, aceptamos Visa, Mastercard...
        Fuente: Pagos.pdf, página 1
   ```

4. **Comandos especiales**:
   - `salir`, `exit`, `quit` - Terminar
   - `stats` - Ver estadísticas del vector store
   - `clear` - Limpiar conversación

### Ejemplos de Uso

#### Ejemplo 1: Primera Ejecución

```bash
$ python main.py

================================================================================
  SETTING UP RAG SYSTEM COMPONENTS
================================================================================

✓ Embedder: HFMultilingualE5Embedding(...) (provider: hf-e5)
✓ Vector Store: Chroma
   Collection: documents
   Directory: ./vectorstore_data
✓ Chunker: chunk_size=1000, overlap=200
✓ Document Processor initialized
✓ Retriever: top_k=2, min_score=0.3
✓ Ingestion Pipeline initialized

================================================================================
  LOADING DOCUMENTS
================================================================================

Processing: FAQ.pdf
  ✓ 5 chunks created
Processing: Manual.pdf
  ✓ 12 chunks created
  
Total documents processed: 2
Total chunks stored: 17

================================================================================
  CHATBOT RAG - INTERACTIVE MODE
================================================================================

Bot: ¡Hola! Pregúntame sobre los documentos cargados.
Usuario> ¿Qué documentos tienes?
Bot: Tengo acceso a los siguientes documentos:
     - FAQ.pdf
     - Manual.pdf
```

#### Ejemplo 2: Testing Rápido

```bash
# Usar dummy embeddings (sin descargar modelos)
export EMBEDDING_PROVIDER=dummy
export VECTOR_STORE_TYPE=memory

python main.py
```

#### Ejemplo 3: Solo Procesar Documentos

```python
# script personalizado
from ingestion.pipeline import IngestionPipeline

pipeline = IngestionPipeline(processor, retriever)
pipeline.ingest_directory("./documents")
```

### Scripts Útiles

```bash
# Ver logs
tail -f logs/chatbot.log

# Limpiar vector store
rm -rf vectorstore_data/

# Limpiar cache de HuggingFace
rm -rf models/

# Ejecutar con debug
PYTHONPATH=. python -m pdb main.py
```

---

## 🐳 Docker

### Quick Start Docker

#### Windows (PowerShell)
```powershell
# Setup automático completo
.\scripts\docker-setup.ps1

# O manual:
Copy-Item .env.docker .env
docker-compose up -d --build
docker-compose exec ollama ollama pull llama3.2
docker-compose exec chatbot python main.py
```

#### Linux/Mac
```bash
# Setup automático completo
chmod +x scripts/docker-setup.sh
./scripts/docker-setup.sh

# O con Make:
make setup
```

### Comandos Docker

```bash
# Ver logs
docker-compose logs -f

# Ejecutar chatbot
docker-compose exec chatbot python main.py

# Ejecutar tests
docker-compose run --rm chatbot python -m pytest

# Shell interactivo
docker-compose exec chatbot bash

# Detener
docker-compose down

# Limpiar TODO
docker-compose down -v
```



## 📂 Estructura del Proyecto

```
ChatBotProyecto/
│
├── 📄 main.py                  # Punto de entrada principal
├── 📄 requirements.txt         # Dependencias Python
├── 📄 .env.example            # Template de configuración
├── 📄 mypy.ini                # Configuración type checking
│
├── 📁 api/                    # (Futuro) API REST
│
├── 📁 chat/                   # Sistema de chat
│   ├── __init__.py
│   ├── models.py              # Modelos: Message, ChatResponse
│   ├── rag_service.py         # Servicio RAG principal
│   ├── security.py            # Validación y seguridad
│   └── llm_clients/           # Clientes LLM
│       ├── base.py            # Clase base abstracta
│       └── ollama_client.py   # Implementación Ollama
│
├── 📁 config/                 # Configuración
│   ├── __init__.py
│   └── settings.py            # Settings con Pydantic
│
├── 📁 core/                   # Core utilities
│   ├── __init__.py
│   └── config.py              # Configuración legacy
│
├── 📁 documents/              # Procesamiento de documentos
│   ├── __init__.py
│   ├── loaders/               # Loaders por tipo
│   │   ├── base.py
│   │   └── pdf_loader.py
│   ├── processor.py           # Procesador principal
│   └── factory.py             # Factory de loaders
│
├── 📁 domain/                 # Modelos de dominio
│   ├── __init__.py
│   └── models.py              # Chunk, SearchResult, etc.
│
├── 📁 embeddings/             # Sistema de embeddings
│   ├── __init__.py
│   ├── base.py                # BaseEmbedding, DummyEmbedding
│   ├── factory.py             # Factory pattern
│   └── providers/
│       └── hf_e5_embedding.py # HuggingFace E5
│
├── 📁 ingestion/              # Pipeline de ingesta
│   ├── __init__.py
│   ├── pipeline.py            # IngestionPipeline
│   └── pdf_loader.py          # Loader de PDFs
│
├── 📁 processing/             # Procesamiento de texto
│   ├── __init__.py
│   └── chunking.py            # TextChunker
│
├── 📁 retrieval/              # Sistema de recuperación
│   ├── __init__.py
│   └── retriever.py           # DocumentRetriever
│
├── 📁 vectorstore/            # Vector stores
│   ├── __init__.py
│   ├── base.py                # BaseVectorStore, InMemory
│   ├── factory.py             # Factory pattern
│   └── implementations/
│       └── chroma.py          # ChromaDB implementation
│
├── 📁 tests/                  # Tests (296+ tests)
│   ├── test_*.py              # Unit tests
│   ├── providers/             # Tests de providers
│   └── compare/               # Comparación de embeddings
│
├── 📁 docs/                   # Documentación
│   ├── FACTORY_PATTERN.md     # Explicación Factory Pattern
│   ├── ANALISIS_MEJORAS.md    # Análisis de código
│   └── GUIA_IMPLEMENTACION_MEJORAS.md
│
├── 📁 scripts/                # Scripts de automatización
│   ├── docker-setup.sh        # Setup Docker (Linux/Mac)
│   ├── docker-setup.ps1       # Setup Docker (Windows)
│   ├── docker-cleanup.sh      # Cleanup (Linux/Mac)
│   └── docker-cleanup.ps1     # Cleanup (Windows)
│
├── 📁 data/                   # Datos procesados (git ignored)
├── 📁 logs/                   # Logs de aplicación (git ignored)
├── 📁 models/                 # Cache de modelos HF (git ignored)
├── 📁 documents/              # PDFs fuente (git ignored)
└── 📁 vectorstore_data/       # ChromaDB data (git ignored)
│
├── 🐳 Docker Files
├── Dockerfile                 # Imagen de la aplicación
├── docker-compose.yml         # Orquestación de servicios
├── .dockerignore             # Exclusiones de build
├── .env.docker               # Template variables Docker
│
├── 📚 Documentación Docker
├── README.docker.md          # Guía completa Docker
├── README.docker.windows.md  # Guía Windows específica
├── DOCKER_QUICKSTART.md      # Referencia rápida
│
└── 🔧 Otros
    ├── Makefile              # Comandos simplificados (Linux/Mac)
    ├── .gitignore           # Exclusiones de Git
    └── mypy.ini             # Configuración mypy
```

---

## 🧪 Testing

### Ejecutar Tests

```bash
# Todos los tests
python -m pytest

# Con verbose
python -m pytest -v

# Con coverage
python -m pytest --cov=. --cov-report=html

# Tests específicos
python -m pytest tests/test_embeddings.py
python -m pytest tests/test_vectorstore.py
python -m pytest tests/test_rag_service.py

# Solo tests rápidos (excluir HF que requiere torch)
python -m pytest --ignore=tests/providers/

# Ver coverage HTML
# Abre htmlcov/index.html en tu navegador
```

### Estadísticas de Testing

```
📊 Coverage Actual:
- 296 tests pasando
- 41 tests de embeddings
- 48 tests de vector store
- 9 tests de factory patterns
- + tests de RAG, seguridad, retrieval, etc.
```

### Tests en Docker

```bash
# Ejecutar todos los tests
docker-compose run --rm chatbot python -m pytest

# Con coverage
docker-compose run --rm chatbot python -m pytest --cov=. --cov-report=html
```

---

## 📚 Documentación Adicional


### Recursos Externos

- **Ollama**: https://ollama.ai/
- **ChromaDB**: https://docs.trychroma.com/
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers/
- **Pydantic**: https://docs.pydantic.dev/
- **Docker**: https://docs.docker.com/

---

## 🤝 Contribución

### Cómo Contribuir

1. **Fork** el repositorio
2. **Crea** una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. **Push** a la rama (`git push origin feature/AmazingFeature`)
5. **Abre** un Pull Request

### Guías de Desarrollo

- Usa **type hints** en todo el código
- Escribe **tests** para nuevas funcionalidades
- Actualiza la **documentación**
- Sigue **PEP 8** (usa `black` para formatear)
- Ejecuta **mypy** para type checking
- Mantén coverage de tests > 80%

### Reportar Bugs

Abre un issue con:
- Descripción del problema
- Pasos para reproducir
- Comportamiento esperado vs actual
- Logs relevantes
- Versión de Python, OS, etc.


---

## 👨‍💻 Autor

**Leandro Tombe**
- GitHub: [@LeandroTombe](https://github.com/LeandroTombe)
- Repositorio: [chatbotproyectof](https://github.com/LeandroTombe/chatbotproyectof)

---

## 🙏 Agradecimientos

- **Ollama** - Por facilitar LLMs locales
- **ChromaDB** - Por un excelente vector database
- **HuggingFace** - Por modelos de embeddings de calidad
- **Comunidad Python** - Por las increíbles librerías

---

## 📞 Soporte

¿Necesitas ayuda?

1. 📖 Revisa la [documentación](#-documentación-adicional)
2. 🐛 Busca en [Issues](https://github.com/LeandroTombe/chatbotproyectof/issues)
3. 💬 Abre un nuevo Issue
4. 📧 Contacta al autor

---

<div align="center">

**⭐ Si te gusta este proyecto, considera darle una estrella en GitHub ⭐**

Hecho con ❤️ usando Python, Ollama y mucho ☕

</div>
