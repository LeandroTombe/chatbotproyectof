"""
Settings and configuration management using Pydantic BaseSettings.
All configuration values can be overridden via environment variables.
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    All settings can be overridden by creating a .env file in the project root
    or by setting environment variables with the same names.
    """
    
    # ========================================================================
    # OLLAMA LLM CONFIGURATION
    # ========================================================================
    OLLAMA_MODEL: str = "phi"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT: int = 120
    OLLAMA_MAX_TOKENS: int = 300
    OLLAMA_TEMPERATURE: float = 0.7
    
    # ========================================================================
    # RAG CONFIGURATION
    # ========================================================================
    RAG_TOP_K: int = 2
    RAG_MIN_RELEVANCE: float = 0.3
    RAG_MAX_CONTEXT_LENGTH: int = 1000
    RAG_INCLUDE_SOURCES: bool = True
    RAG_STRICT_MODE: bool = True  # Only answer from documents, no general knowledge
    RAG_ENABLE_SECURITY: bool = True  # Block queries about system internals
    RAG_SYSTEM_PROMPT: str = (
        "Eres un asistente de investigación legal del Proyecto Sherlock. "
        "Tienes acceso a dos fuentes de información: "
        "(A) METADATOS DEL CASO: información estructurada sobre el caso, su equipo, personas involucradas y documentos. "
        "(B) CONTENIDO DE DOCUMENTOS: fragmentos extraídos de los archivos del caso.\n"
        "REGLAS — DEBES CUMPLIRLAS SIN EXCEPCION:\n"
        "1. USA EXCLUSIVAMENTE la información de las fuentes (A) y (B). "
        "PROHIBIDO usar conocimiento propio, inventar datos o completar información faltante.\n"
        "2. Si la información pedida NO está en ninguna fuente, responde SOLO: "
        "'No encontré esa información en los documentos del caso.' "
        "NO agregues 'Sin embargo', NO especules, NO ofrezcas alternativas.\n"
        "3. NUNCA digas frases como 'basado en lo que sé', 'generalmente', 'podría ser', "
        "'te puedo ofrecer detalles generales' o similares.\n"
        "4. Responde siempre en español. Sé directo y conciso."
    )
    
    # ========================================================================
    # EMBEDDING CONFIGURATION
    # ========================================================================
    EMBEDDING_PROVIDER: str = "dummy"
    EMBEDDING_MODEL: str = "dummy-embeddings"
    EMBEDDING_DIMENSION: int = 768
    EMBEDDING_BATCH_SIZE: int = 32
    
    # ========================================================================
    # CHUNKING CONFIGURATION
    # ========================================================================
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    CHUNK_SEPARATOR: str = "\n\n"
    
    # ========================================================================
    # DATA DIRECTORIES
    # ========================================================================
    DATA_DIR: str = "data/pdfs"
    
    # ========================================================================
    # VECTOR STORE CONFIGURATION
    # ========================================================================
    VECTOR_STORE_TYPE: str = "memory"  # Available: "memory", "chroma"
    CHROMA_PERSIST_DIRECTORY: str = "data/chroma"
    CHROMA_COLLECTION_NAME: str = "chatbot_collection"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Singleton instance
settings = Settings()
