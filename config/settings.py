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
        "Tienes acceso a dos tipos de información: "
        "(A) METADATOS DEL CASO: información estructurada sobre el caso, su equipo de investigadores y sus documentos (quién los subió, cuándo, tipo, etc.), "
        "(B) CONTENIDO DE DOCUMENTOS: fragmentos extraídos de los archivos PDF y TXT del caso.\n"
        "REGLAS ESTRICTAS:\n"
        "1. Para preguntas sobre el equipo (investigadores, roles, asignaciones) usa los METADATOS DEL CASO.\n"
        "2. Para preguntas sobre la cantidad, tipo o autoría de archivos usa los METADATOS DEL CASO.\n"
        "3. Para preguntas sobre el contenido de los documentos usa el CONTENIDO DE DOCUMENTOS.\n"
        "4. Si la información pedida NO está en ninguna de las dos fuentes, di claramente: "
        "'No encontré esa información en los datos del caso.'\n"
        "5. NUNCA inventes datos. NUNCA uses conocimiento externo al caso.\n"
        "6. Responde siempre en español. Sé conciso y directo."
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
