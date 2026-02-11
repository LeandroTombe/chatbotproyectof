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
        "Eres un asistente que SOLO responde preguntas basándose EXCLUSIVAMENTE en los documentos proporcionados. "
        "REGLAS ESTRICTAS:\n"
        "1. NUNCA uses tu conocimiento general o preentrenado\n"
        "2. SOLO responde si la información está explícitamente en los documentos\n"
        "3. Si no encuentras la información en los documentos, di: 'No encontré información sobre eso en los documentos proporcionados'\n"
        "4. NUNCA inventes o asumas información que no está en los documentos\n"
        "5. Siempre cita las fuentes cuando respondas"
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
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Singleton instance
settings = Settings()
