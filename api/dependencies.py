"""
Shared dependencies and application state for the FastAPI server.
All heavy components (embedder, vector store, RAGService, etc.) are
initialized once at startup and reused across requests.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from config.settings import settings
from domain.models import ChunkingConfig
from embeddings.base import EmbeddingConfig
from embeddings.factory import create_embedder
from vectorstore import create_vector_store, BaseVectorStore
from ingestion.chunking import TextChunker
from ingestion.processor import DocumentProcessor
from ingestion.pipeline import IngestionPipeline
from retrieval.retriever import DocumentRetriever
from chat.llm_clients.ollama_client import OllamaClient
from chat.llm_clients.base import LLMConfig
from chat.rag_service import RAGService, RAGConfig
from chat.security import SecurityConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global singletons — populated during lifespan startup
# ---------------------------------------------------------------------------

_rag_service: Optional[RAGService] = None
_ingestion_pipeline: Optional[IngestionPipeline] = None
_vector_store: Optional[BaseVectorStore] = None
_llm_client: Optional[OllamaClient] = None


def get_rag_service() -> RAGService:
    """FastAPI dependency: returns the initialized RAGService."""
    if _rag_service is None:
        raise RuntimeError("RAGService not initialized. Server may still be starting.")
    return _rag_service


def get_ingestion_pipeline() -> IngestionPipeline:
    """FastAPI dependency: returns the initialized IngestionPipeline."""
    if _ingestion_pipeline is None:
        raise RuntimeError("IngestionPipeline not initialized. Server may still be starting.")
    return _ingestion_pipeline


def get_vector_store() -> BaseVectorStore:
    """FastAPI dependency: returns the initialized vector store."""
    if _vector_store is None:
        raise RuntimeError("VectorStore not initialized. Server may still be starting.")
    return _vector_store


def get_llm_client() -> OllamaClient:
    """FastAPI dependency: returns the initialized OllamaClient."""
    if _llm_client is None:
        raise RuntimeError("LLM client not initialized. Server may still be starting.")
    return _llm_client


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def initialize_components() -> None:
    """Initialize all RAG components and store them as module-level singletons.
    Called once during FastAPI lifespan startup.
    """
    global _rag_service, _ingestion_pipeline, _vector_store, _llm_client

    logger.info("Initializing RAG components...")

    # 1. Embedder
    emb_cfg = EmbeddingConfig(
        model_name=settings.EMBEDDING_MODEL,
        dimension=settings.EMBEDDING_DIMENSION,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )
    embedder = create_embedder(provider=settings.EMBEDDING_PROVIDER, config=emb_cfg)
    logger.info("Embedder ready: %s (dim=%d)", settings.EMBEDDING_PROVIDER, settings.EMBEDDING_DIMENSION)

    # 2. Vector store
    try:
        _vector_store = create_vector_store(
            provider=settings.VECTOR_STORE_TYPE,
            dimension=settings.EMBEDDING_DIMENSION,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
        )
        logger.info("Vector store ready: %s", settings.VECTOR_STORE_TYPE)
    except ValueError as exc:
        logger.warning("%s — falling back to in-memory store", exc)
        _vector_store = create_vector_store(provider="memory", dimension=settings.EMBEDDING_DIMENSION)

    # 3. Retriever
    retriever = DocumentRetriever(
        embedder=embedder,
        vector_store=_vector_store,
    )

    # 4. LLM client
    llm_cfg = LLMConfig(
        model_name=settings.OLLAMA_MODEL,
        timeout=settings.OLLAMA_TIMEOUT,
        max_tokens=settings.OLLAMA_MAX_TOKENS,
        temperature=settings.OLLAMA_TEMPERATURE,
    )
    llm_client = OllamaClient(config=llm_cfg, base_url=settings.OLLAMA_BASE_URL)
    _llm_client = llm_client
    logger.info("LLM client ready: %s @ %s", settings.OLLAMA_MODEL, settings.OLLAMA_BASE_URL)

    # 5. RAGService
    rag_cfg = RAGConfig(
        top_k=settings.RAG_TOP_K,
        min_relevance=settings.RAG_MIN_RELEVANCE,
        max_context_length=settings.RAG_MAX_CONTEXT_LENGTH,
        include_sources=settings.RAG_INCLUDE_SOURCES,
        strict_mode=settings.RAG_STRICT_MODE,
    )
    security_cfg = SecurityConfig() if settings.RAG_ENABLE_SECURITY else None
    _rag_service = RAGService(
        retriever=retriever,
        llm_client=llm_client,
        config=rag_cfg,
        security_config=security_cfg,
    )
    logger.info("RAGService ready")

    # 6. Ingestion pipeline
    chunking_cfg = ChunkingConfig(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separator=settings.CHUNK_SEPARATOR,
    )
    chunker = TextChunker(config=chunking_cfg)
    doc_processor = DocumentProcessor(
        embedder=embedder,
        vector_store=_vector_store,
        chunker=chunker,
    )
    _ingestion_pipeline = IngestionPipeline(processor=doc_processor)
    logger.info("IngestionPipeline ready")
    logger.info("All components initialized successfully")
