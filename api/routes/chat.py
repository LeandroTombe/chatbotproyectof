"""
Chat routes — REST endpoints for the RAG chatbot.

Endpoints
---------
POST /api/chat          — Send a message and get an AI response
GET  /api/chat/history  — Retrieve the current conversation history
DELETE /api/chat/history — Clear the conversation history
GET  /api/health        — Health check
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.dependencies import get_rag_service, get_llm_client
from chat.rag_service import RAGService
from chat.llm_clients.ollama_client import OllamaClient
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Welcome message shown when the frontend opens the chat
WELCOME_MESSAGE = (
    "¡Hola! Soy el asistente virtual del Proyecto Sherlock. "
    "¿En qué puedo ayudarte hoy?"
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Request body for POST /api/chat."""
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of document IDs to restrict context retrieval",
    )
    case_context: Optional[str] = Field(
        default=None,
        description=(
            "Structured metadata about the case (investigators, file list, etc.) "
            "that is always injected into the prompt so the LLM can answer "
            "administrative questions without relying on vector search."
        ),
    )

    model_config = {"json_schema_extra": {"example": {"message": "¿Quiénes son los investigadores del caso?"}}}


class SourceDocumentResponse(BaseModel):
    """A single source document included in the response."""
    document_id: str
    chunk_index: int
    content: str
    relevance_score: float
    metadata: dict = Field(default_factory=dict)


class ChatMessageResponse(BaseModel):
    """Response body for POST /api/chat."""
    reply: str
    sources: List[SourceDocumentResponse] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class HistoryMessageResponse(BaseModel):
    """A single message in the conversation history."""
    role: str
    content: str
    timestamp: str


class StartResponse(BaseModel):
    """Response for GET /api/chat/start."""
    welcome: str
    model: str
    ollama_available: bool
    ready: bool


class StatusResponse(BaseModel):
    """Response for GET /api/status."""
    ollama_available: bool
    model: str
    ollama_url: str
    ready: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=ChatMessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a message to the chatbot",
)
def chat(
    body: ChatRequest,
    rag: RAGService = Depends(get_rag_service),
) -> ChatMessageResponse:
    """Send *message* to the RAG chatbot and receive an AI-generated answer
    grounded in the indexed documents.
    """
    try:
        response = rag.chat(
            query=body.message,
            document_ids=body.document_ids,
            case_context=body.case_context,
        )
    except Exception as exc:
        logger.exception("Error generating chat response")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating response: {exc}",
        ) from exc

    sources = [
        SourceDocumentResponse(
            document_id=src.document_id,
            chunk_index=src.chunk_index,
            content=src.content,
            relevance_score=src.relevance_score,
            metadata=src.metadata,
        )
        for src in response.sources
    ]

    return ChatMessageResponse(
        reply=response.content,
        sources=sources,
        metadata=response.metadata,
    )


@router.get(
    "/history",
    response_model=List[HistoryMessageResponse],
    summary="Get current conversation history",
)
def get_history(
    rag: RAGService = Depends(get_rag_service),
) -> List[HistoryMessageResponse]:
    """Return all messages in the current conversation session."""
    history = rag.get_conversation_history()
    return [
        HistoryMessageResponse(
            role=msg.role.value,
            content=msg.content,
            timestamp=msg.timestamp.isoformat(),
        )
        for msg in history.messages
    ]


@router.delete(
    "/history",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Clear conversation history",
)
def clear_history(
    rag: RAGService = Depends(get_rag_service),
) -> None:
    """Clear the current conversation history (system prompt is preserved)."""
    rag.clear_conversation()


@router.get(
    "/start",
    response_model=StartResponse,
    summary="Initialize chat session — call this when the widget opens",
)
def start_chat(
    llm: OllamaClient = Depends(get_llm_client),
    rag: RAGService = Depends(get_rag_service),
) -> StartResponse:
    """Returns the welcome greeting and confirms Ollama is reachable.
    Call this endpoint when the frontend chat widget first opens so the UI
    can display the initial greeting and know the model is ready.
    """
    ollama_ok = llm.is_available()

    # Reset conversation so every new session starts clean
    rag.clear_conversation()

    return StartResponse(
        welcome=WELCOME_MESSAGE,
        model=settings.OLLAMA_MODEL,
        ollama_available=ollama_ok,
        ready=ollama_ok,
    )


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Check Ollama and vector store status",
)
def get_status(
    llm: OllamaClient = Depends(get_llm_client),
) -> StatusResponse:
    """Quick health check: reports whether Ollama is reachable and which model
    is configured.
    """
    ollama_ok = llm.is_available()
    return StatusResponse(
        ollama_available=ollama_ok,
        model=settings.OLLAMA_MODEL,
        ollama_url=settings.OLLAMA_BASE_URL,
        ready=ollama_ok,
    )
