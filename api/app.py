"""
FastAPI application factory.

Creates and configures the FastAPI app with:
- CORS middleware (allows React frontend on any localhost port)
- Lifespan startup/shutdown for component initialization
- Chat and document routes
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import initialize_components
from api.routes.chat import router as chat_router
from api.routes.documents import router as documents_router

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CORS — origins allowed to call this API
# ---------------------------------------------------------------------------

# Add here every origin your React dev/prod server may use.
# The REACT_ORIGIN env var can override this at runtime.
ALLOWED_ORIGINS = [
    "http://localhost:3000",   # Create React App default
    "http://localhost:5173",   # Vite default
    "http://localhost:4173",   # Vite preview
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:4173",
]


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize heavy components on startup; clean up on shutdown."""
    logger.info("Starting up — initializing RAG components...")
    initialize_components()
    logger.info("Startup complete. API is ready.")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and return the configured FastAPI application."""
    app = FastAPI(
        title="ChatBot RAG API",
        description=(
            "REST API for a Retrieval-Augmented Generation chatbot. "
            "Upload PDF documents and ask questions about their content."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    app.include_router(chat_router)
    app.include_router(documents_router)

    # ------------------------------------------------------------------
    # Health check (root)
    # ------------------------------------------------------------------
    @app.get("/health", tags=["health"], summary="Health check")
    def health() -> dict:
        return {"status": "ok", "service": "chatbot-rag"}

    return app
