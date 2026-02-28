"""RAG (Retrieval-Augmented Generation) service for document-based chat."""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from chat.models import (
    Message, MessageRole, ChatResponse, 
    ConversationHistory, SourceDocument
)
from chat.llm_clients.base import BaseLLMClient
from chat.security import QueryValidator, SecurityConfig
from config.settings import settings
from retrieval.retriever import DocumentRetriever


@dataclass
class RAGConfig:
    """Configuration for RAG service.
    
    Attributes:
        top_k: Number of documents to retrieve for context
        min_relevance: Minimum relevance score (0-1) for retrieved docs
        max_context_length: Maximum characters in context
        include_sources: Whether to include source citations
        system_prompt: System prompt for the LLM
        context_template: Template for formatting retrieved context
        strict_mode: If True, refuses to answer without relevant documents
    """
    top_k: int = 3
    min_relevance: float = 0.3
    max_context_length: int = 2000
    include_sources: bool = True
    strict_mode: bool = True
    system_prompt: str = ""  # If empty, RAGService falls back to settings.RAG_SYSTEM_PROMPT
    context_template: str = (
        "Contexto de los documentos:\n\n{context}\n\n"
        "Basándote EXCLUSIVAMENTE en el contexto anterior, responde la siguiente pregunta:"
    )
    no_context_message: str = (
        "NO SE ENCONTRÓ INFORMACIÓN RELEVANTE EN LOS DOCUMENTOS.\n\n"
        "Pregunta del usuario: {query}\n\n"
        "Instrucción: Responde que no tienes esa información en los documentos disponibles."
    )


class RAGService:
    """Service for Retrieval-Augmented Generation.
    
    Combines document retrieval with LLM generation to provide
    contextually-grounded answers to user queries.
    
    Example usage:
        retriever = DocumentRetriever(embedder, vector_store)
        llm_client = OllamaClient(config)
        
        rag_service = RAGService(
            retriever=retriever,
            llm_client=llm_client
        )
        
        response = rag_service.chat("¿Qué es la fotosíntesis?")
        print(response.content)
        print(f"Sources: {len(response.sources)}")
    """
    
    def __init__(
        self,
        retriever: DocumentRetriever,
        llm_client: BaseLLMClient,
        config: Optional[RAGConfig] = None,
        security_config: Optional[SecurityConfig] = None
    ):
        """Initialize RAG service.
        
        Args:
            retriever: Document retriever for finding relevant context
            llm_client: LLM client for generating responses
            config: RAG configuration (uses defaults if not provided)
            security_config: Security configuration (uses defaults if not provided)
        """
        self.retriever = retriever
        self.llm_client = llm_client
        self.config = config or RAGConfig()
        self.conversation_history = ConversationHistory()
        self.validator = QueryValidator(security_config or SecurityConfig())
        
        # Use the configured system prompt; fall back to the global settings value
        effective_prompt = self.config.system_prompt or settings.RAG_SYSTEM_PROMPT
        if effective_prompt:
            self.conversation_history.add_message(
                MessageRole.SYSTEM,
                effective_prompt
            )
    
    def chat(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        **llm_kwargs
    ) -> ChatResponse:
        """Have a conversation with RAG-enhanced responses.
        
        Args:
            query: User's question or message
            document_ids: Optional list of document IDs to search within
            **llm_kwargs: Additional parameters for LLM generation
            
        Returns:
            ChatResponse with answer and sources
        """
        # 0. Security validation
        is_allowed, rejection_message = self.validator.validate(query)
        if not is_allowed:
            # Return rejection response immediately
            return ChatResponse(
                content=rejection_message or "Query blocked by security filter.",
                sources=[],
                metadata={
                    "security_blocked": True,
                    "query": query[:100]  # Truncate for privacy
                }
            )
        
        # 1. Retrieve relevant documents
        # Build where filter if document_ids provided
        where = None
        if document_ids:
            where = {"document_id": {"$in": document_ids}}
        
        search_results = self.retriever.search(
            query=query,
            top_k=self.config.top_k,
            where=where
        )
        
        # Filter by minimum relevance
        relevant_results = [
            r for r in search_results 
            if r.score >= self.config.min_relevance
        ]
        
        # 2. Build context from retrieved documents
        context = self._build_context(relevant_results)
        
        # 3. Format the prompt with context
        if context:
            contextualized_query = self.config.context_template.format(
                context=context
            ) + f"\n\nPregunta: {query}"
        else:
            # No relevant context found
            if self.config.strict_mode:
                # In strict mode, explicitly tell LLM there's no information
                contextualized_query = self.config.no_context_message.format(
                    query=query
                )
            else:
                # In non-strict mode, allow LLM to use general knowledge
                contextualized_query = query
        
        # 4. Add user message to history
        self.conversation_history.add_message(
            MessageRole.USER,
            contextualized_query
        )
        
        # 5. Generate response using LLM
        messages = self.conversation_history.get_context_messages()
        llm_response = self.llm_client.generate(messages, **llm_kwargs)
        
        # 6. Add assistant response to history
        self.conversation_history.add_message(
            MessageRole.ASSISTANT,
            llm_response
        )
        
        # 7. Build source documents
        sources = []
        if self.config.include_sources:
            sources = [
                SourceDocument(
                    document_id=result.chunk.document_id,
                    chunk_index=result.chunk.chunk_index,
                    content=result.chunk.content[:200] + "...",  # Truncate for brevity
                    relevance_score=result.score,
                    metadata=result.chunk.metadata
                )
                for result in relevant_results
            ]
        
        # 8. Build metadata
        metadata = {
            "model": self.llm_client.config.model_name,
            "num_sources": len(sources),
            "has_context": bool(context),
            "query": query
        }
        
        return ChatResponse(
            content=llm_response,
            sources=sources,
            metadata=metadata
        )
    
    def chat_streaming(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        **llm_kwargs
    ):
        """Chat with streaming response (for future implementation).
        
        Note: Requires streaming support in LLM client.
        """
        raise NotImplementedError("Streaming not yet implemented")
    
    def _build_context(self, search_results: List[Any]) -> str:
        """Build context string from search results.
        
        Args:
            search_results: List of SearchResult objects
            
        Returns:
            Formatted context string
        """
        if not search_results:
            return ""
        
        context_parts = []
        total_length = 0
        
        for i, result in enumerate(search_results, 1):
            # Format: [Source 1] (Score: 0.85)
            # <chunk text>
            source_header = f"[Fuente {i}] (Relevancia: {result.score:.2f})"
            chunk_text = result.chunk.content
            
            # Check if adding this would exceed max length
            addition_length = len(source_header) + len(chunk_text) + 4  # +4 for newlines
            if total_length + addition_length > self.config.max_context_length:
                # Truncate and stop
                remaining = self.config.max_context_length - total_length - len(source_header) - 4
                if remaining > 100:  # Only add if meaningful amount remains
                    chunk_text = chunk_text[:remaining] + "..."
                    context_parts.append(f"{source_header}\n{chunk_text}")
                break
            
            context_parts.append(f"{source_header}\n{chunk_text}")
            total_length += addition_length
        
        return "\n\n".join(context_parts)
    
    def get_conversation_history(self) -> ConversationHistory:
        """Get the current conversation history.
        
        Returns:
            ConversationHistory object
        """
        return self.conversation_history
    
    def clear_conversation(self) -> None:
        """Clear the conversation history (except system prompt)."""
        system_messages = [
            m for m in self.conversation_history.messages
            if m.role == MessageRole.SYSTEM
        ]
        self.conversation_history.clear()
        
        # Re-add system messages
        for msg in system_messages:
            self.conversation_history.messages.append(msg)
    
    def set_system_prompt(self, prompt: str) -> None:
        """Update the system prompt.
        
        Args:
            prompt: New system prompt
        """
        self.config.system_prompt = prompt
        
        # Update in conversation history
        # Remove old system messages
        self.conversation_history.messages = [
            m for m in self.conversation_history.messages
            if m.role != MessageRole.SYSTEM
        ]
        
        # Add new system prompt at the beginning
        if prompt:
            system_msg = Message(role=MessageRole.SYSTEM, content=prompt)
            self.conversation_history.messages.insert(0, system_msg)
    
    def get_sources_from_last_response(self) -> List[SourceDocument]:
        """Get source documents from the last response.
        
        Note: This is a convenience method. In production, you'd want
        to maintain a response history.
        
        Returns:
            List of source documents (empty if no last response)
        """
        # This is a simplified version - in production you'd store responses
        return []
