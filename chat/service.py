"""
Chat service for RAG-based conversational AI.
Orchestrates retrieval and response generation.
"""
from typing import List, Optional, Dict, Any, Protocol
from dataclasses import dataclass, field
from datetime import datetime
import logging
import uuid

from retrieval.service import RetrievalService

logger = logging.getLogger(__name__)


class ChatException(Exception):
    """Exception raised for chat errors"""
    pass


@dataclass
class Message:
    """Representa un mensaje en la conversación"""
    id: str
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_user_message(self) -> bool:
        """Verifica si es un mensaje del usuario"""
        return self.role == "user"
    
    def is_assistant_message(self) -> bool:
        """Verifica si es un mensaje del asistente"""
        return self.role == "assistant"


@dataclass
class ChatResponse:
    """Respuesta del chat con contexto RAG"""
    message: Message
    context_used: str
    sources: List[Dict[str, Any]]
    retrieval_results_count: int
    
    def get_answer(self) -> str:
        """Obtiene el texto de la respuesta"""
        return self.message.content


@dataclass
class Conversation:
    """Representa una conversación completa"""
    id: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: Message) -> None:
        """Agrega un mensaje a la conversación"""
        self.messages.append(message)
    
    def get_history(self, limit: Optional[int] = None) -> List[Message]:
        """
        Obtiene el historial de mensajes.
        
        Args:
            limit: Número máximo de mensajes (más recientes)
            
        Returns:
            Lista de mensajes
        """
        if limit is None:
            return self.messages.copy()
        return self.messages[-limit:]
    
    def get_last_user_message(self) -> Optional[Message]:
        """Obtiene el último mensaje del usuario"""
        for msg in reversed(self.messages):
            if msg.is_user_message():
                return msg
        return None


class ResponseGenerator(Protocol):
    """Protocolo para generadores de respuestas"""
    
    def generate(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Message]] = None
    ) -> str:
        """
        Genera una respuesta basada en query y contexto.
        
        Args:
            query: Pregunta del usuario
            context: Contexto recuperado
            conversation_history: Historial de conversación
            
        Returns:
            Respuesta generada
        """
        ...


class SimpleResponseGenerator:
    """
    Generador de respuestas simple basado en templates.
    Para testing y desarrollo sin LLM.
    """
    
    def __init__(self, template: Optional[str] = None):
        """
        Inicializa el generador.
        
        Args:
            template: Template de respuesta. Soporta placeholders: {query}, {context}
        """
        self.template = template or (
            "Basándome en la información disponible:\n\n"
            "{context}\n\n"
            "En respuesta a: {query}\n\n"
            "Respuesta: Esta es una respuesta generada automáticamente. "
            "Para respuestas más elaboradas, integra un modelo de lenguaje."
        )
    
    def generate(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Message]] = None
    ) -> str:
        """
        Genera respuesta usando el template.
        
        Args:
            query: Pregunta del usuario
            context: Contexto recuperado
            conversation_history: Historial (ignorado en versión simple)
            
        Returns:
            Respuesta generada
        """
        if not context.strip():
            return (
                "No encontré información relevante para responder tu pregunta. "
                f"Pregunta: {query}"
            )
        
        return self.template.format(query=query, context=context)


class ChatService:
    """
    Servicio de chat RAG que coordina retrieval y generación de respuestas.
    """
    
    def __init__(
        self,
        retrieval_service: RetrievalService,
        response_generator: Optional[ResponseGenerator] = None,
        max_context_length: int = 2000,
        context_separator: str = "\n\n---\n\n",
        include_sources: bool = True
    ):
        """
        Inicializa el servicio de chat.
        
        Args:
            retrieval_service: Servicio de retrieval
            response_generator: Generador de respuestas (usa SimpleResponseGenerator si None)
            max_context_length: Longitud máxima del contexto en caracteres
            context_separator: Separador entre chunks de contexto
            include_sources: Si True, incluye fuentes en la respuesta
            
        Raises:
            ValueError: Si retrieval_service es None
        """
        if retrieval_service is None:
            raise ValueError("retrieval_service cannot be None")
        
        self.retrieval_service = retrieval_service
        self.response_generator = response_generator or SimpleResponseGenerator()
        self.max_context_length = max_context_length
        self.context_separator = context_separator
        self.include_sources = include_sources
        
        # Almacén de conversaciones activas
        self.conversations: Dict[str, Conversation] = {}
        
        logger.info(
            f"ChatService initialized with max_context_length={max_context_length}, "
            f"include_sources={include_sources}"
        )
    
    def create_conversation(self, conversation_id: Optional[str] = None) -> Conversation:
        """
        Crea una nueva conversación.
        
        Args:
            conversation_id: ID de la conversación (genera uno si None)
            
        Returns:
            Conversación creada
        """
        conv_id = conversation_id or str(uuid.uuid4())
        conversation = Conversation(id=conv_id)
        self.conversations[conv_id] = conversation
        
        logger.info(f"Created conversation {conv_id}")
        return conversation
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Obtiene una conversación por ID.
        
        Args:
            conversation_id: ID de la conversación
            
        Returns:
            Conversación si existe, None si no
        """
        return self.conversations.get(conversation_id)
    
    def chat(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> ChatResponse:
        """
        Procesa un mensaje del usuario y genera respuesta RAG.
        
        Args:
            user_message: Mensaje del usuario
            conversation_id: ID de conversación (crea una nueva si None)
            document_id: Filtrar búsqueda por documento específico
            
        Returns:
            ChatResponse con respuesta y metadata
            
        Raises:
            ChatException: Si hay error procesando el mensaje
            ValueError: Si user_message está vacío
        """
        if not user_message or not user_message.strip():
            raise ValueError("user_message cannot be empty")
        
        user_message = user_message.strip()
        
        # Obtener o crear conversación
        conversation = self._get_or_create_conversation(conversation_id)
        
        logger.info(
            f"Processing message in conversation {conversation.id}: "
            f"'{user_message[:50]}...'"
        )
        
        try:
            # Crear mensaje del usuario
            user_msg = Message(
                id=str(uuid.uuid4()),
                role="user",
                content=user_message
            )
            conversation.add_message(user_msg)
            
            # Recuperar contexto relevante
            context = self._retrieve_context(user_message, document_id)
            
            # Obtener fuentes si está habilitado
            sources = []
            retrieval_count = 0
            if self.include_sources:
                sources, retrieval_count = self._get_sources(user_message, document_id)
            
            # Generar respuesta
            response_text = self._generate_response(
                query=user_message,
                context=context,
                conversation_history=conversation.get_history()
            )
            
            # Crear mensaje del asistente
            assistant_msg = Message(
                id=str(uuid.uuid4()),
                role="assistant",
                content=response_text,
                metadata={
                    "context_length": len(context),
                    "sources_count": len(sources)
                }
            )
            conversation.add_message(assistant_msg)
            
            # Crear respuesta
            chat_response = ChatResponse(
                message=assistant_msg,
                context_used=context,
                sources=sources,
                retrieval_results_count=retrieval_count
            )
            
            logger.info(
                f"Generated response in conversation {conversation.id}: "
                f"{len(response_text)} chars, {len(sources)} sources"
            )
            
            return chat_response
            
        except Exception as e:
            error_msg = f"Error processing chat message: {str(e)}"
            logger.error(error_msg)
            raise ChatException(error_msg) from e
    
    def chat_simple(self, user_message: str) -> str:
        """
        Versión simplificada que retorna solo el texto de respuesta.
        
        Args:
            user_message: Mensaje del usuario
            
        Returns:
            Texto de la respuesta
            
        Raises:
            ChatException: Si hay error procesando el mensaje
        """
        response = self.chat(user_message)
        return response.get_answer()
    
    def _get_or_create_conversation(
        self,
        conversation_id: Optional[str]
    ) -> Conversation:
        """
        Obtiene conversación existente o crea una nueva.
        
        Args:
            conversation_id: ID de conversación
            
        Returns:
            Conversación
        """
        if conversation_id is None:
            return self.create_conversation()
        
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            return self.create_conversation(conversation_id)
        
        return conversation
    
    def _retrieve_context(
        self,
        query: str,
        document_id: Optional[str] = None
    ) -> str:
        """
        Recupera contexto relevante usando el servicio de retrieval.
        
        Args:
            query: Query del usuario
            document_id: Filtrar por documento
            
        Returns:
            Contexto concatenado
            
        Raises:
            ChatException: Si hay error en retrieval
        """
        try:
            context = self.retrieval_service.retrieve_context(
                query_text=query,
                max_context_length=self.max_context_length,
                separator=self.context_separator
            )
            
            logger.debug(f"Retrieved context: {len(context)} chars")
            return context
            
        except Exception as e:
            error_msg = f"Error retrieving context: {str(e)}"
            logger.error(error_msg)
            raise ChatException(error_msg) from e
    
    def _get_sources(
        self,
        query: str,
        document_id: Optional[str] = None
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Obtiene las fuentes de información.
        
        Args:
            query: Query del usuario
            document_id: Filtrar por documento
            
        Returns:
            Tupla de (lista de fuentes, número de resultados)
        """
        try:
            results = self.retrieval_service.retrieve_with_metadata(query)
            
            sources = [
                {
                    "document_name": r["document_name"],
                    "document_id": r["document_id"],
                    "chunk_id": r["chunk_id"],
                    "score": r["score"],
                    "preview": r["content"][:100] + "..." if len(r["content"]) > 100 else r["content"]
                }
                for r in results
            ]
            
            return sources, len(results)
            
        except Exception as e:
            logger.warning(f"Error getting sources: {str(e)}")
            return [], 0
    
    def _generate_response(
        self,
        query: str,
        context: str,
        conversation_history: List[Message]
    ) -> str:
        """
        Genera respuesta usando el generador.
        
        Args:
            query: Query del usuario
            context: Contexto recuperado
            conversation_history: Historial de conversación
            
        Returns:
            Respuesta generada
            
        Raises:
            ChatException: Si hay error generando respuesta
        """
        try:
            response = self.response_generator.generate(
                query=query,
                context=context,
                conversation_history=conversation_history
            )
            
            logger.debug(f"Generated response: {len(response)} chars")
            return response
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            raise ChatException(error_msg) from e
    
    def get_conversation_count(self) -> int:
        """
        Retorna el número de conversaciones activas.
        
        Returns:
            Número de conversaciones
        """
        return len(self.conversations)
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """
        Elimina una conversación.
        
        Args:
            conversation_id: ID de la conversación
            
        Returns:
            True si se eliminó, False si no existía
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Cleared conversation {conversation_id}")
            return True
        return False
    
    def clear_all_conversations(self) -> int:
        """
        Elimina todas las conversaciones.
        
        Returns:
            Número de conversaciones eliminadas
        """
        count = len(self.conversations)
        self.conversations.clear()
        logger.info(f"Cleared all conversations ({count} total)")
        return count
