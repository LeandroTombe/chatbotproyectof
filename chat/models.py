"""Domain models for chat conversations."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from enum import Enum


class MessageRole(str, Enum):
    """Role of a message in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A single message in a conversation.
    
    Attributes:
        role: The role of the message sender (system, user, assistant)
        content: The text content of the message
        timestamp: When the message was created
    """
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """Convert message to dictionary format."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class SourceDocument:
    """Reference to a source document used in generating a response.
    
    Attributes:
        document_id: ID of the source document
        chunk_index: Index of the chunk within the document
        content: The actual text content from the source
        relevance_score: Similarity score (0-1)
        metadata: Additional document metadata
    """
    document_id: str
    chunk_index: int
    content: str
    relevance_score: float
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert source document to dictionary format."""
        return {
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "relevance_score": self.relevance_score,
            "metadata": self.metadata
        }


@dataclass
class ChatResponse:
    """Response from the chat service.
    
    Attributes:
        content: The generated response text
        sources: List of source documents used
        metadata: Additional response metadata (model, tokens, etc.)
        timestamp: When the response was generated
    """
    content: str
    sources: List[SourceDocument] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """Convert response to dictionary format."""
        return {
            "content": self.content,
            "sources": [s.to_dict() for s in self.sources],
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ConversationHistory:
    """Manages conversation history and context.
    
    Attributes:
        messages: List of messages in chronological order
        max_messages: Maximum number of messages to retain
        conversation_id: Unique identifier for this conversation
    """
    messages: List[Message] = field(default_factory=list)
    max_messages: int = 10
    conversation_id: str = ""
    
    def add_message(self, role: MessageRole, content: str) -> Message:
        """Add a message to the conversation history.
        
        Args:
            role: Role of the message sender
            content: Message content
            
        Returns:
            The created Message object
        """
        message = Message(role=role, content=content)
        self.messages.append(message)
        
        # Trim history if it exceeds max_messages
        if len(self.messages) > self.max_messages:
            # Keep system messages and trim oldest user/assistant messages
            system_messages = [m for m in self.messages if m.role == MessageRole.SYSTEM]
            other_messages = [m for m in self.messages if m.role != MessageRole.SYSTEM]
            
            # Keep only the most recent messages
            messages_to_keep = self.max_messages - len(system_messages)
            other_messages = other_messages[-messages_to_keep:]
            
            self.messages = system_messages + other_messages
        
        return message
    
    def get_messages(self, include_system: bool = True) -> List[Message]:
        """Get all messages in the conversation.
        
        Args:
            include_system: Whether to include system messages
            
        Returns:
            List of messages
        """
        if include_system:
            return self.messages.copy()
        return [m for m in self.messages if m.role != MessageRole.SYSTEM]
    
    def get_context_messages(self, max_tokens: Optional[int] = None) -> List[Message]:
        """Get messages for context, optionally limited by token count.
        
        Args:
            max_tokens: Maximum number of tokens (approximate)
            
        Returns:
            List of messages to include in context
        """
        if max_tokens is None:
            return self.get_messages()
        
        # Simple token estimation: ~4 chars per token
        messages = []
        total_chars = 0
        max_chars = max_tokens * 4
        
        # Add messages in reverse order (most recent first)
        for message in reversed(self.messages):
            message_chars = len(message.content)
            if total_chars + message_chars > max_chars and messages:
                break
            messages.insert(0, message)
            total_chars += message_chars
        
        return messages
    
    def clear(self) -> None:
        """Clear all messages from the conversation."""
        self.messages.clear()
    
    def to_dict(self) -> dict:
        """Convert conversation history to dictionary format."""
        return {
            "conversation_id": self.conversation_id,
            "messages": [m.to_dict() for m in self.messages],
            "max_messages": self.max_messages
        }
