"""Tests for chat domain models."""
import pytest
from datetime import datetime
from chat.models import (
    Message, MessageRole, SourceDocument,
    ChatResponse, ConversationHistory
)


class TestMessage:
    """Tests for Message model."""
    
    def test_create_message(self):
        """Test creating a message."""
        msg = Message(
            role=MessageRole.USER,
            content="Hello, world!"
        )
        
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello, world!"
        assert isinstance(msg.timestamp, datetime)
    
    def test_message_roles(self):
        """Test different message roles."""
        system_msg = Message(MessageRole.SYSTEM, "You are helpful")
        user_msg = Message(MessageRole.USER, "Question")
        assistant_msg = Message(MessageRole.ASSISTANT, "Answer")
        
        assert system_msg.role == MessageRole.SYSTEM
        assert user_msg.role == MessageRole.USER
        assert assistant_msg.role == MessageRole.ASSISTANT
    
    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        msg = Message(
            role=MessageRole.USER,
            content="Test content"
        )
        
        msg_dict = msg.to_dict()
        
        assert msg_dict["role"] == "user"
        assert msg_dict["content"] == "Test content"
        assert "timestamp" in msg_dict


class TestSourceDocument:
    """Tests for SourceDocument model."""
    
    def test_create_source_document(self):
        """Test creating a source document."""
        source = SourceDocument(
            document_id="doc1",
            chunk_index=0,
            content="This is the source text",
            relevance_score=0.85
        )
        
        assert source.document_id == "doc1"
        assert source.chunk_index == 0
        assert source.content == "This is the source text"
        assert source.relevance_score == 0.85
        assert source.metadata == {}
    
    def test_source_document_with_metadata(self):
        """Test source document with metadata."""
        source = SourceDocument(
            document_id="doc1",
            chunk_index=0,
            content="Text",
            relevance_score=0.9,
            metadata={"page": 5, "author": "John"}
        )
        
        assert source.metadata["page"] == 5
        assert source.metadata["author"] == "John"
    
    def test_source_document_to_dict(self):
        """Test converting source document to dictionary."""
        source = SourceDocument(
            document_id="doc1",
            chunk_index=0,
            content="Text",
            relevance_score=0.75,
            metadata={"page": 1}
        )
        
        source_dict = source.to_dict()
        
        assert source_dict["document_id"] == "doc1"
        assert source_dict["chunk_index"] == 0
        assert source_dict["content"] == "Text"
        assert source_dict["relevance_score"] == 0.75
        assert source_dict["metadata"] == {"page": 1}


class TestChatResponse:
    """Tests for ChatResponse model."""
    
    def test_create_chat_response(self):
        """Test creating a chat response."""
        response = ChatResponse(
            content="This is the answer"
        )
        
        assert response.content == "This is the answer"
        assert response.sources == []
        assert response.metadata == {}
        assert isinstance(response.timestamp, datetime)
    
    def test_chat_response_with_sources(self):
        """Test chat response with source documents."""
        source1 = SourceDocument("doc1", 0, "Text 1", 0.9)
        source2 = SourceDocument("doc2", 1, "Text 2", 0.8)
        
        response = ChatResponse(
            content="Answer",
            sources=[source1, source2]
        )
        
        assert len(response.sources) == 2
        assert response.sources[0].document_id == "doc1"
        assert response.sources[1].document_id == "doc2"
    
    def test_chat_response_with_metadata(self):
        """Test chat response with metadata."""
        response = ChatResponse(
            content="Answer",
            metadata={"model": "llama2", "tokens": 50}
        )
        
        assert response.metadata["model"] == "llama2"
        assert response.metadata["tokens"] == 50
    
    def test_chat_response_to_dict(self):
        """Test converting chat response to dictionary."""
        source = SourceDocument("doc1", 0, "Text", 0.8)
        response = ChatResponse(
            content="Answer",
            sources=[source],
            metadata={"model": "llama2"}
        )
        
        response_dict = response.to_dict()
        
        assert response_dict["content"] == "Answer"
        assert len(response_dict["sources"]) == 1
        assert response_dict["metadata"]["model"] == "llama2"
        assert "timestamp" in response_dict


class TestConversationHistory:
    """Tests for ConversationHistory model."""
    
    def test_create_empty_history(self):
        """Test creating empty conversation history."""
        history = ConversationHistory()
        
        assert history.messages == []
        assert history.max_messages == 10
        assert history.conversation_id == ""
    
    def test_add_message(self):
        """Test adding a message to history."""
        history = ConversationHistory()
        
        msg = history.add_message(MessageRole.USER, "Hello")
        
        assert len(history.messages) == 1
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert history.messages[0] == msg
    
    def test_add_multiple_messages(self):
        """Test adding multiple messages."""
        history = ConversationHistory()
        
        history.add_message(MessageRole.SYSTEM, "You are helpful")
        history.add_message(MessageRole.USER, "Question")
        history.add_message(MessageRole.ASSISTANT, "Answer")
        
        assert len(history.messages) == 3
        assert history.messages[0].role == MessageRole.SYSTEM
        assert history.messages[1].role == MessageRole.USER
        assert history.messages[2].role == MessageRole.ASSISTANT
    
    def test_max_messages_limit(self):
        """Test that history respects max_messages limit."""
        history = ConversationHistory(max_messages=3)
        
        # Add more messages than the limit
        history.add_message(MessageRole.USER, "Msg 1")
        history.add_message(MessageRole.ASSISTANT, "Msg 2")
        history.add_message(MessageRole.USER, "Msg 3")
        history.add_message(MessageRole.ASSISTANT, "Msg 4")
        history.add_message(MessageRole.USER, "Msg 5")
        
        # Should only keep last 3 messages
        assert len(history.messages) == 3
        assert history.messages[0].content == "Msg 3"
        assert history.messages[1].content == "Msg 4"
        assert history.messages[2].content == "Msg 5"
    
    def test_max_messages_keeps_system_messages(self):
        """Test that system messages are preserved when trimming."""
        history = ConversationHistory(max_messages=4)
        
        history.add_message(MessageRole.SYSTEM, "System prompt")
        history.add_message(MessageRole.USER, "Msg 1")
        history.add_message(MessageRole.ASSISTANT, "Msg 2")
        history.add_message(MessageRole.USER, "Msg 3")
        history.add_message(MessageRole.ASSISTANT, "Msg 4")
        history.add_message(MessageRole.USER, "Msg 5")
        
        assert len(history.messages) == 4
        # System message should be first
        assert history.messages[0].role == MessageRole.SYSTEM
        assert history.messages[0].content == "System prompt"
        # Should have 3 most recent non-system messages
        assert history.messages[-1].content == "Msg 5"
    
    def test_get_messages(self):
        """Test getting all messages."""
        history = ConversationHistory()
        
        history.add_message(MessageRole.SYSTEM, "System")
        history.add_message(MessageRole.USER, "User")
        
        messages = history.get_messages()
        
        assert len(messages) == 2
        assert messages[0].content == "System"
        assert messages[1].content == "User"
    
    def test_get_messages_without_system(self):
        """Test getting messages excluding system messages."""
        history = ConversationHistory()
        
        history.add_message(MessageRole.SYSTEM, "System")
        history.add_message(MessageRole.USER, "User")
        history.add_message(MessageRole.ASSISTANT, "Assistant")
        
        messages = history.get_messages(include_system=False)
        
        assert len(messages) == 2
        assert all(m.role != MessageRole.SYSTEM for m in messages)
    
    def test_get_context_messages_no_limit(self):
        """Test getting context messages without token limit."""
        history = ConversationHistory()
        
        history.add_message(MessageRole.USER, "Hello")
        history.add_message(MessageRole.ASSISTANT, "Hi")
        
        context = history.get_context_messages()
        
        assert len(context) == 2
    
    def test_get_context_messages_with_token_limit(self):
        """Test getting context messages with token limit."""
        history = ConversationHistory()
        
        # Add messages with varying lengths
        history.add_message(MessageRole.USER, "A" * 100)  # ~25 tokens
        history.add_message(MessageRole.ASSISTANT, "B" * 100)  # ~25 tokens
        history.add_message(MessageRole.USER, "C" * 100)  # ~25 tokens
        
        # Request only ~50 tokens worth (should get last 2 messages)
        context = history.get_context_messages(max_tokens=50)
        
        assert len(context) <= 2
        # Should include most recent messages
        assert context[-1].content == "C" * 100
    
    def test_clear_history(self):
        """Test clearing conversation history."""
        history = ConversationHistory()
        
        history.add_message(MessageRole.USER, "Hello")
        history.add_message(MessageRole.ASSISTANT, "Hi")
        
        assert len(history.messages) == 2
        
        history.clear()
        
        assert len(history.messages) == 0
    
    def test_to_dict(self):
        """Test converting history to dictionary."""
        history = ConversationHistory(
            conversation_id="conv123",
            max_messages=5
        )
        
        history.add_message(MessageRole.USER, "Hello")
        
        history_dict = history.to_dict()
        
        assert history_dict["conversation_id"] == "conv123"
        assert history_dict["max_messages"] == 5
        assert len(history_dict["messages"]) == 1
        assert history_dict["messages"][0]["content"] == "Hello"
