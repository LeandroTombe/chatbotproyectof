"""Tests for RAG service."""
import pytest
from unittest.mock import Mock, MagicMock
from chat.rag_service import RAGService, RAGConfig
from chat.models import Message, MessageRole, SourceDocument
from chat.llm_clients.base import LLMConfig
from domain.models import Chunk, SearchResult


class TestRAGConfig:
    """Tests for RAGConfig."""
    
    def test_default_config(self):
        """Test default RAG configuration."""
        config = RAGConfig()
        
        assert config.top_k == 3
        assert config.min_relevance == 0.3
        assert config.max_context_length == 2000
        assert config.include_sources is True
        # system_prompt defaults to empty string; effective prompt comes from settings
        assert config.system_prompt == ""
    
    def test_custom_config(self):
        """Test custom RAG configuration."""
        config = RAGConfig(
            top_k=5,
            min_relevance=0.5,
            max_context_length=3000,
            include_sources=False,
            system_prompt="Custom prompt"
        )
        
        assert config.top_k == 5
        assert config.min_relevance == 0.5
        assert config.max_context_length == 3000
        assert config.include_sources is False
        assert config.system_prompt == "Custom prompt"


class TestRAGService:
    """Tests for RAGService."""
    
    @pytest.fixture
    def mock_retriever(self):
        """Create a mock document retriever."""
        return Mock()
    
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = Mock()
        client.config = LLMConfig(model_name="llama2")
        return client
    
    @pytest.fixture
    def rag_service(self, mock_retriever, mock_llm_client):
        """Create a RAG service instance."""
        return RAGService(
            retriever=mock_retriever,
            llm_client=mock_llm_client
        )
    
    def test_initialization(self, mock_retriever, mock_llm_client):
        """Test RAG service initialization."""
        service = RAGService(mock_retriever, mock_llm_client)
        
        assert service.retriever == mock_retriever
        assert service.llm_client == mock_llm_client
        assert isinstance(service.config, RAGConfig)
        
        # Should have system prompt in conversation
        messages = service.conversation_history.get_messages()
        assert len(messages) == 1
        assert messages[0].role == MessageRole.SYSTEM
    
    def test_initialization_custom_config(self, mock_retriever, mock_llm_client):
        """Test initialization with custom config."""
        config = RAGConfig(top_k=5, system_prompt="Custom")
        service = RAGService(mock_retriever, mock_llm_client, config)
        
        assert service.config.top_k == 5
        
        messages = service.conversation_history.get_messages()
        assert messages[0].content == "Custom"
    
    def test_initialization_no_system_prompt(self, mock_retriever, mock_llm_client):
        """Test initialization with empty system_prompt falls back to settings."""
        config = RAGConfig(system_prompt="")
        service = RAGService(mock_retriever, mock_llm_client, config)
        
        # Falls back to settings.RAG_SYSTEM_PROMPT (which is non-empty)
        messages = service.conversation_history.get_messages()
        assert len(messages) == 1
        assert messages[0].role == MessageRole.SYSTEM
    
    def test_chat_with_relevant_results(self, rag_service, mock_retriever, mock_llm_client):
        """Test chat with relevant search results."""
        # Mock search results
        chunk1 = Chunk(
            id="chunk1",
            document_id="doc1",
            content="La fotosíntesis es el proceso por el cual las plantas producen energía.",
            chunk_index=0,
            embedding=[0.1] * 10
        )
        chunk2 = Chunk(
            id="chunk2",
            document_id="doc2",
            content="Las plantas usan luz solar, agua y CO2 en la fotosíntesis.",
            chunk_index=0,
            embedding=[0.2] * 10
        )
        
        search_results = [
            SearchResult(chunk=chunk1, score=0.85, document_name="doc1.pdf"),
            SearchResult(chunk=chunk2, score=0.75, document_name="doc2.pdf")
        ]
        mock_retriever.search.return_value = search_results
        
        # Mock LLM response
        mock_llm_client.generate.return_value = (
            "La fotosíntesis es el proceso mediante el cual las plantas "
            "convierten la luz solar en energía química."
        )
        
        # Execute chat
        response = rag_service.chat("¿Qué es la fotosíntesis?")
        
        # Verify retriever was called
        mock_retriever.search.assert_called_once_with(
            query="¿Qué es la fotosíntesis?",
            top_k=3,
            where=None
        )
        
        # Verify LLM was called
        assert mock_llm_client.generate.called
        
        # Verify response
        assert "fotosíntesis" in response.content.lower()
        assert len(response.sources) == 2
        assert response.sources[0].document_id == "doc1"
        assert response.sources[1].document_id == "doc2"
        assert response.metadata["model"] == "llama2"
        assert response.metadata["has_context"] is True
    
    def test_chat_no_relevant_results(self, rag_service, mock_retriever, mock_llm_client):
        """Test chat when no relevant results are found."""
        # Mock search results with low relevance
        chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            content="Irrelevant text",
            chunk_index=0,
            embedding=[0.1] * 10
        )
        search_results = [
            SearchResult(chunk=chunk, score=0.2, document_name="doc1.pdf")  # Below min_relevance of 0.3
        ]
        mock_retriever.search.return_value = search_results
        
        mock_llm_client.generate.return_value = (
            "No tengo información sobre eso en los documentos."
        )
        
        response = rag_service.chat("¿Quién ganó el mundial 2022?")
        
        # Should have no sources (filtered by min_relevance)
        assert len(response.sources) == 0
        assert response.metadata["has_context"] is False
    
    def test_chat_with_document_filter(self, rag_service, mock_retriever, mock_llm_client):
        """Test chat with document ID filter."""
        mock_retriever.search.return_value = []
        mock_llm_client.generate.return_value = "Response"
        
        rag_service.chat("Query", document_ids=["doc1", "doc2"])
        
        # Verify document_ids were converted to where filter
        mock_retriever.search.assert_called_once_with(
            query="Query",
            top_k=3,
            where={"document_id": {"$in": ["doc1", "doc2"]}}
        )
    
    def test_chat_updates_conversation_history(self, rag_service, mock_retriever, mock_llm_client):
        """Test that chat updates conversation history."""
        mock_retriever.search.return_value = []
        mock_llm_client.generate.return_value = "Response"
        
        # Initial state: only system prompt
        initial_count = len(rag_service.conversation_history.get_messages())
        
        rag_service.chat("Hello")
        
        # Should add user message and assistant message
        messages = rag_service.conversation_history.get_messages()
        assert len(messages) == initial_count + 2
        
        # Last two should be user and assistant
        assert messages[-2].role == MessageRole.USER
        assert messages[-1].role == MessageRole.ASSISTANT
        assert messages[-1].content == "Response"
    
    def test_chat_multiple_turns(self, rag_service, mock_retriever, mock_llm_client):
        """Test multiple conversation turns."""
        mock_retriever.search.return_value = []
        mock_llm_client.generate.return_value = "Response"
        
        rag_service.chat("First question")
        rag_service.chat("Second question")
        rag_service.chat("Third question")
        
        # Should have system + 6 messages (3 user + 3 assistant)
        messages = rag_service.conversation_history.get_messages()
        assert len(messages) == 7  # 1 system + 6 conversation
    
    def test_build_context_single_result(self, rag_service):
        """Test building context from single search result."""
        chunk = Chunk(
            id="chunk1",
            document_id="doc1",
            content="This is the context text.",
            chunk_index=0,
            embedding=[0.1] * 10
        )
        results = [SearchResult(chunk=chunk, score=0.9, document_name="doc1.pdf")]
        
        context = rag_service._build_context(results)
        
        assert "Fuente 1" in context
        assert "0.90" in context
        assert "This is the context text." in context
    
    def test_build_context_multiple_results(self, rag_service):
        """Test building context from multiple results."""
        chunks = [
            Chunk(id="chunk1", document_id="doc1", content="Context 1", chunk_index=0, embedding=[0.1] * 10),
            Chunk(id="chunk2", document_id="doc2", content="Context 2", chunk_index=0, embedding=[0.2] * 10),
            Chunk(id="chunk3", document_id="doc3", content="Context 3", chunk_index=0, embedding=[0.3] * 10)
        ]
        results = [
            SearchResult(chunk=chunks[0], score=0.9, document_name="doc1.pdf"),
            SearchResult(chunk=chunks[1], score=0.8, document_name="doc2.pdf"),
            SearchResult(chunk=chunks[2], score=0.7, document_name="doc3.pdf")
        ]
        
        context = rag_service._build_context(results)
        
        assert "Fuente 1" in context
        assert "Fuente 2" in context
        assert "Fuente 3" in context
        assert "Context 1" in context
        assert "Context 2" in context
        assert "Context 3" in context
    
    def test_build_context_respects_max_length(self, rag_service):
        """Test that context respects max_context_length."""
        # Create chunk with long text
        long_text = "A" * 1000
        chunks = [
            Chunk(id="chunk1", document_id="doc1", content=long_text, chunk_index=0, embedding=[0.1] * 10),
            Chunk(id="chunk2", document_id="doc2", content=long_text, chunk_index=0, embedding=[0.2] * 10),
            Chunk(id="chunk3", document_id="doc3", content=long_text, chunk_index=0, embedding=[0.3] * 10)
        ]
        results = [SearchResult(chunk=c, score=0.9, document_name=f"doc{i}.pdf") for i, c in enumerate(chunks, 1)]
        
        # Set small max_context_length
        rag_service.config.max_context_length = 500
        
        context = rag_service._build_context(results)
        
        # Context should be truncated
        assert len(context) <= 500
    
    def test_build_context_empty_results(self, rag_service):
        """Test building context with no results."""
        context = rag_service._build_context([])
        
        assert context == ""
    
    def test_get_conversation_history(self, rag_service):
        """Test getting conversation history."""
        history = rag_service.get_conversation_history()
        
        assert history == rag_service.conversation_history
    
    def test_clear_conversation(self, rag_service, mock_retriever, mock_llm_client):
        """Test clearing conversation history."""
        mock_retriever.search.return_value = []
        mock_llm_client.generate.return_value = "Response"
        
        # Add some messages
        rag_service.chat("Hello")
        rag_service.chat("How are you?")
        
        # Should have multiple messages
        assert len(rag_service.conversation_history.get_messages()) > 1
        
        # Clear conversation
        rag_service.clear_conversation()
        
        # Should only have system message left
        messages = rag_service.conversation_history.get_messages()
        assert len(messages) == 1
        assert messages[0].role == MessageRole.SYSTEM
    
    def test_clear_conversation_preserves_system_prompt(self, rag_service):
        """Test that clear_conversation preserves system prompt."""
        original_system = rag_service.conversation_history.messages[0].content
        
        rag_service.clear_conversation()
        
        messages = rag_service.conversation_history.get_messages()
        assert len(messages) == 1
        assert messages[0].content == original_system
    
    def test_set_system_prompt(self, rag_service):
        """Test updating system prompt."""
        new_prompt = "New system prompt"
        
        rag_service.set_system_prompt(new_prompt)
        
        messages = rag_service.conversation_history.get_messages()
        assert messages[0].role == MessageRole.SYSTEM
        assert messages[0].content == new_prompt
        assert rag_service.config.system_prompt == new_prompt
    
    def test_set_system_prompt_removes_old(self, rag_service, mock_retriever, mock_llm_client):
        """Test that setting system prompt removes old system messages."""
        # Add a conversation
        mock_retriever.search.return_value = []
        mock_llm_client.generate.return_value = "Response"
        rag_service.chat("Hello")
        
        # Change system prompt
        rag_service.set_system_prompt("New prompt")
        
        # Should have only one system message at the beginning
        messages = rag_service.conversation_history.get_messages()
        system_messages = [m for m in messages if m.role == MessageRole.SYSTEM]
        assert len(system_messages) == 1
        assert system_messages[0].content == "New prompt"
        assert messages[0].role == MessageRole.SYSTEM
    
    def test_chat_without_sources(self, mock_retriever, mock_llm_client):
        """Test chat with include_sources=False."""
        config = RAGConfig(include_sources=False)
        service = RAGService(mock_retriever, mock_llm_client, config)
        
        # Mock search results
        chunk = Chunk(id="chunk1", document_id="doc1", content="Text", chunk_index=0, embedding=[0.1] * 10)
        mock_retriever.search.return_value = [
            SearchResult(chunk=chunk, score=0.9, document_name="doc1.pdf")
        ]
        mock_llm_client.generate.return_value = "Response"
        
        response = service.chat("Query")
        
        # Should have no sources
        assert len(response.sources) == 0
    
    def test_chat_passes_llm_kwargs(self, rag_service, mock_retriever, mock_llm_client):
        """Test that additional kwargs are passed to LLM."""
        mock_retriever.search.return_value = []
        mock_llm_client.generate.return_value = "Response"
        
        rag_service.chat("Query", temperature=0.9, max_tokens=1024)
        
        # Verify kwargs were passed to generate
        call_kwargs = mock_llm_client.generate.call_args[1]
        assert call_kwargs.get("temperature") == 0.9
        assert call_kwargs.get("max_tokens") == 1024
    
    def test_chat_streaming_not_implemented(self, rag_service):
        """Test that streaming is not yet implemented."""
        with pytest.raises(NotImplementedError):
            rag_service.chat_streaming("Query")
