"""Tests for Ollama LLM client."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from chat.llm_clients.ollama_client import OllamaClient
from chat.llm_clients.base import LLMConfig, LLMConnectionError, LLMResponseError
from chat.models import Message, MessageRole


class TestOllamaClient:
    """Tests for OllamaClient."""
    
    @pytest.fixture
    def config(self):
        """Create a test LLM configuration."""
        return LLMConfig(
            model_name="llama2",
            temperature=0.7,
            max_tokens=512
        )
    
    @pytest.fixture
    def client(self, config):
        """Create an Ollama client instance."""
        return OllamaClient(config)
    
    def test_initialization(self, client, config):
        """Test client initialization."""
        assert client.config == config
        assert client.base_url == "http://localhost:11434"
        assert client.chat_endpoint == "http://localhost:11434/api/chat"
        assert client.tags_endpoint == "http://localhost:11434/api/tags"
    
    def test_initialization_custom_url(self, config):
        """Test initialization with custom base URL."""
        client = OllamaClient(config, base_url="http://custom:8080")
        
        assert client.base_url == "http://custom:8080"
        assert client.chat_endpoint == "http://custom:8080/api/chat"
    
    def test_initialization_url_trailing_slash(self, config):
        """Test that trailing slash is removed from base URL."""
        client = OllamaClient(config, base_url="http://localhost:11434/")
        
        assert client.base_url == "http://localhost:11434"
    
    @patch('chat.llm_clients.ollama_client.requests.post')
    def test_generate_success(self, mock_post, client):
        """Test successful response generation."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "This is the generated response"
            },
            "done": True
        }
        mock_post.return_value = mock_response
        
        messages = [
            Message(MessageRole.USER, "What is AI?")
        ]
        
        result = client.generate(messages)
        
        assert result == "This is the generated response"
        
        # Verify the API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == client.chat_endpoint
        
        payload = call_args[1]['json']
        assert payload['model'] == "llama2"
        assert len(payload['messages']) == 1
        assert payload['messages'][0]['role'] == "user"
        assert payload['messages'][0]['content'] == "What is AI?"
    
    @patch('chat.llm_clients.ollama_client.requests.post')
    def test_generate_with_multiple_messages(self, mock_post, client):
        """Test generation with conversation history."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Response"},
            "done": True
        }
        mock_post.return_value = mock_response
        
        messages = [
            Message(MessageRole.SYSTEM, "You are helpful"),
            Message(MessageRole.USER, "Hello"),
            Message(MessageRole.ASSISTANT, "Hi"),
            Message(MessageRole.USER, "How are you?")
        ]
        
        result = client.generate(messages)
        
        assert result == "Response"
        
        payload = mock_post.call_args[1]['json']
        assert len(payload['messages']) == 4
    
    @patch('chat.llm_clients.ollama_client.requests.post')
    def test_generate_with_custom_parameters(self, mock_post, client):
        """Test generation with custom parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Response"},
            "done": True
        }
        mock_post.return_value = mock_response
        
        messages = [Message(MessageRole.USER, "Test")]
        
        client.generate(
            messages,
            temperature=0.9,
            max_tokens=1024,
            top_p=0.95
        )
        
        payload = mock_post.call_args[1]['json']
        assert payload['options']['temperature'] == 0.9
        assert payload['options']['num_predict'] == 1024
        assert payload['options']['top_p'] == 0.95
    
    @patch('chat.llm_clients.ollama_client.requests.post')
    def test_generate_connection_error(self, mock_post, client):
        """Test handling of connection errors."""
        mock_post.side_effect = Exception("Connection refused")
        
        messages = [Message(MessageRole.USER, "Test")]
        
        with pytest.raises(LLMResponseError):
            client.generate(messages)
    
    @patch('chat.llm_clients.ollama_client.requests.post')
    def test_generate_timeout(self, mock_post, client):
        """Test handling of timeout errors."""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout()
        
        messages = [Message(MessageRole.USER, "Test")]
        
        with pytest.raises(LLMConnectionError) as exc_info:
            client.generate(messages)
        
        assert "timed out" in str(exc_info.value).lower()
    
    @patch('chat.llm_clients.ollama_client.requests.post')
    def test_generate_http_error(self, mock_post, client):
        """Test handling of HTTP errors."""
        import requests
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_post.return_value = mock_response
        
        messages = [Message(MessageRole.USER, "Test")]
        
        with pytest.raises(LLMResponseError):
            client.generate(messages)
    
    @patch('chat.llm_clients.ollama_client.requests.post')
    def test_generate_unexpected_response_format(self, mock_post, client):
        """Test handling of unexpected response format."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "unexpected_key": "value"
        }
        mock_post.return_value = mock_response
        
        messages = [Message(MessageRole.USER, "Test")]
        
        with pytest.raises(LLMResponseError) as exc_info:
            client.generate(messages)
        
        assert "Unexpected response format" in str(exc_info.value)
    
    @patch('chat.llm_clients.ollama_client.requests.get')
    def test_is_available_true(self, mock_get, client):
        """Test is_available when Ollama is running."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        assert client.is_available() is True
    
    @patch('chat.llm_clients.ollama_client.requests.get')
    def test_is_available_false(self, mock_get, client):
        """Test is_available when Ollama is not running."""
        mock_get.side_effect = Exception("Connection refused")
        
        assert client.is_available() is False
    
    @patch('chat.llm_clients.ollama_client.requests.get')
    def test_get_model_info_success(self, mock_get, client):
        """Test getting model information."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2:latest", "size": 3800000000},
                {"name": "mistral:latest", "size": 4100000000}
            ]
        }
        mock_get.return_value = mock_response
        
        info = client.get_model_info()
        
        assert info["model_name"] == "llama2"
        assert info["available"] is True
        assert "llama2:latest" in info["all_models"]
        assert "mistral:latest" in info["all_models"]
        assert info["base_url"] == "http://localhost:11434"
    
    @patch('chat.llm_clients.ollama_client.requests.get')
    def test_get_model_info_model_not_available(self, mock_get, client):
        """Test get_model_info when requested model is not available."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "mistral:latest", "size": 4100000000}
            ]
        }
        mock_get.return_value = mock_response
        
        info = client.get_model_info()
        
        assert info["available"] is False
        assert info["model_details"] is None
    
    @patch('chat.llm_clients.ollama_client.requests.get')
    def test_get_model_info_connection_error(self, mock_get, client):
        """Test get_model_info when connection fails."""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        with pytest.raises(LLMConnectionError) as exc_info:
            client.get_model_info()
        
        assert "Failed to connect" in str(exc_info.value)
    
    @patch('chat.llm_clients.ollama_client.requests.get')
    def test_list_models(self, mock_get, client):
        """Test listing all available models."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2:latest"},
                {"name": "mistral:latest"},
                {"name": "codellama:latest"}
            ]
        }
        mock_get.return_value = mock_response
        
        models = client.list_models()
        
        assert len(models) == 3
        assert "llama2:latest" in models
        assert "mistral:latest" in models
        assert "codellama:latest" in models
    
    @patch('chat.llm_clients.ollama_client.requests.post')
    def test_pull_model_success(self, mock_post, client):
        """Test pulling a model successfully."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_post.return_value = mock_response
        
        result = client.pull_model("llama2")
        
        assert result is True
        
        # Verify API call
        call_args = mock_post.call_args
        assert "pull" in call_args[0][0]
        assert call_args[1]['json']['name'] == "llama2"
    
    @patch('chat.llm_clients.ollama_client.requests.post')
    def test_pull_model_default_model(self, mock_post, client):
        """Test pulling model without specifying name (uses default)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        client.pull_model()
        
        # Should use the configured model name
        payload = mock_post.call_args[1]['json']
        assert payload['name'] == client.config.model_name
    
    @patch('chat.llm_clients.ollama_client.requests.post')
    def test_pull_model_connection_error(self, mock_post, client):
        """Test pull_model when connection fails."""
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError()
        
        with pytest.raises(LLMConnectionError):
            client.pull_model()
    
    @patch('chat.llm_clients.ollama_client.requests.post')
    def test_pull_model_http_error(self, mock_post, client):
        """Test pull_model when HTTP error occurs."""
        import requests
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Model not found"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_post.return_value = mock_response
        
        with pytest.raises(LLMResponseError) as exc_info:
            client.pull_model("nonexistent")
        
        assert "Failed to pull model" in str(exc_info.value)
    
    def test_format_messages(self, client):
        """Test message formatting."""
        messages = [
            Message(MessageRole.SYSTEM, "System prompt"),
            Message(MessageRole.USER, "User message"),
            Message(MessageRole.ASSISTANT, "Assistant response")
        ]
        
        formatted = client.format_messages(messages)
        
        assert len(formatted) == 3
        assert formatted[0] == {"role": "system", "content": "System prompt"}
        assert formatted[1] == {"role": "user", "content": "User message"}
        assert formatted[2] == {"role": "assistant", "content": "Assistant response"}
