"""Ollama LLM client implementation."""
import json
import requests
from typing import List, Dict, Any, Optional, Callable
from chat.llm_clients.base import BaseLLMClient, LLMConfig, LLMConnectionError, LLMResponseError
from chat.models import Message


class OllamaClient(BaseLLMClient):
    """Client for interacting with Ollama LLM.
    
    Ollama runs models locally via HTTP API.
    Default endpoint: http://localhost:11434
    
    Example usage:
        config = LLMConfig(model_name="llama2", temperature=0.7)
        client = OllamaClient(config, base_url="http://localhost:11434")
        
        messages = [Message(role=MessageRole.USER, content="Hello!")]
        response = client.generate(messages)
    """
    
    def __init__(
        self,
        config: LLMConfig,
        base_url: str = "http://localhost:11434"
    ):
        """Initialize Ollama client.
        
        Args:
            config: LLM configuration
            base_url: Base URL for Ollama API
        """
        super().__init__(config)
        self.base_url = base_url.rstrip('/')
        self.chat_endpoint = f"{self.base_url}/api/chat"
        self.tags_endpoint = f"{self.base_url}/api/tags"
    
    def generate(
        self,
        messages: List[Message],
        **kwargs
    ) -> str:
        """Generate a response using Ollama.
        
        Args:
            messages: Conversation history
            **kwargs: Additional parameters (stream, format, etc.)
            
        Returns:
            Generated text response
            
        Raises:
            LLMConnectionError: If cannot connect to Ollama
            LLMResponseError: If Ollama returns an error
        """
        # Format messages for Ollama API
        formatted_messages = self.format_messages(messages)
        
        # Prepare request payload
        payload = {
            "model": self.config.model_name,
            "messages": formatted_messages,
            "stream": kwargs.get("stream", False),
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                "top_p": kwargs.get("top_p", self.config.top_p),
                **(self.config.additional_params or {})
            }
        }
        
        # Add format if specified (e.g., "json")
        if "format" in kwargs:
            payload["format"] = kwargs["format"]
        
        try:
            response = requests.post(
                self.chat_endpoint,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract message content from response
            if "message" in result and "content" in result["message"]:
                return result["message"]["content"]
            else:
                raise LLMResponseError(
                    f"Unexpected response format: {result}"
                )
                
        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Ensure Ollama is running. Error: {e}"
            )
        except requests.exceptions.Timeout as e:
            raise LLMConnectionError(
                f"Request to Ollama timed out after {self.config.timeout}s. "
                f"Error: {e}"
            )
        except requests.exceptions.HTTPError as e:
            raise LLMResponseError(
                f"Ollama API returned error: {e}. "
                f"Response: {e.response.text if e.response else 'N/A'}"
            )
        except Exception as e:
            raise LLMResponseError(f"Unexpected error: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama is running and responsive.
        
        Returns:
            True if Ollama is available, False otherwise
        """
        try:
            response = requests.get(self.tags_endpoint, timeout=5)
            return response.status_code == 200
        except (requests.exceptions.RequestException, OSError):
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models.
        
        Returns:
            Dictionary containing model information
            
        Raises:
            LLMConnectionError: If cannot connect to Ollama
        """
        try:
            response = requests.get(self.tags_endpoint, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Find info for current model
            models = data.get("models", [])
            current_model = next(
                (m for m in models if m.get("name", "").startswith(self.config.model_name)),
                None
            )
            
            return {
                "model_name": self.config.model_name,
                "available": current_model is not None,
                "all_models": [m.get("name") for m in models],
                "all_models_detail": [
                    {
                        "name": m.get("name", ""),
                        "size": m.get("size", 0),
                    }
                    for m in models
                ],
                "model_details": current_model,
                "base_url": self.base_url
            }
            
        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Ensure Ollama is running. Error: {e}"
            )
        except Exception as e:
            raise LLMConnectionError(f"Error getting model info: {e}")
    
    def list_models(self) -> List[str]:
        """List all available models in Ollama.
        
        Returns:
            List of model names
            
        Raises:
            LLMConnectionError: If cannot connect to Ollama
        """
        info = self.get_model_info()
        return info.get("all_models", [])
    
    def pull_model(
        self,
        model_name: Optional[str] = None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ) -> bool:
        """Pull/download a model from Ollama library.

        Streams the download so callers receive live progress via ``on_progress``.

        Args:
            model_name:   Name of model to pull (defaults to configured model)
            on_progress:  Optional callback(status, completed_bytes, total_bytes)
                          called on every progress event from the API.

        Returns:
            True if successful

        Raises:
            LLMConnectionError: If cannot connect to Ollama
            LLMResponseError:   If pull fails
        """
        model = model_name or self.config.model_name
        pull_endpoint = f"{self.base_url}/api/pull"

        try:
            success_received = False
            # stream=True so we get progress events as NDJSON lines
            # timeout=(connect_timeout, read_timeout)
            # 30 s to connect; 7 200 s (2 h) read — large models can be several GB
            with requests.post(
                pull_endpoint,
                json={"name": model, "stream": True},
                stream=True,
                timeout=(30, 7200),
            ) as response:
                response.raise_for_status()

                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        event = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    status    = event.get("status", "")
                    completed = int(event.get("completed", 0) or 0)
                    total     = int(event.get("total", 0) or 0)

                    if on_progress:
                        on_progress(status, completed, total)

                    if status == "success":
                        success_received = True

            return success_received

        except requests.exceptions.ConnectionError as e:
            raise LLMConnectionError(
                f"Failed to connect to Ollama at {self.base_url}. Error: {e}"
            )
        except requests.exceptions.HTTPError as e:
            raise LLMResponseError(
                f"Failed to pull model '{model}': {e}. "
                f"Response: {e.response.text if e.response else 'N/A'}"
            )
