"""Base interface for LLM clients."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from chat.models import Message


@dataclass
class LLMConfig:
    """Configuration for LLM client.
    
    Attributes:
        model_name: Name of the model to use
        temperature: Sampling temperature (0-1)
        max_tokens: Maximum tokens in response
        top_p: Nucleus sampling parameter
        timeout: Request timeout in seconds
        additional_params: Additional model-specific parameters
    """
    model_name: str = "llama2"
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9
    timeout: int = 60
    additional_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


class LLMException(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMConnectionError(LLMException):
    """Exception raised when connection to LLM fails."""
    pass


class LLMResponseError(LLMException):
    """Exception raised when LLM returns an error."""
    pass


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients.
    
    Provides interface for interacting with language models.
    Implementations should handle:
    - Model initialization
    - Message formatting
    - API communication
    - Error handling
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize the LLM client.
        
        Args:
            config: Configuration for the LLM
        """
        self.config = config
    
    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        **kwargs
    ) -> str:
        """Generate a response from the LLM.
        
        Args:
            messages: Conversation history
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
            
        Raises:
            LLMConnectionError: If connection to LLM fails
            LLMResponseError: If LLM returns an error
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM is available and responsive.
        
        Returns:
            True if LLM is available, False otherwise
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.
        
        Returns:
            Dictionary containing model metadata
        """
        pass
    
    def format_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Format messages for the LLM API.
        
        Args:
            messages: List of Message objects
            
        Returns:
            List of formatted message dictionaries
        """
        return [
            {
                "role": msg.role.value,
                "content": msg.content
            }
            for msg in messages
        ]
