"""
Security filters and validators for RAG chat service.
Prevents unauthorized access to system information and prompt injection attempts.
"""
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import re


@dataclass
class SecurityConfig:
    """Configuration for security filters.
    
    Attributes:
        enabled: Whether security filtering is enabled
        blocked_patterns: List of regex patterns that trigger rejection
        blocked_keywords: List of keywords that trigger rejection (case-insensitive)
        rejection_message: Message to return when query is blocked
    """
    enabled: bool = True
    blocked_patterns: List[str] = field(default_factory=lambda: [
        r'\bsystem\s+prompt\b',
        r'\bignore\s+(previous|above|all)\b',
        r'\bpretend\s+you\s+are\b',
        r'\bactúa\s+como\b',
        r'\bolv[ií]date\s+de\b',
        r'\bDAN\b',  # "Do Anything Now" jailbreak
        r'\bjailbreak\b',
    ])
    blocked_keywords: List[str] = field(default_factory=lambda: [
        # Información del sistema (variaciones)
        'usuarios del sistema',
        'usuarios tiene',
        'cantidad de usuarios',
        'cuántos usuarios',
        'base de datos',
        'database',
        'contraseña',
        'password',
        'api key',
        'token',
        'secret',
        'configuración del sistema',
        'config del sistema',
        'variables de entorno',
        'environment variables',
        
        # Información de infraestructura
        'servidor',
        'server',
        'ip address',
        'dirección ip',
        'puerto',
        'port',
        'modelo de ia',
        'modelo estás usando',
        'qué modelo',
        'versión del modelo',
        'arquitectura del sistema',
        'información de la infraestructura',
        'infraestructura',
        
        # Prompt injection attempts
        'system prompt',
        'instrucciones del sistema',
        'eres un asistente',
        'ignore previous',
        'ignora las instrucciones',
        'olvídate de',
        'pretend you are',
        'actúa como si fueras',
        
        # Información privada
        'datos personales',
        'información confidencial',
        'credenciales',
        'credentials',
        'llm',
        'embedding',
        'vector store',
        'ollama config',
    ])
    rejection_message: str = (
        "Lo siento, no puedo responder preguntas sobre el sistema, "
        "infraestructura o información privada de la aplicación. "
        "Solo puedo ayudarte con preguntas sobre los documentos proporcionados."
    )


class QueryValidator:
    """Validates user queries for security risks.
    
    Detects and blocks:
    - Prompt injection attempts
    - Questions about system internals
    - Requests for private/sensitive information
    
    Example:
        validator = QueryValidator()
        is_safe, message = validator.validate("¿Qué dice el documento sobre X?")
        if not is_safe:
            return message  # Rejection message
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize validator with security configuration.
        
        Args:
            config: SecurityConfig instance (uses defaults if not provided)
        """
        self.config = config or SecurityConfig()
    
    def validate(self, query: str) -> Tuple[bool, Optional[str]]:
        """Validate a user query for security risks.
        
        Args:
            query: User's query string
            
        Returns:
            Tuple of (is_allowed, rejection_message)
            - is_allowed: True if query is safe, False if blocked
            - rejection_message: None if allowed, error message if blocked
        """
        if not self.config.enabled:
            return True, None
        
        query_lower = query.lower()
        
        # Check blocked patterns (regex)
        for pattern in self.config.blocked_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return False, self.config.rejection_message
        
        # Check blocked keywords
        for keyword in self.config.blocked_keywords:
            if keyword.lower() in query_lower:
                return False, self.config.rejection_message
        
        # Additional heuristics
        
        # Detect excessive special characters (possible injection)
        # Only check if query is long enough (> 10 chars)
        # Exempt queries that look like they're asking about programming (C++, C#, etc.)
        if len(query) > 10 and not re.search(r'\b[A-Z][+#]', query):
            special_char_ratio = sum(not c.isalnum() and not c.isspace() for c in query) / max(len(query), 1)
            if special_char_ratio > 0.35:  # >35% special chars (increased threshold)
                return False, self.config.rejection_message
        
        # Detect very long queries (possible injection)
        if len(query) > 2000:
            return False, "Tu pregunta es demasiado larga. Por favor, hazla más concisa."
        
        # All checks passed
        return True, None
    
    def is_allowed(self, query: str) -> bool:
        """Quick check if query is allowed.
        
        Args:
            query: User's query string
            
        Returns:
            True if allowed, False if blocked
        """
        is_allowed, _ = self.validate(query)
        return is_allowed
    
    def get_rejection_reason(self, query: str) -> Optional[str]:
        """Get rejection reason for a query.
        
        Args:
            query: User's query string
            
        Returns:
            Rejection message if blocked, None if allowed
        """
        _, message = self.validate(query)
        return message


def create_default_validator(enabled: bool = True) -> QueryValidator:
    """Create a validator with default security settings.
    
    Args:
        enabled: Whether to enable security filtering
        
    Returns:
        QueryValidator instance
    """
    config = SecurityConfig(enabled=enabled)
    return QueryValidator(config)
