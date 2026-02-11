import logging
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List

from embeddings.base import (
    BaseEmbedding,
    EmbeddingConfig,
    EmbeddingException,
)
from embeddings.factory import register_provider

logger = logging.getLogger(__name__)


@register_provider("hf-e5")
class HFMultilingualE5Embedding(BaseEmbedding):
    """
    Embedding provider using intfloat/multilingual-e5 from HuggingFace.
    
    This model requires prefixes:
    - "query: " for search queries
    - "passage: " for documents/chunks
    
    Provider name: "hf-e5"
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize HuggingFace E5 embedding provider.
        
        Args:
            config: Embedding configuration with model_name
        """
        # Set default model name if not provided
        if not config.model_name:
            config.model_name = "intfloat/multilingual-e5-large"
        
        super().__init__(config)
        logger.info(f"Initialized HFMultilingualE5Embedding with model: {config.model_name}")

    def _validate_provider(self) -> None:
        """Validate and load the HuggingFace model."""
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModel.from_pretrained(self.config.model_name)
            self.model.eval()

            # Infer real dimension from model
            with torch.no_grad():
                test = self.tokenizer(
                    ["query: test"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                output = self.model(**test)
                pooled = self._average_pool(
                    output.last_hidden_state,
                    test["attention_mask"],
                )
                inferred_dim = pooled.shape[1]
                
                # Update config dimension if different
                if self.config.dimension != inferred_dim:
                    logger.warning(
                        f"Config dimension ({self.config.dimension}) differs from model "
                        f"dimension ({inferred_dim}). Updating to {inferred_dim}."
                    )
                    self.config.dimension = inferred_dim

            logger.info(f"Model loaded successfully. Dimension: {self.config.dimension}")

        except Exception as e:
            logger.error(f"Failed to initialize HF E5 embedding provider: {e}")
            raise EmbeddingException(
                f"Failed to initialize HF E5 embedding provider: {e}"
            ) from e

    @staticmethod
    def _average_pool(
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply average pooling to token embeddings.
        
        Args:
            last_hidden_states: Output from transformer model
            attention_mask: Attention mask for valid tokens
            
        Returns:
            Pooled embeddings
        """
        masked = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embed_text(self, text: str, prefix: str = "passage") -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            prefix: Prefix type ("query" or "passage"). Default is "passage"
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            EmbeddingException: If text is empty or embedding fails
        """
        if not text or not text.strip():
            raise EmbeddingException("Cannot embed empty text")

        try:
            # E5 requires prefix
            prefixed_text = f"{prefix}: {text}"
            logger.debug(f"Embedding text with prefix '{prefix}': {text[:50]}...")

            batch = self.tokenizer(
                prefixed_text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                output = self.model(**batch)
                pooled = self._average_pool(
                    output.last_hidden_state,
                    batch["attention_mask"],
                )

                if self.config.normalize:
                    pooled = F.normalize(pooled, p=2, dim=1)
                    logger.debug("Applied L2 normalization")

            embedding = pooled[0].tolist()
            logger.debug(f"Generated embedding of dimension {len(embedding)}")
            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise EmbeddingException(
                f"Error generating embedding: {e}"
            ) from e
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        Uses "query: " prefix as required by E5 models.
        
        Args:
            query: Search query text
            
        Returns:
            Embedding vector
        """
        return self.embed_text(query, prefix="query")