"""
Embedding providers.
Auto-imports all providers to register them with the factory.
"""

# Import all providers to auto-register them
try:
    from embeddings.providers.hf_e5_embedding import HFMultilingualE5Embedding
except ImportError:
    # HF E5 requires torch/transformers - optional dependency
    pass

__all__ = [
    "HFMultilingualE5Embedding",
]
