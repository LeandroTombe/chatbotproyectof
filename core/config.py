"""
core.config — backward-compatibility shim.
All settings now live in `config.settings`. Use that module directly.
"""
from pathlib import Path
from config.settings import settings  # noqa: F401

# ChromaDB constants used by vectorstore/implementations/chroma.py
CHROMA_DIR: Path = Path(settings.CHROMA_PERSIST_DIRECTORY)
CHROMA_PERSIST: bool = True
CHROMA_COLLECTION_NAME: str = settings.CHROMA_COLLECTION_NAME

# Re-export the singleton so old code using `from core.config import settings` keeps working
__all__ = ["settings", "CHROMA_DIR", "CHROMA_PERSIST", "CHROMA_COLLECTION_NAME"]