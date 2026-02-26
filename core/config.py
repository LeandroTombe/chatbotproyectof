"""
core.config — backward-compatibility shim.
All settings now live in `config.settings`. Use that module directly.
"""
from config.settings import settings  # noqa: F401

# Re-export the singleton so old code using `from core.config import settings` keeps working
__all__ = ["settings"]