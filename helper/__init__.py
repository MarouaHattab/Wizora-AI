import sys

from . import chunking as _chunking
from .chunking import split_documents, text_splitter
from .settings import (
    EmbeddingProvider,
    LLMProvider,
    ModelName,
    Settings,
    Temperature,
    get_settings,
    print_env_variables,
)

# Existing callers use this module path as well as the package-level function.
# Pre-register its alias so importing it cannot overwrite the public function.
sys.modules[__name__ + ".text_splitter"] = _chunking

__all__ = [
    "Settings",
    "get_settings",
    "print_env_variables",
    "ModelName",
    "Temperature",
    "LLMProvider",
    "EmbeddingProvider",
    "split_documents",
    "text_splitter",
]
