import sys

from . import chat_factory as _chat_factory
from . import embedding_factory as _embedding_factory
from . import provider_types as _provider_types
from .chat_factory import LLMProviderFactory
from .embedding_factory import EmbeddingProviderFactory, resolve_embedding_config
from .embedding_resource import EmbeddingResource
from .provider_types import EmbeddingEnums, LLMEnums

# Preserve old module paths without letting imports replace the public classes.
# These modules contain lightweight factories; their SDK adapters remain lazy.
sys.modules[__name__ + ".LLMProviderFactory"] = _chat_factory
sys.modules[__name__ + ".EmbeddingProviderFactory"] = _embedding_factory
sys.modules[__name__ + ".Enums"] = _provider_types
Enums = _provider_types

__all__ = [
    "LLMProviderFactory",
    "EmbeddingProviderFactory",
    "resolve_embedding_config",
    "LLMEnums",
    "EmbeddingEnums",
    "EmbeddingResource",
]
