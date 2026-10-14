from collections.abc import Mapping
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings
from helper.Config import Settings
from .providers.LocalEmbeddingProvider import LocalEmbeddingProvider
from .Enums import EmbeddingEnums


def resolve_embedding_config(
    config: Settings | Mapping[str, Any],
    provider: str | None = None,
    model_name: str | None = None,
) -> tuple[str, str]:
    if isinstance(config, Mapping):
        configured_provider = config.get("EMBEDDING_MODEL_PROVIDER")
        configured_model = config.get("EMBEDDING_MODEL_ID")
    else:
        configured_provider = getattr(config, "EMBEDDING_MODEL_PROVIDER", None)
        configured_model = getattr(config, "EMBEDDING_MODEL_ID", None)

    effective_provider = provider if provider is not None else configured_provider
    if not isinstance(effective_provider, str) or not effective_provider.strip():
        raise ValueError("EMBEDDING_MODEL_PROVIDER must be a non-empty string.")
    effective_provider = effective_provider.strip().upper()
    supported_providers = {entry.value for entry in EmbeddingEnums}
    if effective_provider not in supported_providers:
        raise ValueError(
            "Unsupported embedding provider. Expected one of: "
            + ", ".join(sorted(supported_providers))
            + "."
        )

    effective_model = model_name if model_name is not None else configured_model
    if not isinstance(effective_model, str) or not effective_model.strip():
        raise ValueError("EMBEDDING_MODEL_ID must be a non-empty model identifier.")
    effective_model = effective_model.strip()
    if effective_model == "all-MiniLM-L6-v2":
        effective_model = "sentence-transformers/all-MiniLM-L6-v2"
    return effective_provider, effective_model


class EmbeddingProviderFactory:
    def __init__(self, config: Settings | Mapping[str, Any]):
        self.config = config

    def create(self, provider: str | None = None, model_name: str | None = None):
        provider, model_name = resolve_embedding_config(self.config, provider, model_name)
        if provider == EmbeddingEnums.LOCAL_EMBEDDING.value:
            return LocalEmbeddingProvider(model_id=model_name)
        return HuggingFaceEmbeddings(model_name=model_name)
