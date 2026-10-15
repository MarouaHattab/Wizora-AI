from collections import OrderedDict
from collections.abc import Mapping
from threading import RLock
from typing import Any
from weakref import WeakValueDictionary

from langchain_huggingface import HuggingFaceEmbeddings

from helper.Config import Settings
from helper.ingestion_errors import ModelInitializationError
from .embedding_resource import EmbeddingResource
from .providers.LocalEmbeddingProvider import (
    LOCAL_MAX_SEQ_LENGTH,
    LOCAL_NORMALIZE_EMBEDDINGS,
    LOCAL_POOLING_MODE,
    LocalEmbeddingProvider,
)
from .Enums import EmbeddingEnums


_RESOURCE_CACHE_SIZE = 2
_resource_lock = RLock()
_resources: OrderedDict[tuple, EmbeddingResource] = OrderedDict()
_live_resources: WeakValueDictionary[tuple, EmbeddingResource] = WeakValueDictionary()


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

    def create(
        self, provider: str | None = None, model_name: str | None = None
    ) -> EmbeddingResource:
        provider, model_name = resolve_embedding_config(self.config, provider, model_name)
        local = provider == EmbeddingEnums.LOCAL_EMBEDDING.value
        pipeline = (
            ("transformer", LOCAL_MAX_SEQ_LENGTH, LOCAL_POOLING_MODE, LOCAL_NORMALIZE_EMBEDDINGS)
            if local else ("sentence-transformers-model-default", False)
        )
        key = (provider, model_name, pipeline)
        with _resource_lock:
            if key in _resources:
                _resources.move_to_end(key)
                return _resources[key]
            if key in _live_resources:
                resource = _live_resources[key]
                _resources[key] = resource
                while len(_resources) > _RESOURCE_CACHE_SIZE:
                    _resources.popitem(last=False)
                return resource
            try:
                if local:
                    resource = LocalEmbeddingProvider(model_id=model_name)
                else:
                    delegate = HuggingFaceEmbeddings(model_name=model_name)
                    resource = EmbeddingResource(
                        delegate._client,
                        provider=provider,
                        model_id=model_name,
                        pipeline="sentence-transformers-model-default",
                        normalize_embeddings=False,
                        delegate=delegate,
                    )
            except ModelInitializationError:
                raise
            except Exception as exc:
                raise ModelInitializationError("The embedding model could not be initialized.") from exc
            _resources[key] = resource
            _live_resources[key] = resource
            while len(_resources) > _RESOURCE_CACHE_SIZE:
                _resources.popitem(last=False)
            return resource
