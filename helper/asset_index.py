import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from llm.EmbeddingProviderFactory import EmbeddingProviderFactory, resolve_embedding_config
from llm.Enums import EmbeddingEnums
from llm.providers.LocalEmbeddingProvider import (
    LOCAL_MAX_SEQ_LENGTH,
    LOCAL_NORMALIZE_EMBEDDINGS,
    LOCAL_POOLING_MODE,
)
from .text_splitter import text_splitter


_SEPARATORS = ("\n\n", "\n", " ", "")
_cache_lock = RLock()
_client = None
_embeddings = {}
_indexes = {}


@dataclass(frozen=True)
class AssetIndex:
    asset_id: str
    index_version: str
    chunk_count: int
    vectorstore: Chroma

    def retrieve(self, question: str) -> list[Document]:
        documents = self.vectorstore.max_marginal_relevance_search(
            question,
            k=min(5, self.chunk_count),
            fetch_k=min(20, self.chunk_count),
            filter={"$and": [
                {"asset_id": self.asset_id},
                {"index_version": self.index_version},
            ]},
        )
        if any(
            document.metadata.get("asset_id") != self.asset_id
            or document.metadata.get("index_version") != self.index_version
            for document in documents
        ):
            raise RuntimeError("Retrieved chunks do not belong to the requested asset index")
        return documents


def _setting(config, name):
    return config.get(name) if isinstance(config, Mapping) else getattr(config, name, None)


def ensure_asset_index(asset_id: str, context: str, config) -> AssetIndex:
    global _client

    if not isinstance(asset_id, str) or not asset_id.strip():
        raise ValueError("An asset_id is required for question answering")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("The asset has no text to index")

    provider, model_id = resolve_embedding_config(config)
    chunk_size = _setting(config, "CHUNK_SIZE")
    chunk_overlap = _setting(config, "CHUNK_OVERLAP")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("CHUNK_SIZE must be a positive integer")
    if (
        isinstance(chunk_overlap, bool)
        or not isinstance(chunk_overlap, int)
        or not 0 <= chunk_overlap < chunk_size
    ):
        raise ValueError("CHUNK_OVERLAP must be an integer from zero to CHUNK_SIZE - 1")

    embedding_config = {"provider": provider, "model_id": model_id}
    if provider == EmbeddingEnums.LOCAL_EMBEDDING.value:
        embedding_config.update({
            "max_seq_length": LOCAL_MAX_SEQ_LENGTH,
            "pooling": LOCAL_POOLING_MODE,
            "normalize_embeddings": LOCAL_NORMALIZE_EMBEDDINGS,
        })
    else:
        embedding_config.update({
            "pipeline": "sentence-transformers-model-default",
            "encode_normalize_embeddings": False,
        })

    embedding_key = json.dumps(embedding_config, sort_keys=True)
    source_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()
    index_config = {
        "scheme": 1,
        "asset_id": asset_id,
        "source_hash": source_hash,
        "embedding": embedding_config,
        "splitter": {
            "type": "recursive_character",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "separators": _SEPARATORS,
        },
    }
    index_version = hashlib.sha256(
        json.dumps(index_config, sort_keys=True).encode("utf-8")
    ).hexdigest()

    # Publish only fully built indexes; retries upsert the same deterministic IDs.
    with _cache_lock:
        if index_version in _indexes:
            return _indexes[index_version]

        texts = text_splitter(
            context,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=list(_SEPARATORS),
        )
        if embedding_key not in _embeddings:
            _embeddings[embedding_key] = EmbeddingProviderFactory(config).create(provider, model_id)
        if _client is None:
            _client = chromadb.EphemeralClient(ChromaSettings(anonymized_telemetry=False))

        ids = []
        metadatas = []
        for chunk_index, text in enumerate(texts):
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            ids.append(hashlib.sha256(
                f"{index_version}:{chunk_index}:{content_hash}".encode("utf-8")
            ).hexdigest())
            metadatas.append({
                "asset_id": asset_id,
                "index_version": index_version,
                "chunk_index": chunk_index,
                "content_hash": content_hash,
                "source_hash": source_hash,
            })

        vectorstore = Chroma.from_texts(
            texts,
            _embeddings[embedding_key],
            ids=ids,
            metadatas=metadatas,
            collection_name=f"wizora_{index_version[:56]}",
            collection_metadata={
                "asset_id": asset_id,
                "index_version": index_version,
                "source_hash": source_hash,
            },
            client=_client,
        )
        index = AssetIndex(asset_id, index_version, len(texts), vectorstore)
        _indexes[index_version] = index
        return index
